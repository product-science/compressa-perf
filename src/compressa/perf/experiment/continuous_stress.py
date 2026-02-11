# File: compressa/perf/experiment/continuous_stress.py

import socket
from urllib.parse import urlparse
import time
import threading
import random
import sqlite3
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from datetime import datetime

from compressa.perf.experiment.inference import InferenceRunner
from compressa.perf.experiment.chain_client import get_entrypoint_addr
from compressa.perf.experiment.analysis import Analyzer
from compressa.perf.data.models import (
    Measurement,
    Metric,
    Parameter,
    Status,
    MetricName,
)
from compressa.perf.db.operations import insert_measurement, insert_parameter, insert_metric
from compressa.utils import get_logger

logger = get_logger(__name__)


class ContinuousStressTestRunner:
    """
    Runs inference requests continuously using shared HTTP client pool.
    Every 'report_freq_min' minutes, it computes metrics on the last window 
    of measurements and stores them in DB with a suffix like "ttft_window_1", etc.
    """

    def __init__(
        self,
        db_path: str,
        node_url: str,
        model_name: str,
        account_address: str = None,
        private_key_hex: str = None,
        experiment_id: int = None,
        prompts: List[str] = None,
        num_runners: int = 10,
        max_tokens: int = 1000,
        report_freq_min: float = 1.0,
        seed: int = 42,
        no_sign: bool = False,
        old_sign: bool = False,
        account_pool: List[Tuple[str, str]] = None,
        transfer_address: str = None,
    ):
        self.db_path = db_path
        self.node_url = node_url
        self.model_name = model_name
        self.account_address = account_address
        self.private_key_hex = private_key_hex
        self.experiment_id = experiment_id
        self.prompts = prompts
        self.num_runners = num_runners
        self.max_tokens = max_tokens
        self.report_freq_sec = report_freq_min * 60
        self.running = True
        self.no_sign = no_sign
        self.old_sign = old_sign
        self.transfer_address = transfer_address

        self.experiment_start_ts = time.time()
        self.window_count = 1

        self.choice_generator = random.Random(seed)
        
        # Account pool setup for random selection
        self.account_pool = account_pool or []
        
        # Initialize runners pool
        self.runners = []
        self.runner_queue = queue.Queue()
        
        # 0. Resolve Node URL to IP to prevent DNS exhaustion
        resolved_node_url = self.node_url
        try:
            parsed = urlparse(self.node_url)
            if parsed.hostname:
                ip_addr = socket.gethostbyname(parsed.hostname)
                # Reconstruct URL with IP
                new_netloc = f"{ip_addr}:{parsed.port}" if parsed.port else ip_addr
                resolved_node_url = parsed._replace(netloc=new_netloc).geturl()
                logger.info(f"Resolved {self.node_url} to {resolved_node_url} to bypass DNS")
        except Exception as e:
            logger.warning(f"Failed to resolve node URL to IP: {e}")

        # Resolve entrypoint address once
        self.entrypoint_addr = ""
        if not self.no_sign:
            logger.info("Resolving entrypoint address...")
            try:
                self.entrypoint_addr = get_entrypoint_addr(self.node_url)
                logger.info(f"Entrypoint address: {self.entrypoint_addr}")
            except Exception as e:
                logger.warning(f"Failed to resolve entrypoint address: {e}")

        # Create independent runners
        logger.info(f"Creating {num_runners} independent runners for continuous stress test")
        
        for i in range(num_runners):
            # Determine account for this runner
            if self.account_pool and len(self.account_pool) > 0:
                # Round-robin assignment of accounts to runners
                acc_addr, acc_key = self.account_pool[i % len(self.account_pool)]
            else:
                acc_addr, acc_key = self.account_address, self.private_key_hex
            
            try:
                runner = InferenceRunner(
                    node_url=resolved_node_url,  # USE IP-BASED URL
                    entrypoint_addr=self.entrypoint_addr,
                    model_name=model_name,
                    account_address=acc_addr,
                    private_key_hex=acc_key,
                    no_sign=no_sign,
                    old_sign=old_sign,
                    transfer_address=self.transfer_address,
                )
                self.runners.append(runner)
                self.runner_queue.put(runner)
            except Exception as e:
                logger.error(f"Failed to create runner {i}: {e}")
                # Clean up already created runners
                for r in self.runners:
                    r.close()
                raise

    def start_test(self):
        """
        Launches two threads:
          1) A worker thread pool sending requests continuously
          2) A metrics thread computing windowed metrics every report_freq_sec
        """
        self.executor = ThreadPoolExecutor(max_workers=self.num_runners)
        
        self._store_continuous_params()

        t_infer = threading.Thread(
            target=self._continuous_inference_loop,
            daemon=True,
        )
        t_infer.start()

        t_metrics = threading.Thread(
            target=self._metrics_loop,
            daemon=True,
        )
        t_metrics.start()

        logger.info("Continuous stress test started (shared client pool enabled). Press Ctrl+C to stop.")

        # Keep main thread alive until user stops
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping continuous stress test.")
            self.running = False
            self.executor.shutdown(wait=False)
            # Clean up resources
            for runner in self.runners:
                runner.close()
    
    def _continuous_inference_loop(self):
        """
        Continuously schedule inference tasks in the thread pool.
        """
        while self.running:
            prompt = self.choice_generator.choice(self.prompts)
            self.executor.submit(self._do_inference_task, prompt)
            # Optimized for high throughput
            time.sleep(0.001)

    def _do_inference_task(self, prompt: str):
        """
        Single inference call. Stores the resulting measurement to DB.
        If the request fails, waits 5 seconds before allowing the thread to continue.
        """
        runner = None
        try:
            # Get an available runner from the queue (blocks until one is free)
            try:
                runner = self.runner_queue.get(timeout=10.0)
            except queue.Empty:
                logger.warning("Timed out waiting for a runner (all busy?)")
                return

            meas: Measurement = runner.run_inference(
                experiment_id=self.experiment_id,
                prompt=prompt,
                max_tokens=self.max_tokens,
            )
            
            insert_measurement(meas)
            
            # Check if the measurement indicates a failed request
            if meas.status == Status.FAILED:
                # logger.error("Request failed")
                # Add back off or just log?
                pass
                
        except Exception as e:
            logger.error(f"Inference task failed with exception: {e}")
            time.sleep(1.0)
        finally:
            # Return runner to the queue so it can be reused
            if runner:
                self.runner_queue.put(runner)

    def _metrics_loop(self):
        """
        Every 'report_freq_sec', compute metrics for the time window
        [start, end], store them in the DB (with a suffix), and log them.
        """
        while self.running:
            time.sleep(self.report_freq_sec)

            window_start = self.experiment_start_ts
            window_end = self.experiment_start_ts + self.window_count * self.report_freq_sec

            with sqlite3.connect(self.db_path) as conn:
                analyzer = Analyzer(conn)
                self._compute_and_store_window_metrics(
                    conn,
                    analyzer,
                    window_start,
                    window_end,
                    self.window_count,
                )
                self.window_count += 1

    def _compute_and_store_window_metrics(
        self,
        conn: sqlite3.Connection,
        analyzer: Analyzer,
        start_ts: float,
        end_ts: float,
        window_index: int,
    ):
        """
        Fetch measurements in [start_ts, end_ts], compute standard metrics via Analyzer,
        then store them with a suffix. Also logs them in real time.
        """
        cursor = conn.cursor()
        sql = """
            SELECT * FROM Measurements
             WHERE experiment_id = ?
               AND start_time >= ?
               AND end_time <= ?
        """
        cursor.execute(sql, (self.experiment_id, start_ts, end_ts))
        rows = cursor.fetchall()
        measurements = []
        for row in rows:
            measurements.append(
                Measurement(
                    id=row[0],
                    experiment_id=row[1],
                    n_input=row[2],
                    n_output=row[3],
                    ttft=row[4],
                    start_time=row[5],
                    end_time=row[6],
                    status=Status(row[7]),
                )
            )

        if not measurements:
            logger.info(f"No measurements found in window {window_index} ({int(start_ts)}-{int(end_ts)}).")
            return

        metrics_dict, io_stats = analyzer.compute_metrics_for_measurements(measurements)
        if not metrics_dict:
            logger.info(f"No valid metrics in window {window_index}. Possibly all failed.")
            return

        now = datetime.now()

        for base_name, value in metrics_dict.items():
            metric_name = f"{base_name}_window_{window_index}"
            metric = Metric(
                id=None,
                experiment_id=self.experiment_id,
                metric_name=metric_name,
                metric_value=value,
                timestamp=now
            )
            insert_metric(metric)

        for io_key, io_val in io_stats.items():
            param_name = f"{io_key}_window_{window_index}"
            param = Parameter(
                id=None,
                experiment_id=self.experiment_id,
                key=param_name,
                value=str(io_val)
            )
            insert_parameter(param)

        avg_ttft = metrics_dict.get(MetricName.TTFT.value, 0.0)
        avg_lat = metrics_dict.get(MetricName.LATENCY.value, 0.0)
        rps = metrics_dict.get(MetricName.RPS.value, 0.0)
        fails = metrics_dict.get(MetricName.FAILED_REQUESTS.value, 0.0)
        logger.info(f"[Window {window_index}] TTFT={avg_ttft:.3f}s, LAT={avg_lat:.3f}s, RPS={rps:.3f}, FAILS={fails}")

    def _store_continuous_params(self):
        """
        Store parameters about the continuous run.
        """
        param_list = [
            ("run_mode", "continuous"),
            ("num_workers", str(self.num_runners)),
            ("max_tokens", str(self.max_tokens)),
            ("report_freq_min", str(int(self.report_freq_sec // 60))),
            ("model_name", self.model_name),
            ("node_url", self.node_url),
            ("no_sign", str(self.no_sign)),
            ("old_sign", str(self.old_sign)),
            ("client_architecture", "independent_runner_pool"),
            ("account_random_selection_enabled", str(len(self.account_pool) > 1)),
            ("account_pool_size", str(len(self.account_pool))),
        ]

        if self.account_address:
            param_list.append(("initial_account_address", self.account_address))
        if self.account_pool:
            # Store all account addresses for reference
            addresses = [account[0] for account in self.account_pool]
            param_list.append(("account_pool_addresses", ",".join(addresses)))
        for k, v in param_list:
            p = Parameter(
                id=None,
                experiment_id=self.experiment_id,
                key=k,
                value=v
            )
            insert_parameter(p)
