# File: compressa/perf/experiment/continuous_stress.py

import time
import threading
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from datetime import datetime

from compressa.perf.experiment.inference import InferenceRunner
from compressa.perf.experiment.chain_client import OptimizedNodeClientManager
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

        self.experiment_start_ts = time.time()
        self.window_count = 1

        self.choice_generator = random.Random(seed)
        
        # Account pool setup for random selection
        self.account_pool = account_pool or []
        
        # Create client managers for each account in the pool
        self._account_client_managers = {}
        self._account_inference_runners = {}
        
        if self.account_pool and len(self.account_pool) > 1:
            # Create persistent client managers for each account
            num_clients = min(5, max(2, num_runners // 10))  # Fewer clients per account
            max_connections_per_client = 20  # Fewer connections per client
            
            logger.info(f"Creating {len(self.account_pool)} client managers for account pool with {num_clients} clients, {max_connections_per_client} connections each")
            
            for i, (acc_address, acc_private_key) in enumerate(self.account_pool):
                client_manager = OptimizedNodeClientManager(
                    node_url=node_url,
                    account_address=acc_address,
                    private_key_hex=acc_private_key,
                    no_sign=no_sign,
                    old_sign=old_sign,
                    num_clients=num_clients,
                    max_connections_per_client=max_connections_per_client,
                )
                
                inference_runner = InferenceRunner(
                    shared_client_manager=client_manager,
                    model_name=self.model_name,
                )
                
                self._account_client_managers[acc_address] = client_manager
                self._account_inference_runners[acc_address] = inference_runner
                
                logger.info(f"Created client manager {i+1}/{len(self.account_pool)} for account {acc_address}")
        else:
            # Create single shared client manager for single account
            num_clients = min(10, max(3, num_runners // 20))  # 3-10 clients based on runner count
            max_connections_per_client = 50
            
            logger.info(f"Creating shared client manager with {num_clients} clients, {max_connections_per_client} connections each for {num_runners} runners")
            
            self._shared_client_manager = OptimizedNodeClientManager(
                node_url=node_url,
                account_address=account_address,
                private_key_hex=private_key_hex,
                no_sign=no_sign,
                old_sign=old_sign,
                num_clients=num_clients,
                max_connections_per_client=max_connections_per_client,
            )

    def start_test(self):
        """
        Launches two threads:
          1) A worker thread pool sending requests continuously
          2) A metrics thread computing windowed metrics every report_freq_sec
        """
        self.executor = ThreadPoolExecutor(max_workers=self.num_runners)
        
        # Create inference runner with shared client manager (only for single account mode)
        if not (self.account_pool and len(self.account_pool) > 1):
            self.inference_runner = InferenceRunner(
                shared_client_manager=self._shared_client_manager,
                model_name=self.model_name,
            )

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
            if hasattr(self, '_shared_client_manager'):
                self._shared_client_manager.close_all()
            if hasattr(self, '_account_client_managers'):
                for client_manager in self._account_client_managers.values():
                    client_manager.close_all()
    
    def _get_random_account(self) -> Tuple[str, str]:
        """
        Get a random account from the pool.
        If no pool exists, return the default account.
        Thread-safe.
        """
        if self.account_pool and len(self.account_pool) > 1:
            # Return a random account from the pool
            return self.choice_generator.choice(self.account_pool)
        elif self.account_pool and len(self.account_pool) == 1:
            # Single account in pool
            return self.account_pool[0]
        else:
            # No pool, use default account
            return self.account_address, self.private_key_hex
    


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
        Randomly selects an account from the pool for each request.
        """
        try:
            # Get a random account for this request
            current_account_address, current_private_key = self._get_random_account()
            
            # If we have multiple accounts, use the persistent client manager for that account
            if self.account_pool and len(self.account_pool) > 1:
                # Use the persistent inference runner for this account
                inference_runner = self._account_inference_runners[current_account_address]
                meas: Measurement = inference_runner.run_inference(
                    experiment_id=self.experiment_id,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                )
            else:
                # Use shared client manager for single account
                meas: Measurement = self.inference_runner.run_inference(
                    experiment_id=self.experiment_id,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                )
            
            insert_measurement(meas)
            
            # Check if the measurement indicates a failed request
            if meas.status == Status.FAILED:
                logger.error(f"Request failed (HTTP/connection error) using account {current_account_address}, waiting 5 seconds before next attempt")
                time.sleep(5.0)
                
        except Exception as e:
            logger.error(f"Inference task failed with exception: {e}, waiting 5 seconds before next attempt")
            time.sleep(5.0)

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
            ("client_architecture", "persistent_pool" if len(self.account_pool) > 1 else "shared_pool"),
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
