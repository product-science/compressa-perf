import time
import os
import json
import logging
import openai
import httpx
import requests
from typing import List, Dict

from compressa.perf.data.models import (
    Measurement,
    Parameter,
    Status,
)
from compressa.perf.db.operations import (
    insert_measurement,
    insert_parameter,
)
from compressa.utils import get_logger, stream_chat
from compressa.perf.experiment.chain_client import (
    _NodeClient,
    get_entrypoint_addr,
    managed_stream_response,
)
from compressa.perf.experiment.utils import SlidingWindowRateLimiter

import socket
from urllib.parse import urlparse

import sqlite3
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
import queue
import random
import threading
from tqdm import tqdm

logger = get_logger(__name__)


class CancelledError(Exception):
    """Raised when a task is cancelled via the cancel_event."""
    pass


class InferenceRunner:
    def __init__(
        self,
        node_url: str,
        entrypoint_addr: str,
        model_name: str,
        account_address: str = None,
        private_key_hex: str = None,
        no_sign: bool = False,
        old_sign: bool = False,
        host_header: str = None,
        transfer_address: str = None,
    ) -> None:
        self.model_name = model_name
        
        # Create private client for this runner
        # We only need 1-2 connections per runner since it's single-threaded
        self._client = _NodeClient(
            node_url=node_url,
            entrypoint_addr=entrypoint_addr,
            transfer_address=transfer_address,
            account_address=account_address,
            private_key_hex=private_key_hex,
            timeout=600.0,
            max_connections=2,  # Minimal pool for single thread
            max_connections_per_host=2,
            max_retries=3,
            backoff_factor=0.5,
            no_sign=no_sign,
            old_sign=old_sign,
            host_header=host_header,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the private client"""
        if hasattr(self, '_client'):
            self._client.close()

    # ---------------------------------------------------------------------
    # Public
    # ---------------------------------------------------------------------
    def run_inference(
        self,
        experiment_id: int,
        prompt: str,
        max_tokens: int,
        cancel_event: threading.Event = None,
    ) -> Measurement:
        # Check for cancellation before starting
        if cancel_event and cancel_event.is_set():
            raise CancelledError("Task cancelled before start")
        
        # Timing instrumentation
        t0 = time.time()
        timings = {}
        
        start_time = time.time()
        first_token_time = -1.0
        ttft = 0.0
        n_chunks = 0
        n_input = 0
        n_output = 0
        response_text = ""
        status = Status.SUCCESS

        try:
            # Phase 1: Use private client (no contention)
            client = self._client
            timings['get_client'] = 0.0
            
            # Phase 2: Send request
            t1 = time.time()
            resp = client.stream_chat_completion(
                messages=[
                    {"role": "system", "content": """You are a science journalist. Your job is to write VERY LONG, detailed blog posts explaining medical research to general audiences.

TASK: You will receive 10 medical papers separated by "########## Paper X #########". Write a SEPARATE 1500-word article for EACH paper. Total output must be approximately 15,000 words.

MANDATORY OUTPUT STRUCTURE - REPEAT THIS EXACTLY 10 TIMES:

================================================================================
ARTICLE [N] OF 10: [Headline]
================================================================================

SECTION 1 - THE DISCOVERY (400 words minimum)
Write 4 detailed paragraphs explaining what researchers found. Use simple language. Include specific numbers and results from the paper.

SECTION 2 - WHY THIS MATTERS TO YOU (300 words minimum)
Write 3 paragraphs about real-world impact. Give concrete examples of how this affects ordinary people's lives.

SECTION 3 - THE SCIENCE EXPLAINED (400 words minimum)
Write 4 paragraphs going deeper into methodology. Explain HOW they did the research. Use analogies.

SECTION 4 - EXPERT CONTEXT (200 words minimum)
Write 2 paragraphs placing this in broader scientific context.

SECTION 5 - WHAT COMES NEXT (200 words minimum)
Write 2 paragraphs about future research directions and unanswered questions.

[End with separator line: ════════════════════════════════════════════════════════════════════════════════]

CRITICAL RULES:
- You MUST write ALL 10 articles. Count them: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.
- Each article MUST be 1500+ words. Do NOT write short summaries.
- Do NOT stop early. Do NOT say "I'll continue with the remaining papers" - just write them.
- Do NOT skip papers. Do NOT combine papers.
- After finishing Article 10, write "=== END OF ALL 10 ARTICLES ===" """},
                    {"role": "user", "content": prompt + """

INSTRUCTIONS: Above are 10 scientific papers. Write a detailed 1500-word blog article for EACH paper.

Start now with "ARTICLE 1 OF 10:" and continue through "ARTICLE 10 OF 10:". Do not stop until you have written all 10 complete articles. Your response should be approximately 15,000 words total."""}
                ],
                model=self.model_name,
                max_tokens=max_tokens,
            )
            timings['send_request'] = time.time() - t1

            # Phase 3: Stream response
            t2 = time.time()
            # Use context manager for proper resource cleanup
            with managed_stream_response(resp) as response:
                for raw_line in response.iter_lines(decode_unicode=True):
                    # Check for cancellation during streaming
                    if cancel_event and cancel_event.is_set():
                        raise CancelledError("Task cancelled during streaming")
                    
                    if not raw_line:
                        continue

                    if raw_line.startswith("data:"):
                        raw_line = raw_line[len("data:"):].strip()

                    if raw_line == "[DONE]":
                        break

                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse JSON: %s", raw_line)
                        continue

                    usage = chunk.get("usage")
                    if usage:  # present on final chunk
                        n_input = usage.get("prompt_tokens", 0)
                        n_output = usage.get("completion_tokens", 0)

                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content") if chunk.get("choices") else None
                    )
                    # Lazy logging optimization
                    # logger.debug("Delta: %s", delta) 
                    if delta is not None:
                        if first_token_time < 0:
                            first_token_time = time.time()
                            ttft = first_token_time - start_time
                        response_text += delta
                        n_chunks += 1
                        
                        # Print progress every 500 tokens
                        if n_chunks % 500 == 0:
                            print(f"\n[{threading.current_thread().name}] === {n_chunks} tokens streamed ===")
                            # Print last ~500 chars of response to show recent content
                            recent = response_text[-1000:] if len(response_text) > 1000 else response_text
                            print(recent)
                            print("..." if len(response_text) > 1000 else "")

            timings['stream_response'] = time.time() - t2

            if n_chunks == 0:
                raise RuntimeError("No content chunks received – server returned empty stream")

            end_time = time.time()
            timings['total'] = end_time - t0
            
            # Log timing breakdown
            logger.debug(
                "[Thread %s] Request timing: get_client=%.3fs, send=%.3fs, stream=%.3fs, total=%.3fs",
                threading.current_thread().name,
                timings['get_client'],
                timings['send_request'],
                timings['stream_response'],
                timings['total']
            )
            
            # Log token usage
            logger.info(
                "[Thread %s] Token usage: input=%d, output=%d",
                threading.current_thread().name,
                n_input,
                n_output,
            )
            
            # Print full response for debugging (unescaped)
            print("\n" + "=" * 80)
            print(f"[Thread {threading.current_thread().name}] FULL RESPONSE ({n_output} tokens):")
            print("=" * 80)
            print(response_text)
            print("=" * 80 + "\n")
            
            logger.debug(
                "Prompt:%s\nResponse text:%s\n%s",
                prompt,
                response_text,
                "#" * 40,
            )

            return Measurement(
                id=None,
                experiment_id=experiment_id,
                n_input=n_input,
                n_output=n_output,
                ttft=ttft,
                start_time=start_time,
                end_time=end_time,
                status=status,
            )

        except requests.exceptions.ConnectionError as exc:
            logger.error(
                "Connection error: %s (chunks=%s, ttft=%.3fs) - consider reducing concurrency", 
                exc, n_chunks, ttft
            )
            end_time = time.time()
            return Measurement.failed(
                experiment_id=experiment_id,
                n_input=n_input,
                n_output=n_output,
                ttft=ttft,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as exc:
            logger.error(
                "API request failed: %s (chunks=%s, ttft=%.3fs)", exc, n_chunks, ttft
            )
            end_time = time.time()
            return Measurement.failed(
                experiment_id=experiment_id,
                n_input=n_input,
                n_output=n_output,
                ttft=ttft,
                start_time=start_time,
                end_time=end_time,
            )


class ExperimentRunner:
    def __init__(
        self,
        node_url: str,
        model_name: str,
        account_address: str = None,
        private_key_hex: str = None,
        num_runners: int = 10,
        no_sign: bool = False,
        old_sign: bool = False,
        transfer_address: str = None,
    ) -> None:
        self.node_url = node_url
        self.model_name = model_name
        self.account_address = account_address
        self.private_key_hex = private_key_hex
        self.num_runners = num_runners
        self.no_sign = no_sign
        self.old_sign = old_sign
        self.transfer_address = transfer_address

    def _store_experiment_parameters(
        self,
        experiment_id: int,
        num_tasks: int,
        max_tokens: int,
        rate_limit_requests: int,
        rate_limit_window: float,
    ) -> None:
        params = [
            ("num_workers", str(self.num_runners)),
            ("num_tasks", str(num_tasks)),
            ("node_url", self.node_url),
            ("max_tokens", str(max_tokens)),
            ("model_name", self.model_name),
            ("no_sign", str(self.no_sign)),
            ("old_sign", str(self.old_sign)),
            ("client_architecture", "independent_runners"),
            ("rate_limit_requests", str(rate_limit_requests)),
            ("rate_limit_window", str(rate_limit_window)),
        ]

        if self.account_address:
            params.append(("requester_address", self.account_address))
        for key, value in params:
            insert_parameter(
                Parameter(id=None, experiment_id=experiment_id, key=key, value=value)
            )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run_experiment(
        self,
        *,
        experiment_id: int,
        prompts: List[str],
        num_tasks: int = 100,
        max_tokens: int = 1000,
        seed: int = 42,
        over_schedule_factor: float = 0.5,
        rate_limit_requests: int = 500,
        rate_limit_window: float = 5.0,
    ) -> None:

        rng = random.Random(seed)
        all_measurements: List[Measurement] = []
        cancel_event = threading.Event()

        # Calculate buffer for over-scheduling to eliminate tail latency
        buffer = max(self.num_runners, int(num_tasks * over_schedule_factor))
        scheduled_tasks = num_tasks + buffer
        logger.info(
            "Over-scheduling: %d tasks (%d needed + %d buffer)",
            scheduled_tasks, num_tasks, buffer
        )

        # 0. Resolve Node URL to IP to prevent DNS exhaustion
        resolved_node_url = self.node_url


        # 1. Resolve entrypoint address ONCE
        entrypoint_addr = ""
        if not self.no_sign:
            logger.info("Resolving entrypoint address...")
            entrypoint_addr = get_entrypoint_addr(self.node_url)
            logger.info("Entrypoint address: %s", entrypoint_addr)

        # 2. Create independent runners
        logger.info("Initializing %d independent runners...", self.num_runners)
        runners = []
        try:
            for i in range(self.num_runners):
                runner = InferenceRunner(
                    node_url=resolved_node_url,  # USE IP-BASED URL
                    entrypoint_addr=entrypoint_addr,
                    model_name=self.model_name,
                    account_address=self.account_address,
                    private_key_hex=self.private_key_hex,
                    no_sign=self.no_sign,
                    old_sign=self.old_sign,
                    transfer_address=self.transfer_address,
                )
                runners.append(runner)
        except Exception as e:
            logger.error("Failed to initialize runners: %s", e)
            for r in runners:
                r.close()
            raise

        # Manual executor management for early shutdown control
        executor = ThreadPoolExecutor(max_workers=self.num_runners)

        try:
            # Rate limiter to control request submission rate
            rate_limiter = SlidingWindowRateLimiter(
                max_requests=rate_limit_requests,
                window_seconds=rate_limit_window,
            )
            logger.info(
                "Rate limiting: max %d requests per %.1f seconds",
                rate_limit_requests, rate_limit_window
            )

            # Thread-safe queue for futures
            futures_queue = queue.Queue()
            submission_done = threading.Event()

            def submit_tasks():
                """Background thread for rate-limited task submission."""
                for i in range(scheduled_tasks):
                    if cancel_event.is_set():
                        break
                    rate_limiter.acquire()
                    future = executor.submit(
                        runners[i % self.num_runners].run_inference,
                        experiment_id,
                        rng.choice(prompts),
                        max_tokens,
                        cancel_event,
                    )
                    futures_queue.put(future)
                submission_done.set()

            # Start background submission thread
            submit_thread = threading.Thread(target=submit_tasks, daemon=True)
            submit_thread.start()

            # Track progress as results come in
            pending_futures = []
            with tqdm(total=num_tasks, desc="Running experiments") as pbar:
                while len(all_measurements) < num_tasks:
                    # Collect new futures from submission thread
                    while True:
                        try:
                            f = futures_queue.get_nowait()
                            pending_futures.append(f)
                        except queue.Empty:
                            break

                    # Check if any futures are done
                    still_pending = []
                    for f in pending_futures:
                        if f.done():
                            try:
                                measurement = f.result()
                                all_measurements.append(measurement)
                                pbar.update(1)
                            except CancelledError:
                                pass
                            except Exception as exc:
                                logger.error("Task failed: %s", exc)
                        else:
                            still_pending.append(f)
                    pending_futures = still_pending

                    # Exit if submission is done and no pending futures
                    if submission_done.is_set() and not pending_futures:
                        break

                    # Small sleep to avoid busy-waiting
                    time.sleep(0.01)

            # Wait for submission thread to finish
            submit_thread.join(timeout=1.0)

        finally:
            # Signal cancellation to running tasks
            cancel_event.set()

            # Cancel pending futures and shut down
            # cancel_futures=True requires Python 3.9+
            executor.shutdown(wait=False, cancel_futures=True)

            # Log summary
            logger.info("=" * 60)
            logger.info("=== EXPERIMENT SUMMARY ===")
            logger.info("=" * 60)
            
            # Experiment summary
            logger.info("  Requested tasks: %d", num_tasks)
            logger.info("  Scheduled tasks (with buffer): %d", scheduled_tasks)
            logger.info("  Completed measurements: %d", len(all_measurements))
            logger.info("  Num runners configured: %d", self.num_runners)
            logger.info("=" * 60)
            
            # Clean up all clients
            for runner in runners:
                runner.close()

        # Persist metadata & results --------------------------------------
        self._store_experiment_parameters(
            experiment_id, num_tasks, max_tokens,
            rate_limit_requests, rate_limit_window
        )
        for m in all_measurements[:num_tasks]:
            insert_measurement(m)

        failed_count = len([m for m in all_measurements if m.status == Status.FAILED])
        logger.info("Number of failed measurements: %d", failed_count)
