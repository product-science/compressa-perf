import time
import os
import json
import logging
import hashlib
import threading
import openai
import httpx
import requests
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Union

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
    OptimizedNodeClientManager,
    managed_stream_response,
)

import sqlite3
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
import random
import math
from tqdm import tqdm

logger = get_logger(__name__)
_MALFORMED_STREAM_CHUNK_WRITE_LOCK = threading.Lock()

ChatMessage = Dict[str, str]
PromptInput = Union[str, List[ChatMessage]]


def _shared_pool_config(num_runners: int) -> Tuple[int, int]:
    """
    Size the shared aiohttp pool to avoid the old 500-connection ceiling.

    We keep 64 connections per client and scale the number of shared clients so
    the total configured pool tracks runner count up to 1024 connections.
    """
    max_connections_per_client = 64
    target_total_connections = min(1024, max(320, num_runners))
    num_clients = max(3, min(16, math.ceil(target_total_connections / max_connections_per_client)))
    return num_clients, max_connections_per_client


@dataclass
class InferenceResult:
    measurement: Measurement
    failure_code: Optional[str] = None


def _json_log(event: str, **fields) -> str:
    payload = {"event": event, **fields}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def _malformed_stream_chunk_output_path(experiment_id: int) -> str:
    output_dir = os.getenv("COMPRESSA_PERF_MALFORMED_STREAM_DIR")
    filename = f"malformed_stream_chunks_experiment_{experiment_id}.jsonl"
    if output_dir:
        return os.path.join(output_dir, filename)
    return os.path.join(os.getcwd(), filename)


def _persist_malformed_stream_chunk(
    *,
    experiment_id: int,
    task_id: int,
    runner_id: int,
    thread_id: int,
    client_index: Optional[int],
    prompt_hash: str,
    model_name: str,
    target_url: str,
    status_code: Optional[int],
    response_headers: Optional[Dict[str, str]],
    server_request_id: Optional[str],
    error: str,
    raw_wire_line: str,
    raw_payload: str,
) -> Optional[str]:
    output_path = _malformed_stream_chunk_output_path(experiment_id)
    output_dir = os.path.dirname(output_path) or "."
    record = {
        "logged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_id": experiment_id,
        "task_id": task_id,
        "runner_id": runner_id,
        "thread_id": thread_id,
        "client_index": client_index,
        "prompt_hash": prompt_hash,
        "model_name": model_name,
        "target_url": target_url,
        "status_code": status_code,
        "response_headers": response_headers,
        "server_request_id": server_request_id,
        "error": error,
        "raw_wire_line": raw_wire_line,
        "raw_payload": raw_payload,
    }
    try:
        os.makedirs(output_dir, exist_ok=True)
        with _MALFORMED_STREAM_CHUNK_WRITE_LOCK:
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return output_path
    except OSError as file_exc:
        logger.warning(
            "Failed to persist malformed stream chunk for request_id=%s to %s: %s",
            server_request_id,
            output_path,
            file_exc,
        )
        return None


def _prompt_fingerprint(prompt: PromptInput) -> str:
    normalized = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def _extract_root_cause(exc: Exception) -> Exception:
    current = exc
    while True:
        next_exc = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if next_exc is None:
            return current
        current = next_exc


def _extract_errno(exc: Exception) -> Optional[int]:
    root = _extract_root_cause(exc)
    errno = getattr(root, "errno", None)
    if errno is not None:
        return errno
    for arg in getattr(root, "args", ()):
        if isinstance(arg, int):
            return arg
    return None


def _extract_response_metadata(response: Optional[requests.Response]) -> Dict[str, Optional[Union[int, str, Dict[str, str]]]]:
    if response is None:
        return {
            "status_code": None,
            "response_headers": None,
            "server_request_id": None,
        }

    # Lowercase set for case-insensitive matching. aiohttp stores headers in a
    # CIMultiDict that we flatten to a plain dict when wrapping the response,
    # which loses case-insensitivity. Servers and proxies send this header
    # under any of "x-request-id", "X-Request-Id", "X-Request-ID", etc.
    interesting_headers = {
        "server",
        "connection",
        "keep-alive",
        "via",
        "x-request-id",
        "retry-after",
        "content-type",
        "transfer-encoding",
    }

    try:
        header_items = list(response.headers.items())
    except Exception:
        header_items = []

    headers: Dict[str, str] = {}
    server_request_id: Optional[str] = None
    for raw_key, value in header_items:
        if value is None:
            continue
        lower_key = str(raw_key).lower()
        if lower_key in interesting_headers:
            headers.setdefault(lower_key, str(value))
        if lower_key == "x-request-id" and server_request_id is None:
            server_request_id = str(value)

    return {
        "status_code": getattr(response, "status_code", None),
        "response_headers": headers or None,
        "server_request_id": server_request_id,
    }


def _extract_received_body_metrics(response: Optional[requests.Response]) -> Dict[str, Optional[int]]:
    """Return body-byte counters collected by the transport wrapper.

    ``http_body_bytes_received`` counts decoded HTTP response-body bytes as
    observed by the client before line splitting/CRLF stripping. This is the
    closest client-side analogue to the proxy's ``bytes_written`` metric.

    ``sse_payload_bytes_received`` counts the SSE payload bytes actually
    delivered to the application loop after trimming trailing CRLF from each
    yielded line.
    """
    if response is None:
        return {
            "http_body_bytes_received": None,
            "sse_payload_bytes_received": None,
        }
    return {
        "http_body_bytes_received": getattr(response, "http_body_bytes_received", None),
        "sse_payload_bytes_received": getattr(response, "sse_payload_bytes_received", None),
    }


def _pool_counters(snapshot: Optional[Dict[str, Union[int, str, Dict[str, str]]]]) -> Dict[str, int]:
    if not snapshot:
        return {
            "num_connections": 0,
            "num_requests": 0,
            "idle_connections": 0,
        }
    return {
        "num_connections": int(snapshot.get("num_connections", 0) or 0),
        "num_requests": int(snapshot.get("num_requests", 0) or 0),
        "idle_connections": int(snapshot.get("idle_connections", 0) or 0),
    }


def _pool_delta(before: Optional[Dict[str, Union[int, str, Dict[str, str]]]], after: Optional[Dict[str, Union[int, str, Dict[str, str]]]]) -> Dict[str, int]:
    before_counters = _pool_counters(before)
    after_counters = _pool_counters(after)
    return {
        "new_tcp_connections_estimate": max(0, after_counters["num_connections"] - before_counters["num_connections"]),
        "new_http_requests_seen": max(0, after_counters["num_requests"] - before_counters["num_requests"]),
        "idle_connection_delta": after_counters["idle_connections"] - before_counters["idle_connections"],
    }


def _pool_totals_delta(before: Optional[Dict[str, int]], after: Optional[Dict[str, int]]) -> Dict[str, int]:
    before_connections = int((before or {}).get("num_connections", 0) or 0)
    before_requests = int((before or {}).get("num_requests", 0) or 0)
    after_connections = int((after or {}).get("num_connections", 0) or 0)
    after_requests = int((after or {}).get("num_requests", 0) or 0)
    return {
        "new_tcp_connections": max(0, after_connections - before_connections),
        "new_http_requests": max(0, after_requests - before_requests),
    }


def _classify_request_failure(exc: Exception, n_chunks: int) -> str:
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) is not None:
        return f"http_{response.status_code}"

    message = str(exc).lower()
    errno = _extract_errno(exc)
    if errno == 61 or "connection refused" in message:
        return "connect_refused"
    if "connect timeout" in message or "connection timed out" in message:
        return "connect_timeout"
    if "read timed out" in message:
        return "read_timeout"
    if "no content chunks received" in message or (n_chunks == 0 and "empty stream" in message):
        return "empty_stream"
    if isinstance(exc, json.JSONDecodeError):
        return "json_parse_error"
    # aiohttp payload / transfer errors (response body could not be read fully).
    # We catch by type name to avoid a hard dependency on aiohttp being importable here.
    exc_type_name = type(exc).__name__
    if exc_type_name in {
        "ClientPayloadError",
        "TransferEncodingError",
        "InvalidChunkSize",
        "ContentLengthError",
    }:
        return "client_payload_error"
    return exc_type_name.lower()


def _parse_sse_line(raw_line: Union[bytes, str]) -> Dict[str, Optional[Union[str, Dict]]]:
    """Best-effort parse of one SSE line captured during streaming.

    Returns a dict with the decoded text, whether it was a ``data:`` line, and
    the parsed JSON payload (if it was valid JSON). Intended for logging when
    the stream fails mid-payload so we can see what the server actually sent.
    """
    if isinstance(raw_line, bytes):
        try:
            decoded = raw_line.decode("utf-8", errors="replace")
        except Exception:
            decoded = repr(raw_line)
    else:
        decoded = raw_line

    stripped = decoded.strip()
    if not stripped:
        return {"raw": decoded, "is_data": False, "parsed": None}

    is_data = stripped.startswith("data:")
    body = stripped[len("data:"):].strip() if is_data else stripped
    if is_data and body == "[DONE]":
        return {"raw": decoded, "is_data": True, "parsed": "[DONE]"}

    parsed: Optional[Union[str, Dict]] = None
    if is_data:
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = None

    return {"raw": decoded, "is_data": is_data, "parsed": parsed}


def _summarize_partial_payload(
    raw_lines: Deque[Union[bytes, str]],
    response_text: str,
    max_text_chars: int = 500,
    max_parsed_chunks: int = 4,
) -> Dict[str, object]:
    """Build a compact, log-friendly summary of whatever body we received.

    Keeps up to 8 raw SSE lines (the whole ``raw_lines`` deque) for transport
    framing visibility, but only emits the last ``max_parsed_chunks`` parsed
    JSON chunks to keep log lines tractable — the parsed versions are a
    redundant representation of the raw lines, so the trailing few are all
    we need for content/finish_reason inspection.
    """
    parsed_lines = [_parse_sse_line(line) for line in raw_lines]
    truncated_text = response_text[-max_text_chars:] if response_text else ""
    return {
        "partial_response_text_tail": truncated_text,
        "partial_response_text_len": len(response_text or ""),
        "last_raw_lines": [entry["raw"] for entry in parsed_lines],
        "last_parsed_chunks": [entry["parsed"] for entry in parsed_lines[-max_parsed_chunks:]],
    }


class InferenceRunner:
    def __init__(
        self,
        shared_client_manager: OptimizedNodeClientManager,
        model_name: str,
    ) -> None:
        self.model_name = model_name
        self._shared_client_manager = shared_client_manager

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't close the shared client manager - it's owned by ExperimentRunner
        pass

    def close(self):
        """No-op since we use shared client manager"""
        pass

    # ---------------------------------------------------------------------
    # Public
    # ---------------------------------------------------------------------
    def run_inference(
        self,
        experiment_id: int,
        task_id: int,
        runner_id: int,
        prompt: PromptInput,
        max_tokens: int,
        min_tokens: int = None,
    ) -> InferenceResult:
        start_time = time.time()
        request_started_perf = time.perf_counter()
        first_token_time = -1.0
        first_token_perf = None
        ttft = 0.0
        n_chunks = 0
        n_input = 0
        n_output = 0
        response_text = ""
        status = Status.SUCCESS
        prompt_hash = _prompt_fingerprint(prompt)
        response = None
        pool_snapshot_before = None
        # Ring buffer of the most recent raw SSE lines. Used for diagnostics when
        # the stream aborts mid-payload (e.g. aiohttp ClientPayloadError).
        recent_raw_lines: Deque[Union[bytes, str]] = deque(maxlen=8)

        try:
            # Get a client from the shared pool
            client_select_started = time.perf_counter()
            client_index, client = self._shared_client_manager.get_client_with_index()
            client_selected_perf = time.perf_counter()
            pool_snapshot_before = client.get_pool_snapshot()

            messages = (
                prompt
                if isinstance(prompt, list)
                else [{"role": "user", "content": prompt}]
            )

            logger.debug(
                _json_log(
                    "request_start",
                    experiment_id=experiment_id,
                    task_id=task_id,
                    runner_id=runner_id,
                    thread_id=threading.get_ident(),
                    client_index=client_index,
                    prompt_hash=prompt_hash,
                    prompt_message_count=len(messages),
                    target_url=client.get_target_url(),
                )
            )

            response = client.stream_chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
            )
            headers_received_perf = time.perf_counter()
            response_metadata = _extract_response_metadata(response)
            response_body_metrics = _extract_received_body_metrics(response)
            pool_snapshot_after_headers = client.get_pool_snapshot()
            pool_delta_after_headers = _pool_delta(pool_snapshot_before, pool_snapshot_after_headers)

            # Use context manager for proper resource cleanup
            with managed_stream_response(response) as streamed_response:
                for raw_line in streamed_response.iter_lines(decode_unicode=False):
                    if not raw_line:
                        continue

                    # Keep the original wire bytes/string for post-mortem logging.
                    recent_raw_lines.append(raw_line)
                    raw_wire_line = (
                        raw_line.decode("utf-8", errors="replace")
                        if isinstance(raw_line, bytes)
                        else str(raw_line)
                    )

                    if isinstance(raw_line, bytes):
                        raw_line = raw_line.decode("utf-8")

                    if raw_line.startswith("data:"):
                        raw_line = raw_line[len("data:"):].strip()

                    if raw_line == "[DONE]":
                        break

                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        saved_path = _persist_malformed_stream_chunk(
                            experiment_id=experiment_id,
                            task_id=task_id,
                            runner_id=runner_id,
                            thread_id=threading.get_ident(),
                            client_index=client_index,
                            prompt_hash=prompt_hash,
                            model_name=self.model_name,
                            target_url=client.get_target_url(),
                            status_code=response_metadata.get("status_code"),
                            response_headers=response_metadata.get("response_headers"),
                            server_request_id=response_metadata.get("server_request_id"),
                            error=str(exc),
                            raw_wire_line=raw_wire_line,
                            raw_payload=raw_line,
                        )
                        logger.warning(
                            "Failed to parse JSON for request_id=%s%s: %s\nFull raw chunk:\n%s",
                            response_metadata.get("server_request_id"),
                            f"; saved to {saved_path}" if saved_path else "",
                            exc,
                            raw_line,
                        )
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
                    logger.debug(f"Delta: {delta}")
                    if delta is not None:
                        if first_token_time < 0:
                            first_token_time = time.time()
                            first_token_perf = time.perf_counter()
                            ttft = first_token_time - start_time
                        response_text += delta
                        n_chunks += 1

            if n_chunks == 0:
                raise RuntimeError("No content chunks received – server returned empty stream")

            end_time = time.time()
            request_finished_perf = time.perf_counter()
            pool_snapshot_after_request = client.get_pool_snapshot()
            pool_delta_after_request = _pool_delta(pool_snapshot_before, pool_snapshot_after_request)
            logger.debug(
                "Messages:%s\nResponse text:%s\n%s",
                json.dumps(messages, ensure_ascii=False),
                response_text,
                "#" * 40,
            )

            logger.debug(
                _json_log(
                    "request_success",
                    experiment_id=experiment_id,
                    task_id=task_id,
                    runner_id=runner_id,
                    thread_id=threading.get_ident(),
                    client_index=client_index,
                    prompt_hash=prompt_hash,
                    chunks=n_chunks,
                    n_input=n_input,
                    n_output=n_output,
                    client_select_ms=round((client_selected_perf - client_select_started) * 1000, 3),
                    headers_ms=round((headers_received_perf - request_started_perf) * 1000, 3),
                    ttft_ms=round((first_token_perf - request_started_perf) * 1000, 3) if first_token_perf else None,
                    stream_ms=round((request_finished_perf - first_token_perf) * 1000, 3) if first_token_perf else None,
                    total_ms=round((request_finished_perf - request_started_perf) * 1000, 3),
                    status_code=response_metadata["status_code"],
                    response_headers=response_metadata["response_headers"],
                    server_request_id=response_metadata["server_request_id"],
                    connection_reused=getattr(response, "connection_reused", None),
                    http_body_bytes_received=response_body_metrics["http_body_bytes_received"],
                    sse_payload_bytes_received=response_body_metrics["sse_payload_bytes_received"],
                    pool_snapshot_before=pool_snapshot_before,
                    pool_snapshot_after_headers=pool_snapshot_after_headers,
                    pool_snapshot_after_request=pool_snapshot_after_request,
                    new_tcp_connections_after_headers_estimate=pool_delta_after_headers["new_tcp_connections_estimate"],
                    new_tcp_connections_during_request_estimate=pool_delta_after_request["new_tcp_connections_estimate"],
                    new_http_requests_seen_after_headers=pool_delta_after_headers["new_http_requests_seen"],
                    new_http_requests_seen_during_request=pool_delta_after_request["new_http_requests_seen"],
                    idle_connection_delta_during_request=pool_delta_after_request["idle_connection_delta"],
                )
            )

            return InferenceResult(
                measurement=Measurement(
                    id=None,
                    experiment_id=experiment_id,
                    n_input=n_input,
                    n_output=n_output,
                    ttft=ttft,
                    start_time=start_time,
                    end_time=end_time,
                    status=status,
                )
            )
        except Exception as exc:
            end_time = time.time()
            request_finished_perf = time.perf_counter()
            failure_code = _classify_request_failure(exc, n_chunks)
            response_for_metadata = getattr(exc, "response", None)
            if response_for_metadata is None:
                response_for_metadata = response
            response_metadata = _extract_response_metadata(response_for_metadata)
            response_body_metrics = _extract_received_body_metrics(response_for_metadata)
            root_cause = _extract_root_cause(exc)
            client_index = locals().get("client_index")
            client_selected_perf = locals().get("client_selected_perf")
            client_select_started = locals().get("client_select_started", request_started_perf)
            headers_received_perf = locals().get("headers_received_perf")
            pool_snapshot_after_failure = client.get_pool_snapshot() if "client" in locals() else None
            pool_delta_after_failure = _pool_delta(pool_snapshot_before, pool_snapshot_after_failure)

            log_level = logging.ERROR if failure_code != "empty_stream" else logging.WARNING
            payload_summary = _summarize_partial_payload(recent_raw_lines, response_text)
            logger.log(
                log_level,
                _json_log(
                    "request_failure",
                    experiment_id=experiment_id,
                    task_id=task_id,
                    runner_id=runner_id,
                    thread_id=threading.get_ident(),
                    client_index=client_index,
                    prompt_hash=prompt_hash,
                    failure_code=failure_code,
                    exception_type=type(exc).__name__,
                    root_cause_type=type(root_cause).__name__,
                    errno=_extract_errno(exc),
                    error=repr(exc),
                    chunks=n_chunks,
                    n_input=n_input,
                    n_output=n_output,
                    partial_response_text_len=payload_summary["partial_response_text_len"],
                    partial_response_text_tail=payload_summary["partial_response_text_tail"],
                    last_raw_lines=payload_summary["last_raw_lines"],
                    last_parsed_chunks=payload_summary["last_parsed_chunks"],
                    client_select_ms=round((client_selected_perf - client_select_started) * 1000, 3) if client_selected_perf else None,
                    headers_ms=round((headers_received_perf - request_started_perf) * 1000, 3) if headers_received_perf else None,
                    ttft_ms=round((first_token_perf - request_started_perf) * 1000, 3) if first_token_perf else None,
                    total_ms=round((request_finished_perf - request_started_perf) * 1000, 3),
                    status_code=response_metadata["status_code"],
                    response_headers=response_metadata["response_headers"],
                    server_request_id=response_metadata["server_request_id"],
                    connection_reused=getattr(response_for_metadata, "connection_reused", None),
                    http_body_bytes_received=response_body_metrics["http_body_bytes_received"],
                    sse_payload_bytes_received=response_body_metrics["sse_payload_bytes_received"],
                    pool_snapshot_before=pool_snapshot_before,
                    pool_snapshot_after_failure=pool_snapshot_after_failure,
                    new_tcp_connections_before_failure_estimate=pool_delta_after_failure["new_tcp_connections_estimate"],
                    new_http_requests_seen_before_failure=pool_delta_after_failure["new_http_requests_seen"],
                    idle_connection_delta_before_failure=pool_delta_after_failure["idle_connection_delta"],
                )
            )

            return InferenceResult(
                measurement=Measurement.failed(
                    experiment_id=experiment_id,
                    n_input=n_input,
                    n_output=n_output,
                    ttft=ttft,
                    start_time=start_time,
                    end_time=end_time,
                ),
                failure_code=failure_code,
            )


class ExperimentRunner:
    def __init__(
        self,
        node_url: str,
        model_name: str,
        account_address: str = None,
        private_key_hex: str = None,
        api_token: str = None,
        num_runners: int = 10,
        no_sign: bool = False,
        old_sign: bool = False,
        transport: str = "requests",
    ) -> None:
        self.node_url = node_url
        self.model_name = model_name
        self.account_address = account_address
        self.private_key_hex = private_key_hex
        self.api_token = api_token
        self.num_runners = num_runners
        self.no_sign = no_sign
        self.old_sign = old_sign
        self.transport = transport
        
        # Create ONE shared client manager for all runners.
        # Scale the shared pool up to 1024 total configured connections.
        num_clients, max_connections_per_client = _shared_pool_config(num_runners)
        self.num_clients = num_clients
        self.max_connections_per_client = max_connections_per_client
        
        logger.info(f"Creating shared client manager with {num_clients} clients, {max_connections_per_client} connections each for {num_runners} runners")
        
        self._shared_client_manager = OptimizedNodeClientManager(
            node_url=node_url,
            account_address=account_address,
            private_key_hex=private_key_hex,
            api_token=api_token,
            no_sign=no_sign,
            old_sign=old_sign,
            num_clients=num_clients,
            max_connections_per_client=max_connections_per_client,
            transport=transport,
        )

    def _store_experiment_parameters(
        self,
        experiment_id: int,
        num_tasks: int,
        max_tokens: int,
    ) -> None:
        params = [
            ("num_workers", str(self.num_runners)),
            ("num_tasks", str(num_tasks)),
            ("node_url", self.node_url),
            ("max_tokens", str(max_tokens)),
            ("model_name", self.model_name),
            ("no_sign", str(self.no_sign)),
            ("old_sign", str(self.old_sign)),
            ("auth_mode", "bearer" if self.api_token else ("unsigned" if self.no_sign else "signed")),
            ("client_architecture", "shared_pool"),  # Now using shared pool
            ("http_transport", self.transport),
            ("shared_pool_clients", str(self.num_clients)),
            ("shared_pool_max_connections_per_client", str(self.max_connections_per_client)),
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
        prompts: List[PromptInput],
        num_tasks: int = 100,
        max_tokens: int = 1000,
        min_tokens: int = None,
        seed: int = 42,
    ) -> None:

        rng = random.Random(seed)
        all_results: List[InferenceResult] = []
        failure_counts: Counter[str] = Counter()
        progress_log_interval = max(1, min(100, num_tasks // 10 or 1))
        final_pool_snapshots: List[Dict[str, Union[int, str, Dict[str, str]]]] = []
        final_pool_totals: Dict[str, int] = {}
        previous_progress_pool_totals: Dict[str, int] = self._shared_client_manager.get_pool_totals()

        # Create runners that share the same client manager
        runners = []
        for runner_id in range(self.num_runners):
            runner = InferenceRunner(
                shared_client_manager=self._shared_client_manager,
                model_name=self.model_name,
            )
            runners.append(runner)

        try:
            with ThreadPoolExecutor(max_workers=self.num_runners) as pool:
                futures = [
                    pool.submit(
                        runners[i % self.num_runners].run_inference,
                        experiment_id,
                        i,
                        i % self.num_runners,
                        rng.choice(prompts),
                        max_tokens,
                        min_tokens,
                    )
                    for i in range(num_tasks)
                ]

                for f in tqdm(as_completed(futures), total=num_tasks, desc="Running experiments"):
                    try:
                        result = f.result()
                        all_results.append(result)
                        if result.failure_code:
                            failure_counts[result.failure_code] += 1

                        completed = len(all_results)
                        if completed % progress_log_interval == 0 or completed == num_tasks:
                            failed = sum(
                                1
                                for run_result in all_results
                                if run_result.measurement.status == Status.FAILED
                            )
                            current_pool_totals = self._shared_client_manager.get_pool_totals()
                            pool_totals_delta = _pool_totals_delta(previous_progress_pool_totals, current_pool_totals)
                            logger.info(
                                _json_log(
                                    "experiment_progress",
                                    experiment_id=experiment_id,
                                    completed=completed,
                                    total=num_tasks,
                                    failed=failed,
                                    failure_counts=dict(sorted(failure_counts.items())),
                                    pool_snapshots=self._shared_client_manager.get_pool_snapshots(),
                                    pool_totals=current_pool_totals,
                                    tcp_connections_opened_total=current_pool_totals.get("num_connections", 0),
                                    new_tcp_connections_since_last_progress=pool_totals_delta["new_tcp_connections"],
                                    http_requests_observed_total=current_pool_totals.get("num_requests", 0),
                                    new_http_requests_since_last_progress=pool_totals_delta["new_http_requests"],
                                )
                            )
                            previous_progress_pool_totals = current_pool_totals
                    except Exception as exc:
                        logger.error("Task failed: %s", exc)
                final_pool_snapshots = self._shared_client_manager.get_pool_snapshots()
                final_pool_totals = self._shared_client_manager.get_pool_totals()

        finally:
            # Close the shared client manager
            self._shared_client_manager.close_all()

        # Persist metadata & results --------------------------------------
        self._store_experiment_parameters(experiment_id, num_tasks, max_tokens)
        all_measurements = [result.measurement for result in all_results]
        for measurement in all_measurements:
            insert_measurement(measurement)

        logger.info(
            _json_log(
                "experiment_complete",
                experiment_id=experiment_id,
                failed_measurements=len([m for m in all_measurements if m.status == Status.FAILED]),
                failure_counts=dict(sorted(failure_counts.items())),
                total_measurements=len(all_measurements),
                pool_snapshots=final_pool_snapshots,
                pool_totals=final_pool_totals,
                tcp_connections_opened_total=final_pool_totals.get("num_connections", 0),
                http_requests_observed_total=final_pool_totals.get("num_requests", 0),
            )
        )
