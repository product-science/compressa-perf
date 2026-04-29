
import time
import json
import hashlib
import base64
from typing import List, Optional, Tuple
import contextlib
import os
import resource
import uuid
import threading
import asyncio
import queue
from urllib.parse import urlparse

import requests
import urllib3
from ecdsa import SigningKey, SECP256k1, util
try:
    import aiohttp
except ImportError:  # pragma: no cover - handled at runtime when transport is selected
    aiohttp = None

from compressa.utils import get_logger

logger = get_logger(__name__)


def check_system_limits():
    """Check and log system limits for file descriptors"""
    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info(f"File descriptor limits: soft={soft_limit}, hard={hard_limit}")
        
        # Warn if limits are too low
        if soft_limit < 10000:
            logger.warning(f"Low file descriptor limit ({soft_limit}). Consider increasing with 'ulimit -n 65536'")
        
        return soft_limit, hard_limit
    except Exception as e:
        logger.error(f"Could not check system limits: {e}")
        return None, None
    

def get_entrypoint_addr(url):
    try:
        parsed = urlparse(url.rstrip('/'))
        host_port = f"{parsed.hostname}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}"
        
        data = requests.get(f"{url}/v1/epochs/current/participants", timeout=10).json()
        participants = data.get('active_participants', {})
        
        for p in participants.get('participants', []):
            participant_url = p.get('inference_url', '')
            if participant_url:
                parsed_participant = urlparse(participant_url.rstrip('/'))
                participant_host_port = f"{parsed_participant.hostname}:{parsed_participant.port or (443 if parsed_participant.scheme == 'https' else 80)}"
                
                if participant_host_port == host_port:
                    return p['index']
       
        return None
    except:
        return None


_AIOHTTP_QUEUE_DONE = object()


class _AiohttpErrorResponse:
    def __init__(self, status_code: int, headers: dict, text: str, reason: str = None):
        self.status_code = status_code
        self.headers = headers
        self.text = text
        self.reason = reason or ""
        self.connection_reused: Optional[bool] = None
        self.http_body_bytes_received = len(text.encode("utf-8"))
        self.sse_payload_bytes_received = self.http_body_bytes_received

    def json(self):
        return json.loads(self.text)

    def close(self):
        return None


class _AiohttpBufferedResponse:
    def __init__(
        self,
        status_code: int,
        headers: dict,
        reason: str,
        line_queue: "queue.Queue[object]",
        close_callback,
        connection_reused: "Optional[bool]" = None,
        byte_counters: Optional[dict] = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.reason = reason or ""
        self._line_queue = line_queue
        self._close_callback = close_callback
        self._closed = False
        self.raw = self
        # True  -> this request rode a warm keep-alive connection.
        # False -> aiohttp opened a new TCP connection for this request.
        # None  -> unknown (trace did not record; e.g. non-aiohttp transport).
        self.connection_reused = connection_reused
        self._byte_counters = byte_counters or {
            "http_body_bytes_received": 0,
            "sse_payload_bytes_received": 0,
        }

    @property
    def http_body_bytes_received(self) -> int:
        return int(self._byte_counters.get("http_body_bytes_received", 0) or 0)

    @property
    def sse_payload_bytes_received(self) -> int:
        return int(self._byte_counters.get("sse_payload_bytes_received", 0) or 0)

    def iter_lines(self, decode_unicode=False):
        while True:
            item = self._line_queue.get()
            if item is _AIOHTTP_QUEUE_DONE:
                break
            if isinstance(item, Exception):
                raise item
            if decode_unicode and isinstance(item, bytes):
                yield item.decode("utf-8")
            else:
                yield item

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._close_callback()


def _split_complete_lines(pending: bytes) -> Tuple[List[bytes], bytes]:
    """Split newline-terminated SSE/body lines from a pending byte buffer.

    Returned lines have trailing ``\\r``/``\\n`` stripped to match the existing
    ``iter_lines()`` contract used by the inference loop. Any final partial line
    is returned as the new ``pending`` buffer so callers can wait for more body
    bytes before yielding it, or flush it once EOF is reached.
    """
    lines: List[bytes] = []
    while True:
        newline_index = pending.find(b"\n")
        if newline_index < 0:
            break
        line_with_newline = pending[: newline_index + 1]
        pending = pending[newline_index + 1 :]
        lines.append(line_with_newline.rstrip(b"\r\n"))
    return lines, pending


class _RequestsBufferedResponse:
    def __init__(self, response: requests.Response) -> None:
        self._response = response
        self.status_code = response.status_code
        self.headers = response.headers
        self.reason = response.reason or ""
        self.raw = response.raw
        self.connection_reused: Optional[bool] = None
        self.http_body_bytes_received = 0
        self.sse_payload_bytes_received = 0

    def iter_lines(self, decode_unicode=False):
        pending = b""
        for chunk in self._response.iter_content(chunk_size=8192, decode_unicode=False):
            if not chunk:
                continue
            self.http_body_bytes_received += len(chunk)
            pending += chunk

            lines, pending = _split_complete_lines(pending)
            for line in lines:
                self.sse_payload_bytes_received += len(line)
                if decode_unicode:
                    yield line.decode("utf-8")
                else:
                    yield line

        if pending:
            self.sse_payload_bytes_received += len(pending)
            if decode_unicode:
                yield pending.decode("utf-8")
            else:
                yield pending

    def close(self):
        self._response.close()




# ---------------------------------------------------------------------------
# High-performance HTTP client optimized for concurrent requests
# ---------------------------------------------------------------------------
class _NodeClient:
    def __init__(
        self,
        node_url: str,
        account_address: str = None,
        private_key_hex: str = None,
        api_token: str = None,
        timeout: float = 1200.0,
        max_connections: int = 100,  # Reduced from 1000
        max_connections_per_host: int = 100,  # Reduced from 1000  
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        no_sign: bool = False,
        old_sign: bool = False,
        client_index: int = 0,
        transport: str = "requests",
    ) -> None:
        self.node_url = node_url.rstrip("/")
        self.account_address = account_address
        self.timeout = timeout
        self.no_sign = no_sign
        self.old_sign = old_sign
        self.client_index = client_index
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.transport = transport
        self.api_token = api_token or os.getenv("GONKA_API_TOKEN")
        self._uses_bearer_auth = bool(self.api_token)
        self._signing_enabled = not self.no_sign and not self._uses_bearer_auth
        self.entrypoint_addr = get_entrypoint_addr(node_url) if self._signing_enabled else ""

        # Check system limits on first initialization
        if not hasattr(_NodeClient, '_limits_checked'):
            check_system_limits()
            _NodeClient._limits_checked = True

        # Deterministic signing key (only if signing is enabled)
        if self._signing_enabled:
            if not account_address:
                raise ValueError("account_address is required when signing is enabled")
            if not private_key_hex:
                raise ValueError("private_key_hex is required when signing is enabled")
            self._signing_key = SigningKey.from_string(
                bytes.fromhex(private_key_hex), curve=SECP256k1
            )

        if self.transport == "aiohttp":
            self._init_aiohttp_client()
        else:
            self._init_requests_client()

    def get_target_url(self) -> str:
        return f"{self.node_url}/v1/chat/completions"

    def get_pool_snapshot(self) -> dict:
        snapshot = {
            "client_index": self.client_index,
            "pool_connections": self.max_connections,
            "pool_maxsize": self.max_connections_per_host,
            "transport": self.transport,
        }
        try:
            if self.transport == "aiohttp":
                snapshot.update(self._get_aiohttp_pool_snapshot())
            else:
                snapshot.update(self._get_requests_pool_snapshot())
        except Exception as exc:
            snapshot["pool_snapshot_error"] = type(exc).__name__
        return snapshot

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Properly close the session and clean up resources"""
        if self.transport == "aiohttp":
            try:
                self._close_aiohttp_client()
            except Exception as e:
                logger.debug(f"Error closing aiohttp client: {e}")
            return
        if hasattr(self, '_session'):
            try:
                self._session.close()
            except Exception as e:
                logger.debug(f"Error closing session: {e}")

    def _init_requests_client(self):
        # Configure urllib3 for high performance
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self.max_connections,
            pool_maxsize=self.max_connections_per_host,
            max_retries=0,
            pool_block=True,
        )
        self._adapter = adapter
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update({
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=30, max=1000'
        })

    def _init_aiohttp_client(self):
        if aiohttp is None:
            raise ImportError("aiohttp is not installed; install it or use --transport requests")

        self._aiohttp_lock = threading.Lock()
        self._aiohttp_num_connections = 0
        self._aiohttp_num_requests = 0
        self._aiohttp_num_reused_connections = 0
        self._aiohttp_ready = threading.Event()
        self._aiohttp_loop = asyncio.new_event_loop()
        self._aiohttp_session = None
        self._aiohttp_connector = None
        self._aiohttp_closed = False
        self._aiohttp_thread = threading.Thread(
            target=self._run_aiohttp_loop,
            name=f"aiohttp-client-{self.client_index}",
            daemon=True,
        )
        self._aiohttp_thread.start()
        self._aiohttp_ready.wait()

    def _run_aiohttp_loop(self):
        asyncio.set_event_loop(self._aiohttp_loop)

        async def _create_session():
            trace_config = aiohttp.TraceConfig()

            async def on_request_start(session, trace_config_ctx, params):
                with self._aiohttp_lock:
                    self._aiohttp_num_requests += 1

            async def on_connection_create_end(session, trace_config_ctx, params):
                with self._aiohttp_lock:
                    self._aiohttp_num_connections += 1
                ctx = getattr(trace_config_ctx, "trace_request_ctx", None)
                if isinstance(ctx, dict):
                    ctx["connection_reused"] = False

            async def on_connection_reuseconn(session, trace_config_ctx, params):
                with self._aiohttp_lock:
                    self._aiohttp_num_reused_connections += 1
                ctx = getattr(trace_config_ctx, "trace_request_ctx", None)
                if isinstance(ctx, dict):
                    ctx["connection_reused"] = True

            trace_config.on_request_start.append(on_request_start)
            trace_config.on_connection_create_end.append(on_connection_create_end)
            trace_config.on_connection_reuseconn.append(on_connection_reuseconn)

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host,
                enable_cleanup_closed=True,
            )
            self._aiohttp_connector = connector
            self._aiohttp_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Connection": "keep-alive",
                    "Keep-Alive": "timeout=30, max=1000",
                },
                trace_configs=[trace_config],
            )

        self._aiohttp_loop.run_until_complete(_create_session())
        self._aiohttp_ready.set()
        self._aiohttp_loop.run_forever()

        async def _cleanup():
            if self._aiohttp_session is not None and not self._aiohttp_session.closed:
                try:
                    await self._aiohttp_session.close()
                except AttributeError as exc:
                    # aiohttp/asyncio can hit a torn-down SSL transport during
                    # shutdown. Suppress it so result reporting still completes.
                    logger.debug(
                        "Ignoring aiohttp shutdown race for client_index=%s: %s",
                        self.client_index,
                        exc,
                    )
                except Exception as exc:
                    logger.debug(
                        "Error closing aiohttp session for client_index=%s: %s",
                        self.client_index,
                        exc,
                    )

        self._aiohttp_loop.run_until_complete(_cleanup())
        self._aiohttp_loop.close()

    def _run_on_aiohttp_loop(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._aiohttp_loop).result()

    def _get_requests_pool_snapshot(self) -> dict:
        pool_manager = getattr(self._adapter, "poolmanager", None)
        if pool_manager is None:
            return {}

        try:
            pools = list(pool_manager.pools.values())
        except NotImplementedError:
            pools = list(getattr(pool_manager.pools, "_container", {}).values())

        idle_connections = 0
        for pool in pools:
            pool_queue = getattr(pool, "pool", None)
            if pool_queue is not None and hasattr(pool_queue, "qsize"):
                idle_connections += pool_queue.qsize()

        return {
            "active_pools": len(pools),
            "num_connections": sum(getattr(pool, "num_connections", 0) for pool in pools),
            "num_requests": sum(getattr(pool, "num_requests", 0) for pool in pools),
            "idle_connections": idle_connections,
        }

    def _get_aiohttp_pool_snapshot(self) -> dict:
        connector = self._aiohttp_connector
        if connector is None:
            return {}
        with self._aiohttp_lock:
            num_connections = self._aiohttp_num_connections
            num_requests = self._aiohttp_num_requests
            reused_connections = self._aiohttp_num_reused_connections

        pooled = getattr(connector, "_conns", {})
        idle_connections = sum(len(conns) for conns in pooled.values())
        active_pools = len(pooled)
        acquired = len(getattr(connector, "_acquired", set()))
        return {
            "active_pools": active_pools,
            "num_connections": num_connections,
            "num_requests": num_requests,
            "idle_connections": idle_connections,
            "reused_connections": reused_connections,
            "acquired_connections": acquired,
        }

    def _close_aiohttp_client(self):
        if getattr(self, "_aiohttp_closed", False):
            return
        self._aiohttp_closed = True
        if getattr(self, "_aiohttp_loop", None) is None:
            return
        try:
            self._aiohttp_loop.call_soon_threadsafe(self._aiohttp_loop.stop)
            if getattr(self, "_aiohttp_thread", None) is not None:
                self._aiohttp_thread.join(timeout=5)
        except Exception as e:
            logger.debug(f"Error closing aiohttp client: {e}")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _sign(self, payload: bytes, timestamp: int, transfer_address: str) -> str:
        """Return a *low‑s* canonical ECDSA signature encoded in base‑64."""
        # Phase 3: Sign hash of payload instead of raw payload
        payload_hash = hashlib.sha256(payload).hexdigest()
        
        # Build signature input: hash + timestamp + transfer_address
        signature_input = payload_hash
        if timestamp > 0:
            signature_input += str(timestamp)
        if transfer_address:
            signature_input += transfer_address
        else:
            logger.warning("Transfer address is None, using entrypoint address")
            signature_input += self.entrypoint_addr
        
        signature_bytes = signature_input.encode('utf-8')
        
        # Debug logging
        logger.debug(f"Signature components (Phase 3 - hash-based):")
        logger.debug(f"  Payload hash: {payload_hash}")
        logger.debug(f"  Timestamp: {timestamp}")
        logger.debug(f"  Transfer address: {transfer_address}")
        logger.debug(f"  Signature input: {signature_input}")
        
        raw_sig = self._signing_key.sign_deterministic(
            signature_bytes, hashfunc=hashlib.sha256, sigencode=util.sigencode_string
        )
        r, s = raw_sig[:32], raw_sig[32:]

        # Force *low‑s* form to avoid malleable sigs
        curve_n = SECP256k1.order
        s_int = int.from_bytes(s, "big")
        if s_int > curve_n // 2:
            s_int = curve_n - s_int
            s = s_int.to_bytes(32, "big")

        signature = base64.b64encode(r + s).decode()
        logger.debug(f"Generated signature: {signature}")
        
        # Log public key information for debugging
        pub_key_bytes = self._signing_key.get_verifying_key().to_string()
        pub_key_b64 = base64.b64encode(pub_key_bytes).decode()
        logger.debug(f"Public key (base64): {pub_key_b64}")
        logger.debug(f"Account address: {self.account_address}")
        
        return signature

    def _old_sign(self, payload: bytes) -> str:
        """Legacy signing method matching the original behavior for backward compatibility."""
        # Original behavior: only sign the payload bytes, no timestamp or transfer address
        
        # Debug logging
        logger.debug(f"Old signature components:")
        logger.debug(f"  Payload length: {len(payload)}")
        logger.debug(f"  Payload (first 100 chars): {payload[:100]}")
        logger.debug(f"  Note: timestamp and transfer_address are ignored in old signing mode")
        
        raw_sig = self._signing_key.sign_deterministic(
            payload, hashfunc=hashlib.sha256, sigencode=util.sigencode_string
        )
        r, s = raw_sig[:32], raw_sig[32:]

        # Apply low-s enforcement (original behavior had this)
        curve_n = SECP256k1.order
        s_int = int.from_bytes(s, "big")
        if s_int > curve_n // 2:
            s_int = curve_n - s_int
            s = s_int.to_bytes(32, "big")

        signature = base64.b64encode(r + s).decode()
        logger.debug(f"Generated old signature: {signature}")
        
        return signature

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def _build_payload_and_headers(
        self,
        *,
        messages: List[dict],
        model: str,
        max_tokens: int,
        min_tokens: int = None,
        temperature: float = 0.8,
    ):
        payload = {
            "temperature": temperature,
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "stream_options": {
                "include_usage": True
            },
            "_nonce": str(int.from_bytes(os.urandom(4), "big"))
        }
        if min_tokens is not None:
            payload["min_tokens"] = min_tokens
        try:
            payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
        except Exception as e:
            logger.error(f"Error encoding payload: {e}")
            raise

        headers = {
            "Content-Type": "application/json",
        }

        if self._uses_bearer_auth:
            headers["Authorization"] = f"Bearer {self.api_token}"
        elif self._signing_enabled:
            timestamp_ns = int(time.time_ns())
            transfer_address = self.entrypoint_addr

            if self.old_sign:
                headers["Authorization"] = self._old_sign(payload_bytes)
            else:
                headers["Authorization"] = self._sign(payload_bytes, timestamp_ns, transfer_address)
            headers["X-Requester-Address"] = self.account_address
            headers["X-Timestamp"] = str(timestamp_ns)

        return payload_bytes, headers

    def _stream_chat_completion_requests(self, payload_bytes: bytes, headers: dict):
        resp = self._session.post(
            self.get_target_url(),
            data=payload_bytes,
            headers=headers,
            stream=True,
            timeout=self.timeout,
        )

        if resp.status_code >= 400:
            try:
                error_data = resp.json()
                error_message = error_data.get("error", "Unknown error")
                logger.error(f"HTTP {resp.status_code} error: {error_message}")
                raise requests.exceptions.HTTPError(f"HTTP {resp.status_code}: {error_message}", response=resp)
            except ValueError:
                try:
                    error_text = resp.text
                    logger.error(f"HTTP {resp.status_code} error: {error_text}")
                    raise requests.exceptions.HTTPError(f"HTTP {resp.status_code}: {error_text}", response=resp)
                except Exception:
                    logger.error(f"HTTP {resp.status_code} error: {resp.reason}")
                    resp.raise_for_status()

        return _RequestsBufferedResponse(resp)

    async def _aiohttp_cancel_response(self, response, pump_task):
        if pump_task is not None and not pump_task.done():
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump_task
        if response is not None and not response.closed:
            response.close()

    async def _aiohttp_pump_response_lines(self, response, line_queue, byte_counters):
        pending = b""
        try:
            while True:
                chunk = await response.content.readany()
                if not chunk:
                    break
                byte_counters["http_body_bytes_received"] += len(chunk)
                pending += chunk
                lines, pending = _split_complete_lines(pending)
                for line in lines:
                    byte_counters["sse_payload_bytes_received"] += len(line)
                    line_queue.put(line)
            if pending:
                byte_counters["sse_payload_bytes_received"] += len(pending)
                line_queue.put(pending)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            line_queue.put(exc)
        finally:
            with contextlib.suppress(Exception):
                response.close()
            line_queue.put(_AIOHTTP_QUEUE_DONE)

    async def _aiohttp_start_stream_request(self, payload_bytes: bytes, headers: dict):
        # Per-request trace context. aiohttp trace callbacks (on_connection_*)
        # flip ``connection_reused`` so each response knows if this specific
        # request rode a warm keep-alive socket or opened a new TCP connection.
        trace_request_ctx = {"connection_reused": None}
        response = await self._aiohttp_session.post(
            self.get_target_url(),
            data=payload_bytes,
            headers=headers,
            trace_request_ctx=trace_request_ctx,
        )
        connection_reused = trace_request_ctx.get("connection_reused")

        if response.status >= 400:
            error_text = await response.text()
            error_response = _AiohttpErrorResponse(
                status_code=response.status,
                headers=dict(response.headers),
                text=error_text,
                reason=response.reason,
            )
            error_response.connection_reused = connection_reused
            response.close()
            try:
                error_data = error_response.json()
                error_message = error_data.get("error", "Unknown error")
            except ValueError:
                error_message = error_text or response.reason
            logger.error(f"HTTP {response.status} error: {error_message}")
            raise requests.exceptions.HTTPError(
                f"HTTP {response.status}: {error_message}",
                response=error_response,
            )

        line_queue = queue.Queue()
        byte_counters = {
            "http_body_bytes_received": 0,
            "sse_payload_bytes_received": 0,
        }
        pump_task = asyncio.create_task(
            self._aiohttp_pump_response_lines(response, line_queue, byte_counters)
        )

        def close_callback():
            try:
                self._run_on_aiohttp_loop(self._aiohttp_cancel_response(response, pump_task))
            except Exception as exc:
                logger.debug(f"Error closing aiohttp response: {exc}")

        return _AiohttpBufferedResponse(
            status_code=response.status,
            headers=dict(response.headers),
            reason=response.reason,
            line_queue=line_queue,
            close_callback=close_callback,
            connection_reused=connection_reused,
            byte_counters=byte_counters,
        )

    def _stream_chat_completion_aiohttp(self, payload_bytes: bytes, headers: dict):
        return self._run_on_aiohttp_loop(self._aiohttp_start_stream_request(payload_bytes, headers))

    def stream_chat_completion(
        self,
        *,
        messages: List[dict],
        model: str,
        max_tokens: int,
        min_tokens: int = None,
        temperature: float = 0.8,
    ):
        """Send a streaming chat/completions request and return the raw response."""
        payload_bytes, headers = self._build_payload_and_headers(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            temperature=temperature,
        )
        if self.transport == "aiohttp":
            return self._stream_chat_completion_aiohttp(payload_bytes, headers)
        return self._stream_chat_completion_requests(payload_bytes, headers)


# ---------------------------------------------------------------------------
# Optimized client manager for high-concurrency scenarios
# ---------------------------------------------------------------------------
class OptimizedNodeClientManager:
    """
    Manages a pool of _NodeClient instances to handle high-concurrency requests.
    This helps distribute load across multiple connection pools.
    """
    
    def __init__(
        self,
        node_url: str,
        account_address: str = None,
        private_key_hex: str = None,
        api_token: str = None,
        timeout: float = 1200.0,
        num_clients: int = 5,  # Reduced from 10
        max_connections_per_client: int = 50,  # Reduced from 500
        no_sign: bool = False,
        old_sign: bool = False,
        transport: str = "requests",
    ):
        self.clients = []
        self.current_client_index = 0
        self._client_lock = threading.Lock()
        self.transport = transport

        logger.info(
            f"Creating {num_clients} optimized HTTP clients with {max_connections_per_client} "
            f"connections each using transport={transport}"
        )
        
        for client_index in range(num_clients):
            client = _NodeClient(
                node_url=node_url,
                account_address=account_address,
                private_key_hex=private_key_hex,
                api_token=api_token,
                timeout=timeout,
                max_connections=max_connections_per_client,
                max_connections_per_host=max_connections_per_client,
                max_retries=3,
                backoff_factor=0.5,
                no_sign=no_sign,
                old_sign=old_sign,
                client_index=client_index,
                transport=transport,
            )
            self.clients.append(client)
    
    def get_client(self) -> _NodeClient:
        """Get the next client using round-robin selection"""
        _, client = self.get_client_with_index()
        return client

    def get_client_with_index(self):
        with self._client_lock:
            client_index = self.current_client_index
            client = self.clients[client_index]
            self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        return client_index, client

    def get_pool_snapshots(self) -> List[dict]:
        return [client.get_pool_snapshot() for client in self.clients]

    def get_pool_totals(self) -> dict:
        snapshots = self.get_pool_snapshots()
        return {
            "clients": len(snapshots),
            "active_pools": sum(snapshot.get("active_pools", 0) for snapshot in snapshots),
            "num_connections": sum(snapshot.get("num_connections", 0) for snapshot in snapshots),
            "num_requests": sum(snapshot.get("num_requests", 0) for snapshot in snapshots),
            "idle_connections": sum(snapshot.get("idle_connections", 0) for snapshot in snapshots),
            "pool_connections": sum(snapshot.get("pool_connections", 0) for snapshot in snapshots),
            "pool_maxsize": sum(snapshot.get("pool_maxsize", 0) for snapshot in snapshots),
        }
    
    def close_all(self):
        """Close all clients"""
        for client in self.clients:
            try:
                client.close()
            except Exception as exc:
                logger.debug(f"Error closing client during manager shutdown: {exc}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()


# ---------------------------------------------------------------------------
# Streaming response wrapper for proper resource cleanup
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def managed_stream_response(response):
    """Context manager for properly handling streaming responses"""
    try:
        yield response
    finally:
        # Ensure response is properly closed
        try:
            if hasattr(response, 'close'):
                response.close()
            # Also close the underlying connection if needed
            if hasattr(response, 'raw') and hasattr(response.raw, 'close'):
                response.raw.close()
        except Exception as e:
            logger.debug(f"Error during response cleanup: {e}")  # Don't fail on cleanup errors
