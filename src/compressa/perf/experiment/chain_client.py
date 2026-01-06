
import time
import json
import hashlib
import base64
from typing import List
import contextlib
import os
import resource
import uuid
import threading
from urllib.parse import urlparse

import requests
import urllib3
from ecdsa import SigningKey, SECP256k1, util

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




# ---------------------------------------------------------------------------
# High-performance HTTP client optimized for concurrent requests
# ---------------------------------------------------------------------------
class _NodeClient:
    def __init__(
        self,
        node_url: str,
        entrypoint_addr: str = "",
        account_address: str = None,
        private_key_hex: str = None,
        timeout: float = 600.0,
        max_connections: int = 10,  # Small pool per runner
        max_connections_per_host: int = 10,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        no_sign: bool = False,
        old_sign: bool = False,
        host_header: str = None,
    ) -> None:
        self.node_url = node_url.rstrip("/")
        self.account_address = account_address
        self.timeout = timeout
        self.no_sign = no_sign
        self.old_sign = old_sign
        self.entrypoint_addr = entrypoint_addr
        self.host_header = host_header

        # Check system limits on first initialization
        if not hasattr(_NodeClient, '_limits_checked'):
            check_system_limits()
            _NodeClient._limits_checked = True

        # Deterministic signing key (only if signing is enabled)
        if not self.no_sign:
            if not account_address:
                raise ValueError("account_address is required when signing is enabled")
            if not private_key_hex:
                raise ValueError("private_key_hex is required when signing is enabled")
            self._signing_key = SigningKey.from_string(
                bytes.fromhex(private_key_hex), curve=SECP256k1
            )

        # Configure urllib3 for high performance
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Create optimized session for high concurrency
        self._session = requests.Session()
        
        # Configure HTTP adapter with conservative settings for stability
        # Retries disabled for maximum performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_connections,
            pool_maxsize=max_connections_per_host,
            max_retries=0,  # Disable retries
            pool_block=True  # Block when pool is full instead of creating new connections
        )
        
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        
        # Configure session defaults for better performance and connection reuse
        self._session.headers.update({
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=30, max=1000'
        })
        
        # Override Host header if provided (critical when using IP address for connection)
        if self.host_header:
            self._session.headers['Host'] = self.host_header

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Properly close the session and clean up resources"""
        if hasattr(self, '_session'):
            try:
                self._session.close()
            except Exception as e:
                logger.debug("Error closing session: %s", e)


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
    def stream_chat_completion(
        self,
        *,
        messages: List[dict],
        model: str,
        max_tokens: int,
        temperature: float = 0.8,
    ):
        """Send a streaming chat/completions request and return the raw response."""
        
        try:
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
            try:
                payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
            except Exception as e:
                logger.error("Error encoding payload: %s", e)
                raise

            headers = {
                "Content-Type": "application/json",
            }

            if not self.no_sign:
                timestamp_ns = int(time.time_ns())
                
                transfer_address = self.entrypoint_addr
                
                if self.old_sign:
                    headers["Authorization"] = self._old_sign(payload_bytes)
                else:
                    headers["Authorization"] = self._sign(payload_bytes, timestamp_ns, transfer_address)
                headers["X-Requester-Address"] = self.account_address
                headers["X-Timestamp"] = str(timestamp_ns)

            resp = self._session.post(
                f"{self.node_url}/v1/chat/completions",
                data=payload_bytes,
                headers=headers,
                stream=True,
                timeout=self.timeout,
            )
            
            # Handle HTTP errors with detailed error messages
            if resp.status_code >= 400:
                error_message = "Unknown error"
                try:
                    # Read content first while response is still open
                    content = resp.content
                    try:
                        error_data = json.loads(content)
                        error_message = error_data.get("error", "Unknown error")
                    except:
                        error_message = content.decode('utf-8', errors='replace')
                except Exception as e:
                    error_message = f"Could not read error body: {e}"
                finally:
                    resp.close()

                logger.error("HTTP %s error: %s", resp.status_code, error_message)
                raise requests.exceptions.HTTPError(f"HTTP {resp.status_code}: {error_message}", response=resp)
            
            return resp  # caller iterates resp.iter_lines(...)
        except Exception:
            raise



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
            logger.debug("Error during response cleanup: %s", e)  # Don't fail on cleanup errors
