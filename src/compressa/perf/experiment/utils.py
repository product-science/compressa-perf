"""Utility classes for experiment execution."""

import collections
import threading
import time


class SlidingWindowRateLimiter:
    """
    Thread-safe sliding window rate limiter.
    
    Ensures no more than `max_requests` are allowed within any 
    `window_seconds` time window.
    """

    def __init__(self, max_requests: int = 100, window_seconds: float = 5.0):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed per window.
            window_seconds: Size of the sliding window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def _cleanup_expired(self, now: float) -> None:
        """Remove timestamps that are outside the current window."""
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def acquire(self) -> None:
        """
        Block until a request slot is available within the rate limit.
        
        This method will wait if the rate limit has been reached,
        and return once a slot becomes available.
        """
        while True:
            with self._lock:
                now = time.time()
                self._cleanup_expired(now)
                
                if len(self._timestamps) < self.max_requests:
                    # Slot available, record this request
                    self._timestamps.append(now)
                    return
                
                # Calculate how long to wait for the oldest request to expire
                wait_time = self._timestamps[0] + self.window_seconds - now
            
            # Wait outside the lock to allow other threads to proceed
            if wait_time > 0:
                time.sleep(wait_time)


