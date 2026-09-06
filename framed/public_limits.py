"""Bounded process-local beta admission control; deployment uses one worker."""
import math
import threading
import time
from collections import deque


class AnalysisLimiter:
    def __init__(self, limit, window, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._attempts = deque()
        self._lock = threading.Lock()

    def admit(self):
        # Global capacity cannot be bypassed by spoofing proxy/IP headers.
        with self._lock:
            now = self.clock()
            while self._attempts and self._attempts[0] <= now - self.window:
                self._attempts.popleft()
            if len(self._attempts) >= self.limit:
                return max(1, math.ceil(self.window - (now - self._attempts[0])))
            self._attempts.append(now)
            return 0
