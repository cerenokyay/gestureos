import time
from dataclasses import dataclass, field

@dataclass
class Cooldown:
    cooldown_sec: float = 0.7
    last: dict = field(default_factory=dict)

    def allow(self, key: str) -> bool:
        now = time.time()
        if now - self.last.get(key, 0.0) < self.cooldown_sec:
            return False
        self.last[key] = now
        return True