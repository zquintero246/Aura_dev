import uuid
from datetime import datetime
import random


class ZWaveController:
    def __init__(self, name: str = "Z-Wave Controller"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.protocol = "Z-Wave"
        self.device_type = "controller"
        self.is_on = True
        self.last_seen = datetime.utcnow().isoformat()
        self.home_id = f"0x{random.getrandbits(32):08X}"
        self.node_count = random.randint(80, 400)
        self.region = "EU"  # EU/US depends on stick; simulated

    def power_on(self):
        self.is_on = True
        self._touch()

    def power_off(self):
        self.is_on = False
        self._touch()

    def toggle(self):
        self.is_on = not self.is_on
        self._touch()

    def update_config(self, region: str | None = None):
        if region in {"EU", "US", "ANZ"}:
            self.region = region
        self._touch()

    def simulate_tick(self):
        if self.is_on:
            delta = random.randint(-20, 50)
            self.node_count = max(0, min(5000, self.node_count + delta))
        self._touch()

    def _touch(self):
        self.last_seen = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "protocol": self.protocol,
            "type": self.device_type,
            "is_on": self.is_on,
            "last_seen": self.last_seen,
            "home_id": self.home_id,
            "node_count": self.node_count,
            "region": self.region,
        }
