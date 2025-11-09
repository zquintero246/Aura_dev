import uuid
from datetime import datetime
import random


class ShellyRelay:
    def __init__(self, name: str = "Shelly Plus 1"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.protocol = "Wi-Fi"
        self.device_type = "relay"
        self.is_on = False
        self.last_seen = datetime.utcnow().isoformat()
        self.power_w = 0.0
        self.energy_wh = 0.0

    def power_on(self):
        self.is_on = True
        self._touch()

    def power_off(self):
        self.is_on = False
        self.power_w = 0.0
        self._touch()

    def toggle(self):
        self.is_on = not self.is_on
        if not self.is_on:
            self.power_w = 0.0
        self._touch()

    def update_config(self, name: str | None = None):
        if name:
            self.name = name
        self._touch()

    def simulate_tick(self):
        # Randomly toggle occasionally to create visible changes
        if random.random() < 0.08:
            self.is_on = not self.is_on
            if not self.is_on:
                self.power_w = 0.0
        # Power varies more aggressively when on
        if self.is_on:
            self.power_w = max(0.0, min(100.0, self.power_w + random.uniform(-10, 20)))
            self.energy_wh = max(0.0, self.energy_wh + self.power_w / 60.0)
        else:
            self.power_w = 0.0
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
            "power_w": round(self.power_w, 2),
            "energy_wh": round(self.energy_wh, 2),
        }
