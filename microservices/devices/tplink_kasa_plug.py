import uuid
from datetime import datetime
import random


class TplinkKasaPlug:
    def __init__(self, name: str = "TP-Link Kasa Plug"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.protocol = "Wi-Fi"
        self.device_type = "smart_plug"
        self.is_on = False
        self.last_seen = datetime.utcnow().isoformat()
        self.power_w = 0.0
        self.voltage_v = 120.0

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
        # Occasional state toggle
        if random.random() < 0.08:
            self.is_on = not self.is_on
        if self.is_on:
            # Wider swings in power
            self.power_w = max(0.0, min(150.0, self.power_w + random.uniform(-10, 25)))
        else:
            self.power_w = 0.0
        # Voltage small jitter
        self.voltage_v = max(100.0, min(240.0, self.voltage_v + random.uniform(-1.0, 1.0)))
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
            "voltage_v": round(self.voltage_v, 1),
        }
