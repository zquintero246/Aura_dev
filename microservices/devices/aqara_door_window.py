import uuid
from datetime import datetime
import random


class AqaraDoorWindowSensor:
    def __init__(self, name: str = "Aqara Door/Window"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.protocol = "Zigbee"
        self.device_type = "contact_sensor"
        self.is_on = True
        self.last_seen = datetime.utcnow().isoformat()
        self.opened = False
        self.battery = 100.0

    def power_on(self):
        self.is_on = True
        self._touch()

    def power_off(self):
        self.is_on = False
        self._touch()

    def toggle(self):
        self.is_on = not self.is_on
        self._touch()

    def update_config(self, opened: bool | None = None):
        if opened is not None:
            self.opened = bool(opened)
        self._touch()

    def simulate_tick(self):
        if self.is_on:
            # Higher chance to toggle for visible activity
            if random.random() < 0.2:
                self.opened = not self.opened
            self.battery = max(0.0, self.battery - random.uniform(0.0, 0.02))
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
            "opened": self.opened,
            "battery": round(self.battery, 2),
        }
