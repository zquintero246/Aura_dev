import uuid
from datetime import datetime
import random


class PhilipsHueBulb:
    def __init__(self, name: str = "Philips Hue Bulb"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.protocol = "Zigbee"
        self.device_type = "light_bulb"
        self.is_on = False
        self.last_seen = datetime.utcnow().isoformat()
        self.brightness = 0  # 0-100
        self.color_temp = 3000  # Kelvin

    def power_on(self):
        self.is_on = True
        if self.brightness == 0:
            self.brightness = 50
        self._touch()

    def power_off(self):
        self.is_on = False
        self._touch()

    def toggle(self):
        self.is_on = not self.is_on
        self._touch()

    def update_config(self, brightness: int | None = None, color_temp: int | None = None):
        if brightness is not None:
            self.brightness = max(0, min(100, int(brightness)))
        if color_temp is not None:
            self.color_temp = max(2000, min(6500, int(color_temp)))
        self._touch()

    def simulate_tick(self):
        # Random toggle sometimes
        if random.random() < 0.05:
            self.is_on = not self.is_on
        if self.is_on and random.random() < 0.25:
            self.brightness = max(0, min(100, self.brightness + random.randint(-20, 20)))
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
            "brightness": self.brightness,
            "color_temp": self.color_temp,
        }
