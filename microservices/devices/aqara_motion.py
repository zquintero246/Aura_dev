import uuid
from datetime import datetime
import random


class AqaraMotionSensor:
    def __init__(self, name: str = "Aqara Motion"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.protocol = "Zigbee"
        self.device_type = "motion_sensor"
        self.is_on = True
        self.last_seen = datetime.utcnow().isoformat()
        self.motion_detected = False
        self.lux = 50.0
        self.battery = 100.0
        self.occupancy_timeout_s = 60

    def power_on(self):
        self.is_on = True
        self._touch()

    def power_off(self):
        self.is_on = False
        self._touch()

    def toggle(self):
        self.is_on = not self.is_on
        self._touch()

    def update_config(self, occupancy_timeout_s: int | None = None):
        if occupancy_timeout_s is not None and occupancy_timeout_s > 0:
            self.occupancy_timeout_s = int(occupancy_timeout_s)
        self._touch()

    def simulate_tick(self):
        if self.is_on:
            # More frequent spikes of motion
            if random.random() < 0.25:
                self.motion_detected = True
            else:
                if random.random() < 0.35:
                    self.motion_detected = False
            # Larger lux swings
            self.lux = max(0.0, min(1000.0, self.lux + random.uniform(-25, 25)))
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
            "motion_detected": self.motion_detected,
            "lux": round(self.lux, 1),
            "battery": round(self.battery, 2),
            "occupancy_timeout_s": self.occupancy_timeout_s,
        }
