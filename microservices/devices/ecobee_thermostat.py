import uuid
from datetime import datetime
import random


class EcobeeThermostat:
    def __init__(self, name: str = "Ecobee Thermostat"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.protocol = "Wi-Fi"
        self.device_type = "thermostat"
        self.is_on = True
        self.last_seen = datetime.utcnow().isoformat()
        self.hvac_mode = "auto"  # off/heat/cool/auto
        self.target_c = 22.0
        self.current_c = 22.0
        self.fan_on = False

    def power_on(self):
        self.is_on = True
        self._touch()

    def power_off(self):
        self.is_on = False
        self._touch()

    def toggle(self):
        self.is_on = not self.is_on
        self._touch()

    def update_config(self, hvac_mode: str | None = None, target_c: float | None = None, fan_on: bool | None = None):
        if hvac_mode in {"off", "heat", "cool", "auto"}:
            self.hvac_mode = hvac_mode
        if target_c is not None:
            self.target_c = max(10.0, min(30.0, float(target_c)))
        if fan_on is not None:
            self.fan_on = bool(fan_on)
        self._touch()

    def simulate_tick(self):
        if not self.is_on:
            self.current_c += random.uniform(-0.1, 0.1)
        else:
            # Occasionally change mode/target for visible movement
            if random.random() < 0.05:
                self.hvac_mode = random.choice(["off","heat","cool","auto"])
            if random.random() < 0.1:
                self.target_c = max(18.0, min(26.0, self.target_c + random.uniform(-1.0, 1.0)))
            if self.hvac_mode == "off":
                self.current_c += random.uniform(-0.1, 0.1)
            else:
                delta = self.target_c - self.current_c
                self.current_c += max(-0.4, min(0.4, delta * 0.2))
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
            "hvac_mode": self.hvac_mode,
            "target_c": round(self.target_c, 1),
            "current_c": round(self.current_c, 1),
            "fan_on": self.fan_on,
        }
