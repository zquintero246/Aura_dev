import uuid
from datetime import datetime
import random


class ZigbeeCoordinator:
    def __init__(self, name: str = "Zigbee Coordinator"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.protocol = "Zigbee"
        self.device_type = "coordinator"
        self.is_on = True
        self.last_seen = datetime.utcnow().isoformat()
        self.network_channel = 15
        self.pan_id = f"0x{random.randint(0x0000, 0xFFFF):04X}"
        self.extended_pan_id = f"0x{random.getrandbits(64):016X}"
        self.connected_devices = random.randint(100, 500)

    def power_on(self):
        self.is_on = True
        self._touch()

    def power_off(self):
        self.is_on = False
        self._touch()

    def toggle(self):
        self.is_on = not self.is_on
        self._touch()

    def update_config(self, channel: int | None = None):
        if channel is not None:
            if 11 <= channel <= 26:
                self.network_channel = channel
        self._touch()

    def simulate_tick(self):
        if self.is_on:
            # Noisy growth/decline to make charts move more
            delta = random.randint(-30, 80)
            self.connected_devices = max(0, min(5000, self.connected_devices + delta))
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
            "network_channel": self.network_channel,
            "pan_id": self.pan_id,
            "extended_pan_id": self.extended_pan_id,
            "connected_devices": self.connected_devices,
        }
