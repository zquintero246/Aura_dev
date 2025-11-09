import uuid
from datetime import datetime
import random


class SonosSpeaker:
    def __init__(self, name: str = "Sonos Speaker"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.protocol = "Wi-Fi"
        self.device_type = "speaker"
        self.is_on = True
        self.last_seen = datetime.utcnow().isoformat()
        self.playback_state = "stopped"  # playing/paused/stopped
        self.volume = 30
        self.muted = False
        self.track = None

    def power_on(self):
        self.is_on = True
        self._touch()

    def power_off(self):
        self.is_on = False
        self.playback_state = "stopped"
        self._touch()

    def toggle(self):
        self.is_on = not self.is_on
        if not self.is_on:
            self.playback_state = "stopped"
        self._touch()

    def update_config(self, volume: int | None = None, muted: bool | None = None):
        if volume is not None:
            self.volume = max(0, min(100, int(volume)))
        if muted is not None:
            self.muted = bool(muted)
        self._touch()

    def simulate_tick(self):
        if self.is_on:
            if random.random() < 0.2:
                self.playback_state = random.choice(["playing", "paused", "stopped"])
                self.track = (
                    random.choice(["Jazz Vibes","Lo-Fi Beats","Classical Essentials","Top 50"]) if self.playback_state != "stopped" else None
                )
            if random.random() < 0.3:
                self.volume = max(0, min(100, int(self.volume + random.randint(-5, 5))))
            if random.random() < 0.05:
                self.muted = not self.muted
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
            "playback_state": self.playback_state,
            "volume": self.volume,
            "muted": self.muted,
            "track": self.track,
        }
