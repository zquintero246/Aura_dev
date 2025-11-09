import uuid
from datetime import datetime
import random


class GoogleChromecast:
    def __init__(self, name: str = "Google Chromecast"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.protocol = "Wi-Fi"
        self.device_type = "chromecast"
        self.is_on = True
        self.last_seen = datetime.utcnow().isoformat()
        self.app_name = "Idle"
        self.playback_state = "idle"  # idle/playing/paused
        self.volume = 50

    def power_on(self):
        self.is_on = True
        self._touch()

    def power_off(self):
        self.is_on = False
        self.playback_state = "idle"
        self._touch()

    def toggle(self):
        self.is_on = not self.is_on
        if not self.is_on:
            self.playback_state = "idle"
        self._touch()

    def update_config(self, volume: int | None = None, app_name: str | None = None):
        if volume is not None:
            self.volume = max(0, min(100, int(volume)))
        if app_name:
            self.app_name = app_name
        self._touch()

    def simulate_tick(self):
        if self.is_on:
            # More dynamic state/volume changes
            r = random.random()
            if r < 0.2:
                self.playback_state = random.choice(["playing", "paused", "idle"])
                self.app_name = (
                    random.choice(["YouTube", "Netflix", "Spotify", "Twitch"]) if self.playback_state != "idle" else "Idle"
                )
            # Nudge volume with small noise
            if random.random() < 0.3:
                self.volume = max(0, min(100, int(self.volume + random.randint(-5, 5))))
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
            "app_name": self.app_name,
            "playback_state": self.playback_state,
            "volume": self.volume,
        }
