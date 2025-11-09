import os
import threading
import time
import json
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

from services.weather_service import get_weather_data
from devices import (
    ZigbeeCoordinator,
    ZWaveController,
    AqaraDoorWindowSensor,
    AqaraMotionSensor,
    ShellyRelay,
    TplinkKasaPlug,
    PhilipsHueBulb,
    EcobeeThermostat,
    GoogleChromecast,
    SonosSpeaker,
)
from services import db as dbsvc

app = Flask(__name__)
# CORS config (dev): allow FE at 127.0.0.1:4028 with common methods/headers
_origins = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "http://127.0.0.1:4028").split(",") if o.strip()]
# Apply globally to all routes, so preflight OPTIONS includes required headers
CORS(
    app,
    origins=_origins or ["http://127.0.0.1:4028"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Weather API is running",
        "endpoints": [
            "/api/dashboard?lat=..&lon=..",
            "/dashboard?lat=..&lon=..",
            "/api/dashboard?user_id=<id>",
            "/dashboard?user_id=<id>",
            "/api/devices",
        ],
        "source": "Azure Maps",
    })


@app.route('/api/dashboard', methods=['GET'])
@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Devuelve clima (temperatura, humedad, condición) y AQI usando lat/lon o user_id.
    Uso: GET /api/dashboard?lat=..&lon=..  ó  GET /api/dashboard?user_id=..
    """
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    user_id = request.args.get('user_id')

    if (not lat or not lon) and user_id:
        row = dbsvc.get_user(user_id)
        if not row:
            return jsonify({"error": "Usuario no encontrado", "details": {"user_id": user_id}}), 404
        try:
            lat, lon = str(row["lat"]), str(row["lon"]) 
        except Exception:
            lat = str(getattr(row, 'lat', ''))
            lon = str(getattr(row, 'lon', ''))

    if not lat or not lon:
        return jsonify({"error": "Se requiere lat y lon o un user_id válido"}), 400

    data = get_weather_data(lat, lon)
    if isinstance(data, dict) and data.get('error'):
        return jsonify(data), int(data.get('details', {}).get('status', 400))
    return jsonify(data)


# --- Devices registry (simulados) ---
DEVICES: dict[str, object] = {}
STATE_PATH = Path(__file__).resolve().parent / 'data' / 'devices_state.json'


def _ensure_devices():
    if DEVICES:
        return
    items = [
        ("zigbee_coordinator", ZigbeeCoordinator()),
        ("zwave_controller", ZWaveController()),
        ("aqara_door", AqaraDoorWindowSensor()),
        ("aqara_motion", AqaraMotionSensor()),
        ("shelly_relay", ShellyRelay()),
        ("tplink_kasa_plug", TplinkKasaPlug()),
        ("philips_hue_bulb", PhilipsHueBulb()),
        ("ecobee_thermostat", EcobeeThermostat()),
        ("google_chromecast", GoogleChromecast()),
        ("sonos_speaker", SonosSpeaker()),
    ]
    # Load previous state if exists
    saved = _load_states()
    for fixed_id, d in items:
        try:
            setattr(d, 'id', fixed_id)
        except Exception:
            pass
        # restore saved fields
        if fixed_id in saved:
            for k, v in saved[fixed_id].items():
                try:
                    if k in ('id', 'name',):
                        continue
                    if hasattr(d, k):
                        setattr(d, k, v)
                except Exception:
                    pass
        else:
            # Seed sensible defaults so the UI shows movement right away
            try:
                if getattr(d, 'device_type', '') == 'light_bulb':
                    d.is_on = True
                    if hasattr(d, 'brightness'):
                        d.brightness = 50
                elif getattr(d, 'device_type', '') == 'smart_plug':
                    d.is_on = True
                    if hasattr(d, 'power_w'):
                        d.power_w = 5.0
                elif getattr(d, 'device_type', '') == 'speaker':
                    d.is_on = True
                    if hasattr(d, 'volume'):
                        d.volume = 30
                elif getattr(d, 'device_type', '') == 'chromecast':
                    d.is_on = True
                    if hasattr(d, 'playback_state'):
                        d.playback_state = 'idle'
                elif getattr(d, 'device_type', '') == 'relay':
                    d.is_on = True
            except Exception:
                pass
        DEVICES[fixed_id] = d


@app.route('/api/devices', methods=['GET'])
def api_devices():
    _ensure_devices()
    def _state(d):
        try:
            return d.to_dict()
        except Exception:
            return {k: v for k, v in d.__dict__.items() if not k.startswith('_')}
    return jsonify({"devices": [_state(d) for d in DEVICES.values()]})


def _load_states() -> dict:
    try:
        if STATE_PATH.exists():
            with open(STATE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_states():
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for did, d in DEVICES.items():
            try:
                st = d.to_dict() if hasattr(d, 'to_dict') else {k: v for k, v in d.__dict__.items() if not k.startswith('_')}
                data[did] = st
            except Exception:
                pass
        with open(STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception:
        pass


# --- Background simulator ---
_tick_thread_started = False


def _bg_tick_loop(interval: float = None):
    iv = interval or float(os.environ.get('DEVICES_TICK_INTERVAL', '1.0'))
    while True:
        try:
            _ensure_devices()
            for d in list(DEVICES.values()):
                try:
                    if hasattr(d, 'simulate_tick'):
                        d.simulate_tick()
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(iv)


def _start_tick_thread():
    global _tick_thread_started
    if _tick_thread_started:
        return
    t = threading.Thread(target=_bg_tick_loop, daemon=True)
    t.start()
    _tick_thread_started = True


@app.before_request
def _ensure_tick():
    # Start simulator on first request
    _start_tick_thread()


def _bootstrap_devices():
    """Bootstrap once at import time for Flask>=3 (no before_first_request).
    Ensures devices are created and the background simulator is running.
    """
    try:
        _ensure_devices()
    except Exception:
        pass
    _start_tick_thread()

# Call immediately so it runs once when the module is imported
_bootstrap_devices()


@app.route('/api/devices/tick_all', methods=['POST'])
def api_devices_tick_all():
    _ensure_devices()
    for d in DEVICES.values():
        try:
            if hasattr(d, 'simulate_tick'):
                d.simulate_tick()
        except Exception:
            pass
    _save_states()
    return jsonify({"ok": True, "count": len(DEVICES)})


@app.route('/api/devices/<dev_id>/power', methods=['POST'])
def api_device_power(dev_id):
    _ensure_devices()
    d = DEVICES.get(dev_id)
    if not d:
        return jsonify({"error": "Device not found"}), 404
    action = (request.json or {}).get('action') if request.is_json else request.form.get('action')
    try:
        if action == 'on' and hasattr(d, 'power_on'):
            d.power_on()
        elif action == 'off' and hasattr(d, 'power_off'):
            d.power_off()
        elif action == 'toggle' and hasattr(d, 'toggle'):
            d.toggle()
        else:
            return jsonify({"error": "Invalid action. Use on/off/toggle"}), 400
        _save_states()
    except Exception as e:
        return jsonify({"error": f"power failed: {e}"}), 400
    # return state
    st = d.to_dict() if hasattr(d, 'to_dict') else {k: v for k, v in d.__dict__.items() if not k.startswith('_')}
    return jsonify(st)


@app.route('/api/devices/<dev_id>/update', methods=['POST'])
def api_device_update(dev_id):
    _ensure_devices()
    d = DEVICES.get(dev_id)
    if not d:
        return jsonify({"error": "Device not found"}), 404
    payload = request.get_json(silent=True) or {}
    try:
        if hasattr(d, 'update_config'):
            import inspect
            sig = inspect.signature(d.update_config)
            allowed = {k: v for k, v in payload.items() if k in sig.parameters}
            d.update_config(**allowed)
        else:
            # Fallback: assign directly simple attributes
            for k, v in payload.items():
                if hasattr(d, k):
                    setattr(d, k, v)
        _save_states()
    except Exception as e:
        return jsonify({"error": f"update failed: {e}"}), 400
    st = d.to_dict() if hasattr(d, 'to_dict') else {k: v for k, v in d.__dict__.items() if not k.startswith('_')}
    return jsonify(st)


if __name__ == '__main__':
    # Debug: list routes on startup
    try:
        print("\n=== Weather API routes ===")
        for rule in app.url_map.iter_rules():
            print(f"{sorted(list(rule.methods))} -> {rule.rule}")
        print("==========================\n")
    except Exception:
        pass
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5050')))
