"""Retrigger fraud-detector-v1 with extra telemetry and acute drift episode."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arbiter import auth, keys

BASE = os.getenv("ARBITER_URL", "http://localhost:8004")
KEY = os.getenv("ARBITER_KEY", "ak_live_mHCOOmZrD_9LxaMjVEAgWQeBq24Mh_te")
MODEL_ID = "fraud-detector-v1"

def main() -> int:
    print(f"Target Arbiter URL : {BASE}")
    print(f"Model to retrigger : {MODEL_ID}")
    print(f"Using API Key      : {KEY[:10]}...{KEY[-6:]}\n")

    key_record = keys.verify_key(KEY)
    if not key_record:
        print(f"ERROR: API Key not recognized in DB.")
        return 1

    user_id = key_record["user_id"]
    token = auth.create_session(user_id)
    auth_headers = {"Authorization": f"Bearer {token}"}
    sdk_headers = {"X-API-Key": KEY}

    with httpx.Client(base_url=BASE, timeout=30.0) as client:
        # Check initial stats
        m_before = client.get(f"/api/models/{MODEL_ID}/metrics", headers=auth_headers).json()
        ev_before = client.get(f"/api/models/{MODEL_ID}/events", headers=auth_headers).json()
        initial_telemetry = m_before.get("telemetry_count", 0)
        initial_events = ev_before.get("count", 0)
        print(f"Before Retrigger:")
        print(f"  Telemetry rows : {initial_telemetry}")
        print(f"  Drift Events   : {initial_events}\n")

        # Reference distributions for fraud-detector-v1
        # Features: mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness
        ref_mean = np.array([14.1, 19.3, 91.9, 654.8, 0.096, 0.104])
        ref_std = np.array([3.5, 4.3, 24.3, 351.9, 0.014, 0.052])

        # Acute shift on mean_radius (+80%) and mean_area (+150%)
        drift_mean = ref_mean.copy()
        drift_std = ref_std.copy()
        drift_mean[0] = ref_mean[0] * 1.8   # mean_radius: 14.1 -> ~25.4
        drift_mean[3] = ref_mean[3] * 2.5   # mean_area: 654.8 -> ~1637.0
        drift_std[0] = ref_std[0] * 1.6
        drift_std[3] = ref_std[3] * 1.8

        rng = np.random.default_rng(int(time.time()))
        EXTRA_ROWS = 60
        queued_count = 0

        print(f"Streaming {EXTRA_ROWS} extra telemetry predictions for {MODEL_ID}...")
        for i in range(EXTRA_ROWS):
            feat = rng.normal(drift_mean, drift_std).tolist()
            drift_score = float(rng.uniform(0.78, 0.94))
            res = client.post(
                f"/predict/{MODEL_ID}",
                headers=sdk_headers,
                json={
                    "model_id": MODEL_ID,
                    "features": feat,
                    "prediction": [1],
                    "drift_score": drift_score,
                },
            )
            if res.status_code == 200 and res.json().get("reasoning_queued"):
                queued_count += 1

        print(f"  -> {EXTRA_ROWS} rows posted. Reasoning queued: {queued_count > 0}")

        print("\nWaiting for Gemma 4 AI Reasoning on the new telemetry window...")
        deadline = time.time() + 90
        new_event = None
        while time.time() < deadline:
            time.sleep(5)
            ev_now = client.get(f"/api/models/{MODEL_ID}/events", headers=auth_headers).json()
            current_events = ev_now.get("count", 0)
            if current_events > initial_events:
                print(f"  New event recorded! Total events now: {current_events}")
                new_event = ev_now.get("events", [])[0]
                break
            else:
                print(f"  Analyzing telemetry... (events: {current_events})", flush=True)

        m_after = client.get(f"/api/models/{MODEL_ID}/metrics", headers=auth_headers).json()
        print("\n" + "=" * 76)
        print("RETRIGGER RESULTS FOR fraud-detector-v1")
        print("=" * 76)
        print(f"Total Telemetry Count : {initial_telemetry} -> {m_after.get('telemetry_count', 0)} (+{EXTRA_ROWS})")
        print(f"Total Drift Events    : {initial_events} -> {m_after.get('event_count', 0)}")
        if new_event:
            print(f"\nFresh AI Judgment:")
            print(f"  Event ID      : {new_event.get('event_id')}")
            print(f"  Action        : {new_event.get('action')}")
            print(f"  Confidence    : {new_event.get('confidence')}")
            print(f"  Root Cause    : {new_event.get('taxonomy')}")
            features = new_event.get('primary_features') or []
            print(f"  Primary Shift : {', '.join(features) if features else 'Identified in metrics'}")
            print(f"\nGemma Reasoning:")
            print(f"  {new_event.get('reasoning')}")
        print("=" * 76)
        print("Dashboard is live and updated at: http://localhost:3000\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
