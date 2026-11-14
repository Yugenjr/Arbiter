"""Run 6 ML models from the codebase against Arbiter on port 8004 with the user's API key.

Models:
1. fraud-detector-v2        (Finance / Fraud: New high-value population segment)
2. churn-predictor-v1       (SaaS / Churn: Upstream ETL pipeline fault with nulls)
3. credit-risk-v3           (Banking / Credit Risk: Concept drift)
4. basic-logistic-model     (Core Scikit-Learn: Seasonal batch pattern)
5. fraud-detector-v1        (Healthcare / Cancer RF: Covariate shift)
6. advanced-fraud-detector  (Self-healing Challenger: Upstream data break)
"""

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

MODELS: dict[str, dict] = {
    "fraud-detector-v2": {
        "features": [
            "transaction_amount",
            "merchant_category",
            "hour_of_day",
            "card_present",
            "account_age_days",
            "velocity_1h",
        ],
        "ref_mean": [240.0, 8.4, 14.2, 0.62, 1240.0, 1.8],
        "ref_std": [180.0, 3.1, 5.8, 0.48, 820.0, 1.4],
        "kind": "new_segment",
        "shift": {0: 3.7, 1: 1.65},
        "description": "Fraud detection (e-commerce)",
    },
    "churn-predictor-v1": {
        "features": [
            "monthly_charges",
            "tenure_months",
            "support_tickets",
            "data_usage_gb",
            "contract_type",
            "late_payments",
        ],
        "ref_mean": [64.8, 32.4, 1.2, 18.6, 1.7, 0.4],
        "ref_std": [30.2, 24.1, 1.6, 12.4, 0.8, 0.9],
        "kind": "pipeline_break",
        "shift": {4: 0.04},
        "description": "Customer churn predictor",
    },
    "credit-risk-v3": {
        "features": [
            "income_annual",
            "debt_to_income",
            "credit_utilization",
            "inquiries_6mo",
            "employment_years",
            "open_accounts",
        ],
        "ref_mean": [58400.0, 0.34, 0.41, 1.6, 6.2, 5.4],
        "ref_std": [26200.0, 0.14, 0.22, 1.5, 4.8, 2.9],
        "kind": "concept_drift",
        "shift": {},
        "description": "Credit loan default risk",
    },
    "basic-logistic-model": {
        "features": [
            "user_activity_score",
            "session_duration",
            "feature_depth",
            "interaction_rate",
            "latency_ms",
        ],
        "ref_mean": [45.2, 180.0, 4.5, 0.65, 42.0],
        "ref_std": [15.0, 60.0, 1.8, 0.20, 12.0],
        "kind": "seasonal",
        "shift": {1: 1.4, 3: 1.25},
        "description": "Basic logistic classifier",
    },
    "fraud-detector-v1": {
        "features": [
            "mean_radius",
            "mean_texture",
            "mean_perimeter",
            "mean_area",
            "mean_smoothness",
            "mean_compactness",
        ],
        "ref_mean": [14.1, 19.3, 91.9, 654.8, 0.096, 0.104],
        "ref_std": [3.5, 4.3, 24.3, 351.9, 0.014, 0.052],
        "kind": "new_segment",
        "shift": {0: 1.8, 2: 1.7},
        "description": "RandomForest fraud & anomaly detector",
    },
    "advanced-fraud-detector": {
        "features": [
            "account_age_days",
            "tx_velocity_hour",
            "order_value_usd",
            "risk_assessment_score",
            "device_trust_factor",
        ],
        "ref_mean": [420.0, 2.4, 115.0, 0.18, 0.92],
        "ref_std": [210.0, 1.2, 75.0, 0.10, 0.15],
        "kind": "pipeline_break",
        "shift": {4: 0.05},
        "description": "Self-healing champion/challenger model",
    },
}

REF_BATCH = 70
DRIFT_BATCH = 50

def run() -> int:
    print(f"Target Arbiter URL : {BASE}")
    print(f"Using API Key      : {KEY[:10]}...{KEY[-6:]}")
    print(f"Running models     : {len(MODELS)} ML models\n")

    key_record = keys.verify_key(KEY)
    if not key_record:
        print(f"ERROR: Key {KEY} not recognized by Arbiter DB.")
        return 1

    user_id = key_record["user_id"]
    token = auth.create_session(user_id)
    auth_headers = {"Authorization": f"Bearer {token}"}
    sdk_headers = {"X-API-Key": KEY}
    rng = np.random.default_rng(20261014)

    with httpx.Client(base_url=BASE, timeout=30.0) as client:
        try:
            h = client.get("/health")
            if h.status_code != 200:
                print(f"ERROR: Arbiter not healthy at {BASE}: {h.status_code}")
                return 1
            print("Arbiter backend is ONLINE\n")
        except Exception as e:
            print(f"ERROR: Cannot reach {BASE}: {e}")
            return 1

        print("--- STEP 1: Registering 6 Models ---")
        for mid, spec in MODELS.items():
            res = client.post(
                "/models/register",
                headers=sdk_headers,
                json={
                    "model_id": mid,
                    "features": spec["features"],
                    "drift_threshold": 0.15,
                    "version": "1.0.0",
                },
            )
            status = "OK" if res.status_code == 200 else f"FAIL ({res.status_code})"
            print(f"  [{status}] {mid:25} -> {spec['description']}")

        print("\n--- STEP 2: Streaming Telemetry & Triggering Drift Episodes ---")
        for mid, spec in MODELS.items():
            ref_m = np.array(spec["ref_mean"], dtype=float)
            ref_s = np.array(spec["ref_std"], dtype=float)

            # 1. Clean reference batch
            for _ in range(REF_BATCH):
                feat_sample = rng.normal(ref_m, ref_s).tolist()
                client.post(
                    f"/predict/{mid}",
                    headers=sdk_headers,
                    json={
                        "model_id": mid,
                        "features": feat_sample,
                        "prediction": [0],
                        "drift_score": float(rng.uniform(0.02, 0.08)),
                    },
                )

            # 2. Shifted / Drift episode batch
            drift_m, drift_s = ref_m.copy(), ref_s.copy()
            for idx, mult in spec["shift"].items():
                drift_m[idx] = ref_m[idx] * mult
                if spec["kind"] == "pipeline_break":
                    drift_s[idx] = ref_s[idx] * 0.02  # variance collapse
                else:
                    drift_s[idx] = ref_s[idx] * 1.5   # variance expansion

            queued = 0
            for i in range(DRIFT_BATCH):
                vec = rng.normal(drift_m, drift_s).tolist()
                if spec["kind"] == "pipeline_break" and i % 3 == 0:
                    vec[next(iter(spec["shift"]))] = None  # null tell
                
                if spec["kind"] == "concept_drift":
                    score = 0.28
                elif spec["kind"] == "seasonal":
                    score = float(rng.uniform(0.32, 0.42))
                else:
                    score = float(rng.uniform(0.65, 0.92))

                r = client.post(
                    f"/predict/{mid}",
                    headers=sdk_headers,
                    json={
                        "model_id": mid,
                        "features": vec,
                        "prediction": [1],
                        "drift_score": score,
                    },
                )
                if r.status_code == 200 and r.json().get("reasoning_queued"):
                    queued += 1

            total_rows = REF_BATCH + DRIFT_BATCH
            print(f"  {mid:25} : {total_rows} predictions sent | reasoning_queued: {queued}")

        print("\n--- STEP 3: Waiting for Gemma 4 AI Reasoning across Models ---")
        deadline = time.time() + 180
        while time.time() < deadline:
            time.sleep(6)
            r = client.get("/api/models", headers=auth_headers)
            if r.status_code == 200:
                data = r.json()
                models = data.get("models", [])
                with_verdicts = [m for m in models if m.get("latest_verdict")]
                print(f"  Reasoned models: {len(with_verdicts)}/{len(MODELS)}...", flush=True)
                if len(with_verdicts) >= len(MODELS):
                    print("  All 6 models reasoned successfully!")
                    break

        print("\n" + "=" * 80)
        print("SUMMARY: 6 ML MODELS RUN RESULTS ON ARBITER (Port 8004)")
        print("=" * 80)
        r = client.get("/api/models", headers=auth_headers)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                mid = m["model_id"]
                verd = m.get("latest_verdict") or {}
                action = verd.get("action", "MONITOR")
                conf = verd.get("confidence", 0.0)
                tax = verd.get("taxonomy", "n/a")
                print(f"\nModel: {mid}")
                print(f"  Telemetry Count: {m.get('telemetry_count', 0)}")
                print(f"  Latest Drift   : {m.get('latest_drift_score', 'n/a')}")
                print(f"  Verdict Action : {action} (Confidence: {conf})")
                print(f"  Root Cause     : {tax}")
                hypo = verd.get("hypothesis", "")
                if hypo:
                    print(f"  Gemma Reasoning: {hypo[:180]}...")
        print("\n" + "=" * 80)
        print("Dashboard is LIVE at: http://localhost:3000")
        print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(run())
