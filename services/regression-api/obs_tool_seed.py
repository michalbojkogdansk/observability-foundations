"""
Seed data for the Observability Dashboard capstone (obs_tool_* tables).

Design goals (per explicit direction):
  - Separate tables, prefixed obs_tool_, never touching spike_metrics/latency_readings
    used by the other exercises.
  - "Presentation-savvy": lean row count, clean baseline signal, and only a
    handful of deliberate, legible incidents rather than continuous noise.
    Hourly granularity over 21 days x 4 services x 3 metrics = ~6,000 rows.

Simulated platform: api-gateway -> payment-service -> database
                                 -> cache
Three planted story beats, spaced out so each panel has one clear thing to show:
  1. Cascade failure, 14 days ago (3h window): database -> cache -> api-gateway.
  2. Isolated random spike, 8 days ago (1h): cache latency only, uncorrelated.
  3. Gradual drift, last 5 days: payment-service latency trending upward,
     for the regression/forecast panel to project.
"""
import random
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras

SERVICES = ["api-gateway", "payment-service", "database", "cache"]
METRICS = ["latency_ms", "error_rate", "cpu_pct"]

BASELINE = {
    "api-gateway":     {"latency_ms": (45, 8),  "error_rate": (0.005, 0.003), "cpu_pct": (30, 5)},
    "payment-service": {"latency_ms": (120, 20), "error_rate": (0.010, 0.005), "cpu_pct": (40, 6)},
    "database":        {"latency_ms": (35, 10), "error_rate": (0.002, 0.0015), "cpu_pct": (55, 8)},
    "cache":           {"latency_ms": (4, 1.5), "error_rate": (0.001, 0.001), "cpu_pct": (20, 4)},
}

DEPENDENCIES = [
    ("api-gateway", "payment-service"),
    ("api-gateway", "database"),
    ("api-gateway", "cache"),
    ("payment-service", "database"),
    ("payment-service", "cache"),
]

DDL = """
CREATE TABLE IF NOT EXISTS obs_tool_services (
    name TEXT PRIMARY KEY,
    description TEXT
);

CREATE TABLE IF NOT EXISTS obs_tool_dependencies (
    from_service TEXT NOT NULL REFERENCES obs_tool_services(name),
    to_service   TEXT NOT NULL REFERENCES obs_tool_services(name),
    PRIMARY KEY (from_service, to_service)
);

CREATE TABLE IF NOT EXISTS obs_tool_metrics (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    service TEXT NOT NULL REFERENCES obs_tool_services(name),
    metric_name TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_obs_tool_metrics_svc_metric_ts
    ON obs_tool_metrics (service, metric_name, ts);
"""

SERVICE_DESCRIPTIONS = {
    "api-gateway": "Entry point, routes requests to payment-service, database, cache",
    "payment-service": "Handles checkout and payment processing",
    "database": "Primary Postgres datastore",
    "cache": "Redis-backed read cache",
}


def _daily_cycle_multiplier(hour_of_day: int) -> float:
    """Mild business-hours bump, not noisy — peaks gently around midday."""
    return 1.0 + 0.25 * max(0.0, 1 - abs(hour_of_day - 13) / 9)


def _gen_value(mean: float, std: float, mult: float = 1.0) -> float:
    v = random.gauss(mean * mult, std * mult * 0.6)
    return max(0.0, v)


def generate_rows(now: datetime | None = None):
    """Returns a list of (ts, service, metric_name, value) tuples, ~6000 rows."""
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=21)
    hours = int((now - start).total_seconds() // 3600)

    cascade_start = now - timedelta(days=14)
    cascade_end = cascade_start + timedelta(hours=3)

    spike_hour = now - timedelta(days=8)

    drift_start = now - timedelta(days=5)

    rows = []
    for h in range(hours):
        ts = start + timedelta(hours=h)
        cyc = _daily_cycle_multiplier(ts.hour)

        in_cascade = cascade_start <= ts < cascade_end
        cascade_ramp = 0.0
        if in_cascade:
            elapsed = (ts - cascade_start).total_seconds() / 3600
            cascade_ramp = 1.0 - abs(elapsed - 1.5) / 1.5  # peaks mid-window

        is_spike_hour = abs((ts - spike_hour).total_seconds()) < 1800
        in_drift = ts >= drift_start
        drift_frac = ((ts - drift_start).total_seconds() / (now - drift_start).total_seconds()) if in_drift else 0.0

        for svc in SERVICES:
            for metric in METRICS:
                mean, std = BASELINE[svc][metric]
                mult = cyc

                if metric == "latency_ms" and svc == "database" and in_cascade:
                    mult *= 1 + 3.0 * cascade_ramp
                if metric == "latency_ms" and svc == "cache" and in_cascade:
                    mult *= 1 + 1.2 * cascade_ramp
                if metric == "error_rate" and svc == "api-gateway" and in_cascade:
                    mult *= 1 + 25.0 * cascade_ramp

                if metric == "latency_ms" and svc == "cache" and is_spike_hour:
                    mult *= 9.0

                if metric == "latency_ms" and svc == "payment-service" and in_drift:
                    mult *= 1 + 0.4 * drift_frac

                value = _gen_value(mean, std, mult)
                if metric == "error_rate":
                    value = min(value, 1.0)
                rows.append((ts, svc, metric, round(value, 4)))

    return rows


def seed(conn):
    """Idempotent: drops and repopulates obs_tool_* data. Schema is created if missing."""
    cur = conn.cursor()
    cur.execute(DDL)

    cur.execute("TRUNCATE obs_tool_metrics")
    cur.execute("DELETE FROM obs_tool_dependencies")
    cur.execute("DELETE FROM obs_tool_services")

    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO obs_tool_services (name, description) VALUES %s",
        [(s, SERVICE_DESCRIPTIONS[s]) for s in SERVICES],
    )
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO obs_tool_dependencies (from_service, to_service) VALUES %s",
        DEPENDENCIES,
    )

    rows = generate_rows()
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO obs_tool_metrics (ts, service, metric_name, value) VALUES %s",
        rows,
        page_size=1000,
    )

    conn.commit()
    cur.close()
    return {"services": len(SERVICES), "dependencies": len(DEPENDENCIES), "metric_rows": len(rows)}
