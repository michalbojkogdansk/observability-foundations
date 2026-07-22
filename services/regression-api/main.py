"""
Regression Analysis API — real scipy/statsmodels math on Neon Postgres data.
Deployed on Render.com, proxied via Cloudflare Worker.
"""
import os, json, time
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
OTEL_HEADERS = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")  # e.g. "Authorization=Basic <base64>"
OTEL_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "regression-api")
TRACING_ENABLED = OTEL_AVAILABLE and bool(OTEL_ENDPOINT)

tracer = None
if TRACING_ENABLED:
    headers_dict = {}
    for pair in OTEL_HEADERS.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            headers_dict[k.strip()] = v.strip()

    resource = Resource.create({"service.name": OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, headers=headers_dict)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(OTEL_SERVICE_NAME)
    Psycopg2Instrumentor().instrument()

DATABASE_URL = os.environ.get("DATABASE_URL", "")
NEON_WHATIF_URL = os.environ.get("NEON_WHATIF_URL", "")

def get_conn(url=None):
    return psycopg2.connect(url or DATABASE_URL)


# ─── Models ───────────────────────────────────────────────────────────────────

class RegressionResult(BaseModel):
    scenario: str
    regression_type: str
    coefficients: list[float]
    r_squared: float
    std_error: float
    confidence_interval_95: list[list[float]]
    residuals_summary: dict
    prediction: dict
    data_points: int
    computation_ms: float


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        print("✅ Database connected")
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")
    yield

app = FastAPI(title="Regression Analysis API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if TRACING_ENABLED:
    FastAPIInstrumentor.instrument_app(app)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM latency_metrics")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"status": "ok", "service": "regression-api", "db": "connected", "latency_rows": count}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


# ─── Data endpoints ──────────────────────────────────────────────────────────

@app.get("/data/latency")
def get_latency_data(limit: int = Query(default=360, le=1000)):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT ts, service, p50_ms, p95_ms, p99_ms, request_count FROM latency_metrics ORDER BY ts LIMIT %s",
        (limit,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"scenario": "latency_drift", "count": len(rows), "data": [
        {**r, "ts": r["ts"].isoformat()} for r in rows
    ]}


@app.get("/data/errors")
def get_error_data(limit: int = Query(default=240, le=1000)):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT ts, service, total_requests, error_count, error_rate, budget_remaining FROM error_budget ORDER BY ts LIMIT %s",
        (limit,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"scenario": "error_budget", "count": len(rows), "data": [
        {**r, "ts": r["ts"].isoformat()} for r in rows
    ]}


@app.get("/data/capacity")
def get_capacity_data(limit: int = Query(default=120, le=1000)):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT ts, resource_type, used_gb, total_gb, utilization_pct, daily_growth_gb FROM capacity_metrics ORDER BY ts LIMIT %s",
        (limit,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"scenario": "capacity_runway", "count": len(rows), "data": [
        {**r, "ts": r["ts"].isoformat()} for r in rows
    ]}


# ─── Regression engines ──────────────────────────────────────────────────────

def _linear_regression(x, y):
    """OLS linear regression with full statistics."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    
    n = len(x)
    x_mean = np.mean(x)
    se_slope = std_err
    se_intercept = std_err * np.sqrt(np.sum(x**2) / n)
    t_crit = stats.t.ppf(0.975, n - 2)
    
    ci_slope = [slope - t_crit * se_slope, slope + t_crit * se_slope]
    ci_intercept = [intercept - t_crit * se_intercept, intercept + t_crit * se_intercept]
    
    return {
        "type": "linear",
        "coefficients": [float(intercept), float(slope)],
        "r_squared": float(r_value**2),
        "std_error": float(std_err),
        "p_value": float(p_value),
        "confidence_intervals": [ci_intercept, ci_slope],
        "residuals": {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
        },
        "fitted": y_pred.tolist(),
    }


def _polynomial_regression(x, y, degree=2):
    """Polynomial regression (default quadratic)."""
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    y_pred = poly(x)
    residuals = y - y_pred
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    rmse = np.sqrt(ss_res / len(x))
    
    return {
        "type": "polynomial",
        "degree": degree,
        "coefficients": [float(c) for c in coeffs],
        "r_squared": float(r_squared),
        "std_error": float(rmse),
        "residuals": {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
        },
        "fitted": y_pred.tolist(),
    }


def _exponential_regression(x, y):
    """Exponential regression y = a * exp(b * x)."""
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    
    try:
        log_y = np.log(np.maximum(y, 1e-10))
        slope, intercept, _, _, _ = stats.linregress(x, log_y)
        p0 = [np.exp(intercept), slope]
        
        popt, pcov = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
        y_pred = exp_func(x, *popt)
        residuals = y - y_pred
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        perr = np.sqrt(np.diag(pcov))
        
        return {
            "type": "exponential",
            "coefficients": [float(popt[0]), float(popt[1])],
            "r_squared": float(r_squared),
            "std_error": float(np.sqrt(ss_res / len(x))),
            "param_errors": [float(perr[0]), float(perr[1])],
            "residuals": {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
            },
            "fitted": y_pred.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exponential fit failed: {str(e)}")


# ─── Analysis endpoints ──────────────────────────────────────────────────────

@app.get("/analyze/latency")
def analyze_latency(slo_ms: float = Query(default=300, description="SLO threshold in ms")):
    """Linear regression on p99 latency — predicts SLO breach date."""
    t0 = time.perf_counter()
    
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT ts, p99_ms FROM latency_metrics ORDER BY ts")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    timestamps = [r[0] for r in rows]
    base_ts = timestamps[0]
    x = np.array([(t - base_ts).total_seconds() / 86400 for t in timestamps])
    y = np.array([r[1] for r in rows])
    
    result = _linear_regression(x, y)
    
    intercept, slope = result["coefficients"]
    if slope > 0:
        breach_day = (slo_ms - intercept) / slope
        breach_date = (base_ts + __import__('datetime').timedelta(days=breach_day)).isoformat()
        days_remaining = breach_day - x[-1]
    else:
        breach_date = None
        days_remaining = None
    
    computation_ms = (time.perf_counter() - t0) * 1000
    
    return {
        "scenario": "latency_drift",
        "slo_ms": slo_ms,
        "regression": result,
        "prediction": {
            "breach_date": breach_date,
            "days_remaining": round(days_remaining, 1) if days_remaining else None,
            "current_p99": float(y[-1]),
            "trend_per_day_ms": round(slope, 3),
        },
        "x_days": x.tolist(),
        "y_values": y.tolist(),
        "data_points": len(rows),
        "computation_ms": round(computation_ms, 2),
        "base_timestamp": base_ts.isoformat(),
    }


@app.get("/analyze/latency/whatif")
def analyze_latency_whatif(
    slo_ms: float = Query(default=300, description="SLO threshold in ms"),
    fix_ms: float = Query(default=30, description="ms improvement applied to future rows"),
):
    """What-if latency regression — reads from Neon what-if branch, applies simulated fix to future rows."""
    t0 = time.perf_counter()

    whatif_url = NEON_WHATIF_URL or DATABASE_URL
    conn = psycopg2.connect(whatif_url)
    cur = conn.cursor()
    cur.execute("SELECT ts, p99_ms FROM latency_metrics ORDER BY ts")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    now_ts = datetime.now(timezone.utc)
    timestamps = [r[0] for r in rows]
    base_ts = timestamps[0]
    x = np.array([(t - base_ts).total_seconds() / 86400 for t in timestamps])

    y = np.array([
        max(r[1] - fix_ms, 50.0) if r[0] > now_ts and fix_ms > 0 else r[1]
        for r in rows
    ])

    result = _linear_regression(x, y)

    intercept, slope = result["coefficients"]
    if slope > 0:
        breach_day = (slo_ms - intercept) / slope
        breach_date = (base_ts + __import__('datetime').timedelta(days=breach_day)).isoformat()
        days_remaining = breach_day - x[-1]
    else:
        breach_date = None
        days_remaining = None

    computation_ms = (time.perf_counter() - t0) * 1000

    return {
        "scenario": "latency_drift_whatif",
        "branch": "whatif-regression",
        "branch_id": "br-bold-poetry-ajrn8b9g",
        "fix_applied_ms": fix_ms,
        "slo_ms": slo_ms,
        "regression": result,
        "prediction": {
            "breach_date": breach_date,
            "days_remaining": round(days_remaining, 1) if days_remaining else None,
            "trend_per_day_ms": round(slope, 3),
        },
        "x_days": x.tolist(),
        "y_values": y.tolist(),
        "data_points": len(rows),
        "computation_ms": round(computation_ms, 2),
        "base_timestamp": base_ts.isoformat(),
    }


@app.get("/analyze/errors")
def analyze_errors(
    budget_hours: float = Query(default=720, description="SLO window in hours (default 30 days)"),
    slo_target: float = Query(default=0.999, description="SLO target (e.g. 0.999 = 99.9%)")
):
    """Polynomial regression on error rate — predicts budget exhaustion."""
    t0 = time.perf_counter()
    
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT ts, error_rate, budget_remaining FROM error_budget ORDER BY ts")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    timestamps = [r[0] for r in rows]
    base_ts = timestamps[0]
    x = np.array([(t - base_ts).total_seconds() / 3600 for t in timestamps])
    y_rate = np.array([r[1] for r in rows])
    y_budget = np.array([r[2] for r in rows])
    
    rate_result = _polynomial_regression(x, y_rate, degree=2)
    budget_result = _polynomial_regression(x, y_budget, degree=2)
    
    coeffs = budget_result["coefficients"]
    roots = np.roots(coeffs)
    real_positive = [r.real for r in roots if np.isreal(r) and r.real > 0]
    
    if real_positive:
        exhaustion_hour = min(real_positive)
        exhaustion_date = (base_ts + __import__('datetime').timedelta(hours=exhaustion_hour)).isoformat()
        hours_remaining = exhaustion_hour - x[-1]
    else:
        exhaustion_date = None
        hours_remaining = None
    
    linear_result = _linear_regression(x, y_budget)
    lin_intercept, lin_slope = linear_result["coefficients"]
    if lin_slope < 0:
        linear_exhaustion_hour = -lin_intercept / lin_slope
        linear_days = (linear_exhaustion_hour - x[-1]) / 24
    else:
        linear_days = None
    
    computation_ms = (time.perf_counter() - t0) * 1000
    
    return {
        "scenario": "error_budget",
        "slo_target": slo_target,
        "error_rate_regression": rate_result,
        "budget_regression": budget_result,
        "linear_comparison": {
            "r_squared": linear_result["r_squared"],
            "linear_exhaustion_days": round(linear_days, 1) if linear_days else None,
        },
        "prediction": {
            "exhaustion_date": exhaustion_date,
            "hours_remaining": round(hours_remaining, 1) if hours_remaining else None,
            "days_remaining": round(hours_remaining / 24, 1) if hours_remaining else None,
            "current_budget_pct": round(float(y_budget[-1]) * 100, 2),
            "current_error_rate": round(float(y_rate[-1]) * 100, 4),
            "acceleration": "Error rate is accelerating" if coeffs[0] < 0 else "Error rate is decelerating",
        },
        "x_hours": x.tolist(),
        "y_error_rate": y_rate.tolist(),
        "y_budget": y_budget.tolist(),
        "data_points": len(rows),
        "computation_ms": round(computation_ms, 2),
        "base_timestamp": base_ts.isoformat(),
    }


@app.get("/analyze/capacity")
def analyze_capacity(
    threshold_gb: float = Query(default=500, description="Capacity limit in GB"),
    alert_pct: float = Query(default=80, description="Alert threshold percentage")
):
    """Exponential regression on disk usage — predicts capacity exhaustion."""
    t0 = time.perf_counter()
    
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT ts, used_gb FROM capacity_metrics ORDER BY ts")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    timestamps = [r[0] for r in rows]
    base_ts = timestamps[0]
    x = np.array([(t - base_ts).total_seconds() / 86400 for t in timestamps])
    y = np.array([r[1] for r in rows])
    
    exp_result = _exponential_regression(x, y)
    lin_result = _linear_regression(x, y)
    
    a, b = exp_result["coefficients"]
    if b > 0:
        exhaustion_day = np.log(threshold_gb / a) / b
        alert_day = np.log((threshold_gb * alert_pct / 100) / a) / b
        days_remaining = exhaustion_day - x[-1]
        days_to_alert = alert_day - x[-1]
        exhaustion_date = (base_ts + __import__('datetime').timedelta(days=exhaustion_day)).isoformat()
        alert_date = (base_ts + __import__('datetime').timedelta(days=alert_day)).isoformat()
    else:
        exhaustion_date = None
        alert_date = None
        days_remaining = None
        days_to_alert = None
    
    lin_intercept, lin_slope = lin_result["coefficients"]
    if lin_slope > 0:
        lin_exhaustion = (threshold_gb - lin_intercept) / lin_slope
        lin_days = lin_exhaustion - x[-1]
    else:
        lin_days = None
    
    doubling_time = np.log(2) / b if b > 0 else None
    
    computation_ms = (time.perf_counter() - t0) * 1000
    
    return {
        "scenario": "capacity_runway",
        "threshold_gb": threshold_gb,
        "alert_pct": alert_pct,
        "exponential_regression": exp_result,
        "linear_regression": lin_result,
        "prediction": {
            "exhaustion_date": exhaustion_date,
            "days_remaining": round(days_remaining, 1) if days_remaining else None,
            "alert_date": alert_date,
            "days_to_alert": round(days_to_alert, 1) if days_to_alert else None,
            "current_gb": round(float(y[-1]), 1),
            "daily_growth_rate_pct": round(b * 100, 3),
            "doubling_time_days": round(doubling_time, 1) if doubling_time else None,
            "linear_vs_exp_days": round(lin_days - days_remaining, 1) if (lin_days and days_remaining) else None,
        },
        "x_days": x.tolist(),
        "y_values": y.tolist(),
        "data_points": len(rows),
        "computation_ms": round(computation_ms, 2),
        "base_timestamp": base_ts.isoformat(),
    }


# ─── Branch info ──────────────────────────────────────────────────────────────

@app.get("/branch-info")
def branch_info():
    return {
        "feature": "neon_branching",
        "whatif_branch": "whatif-regression",
        "branch_id": "br-bold-poetry-ajrn8b9g",
        "description": "Neon creates instant copy-on-write database branches. Fork production data, inject a simulated fix, and compare regression trajectories — without touching production.",
        "use_cases": [
            "What-if: 'What happens to my p99 if we deploy this fix today?'",
            "A/B compare: regression results on production vs. what-if branch data",
            "Safe experimentation: changes affect branch only, not prod",
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


# ─── Spikes & Anomaly Detection ──────────────────────────────────────────────

import uuid
import urllib.request as _urllib_req
import urllib.parse as _urllib_parse

try:
    from kafka import KafkaProducer, KafkaAdminClient
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaProducer = KafkaAdminClient = None

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "")
KAFKA_USERNAME  = os.environ.get("KAFKA_USERNAME", "")
KAFKA_PASSWORD  = os.environ.get("KAFKA_PASSWORD", "")
KAFKA_TOPIC     = os.environ.get("KAFKA_TOPIC", "spike-events")

_REDIS_URL   = os.environ.get("REDIS_URL", "")
_REDIS_TOKEN = os.environ.get("REDIS_TOKEN", "")
_REDIS_KEY   = "spikes:events"


def _kafka_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-256",
        sasl_plain_username=KAFKA_USERNAME,
        sasl_plain_password=KAFKA_PASSWORD,
        value_serializer=lambda v: v if isinstance(v, bytes) else v.encode("utf-8"),
        key_serializer=lambda k: k if isinstance(k, bytes) else k.encode("utf-8"),
        request_timeout_ms=10000,
        api_version=(2, 5, 0),
    )


def _redis_req(method: str, path: str) -> dict:
    url = f"{_REDIS_URL}{path}"
    req = _urllib_req.Request(url, method=method,
                               headers={"Authorization": f"Bearer {_REDIS_TOKEN}"})
    try:
        with _urllib_req.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def _redis_push(event_json: str):
    encoded = _urllib_parse.quote(event_json, safe="")
    _redis_req("POST", f"/rpush/{_REDIS_KEY}/{encoded}")
    _redis_req("POST", f"/ltrim/{_REDIS_KEY}/-200/-1")


def _redis_lrange(start: int = 0, end: int = -1):
    data = _redis_req("GET", f"/lrange/{_REDIS_KEY}/{start}/{end}")
    return data.get("result", [])


def _redis_del():
    _redis_req("GET", f"/del/{_REDIS_KEY}")


INCIDENT_TEMPLATES = {
    "cascade": [
        {"service": "database",        "metric": "latency_ms",  "severity": "critical", "value_range": (800, 2000)},
        {"service": "cache",           "metric": "latency_ms",  "severity": "critical", "value_range": (500, 1200)},
        {"service": "api-gateway",     "metric": "error_rate",  "severity": "high",     "value_range": (0.15, 0.45)},
        {"service": "payment-service", "metric": "error_rate",  "severity": "high",     "value_range": (0.10, 0.35)},
        {"service": "database",        "metric": "cpu_pct",     "severity": "high",     "value_range": (85, 99)},
        {"service": "api-gateway",     "metric": "cpu_pct",     "severity": "warning",  "value_range": (70, 88)},
        {"service": "cache",           "metric": "cpu_pct",     "severity": "warning",  "value_range": (65, 82)},
    ],
    "memory_leak": [
        {"service": "api-gateway",     "metric": "memory_pct",  "severity": "warning",  "value_range": (72, 85)},
        {"service": "api-gateway",     "metric": "memory_pct",  "severity": "high",     "value_range": (85, 93)},
        {"service": "api-gateway",     "metric": "memory_pct",  "severity": "critical", "value_range": (93, 99)},
        {"service": "api-gateway",     "metric": "latency_ms",  "severity": "high",     "value_range": (400, 900)},
        {"service": "api-gateway",     "metric": "error_rate",  "severity": "high",     "value_range": (0.05, 0.18)},
    ],
    "random_spike": [
        {"service": "payment-service", "metric": "latency_ms",  "severity": "critical", "value_range": (1200, 3500)},
        {"service": "payment-service", "metric": "error_rate",  "severity": "high",     "value_range": (0.08, 0.22)},
        {"service": "database",        "metric": "cpu_pct",     "severity": "warning",  "value_range": (75, 90)},
        {"service": "payment-service", "metric": "cpu_pct",     "severity": "high",     "value_range": (80, 96)},
    ],
}


@app.get("/spikes/health")
def spikes_health():
    kafka_ok = False
    kafka_msg = "kafka-python not installed"
    if KAFKA_AVAILABLE and KAFKA_BOOTSTRAP:
        try:
            admin = KafkaAdminClient(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                security_protocol="SASL_SSL",
                sasl_mechanism="SCRAM-SHA-256",
                sasl_plain_username=KAFKA_USERNAME,
                sasl_plain_password=KAFKA_PASSWORD,
                request_timeout_ms=10000,
                api_version=(2, 5, 0),
            )
            admin.close()
            kafka_ok = True
            kafka_msg = f"connected — broker: {KAFKA_BOOTSTRAP.split(':')[0]}"
        except Exception as e:
            kafka_msg = str(e)

    redis_ok = False
    try:
        r = _redis_req("GET", "/ping")
        redis_ok = r.get("result") == "PONG"
    except Exception:
        pass

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), MIN(ts)::date, MAX(ts)::date FROM spike_metrics")
    row = cur.fetchone()
    cur.close(); conn.close()

    return {
        "kafka": {"ok": kafka_ok, "message": kafka_msg, "topic": KAFKA_TOPIC,
                  "broker": KAFKA_BOOTSTRAP.split(":")[0] if KAFKA_BOOTSTRAP else ""},
        "redis": {"ok": redis_ok},
        "timescale": {"rows": row[0], "from": str(row[1]), "to": str(row[2])},
    }


@app.get("/spikes/timescale")
def spikes_timescale(
    service: str = Query(default="api-gateway"),
    hours: int = Query(default=24, ge=1, le=720)
):
    t0 = time.perf_counter()
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT
          time_bucket('1 hour', ts) AS bucket,
          metric_name,
          ROUND(AVG(value)::numeric, 2)  AS avg_val,
          ROUND(MAX(value)::numeric, 2)  AS max_val,
          ROUND(MIN(value)::numeric, 2)  AS min_val,
          ROUND((percentile_cont(0.99) WITHIN GROUP (ORDER BY value))::numeric, 2) AS p99,
          COUNT(*) AS samples
        FROM spike_metrics
        WHERE ts > NOW() - (INTERVAL '1 hour' * %s)
          AND service = %s
        GROUP BY bucket, metric_name
        ORDER BY bucket DESC, metric_name
    """, (hours, service))
    buckets = []
    for r in cur.fetchall():
        d = dict(r)
        if d["bucket"]:
            d["bucket"] = d["bucket"].isoformat()
        buckets.append(d)

    cur.execute("""
        WITH stats AS (
          SELECT metric_name, AVG(value) AS mu, STDDEV(value) AS sigma
          FROM spike_metrics
          WHERE ts > NOW() - (INTERVAL '1 hour' * %s) AND service = %s
          GROUP BY metric_name
        )
        SELECT sm.ts, sm.metric_name, ROUND(sm.value::numeric, 2) AS value,
               ROUND(ABS((sm.value - s.mu) / NULLIF(s.sigma, 0))::numeric, 2) AS z_score
        FROM spike_metrics sm
        JOIN stats s ON sm.metric_name = s.metric_name
        WHERE sm.ts > NOW() - (INTERVAL '1 hour' * %s)
          AND sm.service = %s
          AND ABS((sm.value - s.mu) / NULLIF(s.sigma, 0)) > 2.5
        ORDER BY z_score DESC
        LIMIT 150
    """, (hours, service, hours, service))
    anomalies = []
    for r in cur.fetchall():
        anomalies.append({
            "ts": r["ts"].isoformat(),
            "metric_name": r["metric_name"],
            "value": float(r["value"]),
            "z_score": float(r["z_score"]),
        })

    cur.close(); conn.close()
    ts_ms = (time.perf_counter() - t0) * 1000

    return {
        "service": service,
        "hours": hours,
        "buckets": buckets,
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "timescaledb_ms": round(ts_ms, 2),
    }


@app.get("/spikes/benchmark")
def spikes_benchmark(
    service: str = Query(default="api-gateway"),
    hours: int = Query(default=24, ge=1, le=720)
):
    conn = get_conn()
    cur = conn.cursor()

    # Plain Postgres: date_trunc — standard SQL, no TimescaleDB extension needed
    t0 = time.perf_counter()
    cur.execute("""
        SELECT date_trunc('hour', ts) AS bucket,
               metric_name,
               AVG(value)::numeric AS avg_val,
               MAX(value)::numeric AS max_val,
               percentile_cont(0.99) WITHIN GROUP (ORDER BY value) AS p99,
               COUNT(*) AS samples
        FROM spike_metrics
        WHERE ts > NOW() - (INTERVAL '1 hour' * %s)
          AND service = %s
        GROUP BY bucket, metric_name
        ORDER BY bucket DESC, metric_name
        LIMIT 1000
    """, (hours, service))
    plain_rows = cur.fetchall()
    plain_ms = (time.perf_counter() - t0) * 1000

    # TimescaleDB: time_bucket — chunk-pruned scan, same result faster
    t0 = time.perf_counter()
    cur.execute("""
        SELECT time_bucket('1 hour', ts) AS bucket,
               metric_name,
               AVG(value)::numeric AS avg_val,
               MAX(value)::numeric AS max_val,
               percentile_cont(0.99) WITHIN GROUP (ORDER BY value) AS p99,
               COUNT(*) AS samples
        FROM spike_metrics
        WHERE ts > NOW() - (INTERVAL '1 hour' * %s)
          AND service = %s
        GROUP BY bucket, metric_name
        ORDER BY bucket DESC, metric_name
        LIMIT 1000
    """, (hours, service))
    ts_rows = cur.fetchall()
    ts_ms = (time.perf_counter() - t0) * 1000

    # Get actual total row count for display
    conn2 = get_conn()
    cur2 = conn2.cursor()
    cur2.execute("SELECT COUNT(*) FROM spike_metrics")
    total_rows = cur2.fetchone()[0]
    cur2.close(); conn2.close()

    cur.close(); conn.close()

    return {
        "plain_postgres": {
            "description": "date_trunc('hour') + percentile_cont — standard SQL, full index-range scan",
            "rows_returned": len(plain_rows),
            "ms": round(plain_ms, 2),
        },
        "timescaledb": {
            "description": "time_bucket(1h) + percentile_cont — chunk-pruned scan, identical output",
            "rows_returned": len(ts_rows),
            "ms": round(ts_ms, 2),
        },
        "speedup": round(plain_ms / max(ts_ms, 0.1), 2),
        "total_rows_in_table": total_rows,
        "window": f"{hours}h",
        "service": service,
    }


@app.post("/spikes/trigger")
def spikes_trigger(
    incident: str = Query(default="cascade"),
):
    import random as _random
    rng = _random.Random(int(time.time() * 1000) % 2**32)
    templates = INCIDENT_TEMPLATES.get(incident, INCIDENT_TEMPLATES["cascade"])
    base_ts = datetime.now(timezone.utc)
    events = []

    for i, tmpl in enumerate(templates):
        val = rng.uniform(*tmpl["value_range"])
        event = {
            "id": str(uuid.uuid4()),
            "ts": (base_ts + __import__("datetime").timedelta(seconds=i * 4)).isoformat(),
            "service": tmpl["service"],
            "metric": tmpl["metric"],
            "value": round(val, 4),
            "severity": tmpl["severity"],
            "incident_type": incident,
            "seq": i,
        }
        events.append(event)

    kafka_meta = {"status": "kafka-python not installed"}
    if KAFKA_AVAILABLE and KAFKA_BOOTSTRAP:
        try:
            p = _kafka_producer()
            futures = []
            for ev in events:
                ev_json = json.dumps(ev)
                future = p.send(KAFKA_TOPIC, key=ev["id"], value=ev_json)
                futures.append(future)
                _redis_push(ev_json)
            p.flush(timeout=10)
            p.close()
            kafka_meta = {
                "status": "published",
                "count": len(events),
                "topic": KAFKA_TOPIC,
                "broker": KAFKA_BOOTSTRAP.split(":")[0],
            }
        except Exception as e:
            kafka_meta = {"status": "error", "message": str(e)}
            for ev in events:
                _redis_push(json.dumps(ev))
    else:
        for ev in events:
            _redis_push(json.dumps(ev))
        kafka_meta = {"status": "kafka_unavailable_used_redis"}

    return {
        "incident": incident,
        "events": events,
        "published": len(events),
        "kafka": kafka_meta,
    }


@app.get("/spikes/events")
def spikes_events(limit: int = Query(default=50, ge=1, le=200)):
    raw = _redis_lrange(-limit, -1)
    events = []
    for r in raw:
        try:
            events.append(json.loads(r))
        except Exception:
            pass
    return {"events": list(reversed(events)), "count": len(events), "source": "redis_via_kafka"}


@app.delete("/spikes/events")
def clear_spikes_events():
    _redis_del()
    return {"cleared": True}


# ─── Scenario 3: OpenTelemetry Trace Explorer ─────────────────────────────────

def _span(name: str):
    """No-op-safe span context manager — works whether or not tracing is enabled."""
    if tracer:
        return tracer.start_as_current_span(name)
    from contextlib import nullcontext
    return nullcontext()


@app.get("/spikes/trace-demo")
def spikes_trace_demo(
    service: str = Query(default="api-gateway"),
    hours: int = Query(default=6, ge=1, le=168),
):
    """
    One request, three spans: DB query (Neon), Python compute (z-score anomaly
    detection), Redis cache (read-through). Trace ID returned so the frontend
    can deep-link straight into Grafana Tempo.
    """
    t0 = time.perf_counter()

    with _span("db.query_recent_metrics"):
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT ts, metric_name, value
            FROM spike_metrics
            WHERE ts > NOW() - (INTERVAL '1 hour' * %s) AND service = %s
            ORDER BY ts
        """, (hours, service))
        rows = cur.fetchall()
        cur.close()
        conn.close()

    with _span("compute.zscore_anomaly_detection"):
        by_metric = {}
        for r in rows:
            by_metric.setdefault(r["metric_name"], []).append(float(r["value"]))

        anomalies = []
        for metric_name, values in by_metric.items():
            arr = np.array(values)
            mu, sigma = arr.mean(), arr.std()
            if sigma == 0:
                continue
            z_scores = (arr - mu) / sigma
            for i, z in enumerate(z_scores):
                if abs(z) > 3:
                    anomalies.append({
                        "metric": metric_name,
                        "value": round(float(arr[i]), 4),
                        "z_score": round(float(z), 2),
                    })

    with _span("cache.redis_readthrough") as _:
        cache_key = f"trace-demo:{service}:{hours}h"
        cached = None
        try:
            cached = _redis_req("GET", f"/get/{cache_key}")
        except Exception:
            pass
        cache_hit = bool(cached and cached.get("result"))
        if not cache_hit:
            try:
                _redis_req("POST", f"/setex/{cache_key}/60/1")
            except Exception:
                pass

    computation_ms = (time.perf_counter() - t0) * 1000

    trace_id = None
    if tracer:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.trace_id:
            trace_id = format(ctx.trace_id, "032x")

    return {
        "service": service,
        "window": f"{hours}h",
        "rows_scanned": len(rows),
        "anomalies_found": len(anomalies),
        "anomalies": anomalies[:20],
        "cache_hit": cache_hit,
        "computation_ms": round(computation_ms, 2),
        "trace_id": trace_id,
        "tracing_enabled": TRACING_ENABLED,
    }
