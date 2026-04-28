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

DATABASE_URL = os.environ.get("DATABASE_URL", "")

def get_conn():
    return psycopg2.connect(DATABASE_URL)


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
    # Test DB connection on startup
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

app = FastAPI(title="Regression Analysis API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        # Initial guess from log-linear fit
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
    x = np.array([(t - base_ts).total_seconds() / 86400 for t in timestamps])  # days
    y = np.array([r[1] for r in rows])
    
    result = _linear_regression(x, y)
    
    # Predict SLO breach
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
    x = np.array([(t - base_ts).total_seconds() / 3600 for t in timestamps])  # hours
    y_rate = np.array([r[1] for r in rows])
    y_budget = np.array([r[2] for r in rows])
    
    # Fit polynomial to error rate
    rate_result = _polynomial_regression(x, y_rate, degree=2)
    
    # Fit polynomial to budget remaining
    budget_result = _polynomial_regression(x, y_budget, degree=2)
    
    # Predict budget exhaustion (when budget_remaining hits 0)
    # Solve: c[0]*x^2 + c[1]*x + c[2] = 0
    coeffs = budget_result["coefficients"]
    poly = np.poly1d(coeffs)
    roots = np.roots(coeffs)
    real_positive = [r.real for r in roots if np.isreal(r) and r.real > 0]
    
    if real_positive:
        exhaustion_hour = min(real_positive)
        exhaustion_date = (base_ts + __import__('datetime').timedelta(hours=exhaustion_hour)).isoformat()
        hours_remaining = exhaustion_hour - x[-1]
    else:
        exhaustion_date = None
        hours_remaining = None
    
    # Linear extrapolation for comparison
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
    x = np.array([(t - base_ts).total_seconds() / 86400 for t in timestamps])  # days
    y = np.array([r[1] for r in rows])
    
    # Exponential fit
    exp_result = _exponential_regression(x, y)
    
    # Linear fit for comparison
    lin_result = _linear_regression(x, y)
    
    # Predict capacity exhaustion
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
    
    # Linear prediction for comparison
    lin_intercept, lin_slope = lin_result["coefficients"]
    if lin_slope > 0:
        lin_exhaustion = (threshold_gb - lin_intercept) / lin_slope
        lin_days = lin_exhaustion - x[-1]
    else:
        lin_days = None
    
    # Doubling time
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


# ─── Branch comparison (Neon feature) ────────────────────────────────────────

@app.get("/branch-info")
def branch_info():
    """Return info about what Neon branching enables."""
    return {
        "feature": "neon_branching",
        "description": "Neon creates instant copy-on-write database branches. This lets you fork production data, inject a simulated scenario (e.g., traffic spike), and run regression against the what-if branch — without touching production.",
        "use_cases": [
            "What-if: 'What happens to my p99 if traffic doubles next month?'",
            "A/B compare: regression results on production vs. branch data",
            "Safe experimentation: inject anomalies into branch, not prod",
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
