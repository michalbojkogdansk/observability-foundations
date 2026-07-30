"""
Live event generator for the Observability Dashboard capstone.

Deliberately isolated from Scenario 2's pipeline:
  - Separate Kafka topic (obs-tool-events, not spike-events)
  - Separate Redis buffer key (obs_tool:events, not spikes:events)
Starting/stopping this can never interfere with the existing Spikes exercise.

Manual start/stop only, with a mandatory auto-stop timer — this is meant to be
switched on a few hours before a lecture/demo and left running, not a permanent
background job. If the process restarts (redeploy, Render sleep/wake), the
generator stops naturally; it must be restarted explicitly.
"""
import asyncio
import json
import random
import time
import uuid
from datetime import datetime, timezone

OBS_TOOL_TOPIC = "obs-tool-events"
OBS_TOOL_REDIS_KEY = "obs_tool:events"

SERVICES = ["api-gateway", "payment-service", "database", "cache"]
METRICS = ["latency_ms", "error_rate", "cpu_pct"]

# Same baseline shape as the seeded historical dataset, so live data looks
# continuous with history rather than a different universe.
BASELINE = {
    "api-gateway":     {"latency_ms": (45, 8),  "error_rate": (0.005, 0.003), "cpu_pct": (30, 5)},
    "payment-service": {"latency_ms": (150, 20), "error_rate": (0.010, 0.005), "cpu_pct": (40, 6)},
    "database":        {"latency_ms": (35, 10), "error_rate": (0.002, 0.0015), "cpu_pct": (55, 8)},
    "cache":           {"latency_ms": (4, 1.5), "error_rate": (0.001, 0.001), "cpu_pct": (20, 4)},
}

PUBLISH_INTERVAL_SECONDS = 5
# Small, occasional live incident so a demo has something to point at —
# rare enough to stay "presentation-savvy", not constant noise.
INCIDENT_CHANCE_PER_TICK = 0.03
INCIDENT_DURATION_TICKS = 6  # ~30s at 5s interval


class GeneratorState:
    def __init__(self):
        self.running = False
        self.started_at = None
        self.stop_at = None
        self.task = None
        self.producer = None
        self._incident_ticks_left = 0
        self._incident_service = None

    def status(self):
        remaining = None
        if self.running and self.stop_at:
            remaining = max(0, int(self.stop_at - time.time()))
        return {
            "running": self.running,
            "started_at": self.started_at,
            "stop_at": self.stop_at,
            "remaining_seconds": remaining,
            "topic": OBS_TOOL_TOPIC,
        }


state = GeneratorState()


def _gen_value(mean, std, mult=1.0):
    v = random.gauss(mean * mult, std * mult * 0.6)
    return max(0.0, round(v, 4))


def _next_batch():
    """One tick of events across all services/metrics, with a rare short incident."""
    now = datetime.now(timezone.utc).isoformat()
    events = []

    if state._incident_ticks_left > 0:
        state._incident_ticks_left -= 1
    elif random.random() < INCIDENT_CHANCE_PER_TICK:
        state._incident_ticks_left = INCIDENT_DURATION_TICKS
        state._incident_service = random.choice(["database", "cache", "payment-service"])

    incident_active = state._incident_ticks_left > 0
    incident_svc = state._incident_service

    for svc in SERVICES:
        for metric in METRICS:
            mean, std = BASELINE[svc][metric]
            mult = 1.0
            if incident_active and svc == incident_svc and metric == "latency_ms":
                mult = 4.0
            if incident_active and svc == incident_svc and metric == "error_rate":
                mult = 8.0
            value = _gen_value(mean, std, mult)
            if metric == "error_rate":
                value = min(value, 1.0)
            events.append({
                "id": str(uuid.uuid4()),
                "ts": now,
                "service": svc,
                "metric": metric,
                "value": value,
                "incident": bool(incident_active and svc == incident_svc),
            })
    return events


def _publish_batch_blocking(producer, events, redis_push_fn):
    """Runs on a worker thread — safe to block here, never on the event loop."""
    for ev in events:
        ev_json = json.dumps(ev)
        producer.send(OBS_TOOL_TOPIC, key=ev["id"], value=ev_json)
        redis_push_fn(ev_json)
    producer.flush(timeout=3)


async def _run_loop(kafka_producer_fn, redis_push_fn):
    loop = asyncio.get_running_loop()
    try:
        state.producer = await loop.run_in_executor(None, kafka_producer_fn)
    except Exception as e:
        print(f"[obs_tool_live] failed to create producer: {e}")
        state.running = False
        state.task = None
        return

    try:
        while state.running:
            if state.stop_at and time.time() >= state.stop_at:
                break
            try:
                events = _next_batch()
                await loop.run_in_executor(None, _publish_batch_blocking, state.producer, events, redis_push_fn)
            except Exception as e:
                # Publish failures shouldn't kill the loop mid-demo — log and keep going.
                print(f"[obs_tool_live] publish error: {e}")
            await asyncio.sleep(PUBLISH_INTERVAL_SECONDS)
    finally:
        if state.producer:
            try:
                await loop.run_in_executor(None, state.producer.close)
            except Exception:
                pass
            state.producer = None
        state.running = False
        state.task = None


def start(kafka_producer_fn, redis_push_fn, hours: float):
    if state.running:
        return {"already_running": True, **state.status()}
    state.running = True
    state.started_at = datetime.now(timezone.utc).isoformat()
    state.stop_at = time.time() + hours * 3600
    state.task = asyncio.create_task(_run_loop(kafka_producer_fn, redis_push_fn))
    return {"started": True, **state.status()}


def stop():
    was_running = state.running
    state.running = False
    return {"stopped": was_running, **state.status()}
