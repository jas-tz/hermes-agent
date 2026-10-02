"""One-shot jobs must survive until their run reaches a terminal state.

Regression for the orphaned run of job 6624271384b8: a gateway tick claimed
and started a one-shot (repeat 1/1, last_run_at unset); 19s later a manual
``cronjob(action='run')`` from a separate CLI process called claim_dispatch(),
which saw the dispatch limit reached with last_run_at unset, judged the job
"wedged", and removed it while the first run was still executing. The run
finished six minutes later and mark_job_run() found no job to record into.

Every test uses an isolated temp jobs store; no real cron job is created.
"""

from datetime import timedelta

import pytest

import cron.jobs as jobs_mod
from cron.jobs import (
    _hermes_now,
    _oneshot_run_claim_ttl_seconds,
    claim_dispatch,
    get_due_jobs,
    get_job,
    heartbeat_run_claim,
    load_jobs,
    mark_job_run,
    save_jobs,
)


@pytest.fixture
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.delenv("HERMES_CRON_TIMEOUT", raising=False)
    return tmp_path


@pytest.fixture
def other_process(monkeypatch):
    """Simulate a caller in a different process: its running set is empty."""
    monkeypatch.setattr(jobs_mod, "_job_running_in_this_process", lambda job_id: False)


def _due_oneshot(job_id="race1", *, times=1):
    run_at = (_hermes_now() - timedelta(seconds=5)).isoformat()
    return {
        "id": job_id,
        "name": "JARVIS SCHEDULING TEST",
        "prompt": "x",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "once", "run_at": run_at},
        "next_run_at": run_at,
        "repeat": {"times": times, "completed": 0},
    }


def _at(monkeypatch, t):
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: t)


# --- the reported race --------------------------------------------------------

def test_second_dispatch_while_run_in_flight_keeps_job(tmp_cron_dir, other_process, monkeypatch):
    """The 6624271384b8 sequence: tick claims + dispatches, then a manual run
    from another process tries to dispatch while the first run is running."""
    t0 = _hermes_now()
    save_jobs([_due_oneshot()])

    assert [j["id"] for j in get_due_jobs()] == ["race1"]   # tick: claim stamped
    assert claim_dispatch("race1") is True                   # dispatch slot 1/1

    _at(monkeypatch, t0 + timedelta(seconds=19))             # worker still running
    assert claim_dispatch("race1") is False                  # limit still enforced
    job = get_job("race1")
    assert job is not None, "in-flight one-shot was removed"
    assert job["repeat"]["completed"] == 1                   # not re-claimed
    assert job.get("last_run_at") is None

    mark_job_run("race1", True)                              # run finishes
    done = get_job("race1")
    assert done["last_status"] == "ok" and done["state"] == "completed"


def test_ticker_tick_while_manual_run_in_flight_keeps_job(tmp_cron_dir, other_process, monkeypatch):
    """Dispatch claimed (manual run, no prior tick) → worker still running →
    ticker tick in another process → the job must remain and not re-fire."""
    t0 = _hermes_now()
    save_jobs([_due_oneshot()])

    assert claim_dispatch("race1") is True
    assert get_job("race1").get("run_claim") is not None, "manual dispatch left no in-flight marker"

    for gap in (60, 120, 600):                                # successive 60s ticks
        _at(monkeypatch, t0 + timedelta(seconds=gap))
        assert get_due_jobs() == [], f"re-dispatched at +{gap}s"
        assert get_job("race1") is not None, f"removed at +{gap}s while running"


def test_job_running_in_this_process_is_kept_without_claim(tmp_cron_dir, monkeypatch):
    save_jobs([{**_due_oneshot(), "repeat": {"times": 1, "completed": 1}}])
    monkeypatch.setattr(jobs_mod, "_job_running_in_this_process", lambda job_id: True)
    assert claim_dispatch("race1") is False
    assert get_job("race1") is not None


def test_heartbeat_keeps_long_manual_run_protected_past_ttl(tmp_cron_dir, other_process, monkeypatch):
    t0 = _hermes_now()
    ttl = _oneshot_run_claim_ttl_seconds()
    save_jobs([_due_oneshot()])
    assert claim_dispatch("race1") is True
    owner = get_job("race1")["run_claim"]["by"]

    _at(monkeypatch, t0 + timedelta(seconds=ttl - 60))
    assert heartbeat_run_claim("race1", expected_owner=owner) is True
    _at(monkeypatch, t0 + timedelta(seconds=ttl + 30))       # past the original TTL
    assert claim_dispatch("race1") is False
    assert get_due_jobs() == []
    assert get_job("race1") is not None


# --- terminal outcomes ----------------------------------------------------------

def test_oneshot_success_reaches_terminal_record(tmp_cron_dir, other_process):
    save_jobs([_due_oneshot()])
    assert [j["id"] for j in get_due_jobs()] == ["race1"]
    assert claim_dispatch("race1") is True
    assert get_due_jobs() == [] and get_job("race1") is not None   # mid-run tick

    mark_job_run("race1", True)
    job = get_job("race1")
    assert job["state"] == "completed" and job["enabled"] is False
    assert job["last_status"] == "ok" and job["last_run_at"] is not None
    assert job["run_claim"] is None and job["repeat"]["completed"] == 1
    assert get_due_jobs() == [] and claim_dispatch("race1") is False   # never re-fires
    assert get_job("race1") is not None                                  # retained record


def test_oneshot_failure_reaches_terminal_record(tmp_cron_dir, other_process):
    save_jobs([_due_oneshot()])
    assert [j["id"] for j in get_due_jobs()] == ["race1"]
    assert claim_dispatch("race1") is True
    assert get_due_jobs() == [] and get_job("race1") is not None

    mark_job_run("race1", False, "boom")
    job = get_job("race1")
    assert job["state"] == "completed" and job["enabled"] is False
    assert job["last_status"] == "error" and job["last_error"] == "boom"
    assert job["run_claim"] is None
    assert get_due_jobs() == [] and claim_dispatch("race1") is False
    assert get_job("race1") is not None


# --- protections that must not change ---------------------------------------------

def test_genuinely_wedged_claim_is_still_removed(tmp_cron_dir, other_process, monkeypatch):
    """Claim expired (owner died, no heartbeat) and not running here: the
    existing wedge cleanup still removes the job and writes a diagnostic."""
    t0 = _hermes_now()
    save_jobs([_due_oneshot()])
    assert claim_dispatch("race1") is True
    _at(monkeypatch, t0 + timedelta(seconds=_oneshot_run_claim_ttl_seconds() + 30))
    assert claim_dispatch("race1") is False
    assert load_jobs() == []
    out_dir = tmp_cron_dir / "cron" / "output" / "race1"
    assert any(out_dir.glob("*.md")), "no wedge diagnostic written"


def test_future_dated_claim_does_not_protect_forever(tmp_cron_dir, other_process):
    future = (_hermes_now() + timedelta(hours=2)).isoformat()
    save_jobs([{**_due_oneshot(), "repeat": {"times": 1, "completed": 1},
                "run_claim": {"at": future, "by": "skewed-host"}}])
    assert claim_dispatch("race1") is False
    assert load_jobs() == []


def test_recurring_job_dispatch_unchanged(tmp_cron_dir):
    save_jobs([{
        "id": "rec1", "name": "r", "prompt": "x", "enabled": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "repeat": {"times": 3, "completed": 0},
    }])
    assert claim_dispatch("rec1") is True
    job = get_job("rec1")
    assert job["repeat"]["completed"] == 0          # recurring jobs don't pre-claim
    assert job.get("run_claim") is None             # and get no one-shot claim


def test_run_one_job_gives_manual_run_a_heartbeat_owner(tmp_cron_dir, monkeypatch):
    """A manual run's job dict has no run_claim; run_one_job must hand run_job
    the claim claim_dispatch stored, so the heartbeat keeps it fresh."""
    import cron.scheduler as s

    save_jobs([_due_oneshot()])
    seen = {}

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        seen["run_claim"] = job.get("run_claim")
        seen["stored"] = get_job("race1").get("run_claim")
        return (True, "out", "final", None)

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: "/dev/null")
    monkeypatch.setattr(s, "_deliver_result", lambda *a, **k: None)
    manual_job = {k: v for k, v in get_job("race1").items() if k != "run_claim"}
    assert s.run_one_job(manual_job) is True
    assert seen["run_claim"] is not None and seen["run_claim"] == seen["stored"]
    assert get_job("race1")["state"] == "completed"   # mark_job_run landed
