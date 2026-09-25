"""End-to-end tests for reservation renewals (TTL extension)."""
from __future__ import annotations

import asyncio
import dataclasses
import os
import tempfile
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest
import pytest_asyncio

# Point the ledger at an isolated database before importing the app.
_tmp = tempfile.mkdtemp(prefix="dpledger-test-renew-")
os.environ["DATABASE_PATH"] = os.path.join(_tmp, "renewals.db")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app import db  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

import itertools

_counter = itertools.count()


def _new_dataset_id(name: str) -> str:
    return f"dsren-{name}-{next(_counter)}"


@pytest_asyncio.fixture()
async def client():
    await db.init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register(client, dataset_id: str, eps: float = 1.0, delta: float = 1e-6):
    return await client.post(
        f"/datasets/{dataset_id}",
        json={"epsilon_cap": eps, "delta_cap": delta,
              "description": "test dataset"},
    )


def _res(reservation_id: str, epsilon: float = 0.1,
         delta: float = 1e-7, ttl: int = 600) -> dict:
    return {
        "reservation_id": reservation_id,
        "epsilon": epsilon,
        "delta": delta,
        "ttl_seconds": ttl,
    }


def _renew(renewal_id: str, extend_seconds: int) -> dict:
    return {"renewal_id": renewal_id, "extend_seconds": extend_seconds}


async def _renew_url(client, ds, rid, payload):
    return await client.post(
        f"/datasets/{ds}/reservations/{rid}/renewals", json=payload
    )


async def _backdate(dataset_id: str, reservation_id: str,
                    expires_at: datetime) -> None:
    """Force a reservation's expiry instant into the past/future."""
    async with db.connect() as conn:
        await db.backdate_reservation_expiry(
            conn,
            dataset_id=dataset_id,
            reservation_id=reservation_id,
            expires_at=expires_at.isoformat(),
        )
        await conn.commit()


# --------------------------------------------------------------- basic flow --


@pytest.mark.asyncio
async def test_renewal_extends_expiry_without_changing_amount_or_ttl(client):
    ds = _new_dataset_id("basic")
    await _register(client, ds, eps=1.0, delta=1e-6)

    created = await client.post(
        f"/datasets/{ds}/reservations",
        json=_res("rv", epsilon=0.2, ttl=600),
    )
    assert created.status_code == 201
    first = created.json()
    old_expires = datetime.fromisoformat(first["expires_at"])

    budget_before = (await client.get(f"/datasets/{ds}/budget")).json()

    r = await _renew_url(client, ds, "rv", _renew("ren-1", 300))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["idempotent_replay"] is False
    assert body["reservation_id"] == "rv"
    assert body["renewal_id"] == "ren-1"
    assert body["extend_seconds"] == 300
    assert body["ttl_seconds"] == 600  # original TTL unchanged

    expires_before = datetime.fromisoformat(body["expires_before"])
    expires_after = datetime.fromisoformat(body["expires_after"])
    assert expires_before == old_expires
    assert expires_after - expires_before == timedelta(seconds=300)

    max_expires = datetime.fromisoformat(body["max_expires_at"])
    assert max_expires - datetime.fromisoformat(first["created_at"]) == timedelta(
        seconds=90 * 24 * 3600
    )
    assert expires_after <= max_expires

    # The reservation view reports the new expiry, same ttl, still granted.
    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert got["status"] == "granted"
    assert got["ttl_seconds"] == 600
    assert datetime.fromisoformat(got["expires_at"]) == expires_after

    # Renewal re-locks nothing: reserved epsilon / remaining budget unchanged.
    budget_after = (await client.get(f"/datasets/{ds}/budget")).json()
    assert budget_after["active_reservations"] == 1
    assert budget_after["epsilon_reserved"] == pytest.approx(
        budget_before["epsilon_reserved"], abs=1e-12
    )
    assert budget_after["epsilon_remaining"] == pytest.approx(
        budget_before["epsilon_remaining"], abs=1e-12
    )
    assert budget_after["granted_requests"] == 0


@pytest.mark.asyncio
async def test_renewal_replay_returns_first_before_and_after(client):
    ds = _new_dataset_id("replay")
    await _register(client, ds)
    first = (await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", ttl=600)
    )).json()

    created = await _renew_url(client, ds, "rv", _renew("ren", 120))
    assert created.status_code == 201
    first_body = created.json()

    # A second, different renewal moves the expiry again.
    second = await _renew_url(client, ds, "rv", _renew("ren-2", 600))
    assert second.status_code == 201
    assert datetime.fromisoformat(second.json()["expires_before"]) == \
        datetime.fromisoformat(first_body["expires_after"])

    # Replaying the first renewal_id returns the ORIGINAL before/after instants,
    # not the current expiry.
    for _ in range(3):
        again = await _renew_url(client, ds, "rv", _renew("ren", 120))
        assert again.status_code == 200
        b = again.json()
        assert b["idempotent_replay"] is True
        assert b["expires_before"] == first_body["expires_before"]
        assert b["expires_after"] == first_body["expires_after"]
        assert b["created_at"] == first_body["created_at"]
        assert b["ttl_seconds"] == 600


@pytest.mark.asyncio
async def test_same_renewal_id_different_extend_conflicts(client):
    ds = _new_dataset_id("diffext")
    await _register(client, ds)
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv"))

    ok = await _renew_url(client, ds, "rv", _renew("ren", 100))
    assert ok.status_code == 201

    bad = await _renew_url(client, ds, "rv", _renew("ren", 200))
    assert bad.status_code == 409
    assert bad.json()["error"] == "renewal_id_conflict"
    assert bad.json()["renewal_id"] == "ren"

    # The failed conflict did not move the expiry.
    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert datetime.fromisoformat(got["expires_at"]) == \
        datetime.fromisoformat(ok.json()["expires_after"])


@pytest.mark.asyncio
async def test_same_renewal_id_on_other_reservation_conflicts(client):
    ds = _new_dataset_id("diffres")
    await _register(client, ds, eps=1.0)
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv-a", epsilon=0.1))
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv-b", epsilon=0.1))

    ok = await _renew_url(client, ds, "rv-a", _renew("shared", 60))
    assert ok.status_code == 201

    bad = await _renew_url(client, ds, "rv-b", _renew("shared", 60))
    assert bad.status_code == 409
    assert bad.json()["error"] == "renewal_id_conflict"

    # rv-b's expiry is untouched.
    rv_b = (await client.get(f"/datasets/{ds}/reservations/rv-b")).json()
    assert rv_b["expires_at"] is not None


@pytest.mark.asyncio
async def test_renewal_id_is_scoped_per_dataset(client):
    ds1 = _new_dataset_id("scope-a")
    ds2 = _new_dataset_id("scope-b")
    await _register(client, ds1)
    await _register(client, ds2)
    await client.post(f"/datasets/{ds1}/reservations", json=_res("rv"))
    await client.post(f"/datasets/{ds2}/reservations", json=_res("rv"))

    r1 = await _renew_url(client, ds1, "rv", _renew("dup", 60))
    r2 = await _renew_url(client, ds2, "rv", _renew("dup", 60))
    assert r1.status_code == 201
    assert r2.status_code == 201


# ------------------------------------------------------------- error states --


@pytest.mark.asyncio
async def test_renew_unknown_reservation_404(client):
    ds = _new_dataset_id("missingres")
    await _register(client, ds)
    r = await _renew_url(client, ds, "nope", _renew("ren", 60))
    assert r.status_code == 404
    assert r.json()["error"] == "reservation_not_found"


@pytest.mark.asyncio
async def test_renew_on_missing_dataset_404(client):
    r = await _renew_url(client, "no-such-dataset", "rv", _renew("ren", 60))
    assert r.status_code == 404
    assert r.json()["error"] == "dataset_not_found"


@pytest.mark.asyncio
async def test_renew_expired_reservation_conflicts(client):
    ds = _new_dataset_id("expired")
    await _register(client, ds)
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", ttl=1)
    )
    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    await _backdate(ds, "rv", past)

    r = await _renew_url(client, ds, "rv", _renew("ren", 600))
    assert r.status_code == 409
    assert r.json()["error"] == "reservation_expired"
    # The failed renewal did not revive the reservation.
    assert (await client.get(
        f"/datasets/{ds}/reservations/rv")).json()["status"] == "expired"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_action", ["confirmed", "released", "rejected"])
async def test_renew_terminal_reservations_conflict(client, terminal_action):
    ds = _new_dataset_id(f"terminal-{terminal_action}")
    await _register(client, ds, eps=1.0, delta=1e-6)

    if terminal_action == "rejected":
        await client.post(
            f"/datasets/{ds}/reservations", json=_res("rv", epsilon=5.0)
        )
    else:
        await client.post(
            f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.1)
        )
        if terminal_action == "confirmed":
            await client.post(
                f"/datasets/{ds}/reservations/rv/confirm",
                json={"request_id": "req-1", "query_type": "q"},
            )
        else:
            await client.post(f"/datasets/{ds}/reservations/rv/release")

    r = await _renew_url(client, ds, "rv", _renew("ren", 60))
    assert r.status_code == 409
    assert r.json()["error"] == f"reservation_{terminal_action}"


@pytest.mark.asyncio
async def test_renewal_beyond_90_days_conflicts_and_changes_nothing(client):
    # 1-hour TTL: the renewed expiry may be at most created + 90 days, i.e. the
    # largest single extension is 90 days - 1 hour = 7772400 seconds.
    ds = _new_dataset_id("maxttl")
    await _register(client, ds)
    created = (await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", ttl=3600)
    )).json()
    original_expires = created["expires_at"]

    too_big = await _renew_url(client, ds, "rv", _renew("ren-bad", 90 * 24 * 3600))
    assert too_big.status_code == 409
    assert too_big.json()["error"] == "reservation_renewal_exceeds_max_ttl"

    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert datetime.fromisoformat(got["expires_at"]) == \
        datetime.fromisoformat(original_expires)

    # No renewal record was written for the rejected attempt: its id is reusable.
    exact = await _renew_url(
        client, ds, "rv", _renew("ren-bad", 90 * 24 * 3600 - 3600)
    )
    assert exact.status_code == 201
    expires_after = datetime.fromisoformat(exact.json()["expires_after"])
    created_at = datetime.fromisoformat(created["created_at"])
    assert expires_after == created_at + timedelta(days=90)


@pytest.mark.asyncio
async def test_cumulative_renewals_still_bounded_by_original_creation(client):
    ds = _new_dataset_id("cumulative")
    await _register(client, ds)
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", ttl=3600)
    )

    r1 = await _renew_url(client, ds, "rv", _renew("ren-1", 3600))
    assert r1.status_code == 201

    # Second renewal starts from the already-extended expiry; pushing beyond
    # the original-creation + 90 days ceiling is rejected.
    over = await _renew_url(
        client, ds, "rv", _renew("ren-2", 90 * 24 * 3600)
    )
    assert over.status_code == 409
    assert over.json()["error"] == "reservation_renewal_exceeds_max_ttl"

    # A smaller second renewal is fine.
    r2 = await _renew_url(client, ds, "rv", _renew("ren-3", 120))
    assert r2.status_code == 201
    assert datetime.fromisoformat(r2.json()["expires_before"]) == \
        datetime.fromisoformat(r1.json()["expires_after"])
    assert datetime.fromisoformat(r2.json()["expires_after"]) == \
        datetime.fromisoformat(r1.json()["expires_after"]) + timedelta(seconds=120)


@pytest.mark.asyncio
async def test_renew_payload_validation(client):
    ds = _new_dataset_id("validation")
    await _register(client, ds)
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv"))

    for bad_payload in (
        {"renewal_id": "ren", "extend_seconds": 0},
        {"renewal_id": "ren", "extend_seconds": -10},
        {"renewal_id": "ren", "extend_seconds": 1.5},
        {"renewal_id": "bad id!", "extend_seconds": 60},
        {"renewal_id": "", "extend_seconds": 60},
        {"extend_seconds": 60},
        {"renewal_id": "ren"},
    ):
        r = await _renew_url(client, ds, "rv", bad_payload)
        assert r.status_code == 422, (bad_payload, r.status_code, r.text)


# ------------------------------------------------- interaction with the rest --


@pytest.mark.asyncio
async def test_renewed_reservation_stays_locked_past_old_expiry(client):
    ds = _new_dataset_id("locked")
    await _register(client, ds, eps=0.3, delta=1e-6)

    body = (await client.post(
        f"/datasets/{ds}/reservations",
        json=_res("rv", epsilon=0.25, ttl=1),
    )).json()
    old_expires = datetime.fromisoformat(body["expires_at"])

    renewed = await _renew_url(client, ds, "rv", _renew("ren", 3600))
    assert renewed.status_code == 201
    new_expires = datetime.fromisoformat(renewed.json()["expires_after"])
    assert new_expires - old_expires == timedelta(seconds=3600)

    # Let the ORIGINAL 1-second TTL elapse; without the renewal the
    # reservation would be swept to "expired" on the next read.
    await asyncio.sleep(2)
    got = await client.get(f"/datasets/{ds}/reservations/rv")
    assert got.json()["status"] == "granted"

    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    assert budget["active_reservations"] == 1
    assert budget["epsilon_reserved"] > 0.0

    # A direct request that only fits once the reservation is gone is blocked.
    blocked = await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "q1", "query_type": "q",
              "epsilon": 0.2, "delta": 1e-7},
    )
    assert blocked.json()["allowed"] is False


@pytest.mark.asyncio
async def test_renewal_feeds_release_forecast_with_new_expiry(client):
    ds = _new_dataset_id("forecast")
    await _register(client, ds, eps=1.0, delta=1e-6)
    created = (await client.post(
        f"/datasets/{ds}/reservations",
        json=_res("rv", epsilon=0.2, ttl=600),
    )).json()
    old_expires = datetime.fromisoformat(created["expires_at"])

    renewed = await _renew_url(client, ds, "rv", _renew("ren", 7200))
    new_expires = datetime.fromisoformat(renewed.json()["expires_after"])

    # Deadline after the old expiry but before the renewed one: without the
    # renewal the reservation would be simulated as released; with it, it must
    # stay locked for the whole horizon.
    deadline = (old_expires + timedelta(seconds=600)).astimezone(timezone.utc)
    assert deadline < new_expires
    forecast = (await client.get(
        f"/datasets/{ds}/budget/release-forecast",
        params={"deadline": deadline.isoformat()},
    )).json()
    assert forecast["events"] == []
    assert forecast["at_deadline"]["active_reservations"] == 1
    assert forecast["at_deadline"]["epsilon_reserved"] > 0.0

    # Deadline past the renewed expiry shows the single release at the NEW time.
    deadline2 = (new_expires + timedelta(seconds=10)).astimezone(timezone.utc)
    forecast2 = (await client.get(
        f"/datasets/{ds}/budget/release-forecast",
        params={"deadline": deadline2.isoformat()},
    )).json()
    assert len(forecast2["events"]) == 1
    event = forecast2["events"][0]
    assert event["released_reservation_ids"] == ["rv"]
    assert datetime.fromisoformat(event["expires_at"]) == new_expires
    assert event["active_reservations"] == 0


@pytest.mark.asyncio
async def test_confirm_after_renewal_charges_budget_once(client):
    ds = _new_dataset_id("confirm")
    await _register(client, ds, eps=1.0, delta=1e-6)
    reserved = (await client.post(
        f"/datasets/{ds}/reservations",
        json=_res("rv", epsilon=0.2, ttl=3600),
    )).json()

    renewed = await _renew_url(client, ds, "rv", _renew("ren", 3600))
    assert renewed.status_code == 201

    verdict = await client.post(
        f"/datasets/{ds}/reservations/rv/confirm",
        json={"request_id": "req-1", "query_type": "gaussian_mean"},
    )
    assert verdict.status_code == 201
    v = verdict.json()
    assert v["allowed"] is True
    # Confirmation must not double-charge the (renewed) reservation.
    assert v["epsilon_after"] == pytest.approx(
        reserved["epsilon_after"], abs=1e-9
    )
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    assert budget["active_reservations"] == 0
    assert budget["granted_requests"] == 1


@pytest.mark.asyncio
async def test_renewal_replay_after_reservation_expired_returns_record(client):
    ds = _new_dataset_id("replayexpired")
    await _register(client, ds)
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", ttl=600)
    )
    first = await _renew_url(client, ds, "rv", _renew("ren", 300))
    assert first.status_code == 201
    first_body = first.json()

    # Expire the reservation (push the renewed expiry into the past).
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    await _backdate(ds, "rv", past)
    assert (await client.get(
        f"/datasets/{ds}/reservations/rv")).json()["status"] == "expired"

    # Same renewal_id + same payload: stored record replays with its original
    # before/after instants even though the reservation is now terminal.
    replay = await _renew_url(client, ds, "rv", _renew("ren", 300))
    assert replay.status_code == 200
    assert replay.json()["idempotent_replay"] is True
    assert replay.json()["expires_before"] == first_body["expires_before"]
    assert replay.json()["expires_after"] == first_body["expires_after"]

    # A brand-new renewal on the expired reservation is still rejected.
    fresh = await _renew_url(client, ds, "rv", _renew("ren-late", 60))
    assert fresh.status_code == 409
    assert fresh.json()["error"] == "reservation_expired"


# ------------------------------------------------------------- concurrency --


@pytest.mark.asyncio
async def test_concurrent_identical_renewals_serialise_to_one_extension(client):
    ds = _new_dataset_id("conc")
    await _register(client, ds)
    created = (await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", ttl=7200)
    )).json()
    old_expires = datetime.fromisoformat(created["expires_at"])
    payload = _renew("ren", 300)

    results = await asyncio.gather(*[
        _renew_url(client, ds, "rv", payload) for _ in range(20)
    ])
    statuses = [r.status_code for r in results]
    assert statuses.count(201) == 1
    assert statuses.count(200) == 19

    # The expiry moved exactly once, not 20 times.
    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert datetime.fromisoformat(got["expires_at"]) == \
        old_expires + timedelta(seconds=300)

    # Exactly one immutable renewal row exists.
    async with db.connect() as conn:
        async with conn.execute(
            "SELECT COUNT(*) FROM reservation_renewals "
            "WHERE dataset_id = ? AND renewal_id = 'ren'",
            (ds,),
        ) as cur:
            assert (await cur.fetchone())[0] == 1


# ------------------------------------------------------------- migration v6 --


@pytest.mark.asyncio
async def test_migration_from_v5_schema_adds_renewal_table(
    tmp_path, monkeypatch
):
    old_db_path = str(tmp_path / "v5.db")

    conn = await aiosqlite.connect(old_db_path)
    try:
        for v in (1, 2, 3, 4, 5):
            await conn.executescript(db.MIGRATIONS[v])
        await conn.execute("PRAGMA user_version = 5")
        await conn.commit()
    finally:
        await conn.close()

    patched = dataclasses.replace(settings, db_path=old_db_path)
    monkeypatch.setattr(db, "settings", patched)
    await db.init_db()

    async with db.connect() as c:
        async with c.execute("PRAGMA user_version") as cur:
            assert (await cur.fetchone())[0] == db.LATEST_VERSION
        async with c.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='reservation_renewals'"
        ) as cur:
            assert await cur.fetchone() is not None
        # Pre-existing reservations are untouched.
        async with c.execute("SELECT COUNT(*) FROM reservations") as cur:
            assert (await cur.fetchone())[0] == 0
