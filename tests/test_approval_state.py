import pytest

from backend import store


def test_approval_state_roundtrip(tmp_path):
    db_path = tmp_path / "backend.sqlite3"
    store.init_backend_db(db_path)
    old_path = store.BACKEND_DB_PATH
    store.BACKEND_DB_PATH = db_path
    try:
        run_id = store.create_agent_run("sess", "user", "q", False, [])
        approval_id = store.create_approval(run_id, "催缴", "测试门店")
        pending = store.list_approvals(run_id=run_id, status="pending")
        assert pending[0]["approval_id"] == approval_id
        store.decide_approval(approval_id, "approved", decided_by="user")
        approved = store.list_approvals(run_id=run_id, status="approved")
        assert approved[0]["status"] == "approved"
    finally:
        store.BACKEND_DB_PATH = old_path


def test_invalid_approval_decision_rejected():
    with pytest.raises(ValueError):
        store.decide_approval("missing", "pending")
