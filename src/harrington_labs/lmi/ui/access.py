"""Role-based access for Harrington LMI. Same architecture as other Harrington apps."""
import hashlib
import json
from pathlib import Path
import streamlit as st

_ACCESS_FILE = Path("data/manual/access.json")
_DEFAULT_HASH = hashlib.sha256("REDACTED_DEFAULT_PASSWORD".encode()).hexdigest()


def _load_access() -> dict:
    if _ACCESS_FILE.exists():
        return json.loads(_ACCESS_FILE.read_text())
    return {"admin_hash": _DEFAULT_HASH}


def _save_access(data: dict):
    _ACCESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ACCESS_FILE.write_text(json.dumps(data, indent=2))


def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def is_admin() -> bool:
    return st.session_state.get("lmi_admin", False)


def require_admin() -> bool:
    if is_admin():
        return True
    st.warning("This page requires admin access.")
    pw = st.text_input("Admin password", type="password", key="lmi_admin_pw")
    if pw and st.button("Login", type="primary"):
        access = _load_access()
        if _hash_password(pw) == access.get("admin_hash", _DEFAULT_HASH):
            st.session_state["lmi_admin"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def set_admin_password(old_pw: str, new_pw: str) -> bool:
    access = _load_access()
    if _hash_password(old_pw) != access.get("admin_hash", _DEFAULT_HASH):
        return False
    access["admin_hash"] = _hash_password(new_pw)
    _save_access(access)
    return True


def admin_logout():
    st.session_state["lmi_admin"] = False
