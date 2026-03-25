"""
pages/6_Admin.py — Admin Dashboard
API keys, data management, and system settings.
"""
import json
import streamlit as st
from harrington_common.compute import render_compute_info, backend_info
from harrington_labs.ui import render_header
from harrington_labs.ui import lab_panel, make_figure, show_figure, COLORS
from harrington_labs.lmi.ui.access import require_admin, set_admin_password, admin_logout
from harrington_labs.lmi.domain.lasers import load_custom_lasers, save_custom_lasers, LASER_DB_PATH
from harrington_labs.lmi.domain.materials import load_custom_materials, save_custom_materials, MATERIAL_DB_PATH

st.set_page_config(page_title="Admin", layout="wide")
render_header("Admin", "System settings • API keys • Custom data management")

if not require_admin():
    st.stop()

with st.sidebar:
    if st.button("Logout admin"):
        admin_logout()
        st.rerun()

tab_api, tab_data, tab_compute, tab_password = st.tabs(["API Keys", "Data Management", "Compute", "Change Password"])

# ── API Keys ──
with tab_api:
    with lab_panel():
        st.subheader("API Keys")
        st.caption("Keys are stored in data/manual/config.json.")

        config_path = LASER_DB_PATH.parent / "config.json"
        existing = {}
        if config_path.exists():
            existing = json.loads(config_path.read_text())

        anthropic_key = st.text_input(
            "Anthropic API key",
            value=existing.get("anthropic_api_key", ""),
            type="password",
        )
        openai_key = st.text_input(
            "OpenAI API key",
            value=existing.get("openai_api_key", ""),
            type="password",
        )

        if st.button("Save API Keys", type="primary"):
            existing["anthropic_api_key"] = anthropic_key.strip() or None
            existing["openai_api_key"] = openai_key.strip() or None
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(existing, indent=2))
            st.success("API keys saved.")

# ── Data Management ──
with tab_data:
    with lab_panel():
        st.subheader("Custom Data")

        custom_lasers = load_custom_lasers()
        custom_materials = load_custom_materials()

        col1, col2 = st.columns(2)
        col1.metric("Custom Lasers", len(custom_lasers))
        col2.metric("Custom Materials", len(custom_materials))

        if custom_lasers:
            st.markdown("**Custom Lasers:**")
            for i, l in enumerate(custom_lasers):
                st.caption(f"{i + 1}. {l.name} ({l.wavelength_nm} nm, {l.power_w} W)")

        if custom_materials:
            st.markdown("**Custom Materials:**")
            for i, m in enumerate(custom_materials):
                st.caption(f"{i + 1}. {m.name} ({m.category})")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Delete all custom lasers", type="secondary"):
                save_custom_lasers([])
                st.success("Custom lasers cleared.")
                st.rerun()
        with col2:
            if st.button("Delete all custom materials", type="secondary"):
                save_custom_materials([])
                st.success("Custom materials cleared.")
                st.rerun()


# ── Compute Backend ──
with tab_compute:
    with lab_panel():
        st.subheader("Compute Backend")
        info = backend_info()
        backend = info["backend"]

        if backend == "cupy":
            st.success(f"[GPU] Accelerated — {info.get('gpu_name', 'CUDA GPU')}")
            col1, col2 = st.columns(2)
            col1.metric("CuPy Version", info.get("cupy_version", "?"))
            col2.metric("GPU VRAM", f"{info.get('gpu_memory_mb', 0):.0f} MB")
        elif backend == "numba":
            st.info(f"[JIT] Numba — {info.get('numba_num_threads', '?')} CPU threads")
            st.metric("Numba Version", info.get("numba_version", "?"))
        else:
            st.warning("[CPU] NumPy fallback — install `numba` for 10-100× speedup on physics engines")

        st.metric("NumPy Version", info["numpy_version"])
        st.caption("Set `HARRINGTON_NO_JIT=1` to force NumPy fallback, "
                   "`HARRINGTON_NO_CUDA=1` to disable GPU.")

# ── Change Password ──
with tab_password:
    with lab_panel():
        st.subheader("Change Admin Password")
        old_pw = st.text_input("Current password", type="password", key="old_pw")
        new_pw = st.text_input("New password", type="password", key="new_pw")
        confirm = st.text_input("Confirm new password", type="password", key="confirm_pw")
        if st.button("Update password", type="primary"):
            if not old_pw or not new_pw:
                st.error("Fill in all fields.")
            elif new_pw != confirm:
                st.error("New passwords don't match.")
            elif len(new_pw) < 8:
                st.error("Password must be at least 8 characters.")
            elif set_admin_password(old_pw, new_pw):
                st.success("Password updated.")
            else:
                st.error("Current password is incorrect.")
