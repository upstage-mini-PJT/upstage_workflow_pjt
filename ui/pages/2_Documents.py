"""Page 2 — Poll status; collect additional documents when requested."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import streamlit as st
from config import API_BASE_URL

st.set_page_config(page_title="추가 서류", page_icon="📂", layout="wide")

st.title("📂 Step 2 — 추가 서류 제출")

# ── Guard: require an active session ──────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.warning("세션이 없습니다. 먼저 **1 Upload** 페이지에서 서류를 업로드하세요.")
    st.stop()

thread_id: str = st.session_state["thread_id"]
st.caption(f"세션 ID: `{thread_id}`")


# ── Fetch current status ───────────────────────────────────────────────────────
@st.cache_data(ttl=2, show_spinner=False)
def _fetch_status(tid: str) -> dict:
    try:
        resp = requests.get(f"{API_BASE_URL}/api/workflow/status/{tid}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


status_data = _fetch_status(thread_id)
status: str = status_data.get("status", "unknown")
st.session_state["last_status"] = status

# ── Status: running / pending → auto-refresh ──────────────────────────────────
if status in ("running", "pending"):
    st.info("⏳ AI가 분석 중입니다. 잠시 기다려 주세요…")
    with st.spinner("분석 중…"):
        time.sleep(2)
    st.rerun()

# ── Status: error ─────────────────────────────────────────────────────────────
elif status == "error":
    st.error("오류가 발생했습니다.")
    with st.expander("오류 상세 (traceback)", expanded=True):
        st.code(status_data.get("error", "알 수 없는 오류"), language="text")

# ── Status: complete → redirect hint ─────────────────────────────────────────
elif status == "complete":
    st.success("분석이 완료되었습니다!")
    st.info("👉 왼쪽 사이드바에서 **3 Results** 페이지로 이동하세요.")

# ── Status: awaiting_docs → show upload form ─────────────────────────────────
elif status == "awaiting_docs":
    required_docs: list[str] = status_data.get("required_documents", [])
    required_items: list[dict] = status_data.get("required_document_items", [])

    st.warning("추가 서류가 필요합니다. 아래 서류를 준비하여 제출해주세요.")

    if required_items:
        st.subheader("요청된 서류 목록")
        for item in required_items:
            doc_id = item.get("id", "")
            name = item.get("name", doc_id)
            st.markdown(f"- **{name}** (`{doc_id}`)")
    elif required_docs:
        st.subheader("요청된 서류 목록")
        for doc in required_docs:
            st.markdown(f"- {doc}")

    with st.form("resume_form"):
        uploaded_files = st.file_uploader(
            "서류 업로드 (여러 파일 선택 가능)",
            type=["pdf", "md", "txt", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="요청된 서류를 모두 업로드한 뒤 제출하세요",
        )
        resume_submitted = st.form_submit_button("서류 제출 ▶", type="primary")

    if resume_submitted:
        if not uploaded_files:
            st.error("서류를 업로드해주세요.")
        else:
            with st.spinner("서류를 제출하고 분석을 재개하는 중…"):
                try:
                    files_payload = [
                        (
                            "files",
                            (f.name, f.getvalue(), f.type or "application/octet-stream"),
                        )
                        for f in uploaded_files
                    ]
                    resp = requests.post(
                        f"{API_BASE_URL}/api/workflow/resume/{thread_id}",
                        files=files_payload,
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        st.session_state["last_status"] = "running"
                        st.success("서류가 제출되었습니다! 분석을 재개합니다.")
                        # Clear cache so next poll reflects new status
                        _fetch_status.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"제출 오류 ({resp.status_code}): {detail}")
                except requests.exceptions.ConnectionError:
                    st.error(f"API 서버에 연결할 수 없습니다 ({API_BASE_URL}).")
                except Exception as exc:
                    st.error(f"예상치 못한 오류: {exc}")

else:
    st.info(f"현재 상태: **{status}**")
