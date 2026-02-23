"""Page 1 — Upload the denial notice and enter policy join date."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the ui/ directory is importable regardless of how Streamlit is launched
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import streamlit as st
from config import API_BASE_URL

st.set_page_config(page_title="서류 업로드", page_icon="📄", layout="wide")

st.title("📄 Step 1 — 서류 업로드")
st.markdown("지급거절 명세서 파일과 보험 가입일을 입력한 뒤 **분석 시작** 버튼을 누르세요.")

with st.form("upload_form"):
    denial_file = st.file_uploader(
        "지급거절 명세서",
        type=["pdf", "md", "txt"],
        help="보험사로부터 받은 지급거절 통지서를 업로드하세요 (PDF / Markdown / TXT)",
    )
    join_date = st.text_input(
        "보험 가입일 (YYYYMMDD)",
        placeholder="예: 20220101",
        help="보험 계약 시작일을 8자리 숫자로 입력하세요",
        max_chars=8,
    )
    submitted = st.form_submit_button("분석 시작 ▶", type="primary")

if submitted:
    if not denial_file:
        st.error("지급거절 명세서를 업로드해주세요.")
    elif not join_date or len(join_date.strip()) != 8 or not join_date.strip().isdigit():
        st.error("가입일을 YYYYMMDD 형식으로 입력하세요. 예: 20220101")
    else:
        with st.spinner("API 서버에 연결 중…"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/workflow/start",
                    data={"join_date": join_date.strip()},
                    files={
                        "denial_file": (
                            denial_file.name,
                            denial_file.getvalue(),
                            denial_file.type or "application/octet-stream",
                        )
                    },
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state["thread_id"] = data["thread_id"]
                    st.session_state["last_status"] = "running"
                    st.success(
                        f"분석이 시작되었습니다! 세션 ID: `{data['thread_id']}`"
                    )
                    st.info("👉 왼쪽 사이드바에서 **2 Documents** 페이지로 이동하세요.")
                else:
                    detail = response.json().get("detail", response.text)
                    st.error(f"API 오류 ({response.status_code}): {detail}")
            except requests.exceptions.ConnectionError:
                st.error(
                    f"API 서버에 연결할 수 없습니다 ({API_BASE_URL}). "
                    "서버가 실행 중인지 확인하세요."
                )
            except Exception as exc:
                st.error(f"예상치 못한 오류: {exc}")
