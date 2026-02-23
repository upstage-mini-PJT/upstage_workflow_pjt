import streamlit as st

st.set_page_config(
    page_title="보험금 이의신청 AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚖️ 보험금 이의신청 AI 가이드")

st.markdown("""
이 서비스는 보험금 지급거절 사례를 분석하여 이의신청을 돕는 AI 시스템입니다.

### 사용 방법

1. **서류 업로드** — 지급거절 명세서와 보험 가입일을 입력합니다
2. **추가 서류** — AI가 요청하는 추가 서류를 제출합니다
3. **결과 확인** — 이의신청 가이드와 성공 가능성을 확인합니다

왼쪽 사이드바에서 단계를 선택하거나 직접 이동하세요.
""")

if "thread_id" not in st.session_state:
    st.info("👈 사이드바에서 **1 Upload** 페이지로 이동하여 시작하세요.")
else:
    st.success(f"현재 진행 중인 세션: `{st.session_state['thread_id']}`")
    status = st.session_state.get("last_status", "unknown")
    st.write(f"상태: **{status}**")
