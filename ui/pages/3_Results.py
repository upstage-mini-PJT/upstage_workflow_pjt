"""Page 3 — Poll until complete, then display AnalysisResult."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import streamlit as st
from config import API_BASE_URL

st.set_page_config(page_title="분석 결과", page_icon="📊", layout="wide")

st.title("📊 Step 3 — 분석 결과")

# ── Guard ─────────────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.warning("세션이 없습니다. 먼저 **1 Upload** 페이지에서 시작하세요.")
    st.stop()

thread_id: str = st.session_state["thread_id"]
st.caption(f"세션 ID: `{thread_id}`")


# ── Fetch result ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3, show_spinner=False)
def _fetch_result(tid: str) -> dict:
    try:
        resp = requests.get(f"{API_BASE_URL}/api/workflow/result/{tid}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"status": "error", "error": str(exc), "result": None}


result_data = _fetch_result(thread_id)
status: str = result_data.get("status", "unknown")
st.session_state["last_status"] = status

# ── Not yet complete ───────────────────────────────────────────────────────────
if status in ("running", "pending", "awaiting_docs"):
    msg = {
        "running": "⏳ AI가 분석 중입니다. 잠시 기다려 주세요…",
        "pending": "⏳ 분석 대기 중…",
        "awaiting_docs": "📂 추가 서류 대기 중입니다. **2 Documents** 페이지에서 서류를 제출하세요.",
    }.get(status, f"현재 상태: {status}")

    if status == "awaiting_docs":
        st.warning(msg)
    else:
        st.info(msg)
        with st.spinner("분석 중…"):
            time.sleep(3)
        _fetch_result.clear()
        st.rerun()

elif status == "error":
    st.error(f"오류가 발생했습니다: {result_data.get('error', '알 수 없는 오류')}")

elif status == "complete":
    result: dict = result_data.get("result") or {}

    # ── Success probability ────────────────────────────────────────────────────
    success_prob: dict = result.get("success_probability") or {}
    band = success_prob.get("band", "UNKNOWN")
    band_map = {"HIGH": (0.8, "🟢"), "MEDIUM": (0.5, "🟡"), "LOW": (0.2, "🔴")}
    prob_value, prob_icon = band_map.get(band, (0.0, "⚪"))

    st.subheader("성공 가능성")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(label="등급", value=f"{prob_icon} {band}")
    with col2:
        st.progress(prob_value, text=f"추정 성공 가능성: {int(prob_value * 100)}%")

    pos_drivers = success_prob.get("positive_drivers", [])
    neg_drivers = success_prob.get("negative_drivers", [])
    if pos_drivers or neg_drivers:
        with st.expander("성공 가능성 근거 보기"):
            if pos_drivers:
                st.markdown("**긍정 요인**")
                for d in pos_drivers:
                    st.markdown(f"- ✅ {d}")
            if neg_drivers:
                st.markdown("**부정 요인**")
                for d in neg_drivers:
                    st.markdown(f"- ⚠️ {d}")

    st.divider()

    # ── User guidance (plain summary + issue brief) ────────────────────────────
    user_guidance: dict = result.get("user_guidance") or {}
    plain_summary = user_guidance.get("plain_summary", "")
    if plain_summary:
        st.subheader("한눈 요약")
        st.info(plain_summary)

    issue_brief: list[dict] = user_guidance.get("issue_brief", [])
    if issue_brief:
        st.subheader("핵심 쟁점")
        for item in issue_brief:
            issue_id = item.get("issue_id", "")
            title = item.get("title", issue_id)
            why = item.get("why_it_matters", "")
            needed = item.get("needed_evidence", "")
            with st.expander(f"🔍 {title}"):
                if why:
                    st.markdown(f"**왜 중요한가:** {why}")
                if needed:
                    st.markdown(f"**필요 증빙:** {needed}")

    st.divider()

    # ── Issue tree ─────────────────────────────────────────────────────────────
    issue_tree: dict = result.get("issue_tree") or {}
    nodes: list[dict] = issue_tree.get("nodes", [])
    if nodes:
        st.subheader(f"쟁점 트리 — {issue_tree.get('root_title', '')}")
        for node in nodes:
            nid = node.get("issue_id", "")
            ntitle = node.get("title", nid)
            ndesc = node.get("description", "")
            with st.expander(f"📌 {ntitle}"):
                if ndesc:
                    st.write(ndesc)
                denial_links = node.get("related_denial_reasons", [])
                clause_links = node.get("related_policy_clauses", [])
                if denial_links:
                    st.markdown("**거절 사유 연결:** " + ", ".join(denial_links))
                if clause_links:
                    st.markdown("**약관 조항 연결:** " + ", ".join(clause_links))

    st.divider()

    # ── Recommended actions ────────────────────────────────────────────────────
    actions: list[dict] = result.get("recommended_actions", [])
    if actions:
        st.subheader("지금 할 일")
        for action in actions:
            priority = action.get("priority", "medium")
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
            title = action.get("title", "")
            detail = action.get("detail", "")
            st.checkbox(
                f"{priority_icon} {title}",
                value=False,
                key=f"action_{action.get('action_id', title)}",
                help=detail,
            )

    st.divider()

    # ── Evidence guide ─────────────────────────────────────────────────────────
    evidence_guide: list[dict] = user_guidance.get("evidence_guide", [])
    if evidence_guide:
        st.subheader("근거 자료")
        rows = []
        for item in evidence_guide:
            rows.append({
                "참조": item.get("ref_token", ""),
                "유형": item.get("source_label", ""),
                "제목": item.get("title", ""),
                "관련성": item.get("why_relevant", ""),
            })
        st.table(rows)

    st.divider()

    # ── Gap analysis ───────────────────────────────────────────────────────────
    gap_analysis: list[dict] = result.get("gap_analysis", [])
    if gap_analysis:
        st.subheader("보완 필요 사항 (Gap Analysis)")
        for gap in gap_analysis:
            impact = gap.get("impact", "low")
            impact_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(impact, "⚪")
            missing = gap.get("missing_evidence", "")
            rationale = gap.get("rationale", "")
            st.warning(f"{impact_icon} **{missing}** — {rationale}")

    st.divider()

    # ── Next steps ─────────────────────────────────────────────────────────────
    next_steps: list[str] = user_guidance.get("next_steps", [])
    if next_steps:
        st.subheader("다음 단계")
        for step in next_steps:
            st.markdown(f"- {step}")

    st.divider()

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    disclaimer = user_guidance.get("disclaimer", "")
    if disclaimer:
        st.caption(f"⚠️ {disclaimer}")

    # ── Download full JSON ─────────────────────────────────────────────────────
    st.subheader("전체 결과 다운로드")
    st.download_button(
        label="📥 JSON 다운로드",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name=f"analysis_{thread_id[:8]}.json",
        mime="application/json",
    )

else:
    st.info(f"현재 상태: **{status}**")
