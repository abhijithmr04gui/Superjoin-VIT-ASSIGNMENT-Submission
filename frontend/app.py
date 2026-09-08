"""
Minimal Streamlit UI for the Fact Knowledge Layer. Talks only to the
FastAPI backend over HTTP (API_BASE_URL) - it holds no pipeline logic of
its own, so the API is the real interface being demonstrated.

Run with: streamlit run frontend/app.py
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Fact Knowledge Layer", layout="wide")
st.title("📄 PDF Fact Knowledge Layer")
st.caption(f"Backend: {API_BASE_URL}")

REL_COLORS = {
    "CORROBORATED": "🟢",
    "CONTRADICTED": "🔴",
    "LIKELY_CONTRADICTION": "🟠",
    "CONTEXTUALLY_RECONCILED": "🔵",
    "UNCERTAIN": "⚪",
    "UNRELATED": "⚫",
}


def api_get(path: str, **params):
    r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def api_post_file(path: str, file_bytes: bytes, filename: str):
    files = {"file": (filename, file_bytes, "application/pdf")}
    r = requests.post(f"{API_BASE_URL}{path}", files=files, timeout=600)
    r.raise_for_status()
    return r.json()


tab_upload, tab_facts, tab_relationships, tab_search, tab_failures = st.tabs(
    ["Upload & Documents", "Facts", "Relationships", "Search", "Failures / Health"]
)

with tab_upload:
    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded is not None and st.button("Process document"):
        try:
            result = api_post_file("/documents/upload", uploaded.getvalue(), uploaded.name)
        except requests.HTTPError as exc:
            st.error(f"Upload failed: {exc.response.text}")
            result = None

        if result:
            document_id = result["document"]["id"]
            latest = result["run"]
            status_box = st.empty()
            with st.spinner("Processing (ingest → extract → normalize → compare)..."):
                while latest["status"] not in ("completed", "failed"):
                    status_box.info(f"Stage: {latest.get('current_stage') or latest['status']}")
                    time.sleep(1.5)
                    runs = api_get(f"/documents/{document_id}/runs")
                    if runs:
                        latest = runs[0]

            if latest["status"] == "failed":
                st.error("Processing failed — check the Failures tab.")
            else:
                st.success(f"Document '{result['document']['filename']}' -> status: {latest['status']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Facts extracted", latest.get("facts_extracted", 0))
            c2.metric("Relationships created", latest.get("relationships_created", 0))
            c3.metric("Errors recorded", latest.get("error_count", 0))

    st.divider()
    st.subheader("Documents")
    try:
        docs = api_get("/documents")
    except requests.RequestException as exc:
        st.error(f"Could not reach API at {API_BASE_URL}: {exc}")
        docs = []

    for doc in docs:
        with st.expander(f"{doc['filename']} — {doc['status']} ({doc['num_pages']} pages)"):
            st.json(doc)
            if doc["status"] == "failed" and doc.get("error_message"):
                st.error(doc["error_message"])


def render_fact_card(fact: dict, prefix: str = ""):
    badge = "✅" if fact["is_valid"] else "⚠️"
    st.markdown(
        f"**{badge} {fact['entity']}** → `{fact['predicate']}` → **{fact['object_text']}**"
    )
    cols = st.columns(4)
    cols[0].caption(f"Type: {fact['fact_type']} / {fact['value_type']}")
    cols[1].caption(f"Confidence: {fact['confidence_level']} ({fact['confidence']:.2f})")
    cols[2].caption(f"Period/date: {fact.get('reporting_period') or fact.get('date') or '—'}")
    cols[3].caption(f"Scope: {fact.get('scope') or '—'}")
    with st.container(border=True):
        st.markdown("**Evidence**")
        ev = fact["evidence"]
        st.markdown(f"📄 *{ev['filename']}* — page {ev['page_number']}")
        st.markdown(f"> {ev['source_text']}")
    if not fact["is_valid"]:
        st.caption(f"Validation note: {fact.get('validation_notes')}")


with tab_facts:
    st.subheader("Browse extracted facts")
    docs = api_get("/documents")
    doc_options = {d["filename"]: d["id"] for d in docs}
    selected_doc = st.selectbox("Filter by document (optional)", ["All"] + list(doc_options.keys()))
    entity_filter = st.text_input("Filter by entity contains", "")

    if selected_doc != "All":
        facts = api_get(f"/documents/{doc_options[selected_doc]}/facts")
    else:
        facts = api_get("/facts", entity=entity_filter or None, only_valid=False)

    st.caption(f"{len(facts)} fact(s)")
    for fact in facts:
        render_fact_card(fact)
        if st.button("Show related facts", key=f"rel-{fact['id']}"):
            rels = api_get(f"/facts/{fact['id']}/relationships")
            if not rels:
                st.info("No related facts found yet.")
            for rel in rels:
                other = rel["fact_b"] if rel["fact_a"]["id"] == fact["id"] else rel["fact_a"]
                icon = REL_COLORS.get(rel["relationship_type"], "•")
                st.markdown(f"{icon} **{rel['relationship_type']}** (confidence {rel['confidence']:.2f}) — {other['entity']} {other['predicate']} {other['object_text']}")
                st.caption(rel["explanation"])
        st.divider()

with tab_relationships:
    st.subheader("Cross-document relationships")
    rel_type = st.selectbox(
        "Filter by type",
        ["All", "CORROBORATED", "CONTRADICTED", "LIKELY_CONTRADICTION", "CONTEXTUALLY_RECONCILED", "UNCERTAIN"],
    )
    rels = api_get("/relationships", relationship_type=None if rel_type == "All" else rel_type)
    st.caption(f"{len(rels)} relationship(s)")
    for rel in rels:
        icon = REL_COLORS.get(rel["relationship_type"], "•")
        with st.container(border=True):
            st.markdown(f"### {icon} {rel['relationship_type']}  ·  confidence {rel['confidence']:.2f} ({rel['confidence_level']})")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Fact A**")
                render_fact_card(rel["fact_a"])
            with col_b:
                st.markdown("**Fact B**")
                render_fact_card(rel["fact_b"])
            st.markdown(f"**Explanation:** {rel['explanation']}")
            er = rel["contextual_dimensions"].get("entity_resolution")
            if er:
                st.caption(
                    f"Entity match resolved via LLM (confidence {er.get('confidence', 0):.2f}): "
                    f"{er.get('reasoning', '')}"
                )
            st.json(rel["contextual_dimensions"])

with tab_search:
    st.subheader("Semantic search across facts")
    query = st.text_input("Search query")
    if query:
        results = api_get("/search", q=query, top_k=10)
        for r in results:
            st.caption(f"score: {r['score']:.3f}")
            render_fact_card(r["fact"])
            st.divider()

with tab_failures:
    st.subheader("Extraction / reasoning failures")
    try:
        health = api_get("/health")
        st.info(f"API health: {health}")
    except requests.RequestException:
        st.error("API unreachable")

    failures = api_get("/failures")
    st.caption(f"{len(failures)} recorded failure(s)")
    for f in failures:
        with st.container(border=True):
            st.markdown(f"**[{f['severity'].upper()}] {f['stage']} / {f['error_type']}** — status: {f['final_status']}")
            st.write(f["error_message"])
            if f.get("suggested_improvement"):
                st.caption(f"Suggested improvement: {f['suggested_improvement']}")
