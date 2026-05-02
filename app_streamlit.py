from __future__ import annotations

import tempfile
from pathlib import Path
from time import strftime, localtime

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.config import load_settings
from src.evaluator import evaluate_pipeline
from src.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="Adaptive Compliance Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #5f6368;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.9rem;
        border: 1px solid #e6e9ef;
        background: #ffffff;
        box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    }
    .source-card {
        padding: 0.9rem;
        border-radius: 0.8rem;
        border: 1px solid #e7e7e7;
        background: #fafafa;
        margin-bottom: 0.7rem;
    }
    .status-ok {
        padding: 0.7rem 0.9rem;
        background: #ecfdf3;
        border: 1px solid #abefc6;
        border-radius: 0.7rem;
        color: #067647;
        font-weight: 600;
    }
    .status-warn {
        padding: 0.7rem 0.9rem;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 0.7rem;
        color: #b45309;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_pipeline() -> RAGPipeline:
    settings = load_settings("config.yaml")
    return RAGPipeline(settings)


pipeline = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "upload_messages" not in st.session_state:
    st.session_state.upload_messages = []

st.markdown('<div class="main-title">Adaptive Drift-Aware Financial Compliance Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload regulatory documents, ask compliance questions, inspect retrieved evidence, and monitor knowledge drift.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Knowledge Base")
    status_text = pipeline.status()
    st.text(status_text)

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, CSV, or DOCX files",
        type=["pdf", "txt", "csv", "docx"],
        accept_multiple_files=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        ingest_clicked = st.button("Index files", use_container_width=True)
    with col_b:
        clear_clicked = st.button("Clear KB", use_container_width=True)

    if clear_clicked:
        pipeline.clear()
        st.session_state.messages = []
        st.session_state.last_response = None
        st.session_state.upload_messages = ["Knowledge base cleared."]
        st.rerun()

    if ingest_clicked:
        st.session_state.upload_messages = []
        if not uploaded_files:
            st.session_state.upload_messages.append("Upload at least one document first.")
        else:
            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = Path(tmp.name)
                try:
                    added, message = pipeline.ingest_file(tmp_path)
                    st.session_state.upload_messages.append(message)
                except Exception as exc:
                    st.session_state.upload_messages.append(f"Failed to index {uploaded_file.name}: {exc}")
                finally:
                    tmp_path.unlink(missing_ok=True)
            st.rerun()

    for msg in st.session_state.upload_messages:
        st.info(msg)

    st.divider()
    st.caption("Tip: use the sample documents in the sample_data folder for a quick demo.")

chunks = pipeline.vector_store.count()
docs = len(pipeline.vector_store.documents())
events = len(pipeline.drift_detector.events)
last_conf = None
if pipeline.drift_detector.query_confidence_history:
    last_conf = pipeline.drift_detector.query_confidence_history[-1]

metric_cols = st.columns(4)
metric_cols[0].metric("Indexed chunks", chunks)
metric_cols[1].metric("Documents", docs)
metric_cols[2].metric("Drift events", events)
metric_cols[3].metric("Last confidence", "—" if last_conf is None else f"{last_conf:.2f}")

chat_tab, sources_tab, drift_tab, evaluation_tab, guide_tab = st.tabs(
    ["Chat", "Retrieved Sources", "Drift Monitoring", "Evaluation", "Run Guide"]
)

with chat_tab:
    left, right = st.columns([2.2, 1])
    with left:
        st.subheader("Compliance Chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask a compliance question, for example: What are the AML monitoring requirements?")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Retrieving evidence and preparing answer..."):
                response = pipeline.ask(prompt)
                st.session_state.last_response = response

            answer_text = (
                f"{response.answer}\n\n"
                f"---\n"
                f"**Retrieval status:** {response.drift_message}\n\n"
                f"**Confidence:** {response.confidence:.2f}  \n"
                f"**Latency:** {response.latency_seconds:.3f} seconds"
            )
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
            with st.chat_message("assistant"):
                st.markdown(answer_text)

    with right:
        st.subheader("System Status")
        if chunks == 0:
            st.markdown('<div class="status-warn">No documents indexed yet. Upload documents from the sidebar.</div>', unsafe_allow_html=True)
        elif last_conf is not None and last_conf < pipeline.settings.retrieval.min_confidence:
            st.markdown('<div class="status-warn">Low retrieval confidence detected. Review sources or add newer documents.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-ok">Knowledge base ready.</div>', unsafe_allow_html=True)

        st.write("**Indexed documents**")
        docs_list = pipeline.vector_store.documents()
        if docs_list:
            for doc in docs_list:
                st.write(f"• {doc}")
        else:
            st.write("No documents yet.")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.rerun()

with sources_tab:
    st.subheader("Retrieved Evidence")
    response = st.session_state.last_response
    if not response or not response.sources:
        st.info("Ask a question first. The retrieved source chunks will appear here.")
    else:
        for idx, src in enumerate(response.sources, start=1):
            st.markdown(
                f"""
                <div class="source-card">
                <b>Source {idx}: {src.document_name}</b><br>
                Chunk: {src.chunk_id} &nbsp; | &nbsp; Score: {src.score:.3f} &nbsp; | &nbsp; Distance: {src.distance:.3f}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(src.text)
            st.divider()

with drift_tab:
    st.subheader("Drift Monitoring")
    history = pipeline.drift_detector.query_confidence_history
    if history:
        df = pd.DataFrame({"query_number": range(1, len(history) + 1), "confidence": history})
        st.line_chart(df, x="query_number", y="confidence")
    else:
        st.info("No confidence history yet. Ask questions to build the drift graph.")

    latest_events = pipeline.drift_detector.latest_events(limit=20)
    st.write("### Latest drift / confidence events")
    if latest_events:
        rows = [
            {
                "time": strftime("%Y-%m-%d %H:%M:%S", localtime(event.timestamp)),
                "type": event.event_type,
                "severity": event.severity,
                "score": round(event.score, 3),
                "message": event.message,
            }
            for event in latest_events
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No drift events recorded yet.")

with evaluation_tab:
    st.subheader("Evaluation Dashboard")
    st.write(
        "This tab evaluates whether the chatbot can answer supported compliance questions from the indexed knowledge base "
        "and reject unsupported/out-of-scope questions. It also shows retrieval confidence, keyword recall, latency, "
        "and a confusion matrix."
    )

    st.info(
        "Dataset used: a small labelled synthetic financial-compliance QA dataset stored in "
        "`sample_data/eval_questions.json`. It contains AML/KYC and sanctions questions based on the sample documents, "
        "plus unrelated out-of-scope questions for negative testing. You can replace this file with a larger real dataset "
        "from FCA, FATF, AML, KYC, sanctions, or internal policy documents."
    )

    eval_path = Path("sample_data/eval_questions.json")
    col1, col2 = st.columns([1, 1])
    with col1:
        keyword_threshold = st.slider("Keyword recall pass threshold", 0.0, 1.0, 0.35, 0.05)
    with col2:
        confidence_threshold = st.slider(
            "Retrieval confidence threshold",
            0.0,
            1.0,
            float(pipeline.settings.retrieval.min_confidence),
            0.05,
        )

    if st.button("Run evaluation", use_container_width=False):
        if not eval_path.exists():
            st.error("sample_data/eval_questions.json not found.")
        else:
            with st.spinner("Running evaluation..."):
                report = evaluate_pipeline(
                    pipeline,
                    eval_path,
                    confidence_threshold=confidence_threshold,
                    keyword_recall_threshold=keyword_threshold,
                )
            st.session_state.evaluation_report = report

    report = st.session_state.get("evaluation_report")
    if report:
        c = report["confusion"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Questions", report["num_questions"])
        m2.metric("Accuracy", f"{c['accuracy']:.2f}")
        m3.metric("Precision", f"{c['precision']:.2f}")
        m4.metric("Recall", f"{c['recall']:.2f}")
        m5.metric("F1-score", f"{c['f1']:.2f}")

        m6, m7, m8 = st.columns(3)
        m6.metric("Avg keyword recall", f"{report['average_keyword_recall']:.2f}")
        m7.metric("Avg confidence", f"{report['average_confidence']:.2f}")
        m8.metric("Avg latency", f"{report['average_latency_seconds']:.3f}s")

        st.write("### Confusion Matrix")
        st.caption("Positive class = supported/answerable compliance question. Negative class = unsupported/out-of-scope question.")
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        matrix = c["matrix"]
        ax.imshow(matrix)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred unsupported", "Pred supported"], rotation=20, ha="right")
        ax.set_yticklabels(["Actual unsupported", "Actual supported"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=14)
        ax.set_title("Answerability Confusion Matrix")
        st.pyplot(fig)

        st.write("### Detailed Evaluation Results")
        results_df = pd.DataFrame(report["results"])
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.write("### Evaluation Meaning")
        st.markdown(
            f"""
            - **TP:** supported compliance question correctly answered as supported: `{c['tp']}`
            - **TN:** out-of-scope question correctly rejected as unsupported: `{c['tn']}`
            - **FP:** out-of-scope question wrongly treated as supported: `{c['fp']}`
            - **FN:** supported compliance question wrongly treated as unsupported: `{c['fn']}`
            """
        )
    else:
        st.warning("Index the sample documents first, then click **Run evaluation**.")

with guide_tab:
    st.subheader("How to run")
    st.code("streamlit run app_streamlit.py", language="bash")
    st.write("For the original Gradio interface, run:")
    st.code("python app.py", language="bash")
    st.write("Recommended demo flow:")
    st.markdown(
        """
        1. Upload the sample files from `sample_data`.
        2. Click **Index files**.
        3. Ask a compliance question in the chat.
        4. Open **Retrieved Sources** to inspect evidence.
        5. Open **Drift Monitoring** to view confidence and drift events.
        6. Run **Evaluation** for a simple quantitative report.
        """
    )
