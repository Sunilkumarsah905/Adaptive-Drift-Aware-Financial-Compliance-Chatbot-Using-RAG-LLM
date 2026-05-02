from __future__ import annotations

from pathlib import Path
import tempfile
import time

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_settings
from src.evaluator import evaluate_pipeline, format_evaluation_report
from src.rag_pipeline import RAGPipeline


settings = load_settings("config.yaml")
pipeline = RAGPipeline(settings)


def upload_files(files):
    if not files:
        return "No files selected.", pipeline.status()
    messages = []
    for file in files:
        try:
            _, msg = pipeline.ingest_file(file.name)
            messages.append(msg)
        except Exception as exc:
            messages.append(f"Failed to index {getattr(file, 'name', 'file')}: {exc}")
    return "\n".join(messages), pipeline.status()


def chat(message, history):
    """Gradio Chatbot callback using the modern messages format.

    Newer Gradio versions expect each chat item to be a dictionary with
    role/content keys, not a tuple. This prevents the UI error:
    "Data incompatible with messages format".
    """
    history = history or []
    if not message or not message.strip():
        return history, "", "Please enter a question."

    response = pipeline.ask(message.strip())
    bot_message = (
        f"{response.answer}\n\n"
        f"---\n"
        f"{response.drift_message}\n"
        f"Latency: {response.latency_seconds:.3f} seconds"
    )
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": bot_message},
    ]
    return history, "", response.drift_message


def show_sources(query):
    if not query or not query.strip():
        return pd.DataFrame(columns=["Rank", "Document", "Chunk", "Confidence", "Extract"])
    results = pipeline.vector_store.search(query.strip(), top_k=settings.retrieval.top_k)
    rows = []
    for rank, result in enumerate(results, start=1):
        rows.append(
            {
                "Rank": rank,
                "Document": result.document_name,
                "Chunk": result.chunk_id,
                "Confidence": round(result.score, 3),
                "Extract": result.text[:400] + ("..." if len(result.text) > 400 else ""),
            }
        )
    return pd.DataFrame(rows)


def plot_drift_history():
    history = pipeline.drift_detector.query_confidence_history
    output = Path("storage/drift_confidence.png")
    output.parent.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(9, 4.8))
    if history:
        plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.axhline(y=settings.retrieval.min_confidence, linestyle="--", label="Minimum confidence threshold")
    plt.title("Retrieval Confidence / Drift Monitoring")
    plt.xlabel("Query Number")
    plt.ylabel("Confidence Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return str(output)


def drift_events_table():
    events = pipeline.drift_detector.latest_events(limit=30)
    rows = []
    for event in events:
        rows.append(
            {
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp)),
                "Type": event.event_type,
                "Severity": event.severity,
                "Score": round(event.score, 3),
                "Message": event.message,
            }
        )
    return pd.DataFrame(rows)


def run_evaluation():
    eval_path = Path("sample_data/eval_questions.json")
    if not eval_path.exists():
        return "Evaluation file not found: sample_data/eval_questions.json"
    report = evaluate_pipeline(pipeline, eval_path)
    Path("storage/evaluation_report.json").write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
    return format_evaluation_report(report)


def clear_kb():
    return pipeline.clear(), pipeline.status()


with gr.Blocks(title=settings.app.title, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Adaptive Drift-Aware Financial Compliance Chatbot\n"
        "Upload regulatory documents, ask compliance questions, inspect sources, and monitor drift indicators."
    )

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(height=480, label="Compliance Chatbot", type="messages")
        with gr.Row():
            msg = gr.Textbox(label="Ask a compliance question", placeholder="Example: What are the customer due diligence requirements?", scale=5)
            send = gr.Button("Send", variant="primary", scale=1)
        status_box = gr.Textbox(label="Retrieval / Drift Status", interactive=False)
        send.click(chat, inputs=[msg, chatbot], outputs=[chatbot, msg, status_box])
        msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot, msg, status_box])

    with gr.Tab("Upload / Index Documents"):
        file_upload = gr.File(label="Upload PDF, TXT, CSV, or DOCX files", file_count="multiple")
        upload_btn = gr.Button("Index Documents", variant="primary")
        upload_result = gr.Textbox(label="Indexing Result", lines=6)
        kb_status = gr.Textbox(label="Knowledge Base Status", value=pipeline.status(), lines=5)
        clear_btn = gr.Button("Clear Knowledge Base", variant="stop")
        upload_btn.click(upload_files, inputs=file_upload, outputs=[upload_result, kb_status])
        clear_btn.click(clear_kb, outputs=[upload_result, kb_status])

    with gr.Tab("Retrieved Sources"):
        source_query = gr.Textbox(label="Question/query to inspect")
        source_btn = gr.Button("Retrieve Sources")
        source_table = gr.Dataframe(label="Top Retrieved Sources", wrap=True)
        source_btn.click(show_sources, inputs=source_query, outputs=source_table)

    with gr.Tab("Drift Monitoring"):
        drift_btn = gr.Button("Refresh Drift Graph")
        drift_img = gr.Image(label="Confidence Trend")
        event_btn = gr.Button("Show Drift Events")
        event_table = gr.Dataframe(label="Drift Events", wrap=True)
        drift_btn.click(plot_drift_history, outputs=drift_img)
        event_btn.click(drift_events_table, outputs=event_table)

    with gr.Tab("Evaluation"):
        gr.Markdown("Run the sample evaluation questions after indexing sample documents.")
        eval_btn = gr.Button("Run Evaluation", variant="primary")
        eval_report = gr.Markdown()
        eval_btn.click(run_evaluation, outputs=eval_report)


def main() -> None:
    demo.launch(server_name=settings.app.host, server_port=settings.app.port)


if __name__ == "__main__":
    main()
