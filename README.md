# Adaptive Drift-Aware Financial Compliance Chatbot

A complete Python prototype for an **Adaptive Drift-Aware Financial Compliance Chatbot using Retrieval-Augmented Generation (RAG)**. The project includes document ingestion, semantic search, source-grounded answers, a Gradio UI, regulatory drift monitoring, persistence, and an evaluation script.

## What this project implements

- Upload financial compliance documents: `.pdf`, `.txt`, `.csv`, `.docx`
- Chunk documents and create semantic embeddings
- Store/retrieve chunks using FAISS vector search
- Generate answers grounded in retrieved sources
- Show source citations with document name and chunk ID
- Detect low-confidence retrieval and regulatory drift indicators
- Compare newly uploaded documents with previous knowledge base state
- Save and reload vector index between runs
- Gradio web UI for chat, upload, drift monitoring, and evaluation
- Command-line evaluation script using sample compliance questions

## Project structure

```text
adaptive_drift_compliance_chatbot/
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
├── run_app.bat
├── run_app.sh
├── sample_data/
│   ├── aml_policy_v1.txt
│   ├── aml_policy_v2_update.txt
│   ├── sanctions_guidance.txt
│   └── eval_questions.json
├── scripts/
│   └── evaluate.py
├── src/
│   ├── answer_generator.py
│   ├── config.py
│   ├── document_loader.py
│   ├── drift_detector.py
│   ├── evaluator.py
│   ├── rag_pipeline.py
│   ├── text_splitter.py
│   └── vector_store.py
└── storage/
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the UI

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal.

## First-time demo steps

1. Run `python app.py`.
2. Open the UI.
3. Go to **Upload / Index Documents**.
4. Upload files from `sample_data/`.
5. Ask questions such as:
   - What are the customer due diligence requirements?
   - When should enhanced due diligence be applied?
   - What must firms do for sanctions screening?
6. Open **Drift Monitoring** to inspect retrieval confidence and drift alerts.
7. Use **Evaluation** to run the sample questions.

## Optional OpenAI generation

The project works without OpenAI by using a local extractive answer generator. To use OpenAI for richer generation, install `openai`, set `USE_OPENAI=true` in `config.yaml`, and export `OPENAI_API_KEY` in your environment.

## Academic scope

This is a dissertation-ready prototype, not a production compliance system. It demonstrates the architecture proposed in the research plan: ingestion, embeddings, vector retrieval, grounded answer generation, UI, drift monitoring, and measurable evaluation.

## Streamlit UI

A cleaner Streamlit interface is also included.

Run it with:

```bash
streamlit run app_streamlit.py
```

Windows shortcut:

```cmd
run_streamlit.bat
```

Mac/Linux shortcut:

```bash
./run_streamlit.sh
```

The Streamlit UI includes a sidebar for document upload, dashboard metrics, chat, retrieved sources, drift monitoring, and evaluation tabs.
