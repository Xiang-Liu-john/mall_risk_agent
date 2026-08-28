# Project Structure

```text
mall_risk_agent/
├── frontend/
│   ├── streamlit_app.py            # Streamlit UI and Agent interaction page
│   └── __init__.py
├── backend/
│   ├── api.py                      # FastAPI backend API prototype
│   ├── store.py                    # SQLite audit/session persistence
│   └── __init__.py
├── data/                           # Generated mall operation datasets
├── rag_docs/                       # Built-in business policy documents
├── rag_sources/                    # User-provided RAG files
├── rag_store.py                    # Local RAG indexing and retrieval
├── risk_rules.py                   # Risk rule engine
├── risk_rule_config.json           # Versioned risk rule configuration
└── generate_data.py                # Synthetic current and 12-month history data generator
```

## Frontend

`frontend/streamlit_app.py` owns the page layout, charts, sidebar filters, chat UI, streaming output, Agent planner, tool calling orchestration and Verifier display.

## Backend

`backend/store.py` writes sessions, messages, Agent runs, steps and evidence into SQLite.

`backend/api.py` exposes a FastAPI prototype for health checks, stats and recent Agent runs. The current Streamlit app can run without starting this API because it imports the persistence layer directly.

## Entry Point

Use `streamlit run frontend/streamlit_app.py` to start the frontend. There is no duplicate root-level `app.py`.
