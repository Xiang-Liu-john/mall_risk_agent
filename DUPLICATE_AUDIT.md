# Duplicate Audit

This file records items that look duplicated or obsolete after the frontend/backend split.

## Code

Resolved:

- `app.py` is now only a compatibility launcher.
- The real Streamlit app is `frontend/streamlit_app.py`.
- Backend code is in `backend/api.py` and `backend/store.py`.
- Legacy local RAG tokenization/chunking inside the frontend has been removed.
- RAG tokenization, chunking, indexing and retrieval now live in `rag_store.py`.

## Data

Keep:

- `data/购物中心100家门店经营数据_优化命名.csv`
- `data/门店经营月度历史数据_12个月.csv`

Candidate duplicate/obsolete:

- `data/购物中心100家门店经营数据.csv`

Reason: older generated dataset name. The app default uses the optimized-name CSV.

## RAG Sources

Keep text-first sources where possible:

- `rag_sources/contracts/*.md`
- `rag_sources/inspection/*.md`
- `rag_sources/policies/*.md`
- `rag_sources/tables/*.xlsx`
- `rag_sources/images/*.png`

Candidate duplicates:

- Chinese Office/PDF files that mirror the English/Markdown source files.
- CSV copies of the same sample tables where an XLSX version already exists.

Reason: indexing both versions can make the same policy or table appear multiple times in retrieved evidence.

## Temporary Logs

Candidate cleanup:

- `tmp/streamlit_8502.err.log`
- `tmp/streamlit_8502.out.log`
- `tmp/streamlit_live.log`
- `tmp/streamlit_run.log`
- `tmp/streamlit_stderr.log`
- `tmp/streamlit_stdout.log`

Reason: local run logs should not be committed.
