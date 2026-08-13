# RAG source inbox

Put source files here, then click "重建 RAG 数据库" in the Streamlit app.

Supported formats:

- `.txt`, `.md`: plain text knowledge
- `.csv`: tabular data
- `.xlsx`: spreadsheet data, requires `openpyxl`
- `.docx`: Word document text
- `.pdf`: PDF text, requires `pypdf`
- `.png`, `.jpg`, `.jpeg`: image OCR/vision extraction when `OPENAI_API_KEY` is configured; otherwise image metadata only

Suggested folders:

```text
rag_sources/
  contracts/
  policies/
  inspection/
  tables/
  images/
```
