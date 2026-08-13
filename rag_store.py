from __future__ import annotations

import hashlib
import base64
import json
import math
import mimetypes
import re
import sqlite3
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_DIR = Path(__file__).parent
RAG_DOCS_DIR = PROJECT_DIR / "rag_docs"
RAG_SOURCES_DIR = PROJECT_DIR / "rag_sources"
RAG_DB_DIR = PROJECT_DIR / "rag_db"
RAG_DB_PATH = RAG_DB_DIR / "mall_rag.sqlite3"
VECTOR_SIZE = 96
SUPPORTED_SOURCE_SUFFIXES = {".txt", ".md", ".csv", ".xlsx", ".docx", ".pdf", ".png", ".jpg", ".jpeg"}
SECRETS_PATH = PROJECT_DIR / ".streamlit" / "secrets.toml"


@dataclass(frozen=True)
class RagSearchResult:
    source: str
    title: str
    content: str
    score: float
    metadata: dict


def tokenize(text: str) -> list[str]:
    normalized = str(text).lower()
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized)
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    tokens.extend(chinese[i : i + 2] for i in range(max(0, len(chinese) - 1)))
    return [token for token in tokens if token.strip()]


def split_text(text: str, chunk_size: int = 520, overlap: int = 80) -> list[str]:
    clean = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def embed_text(text: str) -> list[float]:
    vector = [0.0] * VECTOR_SIZE
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % VECTOR_SIZE
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def connect(db_path: Path = RAG_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS chunks_fts;
        DROP TABLE IF EXISTS chunks;

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            embedding_json TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            title,
            content,
            source,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61'
        );
        """
    )
    conn.commit()


def insert_chunk(conn: sqlite3.Connection, source: str, title: str, content: str, metadata: dict) -> None:
    embedding = embed_text(f"{title}\n{content}")
    cur = conn.execute(
        """
        INSERT INTO chunks (source, title, content, metadata_json, embedding_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (source, title, content, json.dumps(metadata, ensure_ascii=False), json.dumps(embedding)),
    )
    row_id = cur.lastrowid
    conn.execute(
        "INSERT INTO chunks_fts(rowid, title, content, source) VALUES (?, ?, ?, ?)",
        (row_id, title, content, source),
    )


def load_markdown_docs() -> Iterable[tuple[str, str, str, dict]]:
    if not RAG_DOCS_DIR.exists():
        return []
    docs = []
    for path in sorted(RAG_DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = path.stem.replace("_", " ")
        for index, chunk in enumerate(split_text(text), start=1):
            docs.append(("制度与模板", f"{title} #{index}", chunk, {"file": str(path.name), "chunk": index}))
    return docs


def read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", text)
    replacements = {
        "&lt;": "<",
        "&gt;": ">",
        "&amp;": "&",
        "&quot;": '"',
        "&apos;": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return f"PDF 文件：{path.name}。当前环境未安装 pypdf，暂未抽取正文。"
    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        pages.append(f"第 {page_number} 页\n{page.extract_text() or ''}")
    return "\n\n".join(pages)


def read_spreadsheet_text(path: Path) -> str:
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as exc:
        return f"表格文件：{path.name}。当前环境暂未成功读取 xlsx：{exc}"
    parts = []
    for sheet_name, sheet_df in sheets.items():
        preview = sheet_df.fillna("").astype(str).head(200)
        parts.append(f"Sheet：{sheet_name}\n{preview.to_csv(index=False)}")
    return "\n\n".join(parts)


def read_csv_text(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            return df.fillna("").astype(str).head(500).to_csv(index=False)
        except Exception as exc:
            last_error = exc
    return f"CSV 文件：{path.name}。读取失败：{last_error}"


def load_ai_config() -> dict[str, str]:
    if not SECRETS_PATH.exists():
        return {}
    try:
        data = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {key: str(value) for key, value in data.items() if value not in ("", None)}


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def read_image_with_vision_model(path: Path) -> str | None:
    config = load_ai_config()
    api_key = config.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        base_url = config.get("OPENAI_BASE_URL")
        model = config.get("OPENAI_VISION_MODEL") or config.get("OPENAI_MODEL") or "gpt-4o-mini"
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=45) if base_url else OpenAI(api_key=api_key, timeout=45)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是企业知识库 OCR 助手。请提取图片中的可见文字，并补充对业务风险有用的结构化描述。"
                        "如果图片是表格，请尽量保留行列关系；如果文字不清晰，请明确说明。"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请识别这张图片的文字内容，并按以下结构输出：\n"
                                "### OCR 文本\n"
                                "### 图片内容摘要\n"
                                "### 可用于 RAG 检索的关键词"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_to_data_url(path)}},
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"OCR/视觉模型暂未成功解析图片：{exc}"


def read_image_text(path: Path) -> str:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        metadata_text = (
            f"图片文件：{path.name}。尺寸：{width}x{height}，颜色模式：{mode}。"
        )
        vision_text = read_image_with_vision_model(path)
        if vision_text:
            return f"{metadata_text}\n\n### OCR/视觉模型解析\n{vision_text}"
        return f"{metadata_text}\n当前 RAG 已记录该图片元数据；未配置 OCR/视觉模型时不会抽取图片文字。"
    except Exception as exc:
        return f"图片文件：{path.name}。暂未成功读取图片元数据：{exc}"


def read_source_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".csv":
        return read_csv_text(path)
    if suffix == ".xlsx":
        return read_spreadsheet_text(path)
    if suffix == ".docx":
        return read_docx_text(path)
    if suffix == ".pdf":
        return read_pdf_text(path)
    if suffix in {".png", ".jpg", ".jpeg"}:
        return read_image_text(path)
    return ""


def source_category(path: Path) -> str:
    parts = {part.lower() for part in path.relative_to(RAG_SOURCES_DIR).parts[:-1]}
    name = path.name.lower()
    if "contract" in parts or "contracts" in parts or "合同" in name:
        return "合同与条款资料"
    if "policy" in parts or "policies" in parts or "制度" in name:
        return "制度与模板"
    if "inspection" in parts or "巡检" in name:
        return "巡检与运营记录"
    if "image" in parts or path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return "图片资料"
    if path.suffix.lower() in {".csv", ".xlsx"}:
        return "表格资料"
    return "外部导入资料"


def load_source_docs() -> Iterable[tuple[str, str, str, dict]]:
    if not RAG_SOURCES_DIR.exists():
        return []
    docs = []
    for path in sorted(RAG_SOURCES_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
            continue
        text = read_source_file(path)
        if not text.strip():
            continue
        relative_path = path.relative_to(RAG_SOURCES_DIR)
        category = source_category(path)
        chunks = split_text(text) or [text]
        for index, chunk in enumerate(chunks, start=1):
            docs.append(
                (
                    category,
                    f"{path.stem} #{index}",
                    chunk,
                    {
                        "file": str(relative_path),
                        "chunk": index,
                        "suffix": path.suffix.lower(),
                    },
                )
            )
    return docs


def load_interview_docs() -> Iterable[tuple[str, str, str, dict]]:
    path = PROJECT_DIR / "interview_source_extract.txt"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [
        ("Agent 设计方案草稿", f"Agent 设计草稿 #{index}", chunk, {"file": path.name, "chunk": index})
        for index, chunk in enumerate(split_text(text), start=1)
    ]


def store_row_to_content(row: pd.Series) -> str:
    reasons = row.get("触发依据", [])
    if isinstance(reasons, list):
        reason_text = "；".join(str(item) for item in reasons) or "未触发明显风险规则"
    else:
        reason_text = str(reasons) if reasons else "未触发明显风险规则"
    return (
        f"门店：{row.get('门店名称')}。门店ID：{row.get('门店ID')}。业态：{row.get('业态分类')}。楼层：{row.get('楼层')}。"
        f"经营场景：{row.get('经营场景', '未标注')}。风险等级：{row.get('计算风险等级')}。"
        f"风险得分：{row.get('计算风险得分')}/100。经营风险分：{row.get('经营风险分')}。"
        f"财务风险分：{row.get('财务风险分')}。运营风险分：{row.get('运营风险分')}。合同风险分：{row.get('合同风险分')}。"
        f"本月销售额：{row.get('本月销售额')}。销售环比：{row.get('销售环比(%)')}%。"
        f"进店率：{row.get('进店率(%)')}%。成交转化率：{row.get('成交转化率(%)')}%。"
        f"租售比：{row.get('租售比(%)')}%。欠费总额：{row.get('欠费总额(元)')} 元。"
        f"欠费天数：{row.get('欠费天数', 0)}。保证金覆盖率：{row.get('保证金覆盖率(%)', 100)}%。"
        f"水电费波动：{row.get('水电费波动(%)')}%。近90天投诉数：{row.get('近90天投诉数', 0)}。"
        f"触发依据：{reason_text}。"
    )


def load_store_docs(data: pd.DataFrame | None) -> Iterable[tuple[str, str, str, dict]]:
    if data is None or data.empty:
        return []
    docs = []
    for _, row in data.iterrows():
        title = str(row.get("门店名称", "未知门店"))
        docs.append(
            (
                "门店经营数据",
                title,
                store_row_to_content(row),
                {
                    "store_id": str(row.get("门店ID", "")),
                    "category": str(row.get("业态分类", "")),
                    "floor": str(row.get("楼层", "")),
                },
            )
        )
    return docs


def build_rag_database(data: pd.DataFrame | None = None, db_path: Path = RAG_DB_PATH) -> dict[str, int | str]:
    with connect(db_path) as conn:
        init_db(conn)
        counts = {"门店经营数据": 0, "制度与模板": 0, "Agent 设计方案草稿": 0}
        for source, title, content, metadata in [
            *load_store_docs(data),
            *load_markdown_docs(),
            *load_source_docs(),
            *load_interview_docs(),
        ]:
            insert_chunk(conn, source, title, content, metadata)
            counts[source] = counts.get(source, 0) + 1
        conn.commit()
    return {"db_path": str(db_path), **counts}


def rag_database_stats(db_path: Path = RAG_DB_PATH) -> dict[str, int | str]:
    if not db_path.exists():
        return {"db_path": str(db_path), "total_chunks": 0}
    with connect(db_path) as conn:
        rows = conn.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source").fetchall()
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return {"db_path": str(db_path), "total_chunks": total, **{source: count for source, count in rows}}


def search_rag_database(query: str, top_k: int = 8, db_path: Path = RAG_DB_PATH) -> list[RagSearchResult]:
    if not db_path.exists():
        return []
    query_vector = embed_text(query)
    query_tokens = tokenize(query)
    query_terms = " OR ".join(query_tokens[:8])
    fts_scores: dict[int, float] = {}
    with connect(db_path) as conn:
        if query_terms:
            try:
                rows = conn.execute(
                    """
                    SELECT rowid, bm25(chunks_fts) AS rank
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    LIMIT 40
                    """,
                    (query_terms,),
                ).fetchall()
                fts_scores = {row_id: 1 / (1 + abs(rank)) for row_id, rank in rows}
            except sqlite3.OperationalError:
                fts_scores = {}
        rows = conn.execute("SELECT id, source, title, content, metadata_json, embedding_json FROM chunks").fetchall()

    results = []
    image_query = any(term in query for term in ["图片", "截图", "OCR", "ocr", "图里", "照片", "催款函", "催缴函"])
    source_query = any(
        term in query.lower()
        for term in ["合同", "条款", "制度", "流程", "台账", "xlsx", "csv", "pdf", "docx", "图片", "ocr", "巡检", "资料", "文件"]
    )
    for row_id, source, title, content, metadata_json, embedding_json in rows:
        metadata = json.loads(metadata_json)
        file_name = str(metadata.get("file", ""))
        haystack = f"{source}\n{title}\n{file_name}\n{content}".lower()
        vector_score = cosine(query_vector, json.loads(embedding_json))
        lexical_score = fts_scores.get(row_id, 0.0)
        score = 0.7 * vector_score + 0.3 * lexical_score

        exact_hits = sum(1 for term in query_tokens[:16] if len(term) >= 2 and term.lower() in haystack)
        if exact_hits:
            score += min(0.24, exact_hits * 0.035)
        if any(term.lower() in f"{title}\n{file_name}".lower() for term in query_tokens[:16] if len(term) >= 2):
            score += 0.16

        if source == "门店经营数据" and any(term in query for term in ["门店", "欠费", "销售", "租售比", "风险"]):
            score += 0.08
        if source != "门店经营数据" and source_query:
            score += 0.12
        if source == "图片资料" and image_query:
            score += 0.22
        results.append(
            RagSearchResult(
                source=source,
                title=title,
                content=content,
                score=score,
                metadata=metadata,
            )
        )
    results.sort(key=lambda item: item.score, reverse=True)
    return [result for result in results[:top_k] if result.score > 0] or results[:top_k]
