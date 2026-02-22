from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree as ET

import fitz
from PIL import Image
from pptx import Presentation

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None

try:
    from docx import Document
except Exception:  # pragma: no cover - optional dependency
    Document = None

try:
    from openpyxl import load_workbook
except Exception:  # pragma: no cover - optional dependency
    load_workbook = None

try:
    import xlrd
except Exception:  # pragma: no cover - optional dependency
    xlrd = None

try:
    from striprtf.striprtf import rtf_to_text
except Exception:  # pragma: no cover - optional dependency
    rtf_to_text = None


Block = Dict[str, object]
DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
ODF_TEXT_NS = {
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
}


def _table_to_text(rows: List[List[object]]) -> str:
    cleaned_rows: List[List[str]] = []
    width = 0
    for row in rows:
        values = [str(cell).strip() if cell is not None else "" for cell in row]
        width = max(width, len(values))
        cleaned_rows.append(values)
    if width == 0:
        return ""
    normalized = [row + ([""] * (width - len(row))) for row in cleaned_rows]
    table_lines = ["\t".join(row).rstrip() for row in normalized if any(cell for cell in row)]
    return "\n".join(table_lines).strip()


def _image_from_blob(image_blob: bytes) -> Image.Image | None:
    try:
        image = Image.open(io.BytesIO(image_blob))
        image.load()
        return image
    except Exception:
        return None


def extract_pdf(path: Path) -> List[Block]:
    blocks: List[Block] = []
    
    with fitz.open(str(path)) as doc:
        for page_index, page in enumerate(doc, start=1):
            # Instead of extracting text, we render the entire page as a high-res image 
            # so GPT-4o can read it perfectly, preserving all tables and layout constraints.
            pix = page.get_pixmap(dpi=200)
            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            blocks.append({"page": page_index, "type": "image", "image": image})
            
    return blocks


def extract_pptx(path: Path) -> List[Block]:
    blocks: List[Block] = []
    # Similar to PDF, for a true Vision RAG, we would ideally render the slide to an image. 
    # Since python-pptx doesn't natively render slides to images, we'll keep the text/table
    # extraction for now, but extract images directly as well. 
    prs = Presentation(str(path))
    for slide_index, slide in enumerate(prs.slides, start=1):
        slide_text: List[str] = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text = str(shape.text).strip()
                if text:
                    slide_text.append(text)
            if getattr(shape, "has_table", False):
                rows = [[cell.text.strip() for cell in row.cells] for row in shape.table.rows]
                table_text = _table_to_text(rows)
                if table_text:
                    blocks.append({"page": slide_index, "type": "table", "text": table_text})
            if shape.shape_type == 13:  # picture
                image_bytes = shape.image.blob
                image = _image_from_blob(image_bytes)
                if image is not None:
                     blocks.append({"page": slide_index, "type": "image", "image": image})
        if slide_text:
            blocks.append({"page": slide_index, "type": "text", "text": "\n".join(slide_text)})
    return blocks


def _docx_text_from_xml(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    paragraphs: List[str] = []
    for paragraph in root.findall(".//w:p", DOCX_NS):
        pieces: List[str] = []
        for node in paragraph.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "t" and node.text:
                pieces.append(node.text)
            elif tag == "tab":
                pieces.append("\t")
            elif tag in {"br", "cr"}:
                pieces.append("\n")
        text = "".join(pieces).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs).strip()


def extract_docx(path: Path) -> List[Block]:
    if Document is not None:
        return _extract_docx_with_python_docx(path)

    blocks: List[Block] = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        part_names = ["word/document.xml"]
        part_names.extend(
            sorted(
                name
                for name in names
                if name.startswith("word/header") and name.endswith(".xml")
            )
        )
        part_names.extend(
            sorted(
                name
                for name in names
                if name.startswith("word/footer") and name.endswith(".xml")
            )
        )

        page = 1
        for part in part_names:
            if part not in names:
                continue
            text = _docx_text_from_xml(archive.read(part))
            if text:
                blocks.append({"page": page, "type": "text", "text": text})
                page += 1
    return blocks


def _extract_docx_with_python_docx(path: Path) -> List[Block]:
    blocks: List[Block] = []
    doc = Document(str(path))

    paragraph_text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()).strip()
    if paragraph_text:
        blocks.append({"page": 1, "type": "text", "text": paragraph_text})

    for table_idx, table in enumerate(doc.tables, start=1):
        rows: List[List[str]] = []
        for row in table.rows:
            rows.append([cell.text.strip() for cell in row.cells])
        table_text = _table_to_text(rows)
        if table_text:
            blocks.append({"page": 1, "type": "table", "text": f"Table {table_idx}\n{table_text}"})

    seen_parts = set()
    for rel in doc.part.rels.values():
        if "image" not in rel.reltype:
            continue
        target_part = getattr(rel, "target_part", None)
        partname = str(getattr(target_part, "partname", ""))
        if not target_part or partname in seen_parts:
            continue
        seen_parts.add(partname)
        image = _image_from_blob(target_part.blob)
        if image is not None:
            blocks.append({"page": 1, "type": "image", "image": image})
    return blocks


def extract_xlsx(path: Path) -> List[Block]:
    if load_workbook is None:
        raise RuntimeError("openpyxl is required to parse spreadsheet files")

    blocks: List[Block] = []
    workbook = load_workbook(str(path), data_only=True)
    try:
        for sheet_index, sheet in enumerate(workbook.worksheets, start=1):
            rows: List[List[object]] = []
            for row in sheet.iter_rows(values_only=True):
                values = list(row)
                if any(value not in (None, "") for value in values):
                    rows.append(values)

            table_text = _table_to_text(rows)
            if table_text:
                blocks.append(
                    {
                        "page": sheet_index,
                        "type": "table",
                        "text": f"Sheet: {sheet.title}\n{table_text}",
                    }
                )

            for image_ref in getattr(sheet, "_images", []):
                image_bytes = None
                if hasattr(image_ref, "_data"):
                    try:
                        image_bytes = image_ref._data()
                    except Exception:
                        image_bytes = None
                if not image_bytes:
                    continue
                image = _image_from_blob(image_bytes)
                if image is not None:
                    blocks.append({"page": sheet_index, "type": "image", "image": image})
    finally:
        workbook.close()
    return blocks


def extract_xls(path: Path) -> List[Block]:
    if xlrd is None:
        raise RuntimeError("xlrd is required to parse .xls spreadsheet files")

    blocks: List[Block] = []
    workbook = xlrd.open_workbook(str(path), on_demand=True)
    try:
        for sheet_index in range(workbook.nsheets):
            sheet = workbook.sheet_by_index(sheet_index)
            rows: List[List[object]] = []
            for row_index in range(sheet.nrows):
                values = sheet.row_values(row_index)
                if any(value not in (None, "") for value in values):
                    rows.append(values)
            table_text = _table_to_text(rows)
            if table_text:
                blocks.append(
                    {
                        "page": sheet_index + 1,
                        "type": "table",
                        "text": f"Sheet: {sheet.name}\n{table_text}",
                    }
                )
    finally:
        workbook.release_resources()
    return blocks


def extract_html(path: Path) -> List[Block]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if BeautifulSoup is not None:
        soup = BeautifulSoup(raw, "html.parser")
        text = soup.get_text(separator="\n").strip()
    else:
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [{"page": 1, "type": "text", "text": text}]


def extract_rtf(path: Path) -> List[Block]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = rtf_to_text(raw).strip() if rtf_to_text is not None else raw.strip()
    if not text:
        return []
    return [{"page": 1, "type": "text", "text": text}]


def extract_odf(path: Path) -> List[Block]:
    blocks: List[Block] = []
    with zipfile.ZipFile(path) as archive:
        if "content.xml" not in archive.namelist():
            return blocks
        content_xml = archive.read("content.xml")
    root = ET.fromstring(content_xml)
    text_parts = [
        node.text.strip()
        for node in root.findall(".//text:p", ODF_TEXT_NS)
        if node.text and node.text.strip()
    ]
    if text_parts:
        blocks.append({"page": 1, "type": "text", "text": "\n".join(text_parts)})

    table_rows: List[List[object]] = []
    for row_node in root.findall(".//table:table-row", ODF_TEXT_NS):
        row_values: List[str] = []
        for cell_node in row_node.findall(".//table:table-cell", ODF_TEXT_NS):
            cell_text = "".join(cell_node.itertext()).strip()
            row_values.append(cell_text)
        if any(value for value in row_values):
            table_rows.append(row_values)
    table_text = _table_to_text(table_rows)
    if table_text:
        blocks.append({"page": 1, "type": "table", "text": table_text})
    return blocks


def extract_image(path: Path) -> List[Block]:
    image = Image.open(path)
    blocks: List[Block] = []
    blocks.append({"page": 1, "type": "image", "image": image})
    return blocks


def extract_text(path: Path) -> List[Block]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [{"page": 1, "type": "text", "text": text}]


def extract_generic(path: Path) -> List[Block]:
    suffix = path.suffix.lower()
    if suffix in {".htm", ".html", ".xhtml", ".xml"}:
        return extract_html(path)
    if suffix in {".rtf"}:
        return extract_rtf(path)
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        return extract_xlsx(path)
    if suffix in {".xls"}:
        return extract_xls(path)
    if suffix in {".odt", ".ods", ".odp"}:
        return extract_odf(path)

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "word/document.xml" in names:
                return extract_docx(path)
            if "xl/workbook.xml" in names:
                return extract_xlsx(path)
            if "ppt/presentation.xml" in names:
                return extract_pptx(path)
            if "content.xml" in names:
                return extract_odf(path)

    try:
        return extract_pdf(path)
    except Exception:
        pass

    blocks = extract_text(path)
    if blocks:
        return blocks

    ext = suffix or "[no extension]"
    raise ValueError(f"Unsupported file type: {ext}")

