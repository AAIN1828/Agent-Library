"""Internal pipeline steps for from-scratch PDF document generation.

Steps: LLM structures document content (title, summary, sections) -> PyMuPDF renders brand-new PDF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

from .runtime_config import (
    AZURE_OPENAI_API_BASE,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    GPT4_LLM_MODEL_DEPLOYMENT_NAME,
    ensure_azure_openai_config,
)

SYSTEM_PROMPT = """
You are an expert document writer and editor.

Your task is to take user instructions and generate a fully realized, professional document structure.
You must return ONLY valid JSON matching this schema:

{
  "title": "<Document Title>",
  "summary": "<Executive Summary or Brief Description>",
  "sections": [
    {
      "heading": "<Section Heading>",
      "content": "<Detailed paragraph text for this section>"
    }
  ]
}

Rules:
- Make content rich, complete, and tailored to the requested document type and instruction.
- Provide logical section headings and coherent paragraphs.
- Do NOT return markdown or code blocks outside the JSON object.
"""


def _get_llm_client() -> AzureOpenAI:
    ensure_azure_openai_config()
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_API_BASE.rstrip("/"),
    )


def generate_document_content(
    instruction: str,
    document_type: str | None = None,
    *,
    client: AzureOpenAI | None = None,
) -> dict[str, Any]:
    """Generate structured document content (title, summary, sections) using Azure OpenAI."""
    llm = client or _get_llm_client()
    doc_type_str = f"Document Type: {document_type}\n" if document_type else ""
    user_prompt = f"{doc_type_str}Instruction: {instruction}"

    response = llm.chat.completions.create(
        model=GPT4_LLM_MODEL_DEPLOYMENT_NAME,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        parsed = {}

    if not isinstance(parsed, dict):
        parsed = {}

    title = str(parsed.get("title") or "Generated Document").strip()
    summary = str(parsed.get("summary") or "").strip()
    raw_sections = parsed.get("sections")
    sections: list[dict[str, str]] = []

    if isinstance(raw_sections, list):
        for sec in raw_sections:
            if isinstance(sec, dict):
                heading = str(sec.get("heading") or "").strip()
                body = str(sec.get("content") or "").strip()
                if heading or body:
                    sections.append({"heading": heading, "content": body})

    if not sections:
        sections = [
            {
                "heading": "Overview",
                "content": str(instruction),
            }
        ]

    return {
        "title": title,
        "summary": summary,
        "sections": sections,
    }


def render_pdf_from_content(
    content: dict[str, Any],
    output_pdf_path: str | Path,
) -> dict[str, Any]:
    """Render structured document content into a brand-new PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (pymupdf) is required for PDF generation. "
            "Install it with: pip install pymupdf"
        ) from exc

    output_path = Path(output_pdf_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    try:
        # Standard A4 size: 595 x 842 points
        page_width, page_height = 595.0, 842.0
        margin_left = 50.0
        margin_right = 50.0
        margin_top = 50.0
        margin_bottom = 50.0
        usable_width = page_width - margin_left - margin_right

        page = doc.new_page(width=page_width, height=page_height)
        y_cursor = margin_top

        def check_space_and_advance(height_needed: float) -> fitz.Page:
            nonlocal page, y_cursor
            if y_cursor + height_needed > page_height - margin_bottom:
                page = doc.new_page(width=page_width, height=page_height)
                y_cursor = margin_top
            return page

        # Title
        title_text = str(content.get("title") or "Untitled Document")
        title_rect = fitz.Rect(margin_left, y_cursor, page_width - margin_right, y_cursor + 40)
        page.insert_textbox(
            title_rect,
            title_text,
            fontsize=20,
            fontname="helv",
            color=(0.1, 0.1, 0.3),
            align=fitz.TEXT_ALIGN_LEFT,
        )
        y_cursor += 45.0

        # Summary if present
        summary_text = str(content.get("summary") or "").strip()
        if summary_text:
            page = check_space_and_advance(50.0)
            summary_rect = fitz.Rect(margin_left, y_cursor, page_width - margin_right, y_cursor + 60)
            rc = page.insert_textbox(
                summary_rect,
                f"Executive Summary:\n{summary_text}",
                fontsize=11,
                fontname="helv",
                color=(0.3, 0.3, 0.3),
                align=fitz.TEXT_ALIGN_LEFT,
            )
            y_cursor += (60.0 - (rc if rc >= 0 else 0)) + 15.0

        # Sections
        sections = content.get("sections") or []
        section_titles: list[str] = []

        for sec in sections:
            heading = str(sec.get("heading") or "").strip()
            body = str(sec.get("content") or "").strip()
            if heading:
                section_titles.append(heading)

            # Insert Heading
            if heading:
                min_body_space = 70.0 if body else 0.0
                heading_total_space = 30.0 + min_body_space
                page = check_space_and_advance(heading_total_space)
                h_rect = fitz.Rect(margin_left, y_cursor, page_width - margin_right, y_cursor + 25.0)
                page.insert_textbox(
                    h_rect,
                    heading,
                    fontsize=14,
                    fontname="helv",
                    color=(0.15, 0.15, 0.15),
                    align=fitz.TEXT_ALIGN_LEFT,
                )
                y_cursor += 30.0

            # Insert Body Content
            remaining_body = body
            max_y = page_height - margin_bottom
            while remaining_body:
                avail_h = max_y - y_cursor
                if avail_h < 40.0:
                    page = doc.new_page(width=page_width, height=page_height)
                    y_cursor = margin_top
                    avail_h = max_y - y_cursor

                b_rect = fitz.Rect(margin_left, y_cursor, page_width - margin_right, y_cursor + avail_h)

                test_doc = fitz.open()
                test_page = test_doc.new_page(width=page_width, height=page_height)
                rc_test = test_page.insert_textbox(
                    b_rect,
                    remaining_body,
                    fontsize=10,
                    fontname="helv",
                    color=(0.2, 0.2, 0.2),
                    align=fitz.TEXT_ALIGN_LEFT,
                )
                test_doc.close()

                if rc_test >= 0:
                    page.insert_textbox(
                        b_rect,
                        remaining_body,
                        fontsize=10,
                        fontname="helv",
                        color=(0.2, 0.2, 0.2),
                        align=fitz.TEXT_ALIGN_LEFT,
                    )
                    used_h = avail_h - rc_test
                    y_cursor += used_h + 15.0
                    remaining_body = ""
                else:
                    low, high, fit_idx = 0, len(remaining_body), 0
                    while low <= high:
                        mid = (low + high) // 2
                        t_doc = fitz.open()
                        t_page = t_doc.new_page(width=page_width, height=page_height)
                        rc_m = t_page.insert_textbox(
                            b_rect,
                            remaining_body[:mid],
                            fontsize=10,
                            fontname="helv",
                            color=(0.2, 0.2, 0.2),
                            align=fitz.TEXT_ALIGN_LEFT,
                        )
                        t_doc.close()
                        if rc_m >= 0:
                            fit_idx = mid
                            low = mid + 1
                        else:
                            high = mid - 1

                    if fit_idx > 0:
                        last_space = remaining_body[:fit_idx].rfind(" ")
                        if last_space > 0:
                            fit_idx = last_space + 1
                        fitted_text = remaining_body[:fit_idx]
                        page.insert_textbox(
                            b_rect,
                            fitted_text,
                            fontsize=10,
                            fontname="helv",
                            color=(0.2, 0.2, 0.2),
                            align=fitz.TEXT_ALIGN_LEFT,
                        )
                        remaining_body = remaining_body[fit_idx:].lstrip()

                    page = doc.new_page(width=page_width, height=page_height)
                    y_cursor = margin_top

        doc.save(output_path)
        page_count = len(doc)
    finally:
        doc.close()

    return {
        "generated_pdf_path": str(output_path),
        "title": title_text,
        "sections": section_titles if section_titles else ["Main"],
        "page_count": page_count,
        "summary": summary_text,
    }


def create_full_document(
    instruction: str,
    output_pdf_path: str | Path,
    *,
    document_type: str | None = None,
    client: AzureOpenAI | None = None,
) -> dict[str, Any]:
    """End-to-end generate structured content via LLM -> render PDF via PyMuPDF."""
    content = generate_document_content(
        instruction=instruction,
        document_type=document_type,
        client=client,
    )
    rendered = render_pdf_from_content(content, output_pdf_path=output_pdf_path)
    return rendered


__all__ = [
    "SYSTEM_PROMPT",
    "generate_document_content",
    "render_pdf_from_content",
    "create_full_document",
]
