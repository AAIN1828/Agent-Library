"""Internal pipeline steps for instruction-driven in-place PDF text updates.

Steps: extract page-wise text → LLM exact old→new patches → apply overlays
on the original PDF (images/layout/vectors preserved; no rebuild).
"""

from __future__ import annotations

import json
from datetime import datetime
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
You are a deterministic document update engine.

You receive a PDF converted into page-wise text JSON.
The document already contains existing values.
Instructions describe the desired final state.

Rules:
- Identify ALL conflicting text
- Replace ONLY the smallest possible substrings
- Prefer entity-level replacements
- Do NOT paraphrase
- Do NOT rewrite sentences
- Preserve layout and formatting
- Generate multiple updates if conflicts repeat
- Be page-aware

Return ONLY valid JSON.

Format:
{
  "updates": [
    {
      "page": <page_number>,
      "old": "<exact text from document>",
      "new": "<replacement text>"
    }
  ]
}
"""

PAGE_MARGIN_RIGHT = 50


def _get_llm_client() -> AzureOpenAI:
    ensure_azure_openai_config()
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_API_BASE.rstrip("/"),
    )


def save_json(data: dict[str, Any] | list[Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def extract_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract page-wise text for analysis only — never used to rebuild the PDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (pymupdf) is required for PDF extraction. "
            "Install it with: pip install pymupdf"
        ) from exc

    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(path)
    try:
        pages: list[dict[str, Any]] = []
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            blocks = [line.strip() for line in text.split("\n") if line.strip()]
            pages.append({"page_number": page_number, "blocks": blocks})
        return pages
    finally:
        doc.close()


def generate_update_patch(
    pages_json: list[dict[str, Any]],
    instruction: str,
    *,
    client: AzureOpenAI | None = None,
) -> list[dict[str, Any]]:
    """Ask the LLM for exact substring replacements (no paraphrasing)."""
    llm = client or _get_llm_client()
    user_prompt = f"""
DOCUMENT (PAGE-WISE JSON):
{json.dumps(pages_json, indent=2)}

CHANGE INSTRUCTIONS:
{instruction}
"""
    response = llm.chat.completions.create(
        model=GPT4_LLM_MODEL_DEPLOYMENT_NAME,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {}
    updates = parsed.get("updates", []) if isinstance(parsed, dict) else []
    if not isinstance(updates, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in updates:
        if not isinstance(item, dict):
            continue
        old = str(item.get("old") or "")
        new = str(item.get("new") or "")
        if not old or old == new:
            continue
        try:
            page = int(item.get("page") or 0)
        except (TypeError, ValueError):
            continue
        if page < 1:
            continue
        cleaned.append({"page": page, "old": old, "new": new})
    return cleaned


def apply_updates_inplace(
    input_pdf_path: str | Path,
    updates: list[dict[str, Any]],
    output_pdf_path: str | Path,
) -> list[dict[str, Any]]:
    """Apply patches via white overlay + redraw so images/layout stay intact."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (pymupdf) is required for PDF updates. "
            "Install it with: pip install pymupdf"
        ) from exc

    input_path = Path(input_pdf_path).expanduser().resolve()
    output_path = Path(output_pdf_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(input_path)
    applied: list[dict[str, Any]] = []
    try:
        for upd in updates:
            page_index = int(upd["page"]) - 1
            old_text = str(upd["old"])
            new_text = str(upd["new"])
            if page_index < 0 or page_index >= len(doc):
                continue
            page = doc[page_index]
            page_width = page.rect.width
            page_dict = page.get_text("dict")
            matched = False

            for block in page_dict.get("blocks") or []:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    spans = line.get("spans") or []
                    line_text = "".join(str(s.get("text") or "") for s in spans)
                    if old_text not in line_text:
                        continue
                    matched = True
                    line_rect = fitz.Rect(line["bbox"])
                    page.draw_rect(
                        line_rect,
                        color=(1, 1, 1),
                        fill=(1, 1, 1),
                        overlay=True,
                    )
                    current_x = spans[0]["origin"][0] if spans else line_rect.x0
                    for span in spans:
                        original_font = str(span.get("font") or "").lower()
                        is_bold = "bold" in original_font or (int(span.get("flags") or 0) & 2)
                        font_to_use = "helv-bold" if is_bold else "helv"
                        original_text = str(span.get("text") or "")
                        content_to_draw = (
                            original_text.replace(old_text, new_text)
                            if old_text in original_text
                            else original_text
                        )
                        color_int = int(span.get("color") or 0)
                        rgb_color = (
                            ((color_int >> 16) & 255) / 255,
                            ((color_int >> 8) & 255) / 255,
                            (color_int & 255) / 255,
                        )
                        size = float(span.get("size") or 11)
                        try:
                            text_w = fitz.get_text_length(
                                content_to_draw,
                                fontsize=size,
                                fontname=font_to_use,
                            )
                        except Exception:
                            font_to_use = "helv"
                            text_w = fitz.get_text_length(
                                content_to_draw,
                                fontsize=size,
                                fontname=font_to_use,
                            )
                        final_size = size
                        if current_x + text_w > page_width - PAGE_MARGIN_RIGHT:
                            available_w = page_width - PAGE_MARGIN_RIGHT - current_x
                            if available_w > 0 and text_w > 0:
                                final_size = size * (available_w / text_w)
                        origin_y = span["origin"][1]
                        page.insert_text(
                            (current_x, origin_y),
                            content_to_draw,
                            fontsize=final_size,
                            color=rgb_color,
                            fontname=font_to_use,
                            overlay=True,
                        )
                        current_x += fitz.get_text_length(
                            content_to_draw,
                            fontsize=final_size,
                            fontname=font_to_use,
                        )
            if matched:
                applied.append({"page": page_index + 1, "old": old_text, "new": new_text})

        doc.save(output_path, incremental=False)
    finally:
        doc.close()
    return applied


def update_document(
    input_pdf_path: str | Path,
    instruction: str,
    output_pdf_path: str | Path,
    *,
    json_dir: str | Path | None = None,
) -> dict[str, Any]:
    """End-to-end extract → patch → apply pipeline (layout & image preserving)."""
    instruction = (instruction or "").strip()
    if not instruction:
        raise ValueError("instruction text is required")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(json_dir) if json_dir else Path(output_pdf_path).parent / "json_artifacts"

    pages = extract_pdf_pages(input_pdf_path)
    save_json(pages, artifact_dir / f"extracted_pages_{timestamp}.json")

    updates = generate_update_patch(pages, instruction)
    save_json({"updates": updates}, artifact_dir / f"update_patch_{timestamp}.json")

    applied = apply_updates_inplace(
        input_pdf_path=input_pdf_path,
        updates=updates,
        output_pdf_path=output_pdf_path,
    )
    return {
        "updated_pdf_path": str(Path(output_pdf_path).expanduser().resolve()),
        "applied_updates": applied,
        "proposed_updates": updates,
        "page_count": len(pages),
        "artifact_dir": str(artifact_dir.resolve()),
    }


__all__ = [
    "SYSTEM_PROMPT",
    "save_json",
    "extract_pdf_pages",
    "generate_update_patch",
    "apply_updates_inplace",
    "update_document",
]
