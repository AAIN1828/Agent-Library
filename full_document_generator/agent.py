"""Full Document Generator — from-scratch PDF document authoring package.

Generates a brand-new PDF document from natural language instructions.
Unlike document_generation or patch_generation (which update existing PDFs in-place),
this agent structures complete document content (title, summary, sections) via Azure OpenAI
and renders a newly minted PDF file using PyMuPDF (fitz).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .chain_helpers import (
    create_full_document,
    generate_document_content,
    render_pdf_from_content,
)


class FullDocumentGeneratorRunner:
    """Public entrypoint for from-scratch PDF document generation."""

    def run(
        self,
        content: str | None = None,
        instruction: str | None = None,
        document_type: str | None = None,
        output_pdf_path: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """
        Generate a new PDF document from scratch based on prompt instructions.

        Required:
          - instruction / content: Natural language instructions describing the document to generate.

        Optional:
          - document_type: Optional contextual descriptor influencing document tone and structure
            (e.g., "report", "memo", "proposal", "contract").
          - output_pdf_path: Destination path for the generated PDF. Defaults to a timestamped file.
        """
        instr = (instruction or content or "").strip()
        if not instr:
            raise ValueError("instruction (or content) is required")

        doc_type = (document_type or "").strip() or None

        if output_pdf_path and str(output_pdf_path).strip():
            dest = Path(output_pdf_path).expanduser().resolve()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = (Path.cwd() / "generated_documents" / f"document_{timestamp}.pdf").resolve()

        result = create_full_document(
            instruction=instr,
            output_pdf_path=dest,
            document_type=doc_type,
        )

        return {
            "generated_pdf_path": result["generated_pdf_path"],
            "title": result["title"],
            "sections": result["sections"],
            "page_count": result["page_count"],
            "summary": result["summary"],
            "dispatch_output": "generated_new_pdf_document",
        }


__all__ = [
    "FullDocumentGeneratorRunner",
    "generate_document_content",
    "render_pdf_from_content",
    "create_full_document",
]
