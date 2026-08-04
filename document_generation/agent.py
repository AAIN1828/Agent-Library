"""
Document Generation — instruction-driven in-place PDF text updater.

Takes an existing PDF + change instructions, generates exact old→new
substring patches via Azure OpenAI, and applies them with overlay masking
so images, layout, and vectors are preserved (no PDF rebuild / paraphrasing).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chain_helpers import (
    apply_updates_inplace,
    extract_pdf_pages,
    generate_update_patch,
    update_document,
)


class DocumentGenerationRunner:
    """Public entrypoint for the PDF in-place update package."""

    def run(
        self,
        pdf_path: str | None = None,
        instruction: str | None = None,
        *,
        input_pdf_path: str | None = None,
        output_pdf_path: str | None = None,
        json_dir: str | None = None,
        change_instructions: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """
        Update an existing PDF in place according to instruction text.

        Required:
          - pdf_path / input_pdf_path: source PDF
          - instruction / change_instructions: natural-language change brief

        Optional:
          - output_pdf_path: destination (default: ``<stem>_updated.pdf`` beside input)
          - json_dir: directory for extract/patch JSON artifacts
        """
        source = (input_pdf_path or pdf_path or "").strip()
        instr = (instruction or change_instructions or "").strip()
        if not source:
            raise ValueError("pdf_path (or input_pdf_path) is required")
        if not instr:
            raise ValueError("instruction (or change_instructions) is required")

        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"PDF not found: {source_path}")

        if output_pdf_path and str(output_pdf_path).strip():
            dest = Path(output_pdf_path).expanduser().resolve()
        else:
            dest = source_path.with_name(f"{source_path.stem}_updated{source_path.suffix}")

        result = update_document(
            input_pdf_path=source_path,
            instruction=instr,
            output_pdf_path=dest,
            json_dir=json_dir,
        )
        return {
            "input_pdf_path": str(source_path),
            "updated_pdf_path": result["updated_pdf_path"],
            "applied_updates": result["applied_updates"],
            "proposed_updates": result["proposed_updates"],
            "page_count": result["page_count"],
            "update_count": len(result["applied_updates"]),
            "artifact_dir": result["artifact_dir"],
            "dispatch_output": "updated_pdf_and_applied_patches",
        }


__all__ = [
    "DocumentGenerationRunner",
    "extract_pdf_pages",
    "generate_update_patch",
    "apply_updates_inplace",
    "update_document",
]
