"""Convert selected AgentGate PDF figures to PNG beside their source PDFs.

Run from the AgentGate repository root:
    python convert_figures_to_png.py

Dependency:
    pip install pymupdf
"""

from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent

FIGURES = [
    "architecture_agengate.pdf",
    "training_analysis/loss_curve.pdf",
    "interpretability_results/repetition/repetition_loss_combined.pdf",
    "interpretability_results/heads/head_specialization_combined.pdf",
    "interpretability_results/ablation/ablation_effects.pdf",
    "interpretability_results/dependency/dependency_effects.pdf",
]

DPI = 200


def main():
    for relative_path in FIGURES:
        source = ROOT / relative_path
        if not source.is_file():
            print(f"[MISSING] {relative_path}")
            continue

        if source.suffix.lower() == ".png":
            print(f"[SKIP] Already PNG: {relative_path}")
            continue

        if source.suffix.lower() != ".pdf":
            print(f"[SKIP] Unsupported format: {relative_path}")
            continue

        output = source.with_suffix(".png")
        with fitz.open(source) as pdf:
            if not pdf.page_count:
                print(f"[SKIP] Empty PDF: {relative_path}")
                continue
            page = pdf[0]  # Figure PDFs should generally have a single page.
            pix = page.get_pixmap(dpi=DPI, alpha=False)
            pix.save(output)
            extra = f" (first page of {pdf.page_count})" if pdf.page_count > 1 else ""
        print(f"[OK] {output.relative_to(ROOT)}{extra}")


if __name__ == "__main__":
    main()
