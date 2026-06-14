"""
make_pdf.py — convierte INFORME.md a INFORME.pdf.

Markdown → HTML (librería `markdown`, con tablas y bloques de código) → PDF
(Microsoft Edge en modo headless, que renderiza HTML/CSS y Unicode con fidelidad).

Uso:
    python make_pdf.py
    python make_pdf.py --input OTRO.md --output OTRO.pdf
"""
import argparse
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent

CSS = """
@page { size: A4; margin: 1.8cm 2cm; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 11pt;
       line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 21pt; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 15pt; border-bottom: 1px solid #ccc; padding-bottom: 4px;
     margin-top: 1.5em; page-break-after: avoid; }
h3 { font-size: 12.5pt; margin-top: 1.2em; page-break-after: avoid; }
p, li { font-size: 11pt; }
code { font-family: Consolas, 'Courier New', monospace; background: #f2f2f2;
       padding: 1px 4px; border-radius: 3px; font-size: 10pt; }
pre { background: #f6f8fa; border: 1px solid #e0e0e0; border-radius: 5px;
      padding: 10px 12px; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 9.5pt; line-height: 1.35; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 5px 8px; text-align: left; }
th { background: #eaeef2; }
blockquote { border-left: 4px solid #ccc; margin: 10px 0; padding: 4px 12px;
             color: #444; background: #fafafa; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
"""

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def main():
    ap = argparse.ArgumentParser(description="INFORME.md -> PDF (Edge headless)")
    ap.add_argument("--input", default=str(ROOT / "INFORME.md"))
    ap.add_argument("--output", default=str(ROOT / "INFORME.pdf"))
    args = ap.parse_args()

    md_path = Path(args.input)
    pdf_path = Path(args.output)
    html_path = md_path.with_suffix(".html")

    body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    html = (f"<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML intermedio: {html_path}")

    edge = next((e for e in EDGE_CANDIDATES if Path(e).exists()), None)
    if not edge:
        print("No se encontró Microsoft Edge. HTML generado; ábrelo e "
              "imprime a PDF desde el navegador.")
        sys.exit(1)

    file_url = html_path.resolve().as_uri()
    subprocess.run([edge, "--headless=new", "--disable-gpu",
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path.resolve()}", file_url], check=False)

    if pdf_path.exists():
        print(f"PDF generado: {pdf_path}  ({pdf_path.stat().st_size // 1024} KB)")
    else:
        print("Edge no generó el PDF; revisa el HTML manualmente.")
        sys.exit(1)


if __name__ == "__main__":
    main()
