import argparse
import subprocess
import sys
from pathlib import Path
import re

try:
    import markdown  # type: ignore
except Exception:
    markdown = None


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <base href="{base_href}">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.55; color: #222; margin: 40px; }}
    h1, h2, h3 {{ color: #111; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }}
    pre {{ background: #f6f8fa; padding: 12px; border-radius: 6px; overflow: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; }}
    img {{ max-width: 100%; height: auto; }}
    .mermaid {{ page-break-inside: avoid; }}
  </style>
  <script>window.mermaid={{}};</script>
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({{ startOnLoad: true }});</script>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [["\\(","\\)"]], displayMath: [["\\[","\\]"]] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
  <title>{title}</title>
</head>
<body>
{body}
</body>
</html>
"""


def replace_mermaid_pre_with_div(html: str) -> str:
    # Convert <pre><code class="language-mermaid">...</code></pre> into <div class="mermaid">...</div>
    pattern = re.compile(r"<pre><code class=\"language-mermaid\">([\s\S]*?)</code></pre>")
    return re.sub(pattern, lambda m: f"<div class=\"mermaid\">{m.group(1)}</div>", html)


def convert_markdown_to_html(md_path: Path, out_html: Path) -> None:
    if markdown is None:
        raise RuntimeError("The 'markdown' package is required. Run: pip install markdown")

    text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=[
            'extra',
            'admonition',
            'sane_lists',
            'smarty',
            'toc',
            'tables',
            'fenced_code',
            'codehilite',
        ],
        extension_configs={
            'codehilite': {'guess_lang': False}
        }
    )

    html_body = replace_mermaid_pre_with_div(html_body)

    base_href = md_path.parent.resolve().as_uri() + '/'
    title = md_path.stem
    full_html = HTML_TEMPLATE.format(base_href=base_href, title=title, body=html_body)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(full_html, encoding="utf-8")


def find_browser_exe() -> Path:
    candidates = [
        Path(r"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe"),
        Path(r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"),
        Path(r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"),
        Path(r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Headless browser not found (Edge/Chrome). Please install Microsoft Edge or Google Chrome.")


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    browser = find_browser_exe()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    file_url = html_path.resolve().as_uri()
    cmd = [
        str(browser),
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path.resolve()}",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=15000",
        file_url,
    ]
    subprocess.run(cmd, check=True)


def export(md_paths: list[Path], out_dir: Path) -> None:
    for md in md_paths:
        html_out = out_dir / (md.stem + ".html")
        pdf_out = out_dir / (md.stem + ".pdf")
        convert_markdown_to_html(md, html_out)
        render_pdf(html_out, pdf_out)
        print(f"Exported: {pdf_out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export project Markdown docs to PDF with Mermaid and MathJax using headless Edge/Chrome")
    parser.add_argument("--out-dir", default=str(Path("outputs") / "reports"), help="Output directory for HTML/PDF files")
    parser.add_argument("--docs", nargs='*', help="Markdown files to export; defaults to the 4 key docs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    default_docs = [
        project_root / "docs" / "HSBC_REPORT_FR.md",
        project_root / "docs" / "HSBC_REPORT_EN.md",
        project_root / "docs" / "FORMULES_ET_EXEMPLES.md",
        project_root / "docs" / "FORMULAS_AND_EXAMPLES.md",
    ]
    md_paths = [Path(p) for p in (args.docs if args.docs else default_docs)]
    out_dir = Path(args.out_dir)

    export(md_paths, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



