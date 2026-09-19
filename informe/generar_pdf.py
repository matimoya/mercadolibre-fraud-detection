# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown-it-py", "playwright", "pypdf"]
# ///
"""Genera informe.pdf desde informe.md, con el diseño del enunciado.

La franja amarilla con el logo de Mercado Libre se repite en cada página, la
letra es Lato y el número de página va abajo a la derecha. Antes de armar el
PDF vuelve a extraer las figuras de los notebooks, para que muestre las de la
última ejecución. Imprime con el Google Chrome instalado. Ejecutar desde la
raíz: `uv run informe/generar_pdf.py`.
"""

import base64
import os
import re
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt
from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter

RAIZ = Path(__file__).resolve().parents[1]
INFORME = RAIZ / "informe"
FUENTE = INFORME / "informe.md"
DESTINO = INFORME / "informe.pdf"
LOGO = INFORME / "recursos" / "logo_mercadolibre.png"

AUTOR = "Matias Moyano"
TITULO = "Detección de fraude: maximizar la ganancia"

# Párrafos con estilo propio, reconocidos por cómo empiezan en el markdown.
CLASES_POR_INICIO = {
    "ganancia =": "formula",
    "**Hipótesis central.**": "destacado",
    "*Detalle:": "detalle",
}

# Medidas tomadas del enunciado: franja, logo y texto de la franja.
ENCABEZADO = """
<style>
  .franja {{ position: absolute; top: 3.4mm; left: 0; width: 211mm; height: 24.7mm;
            background: #FFF159; -webkit-print-color-adjust: exact; }}
  .franja img {{ position: absolute; left: 25.4mm; top: 6.6mm; width: 16.6mm; }}
  .franja span {{ position: absolute; right: 26.4mm; top: 1.2mm; color: #000;
                 font: 700 11pt/24.7mm Lato, sans-serif; }}
</style>
<div class="franja"><img src="data:image/png;base64,{logo}"><span>Fraud Prevention Fintech</span></div>
"""

PIE = f"""
<div style="position: absolute; left: 25.4mm; right: 25.4mm; bottom: 9mm; display: flex;
            justify-content: space-between; font: 8.5pt Lato, sans-serif; color: #999;">
  <span>{AUTOR} · {TITULO}</span><span class="pageNumber" style="color: #666;"></span>
</div>
"""

ESTILO = """
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font: 10.5pt/1.45 Lato, sans-serif; color: #666; text-align: justify; }
h1, h2 { color: #000; text-align: left; break-after: avoid; }
h1 { font-size: 15pt; margin: 18pt 0 6pt; }
h2 { font-size: 12.5pt; margin: 14pt 0 5pt; }
h1.titulo { font-size: 21pt; margin: 0 0 5pt; }
p.subtitulo { margin: 0; }
p.autor { margin: 2pt 0 16pt; color: #434343; }
strong { color: #434343; }
h1 strong, h2 strong { color: inherit; }
p { margin: 0 0 8pt; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin-bottom: 3pt; }
table { width: 100%; border-collapse: collapse; margin: 4pt 0 11pt; font-size: 9pt;
        line-height: 1.3; text-align: left; }
th, td { border: 0.75pt solid #D9D9D9; padding: 3.5pt 6pt; vertical-align: top; }
th { background: #F2F2F2; color: #434343; font-weight: 700; }
table { break-inside: avoid; }
p:has(+ table), p:has(+ figure) { break-after: avoid; }
figure { margin: 6pt 0 12pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 100mm; }
figcaption { margin-top: 3pt; font-size: 9pt; font-style: italic; }
p.formula { padding: 5pt 8pt; background: #F2F2F2; color: #434343; text-align: left;
            font: 9pt/1.4 Menlo, monospace; }
p.destacado { padding: 2pt 0 2pt 9pt; border-left: 3pt solid #FFE600; }
p.detalle { font-size: 9pt; color: #888; }
"""


def cuerpo_html(texto: str) -> str:
    """El markdown a HTML, con las clases de título, autor, figuras y párrafos especiales."""
    parser = MarkdownIt("commonmark").enable("table")
    tokens = parser.parse(texto)
    titulo_visto = subtitulo_visto = False
    for posicion, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag == "h1" and not titulo_visto:
            token.attrSet("class", "titulo")
            titulo_visto = True
        elif token.type == "paragraph_open":
            contenido = tokens[posicion + 1].content
            if titulo_visto and not subtitulo_visto:
                token.attrSet("class", "subtitulo")
                subtitulo_visto = True
            for inicio, clase in CLASES_POR_INICIO.items():
                if contenido.replace("\\", "").startswith(inicio.replace("\\", "")):
                    token.attrSet("class", clase)
    html = parser.renderer.render(tokens, parser.options, {})

    subtitulo_fin = html.index("</p>", html.index('class="subtitulo"')) + len("</p>")
    html = html[:subtitulo_fin] + f'\n<p class="autor">{AUTOR}</p>' + html[subtitulo_fin:]

    # Una imagen seguida de un párrafo solo en cursiva es una figura con su epígrafe.
    return re.sub(
        r"<p>(<img [^>]*>)</p>\n(?:<p><em>((?:(?!</p>).)*)</em></p>\n)?",
        lambda figura: f"<figure>{figura[1]}<figcaption>{figura[2] or ''}</figcaption></figure>\n",
        html,
    )


def documento(cuerpo: str) -> str:
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<base href="{INFORME.as_uri()}/">
<title>{TITULO}</title>
<style>{ESTILO}</style>
</head><body>{cuerpo}</body></html>"""


def imprimir(html: str, destino: Path) -> None:
    logo = base64.b64encode(LOGO.read_bytes()).decode()
    with tempfile.TemporaryDirectory() as carpeta:
        pagina_html = Path(carpeta) / "informe.html"
        pagina_html.write_text(html, encoding="utf-8")
        with sync_playwright() as playwright:
            navegador = playwright.chromium.launch(channel="chrome")
            pagina = navegador.new_page()
            pagina.goto(pagina_html.as_uri(), wait_until="load")
            pagina.evaluate("document.fonts.ready")
            pagina.pdf(
                path=str(destino),
                format="A4",
                print_background=True,
                display_header_footer=True,
                header_template=ENCABEZADO.format(logo=logo),
                footer_template=PIE,
                margin={"top": "36mm", "bottom": "22mm", "left": "25.4mm", "right": "25.4mm"},
            )
            navegador.close()


def firmar(ruta: Path) -> None:
    """Título y autor en los metadatos del PDF."""
    escritor = PdfWriter(clone_from=PdfReader(ruta))
    escritor.add_metadata(
        {
            "/Title": TITULO,
            "/Author": AUTOR,
            "/Subject": "Data Scientist Technical Challenge — Fraud Prevention Fintech",
        }
    )
    with ruta.open("wb") as archivo:
        escritor.write(archivo)


def main() -> None:
    os.chdir(RAIZ)
    import extraer_figuras  # pylint: disable=import-outside-toplevel

    extraer_figuras.extraer()
    imprimir(documento(cuerpo_html(FUENTE.read_text(encoding="utf-8"))), DESTINO)
    firmar(DESTINO)
    paginas = len(PdfReader(DESTINO).pages)
    print(
        f"\n{DESTINO.relative_to(RAIZ)}: {paginas} páginas, {DESTINO.stat().st_size / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    main()
