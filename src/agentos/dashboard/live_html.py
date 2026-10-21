"""Backward-compatible single-string build of the live dashboard.

The dashboard frontend now lives as separate static files under
``agentos/dashboard/static/`` (index.html / style.css / app.js), served
directly by ``server.py``. This module preserves the historical
``LIVE_DASHBOARD_HTML`` symbol for any consumer that needs the whole page as
one self-contained string: it reads the three static files and inlines the
CSS and JS back into the HTML.
"""

from __future__ import annotations

from pathlib import Path

_STATIC = Path(__file__).parent / "static"


def _build_inline_html() -> str:
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    css = (_STATIC / "style.css").read_text(encoding="utf-8")
    js = (_STATIC / "app.js").read_text(encoding="utf-8")
    html = html.replace(
        '<link rel="stylesheet" href="/style.css">',
        "<style>\n" + css + "\n</style>",
    )
    html = html.replace(
        '<script src="/app.js"></script>',
        "<script>\n" + js + "\n</script>",
    )
    return html


LIVE_DASHBOARD_HTML = _build_inline_html()
