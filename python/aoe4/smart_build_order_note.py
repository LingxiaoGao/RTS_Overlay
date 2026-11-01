"""Render a smart build-order note to HTML with inline icon images.

Pure helper with no Qt imports. Icon markers use ``@relative/path.webp@``
(also png/jpg/jpeg). Existing files under the assets root become escaped
``<img>`` tags; missing files become a readable escaped label. Ordinary text
is HTML-escaped, and common English guide terms are translated to Simplified
Chinese.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Union


_ICON_RE = re.compile(r"@([^@\s]+\.(?:webp|png|jpg|jpeg))@", re.IGNORECASE)

_TERMS = (
    ("enemy base", "敌方基地"),
    ("age up", "升级时代"),
    ("villagers", "村民"),
    ("villager", "村民"),
    ("Research", "研究"),
    ("Produce", "生产"),
    ("Create", "创建"),
    ("Queue", "排队"),
    ("Rally", "集结点"),
    ("Build", "建造"),
    ("First", "首先"),
    ("Send", "派遣"),
    ("Move", "移动"),
    ("until", "直到"),
    ("after", "之后"),
    ("once", "一旦"),
    ("then", "然后"),
    ("back", "返回"),
    ("with", "带有"),
    ("from", "从"),
    ("food", "食物"),
    ("wood", "木材"),
    ("gold", "黄金"),
    ("stone", "石料"),
    ("house", "房屋"),
    ("sheep", "绵羊"),
    ("deer", "鹿"),
    ("to", "到"),
)

_TERM_RE = re.compile(
    r"\b(" + "|".join(re.escape(term) for term, _ in _TERMS) + r")\b",
    re.IGNORECASE,
)
_TERM_MAP = {term.lower(): zh for term, zh in _TERMS}


def _translate(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        return _TERM_MAP[match.group(1).lower()]

    return _TERM_RE.sub(_replace, text)


def _label_for(path: str) -> str:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    return stem.replace("_", " ").replace("-", " ").strip() or name


def render_note_html(
    note: str, assets_root: Union[str, Path], icon_size: int
) -> str:
    """Render a build-order note to escaped HTML with inline icons."""
    if icon_size <= 0:
        raise ValueError("icon_size must be positive")

    root = Path(assets_root)
    source = html.unescape(note)
    parts: list[str] = []
    position = 0

    for match in _ICON_RE.finditer(source):
        parts.append(_translate(html.escape(source[position:match.start()])))
        position = match.end()
        rel = match.group(1)
        candidate = root / rel
        if candidate.is_file():
            src = html.escape(str(candidate.resolve()))
            parts.append(
                f'<img src="{src}" width="{icon_size}" height="{icon_size}">'
            )
        else:
            parts.append(html.escape(_label_for(rel)))

    parts.append(_translate(html.escape(source[position:])))
    return "".join(parts)
