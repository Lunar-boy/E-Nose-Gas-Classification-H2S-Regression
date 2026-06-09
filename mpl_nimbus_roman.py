"""Shared Matplotlib font configuration for Nimbus Roman plots."""

from __future__ import annotations

import difflib
import sys

import matplotlib as mpl
from matplotlib import font_manager


_PREFERRED_FONTS = ("Nimbus Roman", "Nimbus Roman No9 L")
_FALLBACK_SERIF_FONTS = ("Times New Roman", "Times", "DejaVu Serif")
_HAS_WARNED = False


def _installed_font_names() -> set[str]:
    return {entry.name for entry in font_manager.fontManager.ttflist}


def _resolve_font() -> tuple[str, list[str], bool]:
    names = _installed_font_names()
    for preferred in _PREFERRED_FONTS:
        if preferred in names:
            serif_list = list(dict.fromkeys((*_PREFERRED_FONTS, *_FALLBACK_SERIF_FONTS)))
            return preferred, serif_list, True

    sorted_names = sorted(names)
    close_matches: list[str] = []
    for preferred in _PREFERRED_FONTS:
        close_matches.extend(difflib.get_close_matches(preferred, sorted_names, n=4, cutoff=0.45))
    token_matches = [
        name for name in sorted_names
        if any(token in name for token in ("Nimbus", "Roman", "Times"))
    ]
    suggestions = list(dict.fromkeys(close_matches + token_matches))[:8]
    return "serif", suggestions, False


def set_global_font() -> None:
    """Prefer Nimbus Roman everywhere, with a non-fatal serif fallback."""
    global _HAS_WARNED

    font_name, extra_list, found = _resolve_font()
    if found:
        serif_list = list(dict.fromkeys((font_name, *extra_list)))
    else:
        serif_list = list(dict.fromkeys((*_PREFERRED_FONTS, *extra_list, *_FALLBACK_SERIF_FONTS)))
        if not _HAS_WARNED:
            if extra_list:
                suggestion_text = ", ".join(extra_list)
            else:
                suggestion_text = "none"
            print(
                "Warning: Nimbus Roman is not installed. "
                f"Close installed matches: {suggestion_text}. Falling back to generic serif.",
                file=sys.stderr,
            )
            _HAS_WARNED = True

    mpl.rcParams.update(
        {
            "font.family": font_name,
            "font.serif": serif_list,
            "mathtext.fontset": "custom",
            "mathtext.rm": font_name,
            "mathtext.it": f"{font_name}:italic",
            "mathtext.bf": f"{font_name}:bold",
            "mathtext.cal": f"{font_name}:italic",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
