"""Byte <-> printable-unicode map used by byte-level BPE (from GPT-2).

Byte-level BPE works on the UTF-8 *bytes* of the text, so the base vocabulary is
exactly 256 symbols and *no input can ever be out-of-vocabulary*. The merge
tables are stored as text, so each byte is mapped to a visible unicode
character (control characters and spaces would be unreadable / ambiguous).

Printable Latin-1 bytes map to themselves; the remaining bytes are shifted into
the range U+0100.. so that the map is a bijection of 256 distinct characters.
Example: the space byte 0x20 becomes 'Ġ' (U+0120) - the familiar GPT-2 marker.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def bytes_to_unicode() -> dict[int, str]:
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = printable[:]
    shifted = 0
    for b in range(256):
        if b not in printable:
            byte_values.append(b)
            printable.append(256 + shifted)
            shifted += 1
    return dict(zip(byte_values, (chr(c) for c in printable)))


@lru_cache(maxsize=None)
def unicode_to_bytes() -> dict[str, int]:
    return {c: b for b, c in bytes_to_unicode().items()}


def text_to_symbols(text: str) -> list[str]:
    """UTF-8 encode ``text`` and map every byte to its visible character."""
    table = bytes_to_unicode()
    return [table[b] for b in text.encode("utf-8")]


def symbols_to_text(symbols: str) -> str:
    """Inverse of :func:`text_to_symbols` (invalid UTF-8 -> U+FFFD)."""
    table = unicode_to_bytes()
    return bytes(table[c] for c in symbols).decode("utf-8", errors="replace")
