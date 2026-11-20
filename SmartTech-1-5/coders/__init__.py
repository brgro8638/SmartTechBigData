"""Entropy coders compared by the app. Each exposes encode(image) and decode(stream)."""

from . import arithmetic, cabac, huffman

CODERS = {
    "Huffman": huffman,
    "Arithmetic": arithmetic,
    "CABAC": cabac,
}
