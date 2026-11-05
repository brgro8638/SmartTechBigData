"""Shared helpers for the three entropy coders.

Every coder takes a 2D integer array and returns a self-contained byte stream:
the decoder receives nothing but those bytes, so the reported size is honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Encoded:
    """Output of an encoder, with the header size split out for reporting."""

    stream: bytes
    header_bytes: int
    stats: dict = field(default_factory=dict)

    @property
    def payload_bytes(self) -> int:
        return len(self.stream) - self.header_bytes


def write_varint(output: bytearray, value: int) -> None:
    """LEB128: small numbers take one byte, which keeps symbol tables compact."""
    if value < 0:
        raise ValueError("varint requires a non-negative value")
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)


def read_varint(data: bytes, position: int) -> tuple[int, int]:
    """Return (value, next_position)."""
    value = 0
    shift = 0
    while True:
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, position
        shift += 7


def zigzag(value: int) -> int:
    """Map signed to unsigned so small residuals stay small varints."""
    return value * 2 if value >= 0 else -value * 2 - 1


def unzigzag(value: int) -> int:
    return value >> 1 if value & 1 == 0 else -(value + 1) >> 1


def write_shape(output: bytearray, shape: tuple[int, int]) -> None:
    write_varint(output, shape[0])
    write_varint(output, shape[1])


def read_shape(data: bytes, position: int) -> tuple[tuple[int, int], int]:
    height, position = read_varint(data, position)
    width, position = read_varint(data, position)
    return (height, width), position


def check_image(image: np.ndarray) -> np.ndarray:
    """Coders only accept 2D integer arrays; fail early with a clear message."""
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"Expected a 2D integer array, got {image.dtype} {image.shape}")
    return image


def bits_to_bytes(bits: list[int]) -> bytes:
    """Pack MSB-first bits, padding the last byte with zeros."""
    output = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        if bit:
            output[index >> 3] |= 0x80 >> (index & 7)
    return bytes(output)


class BitReader:
    """MSB-first reader. Reading past the end yields zeros, as the coders expect."""

    __slots__ = ("data", "position", "size")

    def __init__(self, data: bytes, start: int = 0):
        self.data = data
        self.position = start * 8
        self.size = len(data) * 8

    def read(self) -> int:
        position = self.position
        self.position = position + 1
        if position >= self.size:
            return 0
        return (self.data[position >> 3] >> (7 - (position & 7))) & 1
