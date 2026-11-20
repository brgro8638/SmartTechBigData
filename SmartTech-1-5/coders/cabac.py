"""CABAC: Context-based Adaptive Binary Arithmetic Coding.

The three parts of the name map onto the three parts of this file:

1. Binarization: each integer symbol becomes a string of binary decisions (bins).
2. Context modelling: each bin picks an adaptive probability from already-coded
   neighbours (left and top), so the model needs no header.
3. Binary arithmetic coding: the M-coder from H.264/AVC (ITU-T H.264 clause 9.3),
   with its 64-state probability machine and rangeTabLPS multiplication-free table.

Deviation from H.264: there are no per-slice init tables, since those are tuned to
video. All contexts start at p = 0.5. The binarizations are chosen for image data.
"""

from __future__ import annotations

import numpy as np

from .common import (
    BitReader,
    Encoded,
    bits_to_bytes,
    check_image,
    read_shape,
    read_varint,
    unzigzag,
    write_shape,
    write_varint,
    zigzag,
)

# H.264 Table 9-44: LPS sub-range per probability state and quantized range.
RANGE_TAB_LPS = (
    (128, 176, 208, 240), (128, 167, 197, 227), (128, 158, 187, 216), (123, 150, 178, 205),
    (116, 142, 169, 195), (111, 135, 160, 185), (105, 128, 152, 175), (100, 122, 144, 166),
    (95, 116, 137, 158), (90, 110, 130, 150), (85, 104, 123, 142), (81, 99, 117, 135),
    (77, 94, 111, 128), (73, 89, 105, 122), (69, 85, 100, 116), (66, 80, 95, 110),
    (62, 76, 90, 104), (59, 72, 86, 99), (56, 69, 81, 94), (53, 65, 77, 89),
    (51, 62, 73, 85), (48, 59, 69, 80), (46, 56, 66, 76), (43, 53, 63, 72),
    (41, 50, 59, 69), (39, 48, 56, 65), (37, 45, 54, 62), (35, 43, 51, 59),
    (33, 41, 48, 56), (32, 39, 46, 53), (30, 37, 43, 50), (29, 35, 41, 48),
    (27, 33, 39, 45), (26, 31, 37, 43), (24, 30, 35, 41), (23, 28, 33, 39),
    (22, 27, 32, 37), (21, 26, 30, 35), (20, 24, 29, 33), (19, 23, 27, 31),
    (18, 22, 26, 30), (17, 21, 25, 28), (16, 20, 23, 27), (15, 19, 22, 25),
    (14, 18, 21, 24), (14, 17, 20, 23), (13, 16, 19, 22), (12, 15, 18, 21),
    (12, 14, 17, 20), (11, 14, 16, 19), (11, 13, 15, 18), (10, 12, 15, 17),
    (10, 12, 14, 16), (9, 11, 13, 15), (9, 11, 12, 14), (8, 10, 12, 14),
    (8, 9, 11, 13), (7, 9, 11, 12), (7, 9, 10, 12), (7, 8, 10, 11),
    (6, 8, 9, 11), (6, 7, 9, 10), (6, 7, 8, 9), (2, 2, 2, 2),
)
# H.264 Table 9-45: next state after coding the least / most probable symbol.
NEXT_STATE_LPS = (
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
)
NEXT_STATE_MPS = tuple(min(state + 1, 62) for state in range(63)) + (63,)

# Binarization of signed residuals (UEG0-style, as coeff_abs_level_minus1 in H.264).
MAGNITUDE_CLASSES = 8  # Neighbour activity classes, by bit length of |left| + |top|.
UNARY_CUTOFF = 14      # Bins 0..13: "is the magnitude larger than i?"
ESCAPE_CONTEXTS = 16   # Context-coded Exp-Golomb prefix for rare large magnitudes.
CLASS_SPAN = UNARY_CUTOFF + ESCAPE_CONTEXTS

# Binarization of raw values: a bit tree, split into classes by the neighbours' mean.
TREE_CLASSES = 8

MODE_RESIDUAL = 0
MODE_RAW = 1


class ArithmeticEncoder:
    """M-coder encoder: 9-bit range, 10-bit low, carry handled by outstanding bits."""

    __slots__ = ("state", "mps", "ones", "zeros", "low", "range", "outstanding", "first", "bits", "bypass_bins")

    def __init__(self, contexts: int):
        self.state = [0] * contexts
        self.mps = [0] * contexts
        self.zeros = [0] * contexts
        self.ones = [0] * contexts
        self.low = 0
        self.range = 510
        self.outstanding = 0
        self.first = True
        self.bits: list[int] = []
        self.bypass_bins = 0

    def _put(self, bit: int) -> None:
        if self.first:
            # The first bit out of the M-coder is always redundant.
            self.first = False
        else:
            self.bits.append(bit)
        if self.outstanding:
            self.bits.extend([1 - bit] * self.outstanding)
            self.outstanding = 0

    def _renormalize(self) -> None:
        range_ = self.range
        low = self.low
        while range_ < 256:
            if low < 256:
                self._put(0)
            elif low >= 512:
                low -= 512
                self._put(1)
            else:
                low -= 256
                self.outstanding += 1
            range_ <<= 1
            low <<= 1
        self.range = range_
        self.low = low

    def encode_bin(self, context: int, bin_value: int) -> None:
        """Context-coded bin: split the range by the state's LPS share, then adapt."""
        state = self.state[context]
        lps_range = RANGE_TAB_LPS[state][(self.range >> 6) & 3]
        self.range -= lps_range
        if bin_value == self.mps[context]:
            self.state[context] = NEXT_STATE_MPS[state]
        else:
            self.low += self.range
            self.range = lps_range
            if state == 0:
                self.mps[context] = 1 - self.mps[context]
            self.state[context] = NEXT_STATE_LPS[state]
        if bin_value:
            self.ones[context] += 1
        else:
            self.zeros[context] += 1
        if self.range < 256:
            self._renormalize()

    def encode_bypass(self, bin_value: int) -> None:
        """Equiprobable bin: exactly one bit, no model. Used for noise-like bins."""
        self.bypass_bins += 1
        low = self.low << 1
        if bin_value:
            low += self.range
        if low >= 1024:
            self._put(1)
            low -= 1024
        elif low < 512:
            self._put(0)
        else:
            low -= 512
            self.outstanding += 1
        self.low = low

    def finish(self) -> bytes:
        """Terminate as H.264 does: flush low so the decoder can resolve the last bins."""
        self.range -= 2
        self.low += self.range
        self.range = 2
        self._renormalize()
        self._put((self.low >> 9) & 1)
        self.bits.append((self.low >> 8) & 1)
        self.bits.append(1)
        return bits_to_bytes(self.bits)


class ArithmeticDecoder:
    """Mirror of ArithmeticEncoder. Same state machine, so no probabilities are sent."""

    __slots__ = ("state", "mps", "range", "offset", "read")

    def __init__(self, contexts: int, stream: bytes, start: int):
        self.state = [0] * contexts
        self.mps = [0] * contexts
        self.read = BitReader(stream, start).read
        self.range = 510
        offset = 0
        for _ in range(9):
            offset = (offset << 1) | self.read()
        self.offset = offset

    def decode_bin(self, context: int) -> int:
        state = self.state[context]
        lps_range = RANGE_TAB_LPS[state][(self.range >> 6) & 3]
        range_ = self.range - lps_range
        if self.offset >= range_:
            bin_value = 1 - self.mps[context]
            self.offset -= range_
            range_ = lps_range
            if state == 0:
                self.mps[context] = bin_value
            self.state[context] = NEXT_STATE_LPS[state]
        else:
            bin_value = self.mps[context]
            self.state[context] = NEXT_STATE_MPS[state]
        if range_ < 256:
            offset = self.offset
            read = self.read
            while range_ < 256:
                range_ <<= 1
                offset = (offset << 1) | read()
            self.offset = offset
        self.range = range_
        return bin_value

    def decode_bypass(self) -> int:
        offset = (self.offset << 1) | self.read()
        if offset >= self.range:
            self.offset = offset - self.range
            return 1
        self.offset = offset
        return 0


def _magnitude_class(left: int, top: int) -> int:
    """Busy neighbours predict a large residual here (idea from H.264's mvd contexts)."""
    return min((abs(left) + abs(top)).bit_length(), MAGNITUDE_CLASSES - 1)


def _tree_class(left: int, top: int, depth: int) -> int:
    """Top 3 bits of the neighbours' mean; raw pixels resemble their neighbours."""
    shift = depth + 1 - 3
    total = left + top
    value = total >> shift if shift >= 0 else total << -shift
    return min(value, TREE_CLASSES - 1)


def encode_residual(encoder, value: int, activity: int, labeller=None) -> None:
    """Significance + truncated unary + Exp-Golomb escape + sign.

    Small magnitudes spend only context-coded bins; the mantissa and sign are close
    to random, so they bypass the model and cost exactly one bit each.
    """
    magnitude = abs(value)
    base = activity * CLASS_SPAN
    encode_bin = encoder.encode_bin
    for index in range(min(magnitude, UNARY_CUTOFF)):
        if labeller:
            labeller("significance" if index == 0 else "unary")
        encode_bin(base + index, 1)
    if magnitude < UNARY_CUTOFF:
        if labeller:
            labeller("significance" if magnitude == 0 else "unary")
        encode_bin(base + magnitude, 0)
    else:
        # Exp-Golomb order 0 of the remainder.
        remainder = magnitude - UNARY_CUTOFF + 1
        group = remainder.bit_length() - 1
        escape = base + UNARY_CUTOFF
        for index in range(group):
            if labeller:
                labeller("escape prefix")
            encode_bin(escape + min(index, ESCAPE_CONTEXTS - 1), 1)
        if labeller:
            labeller("escape prefix")
        encode_bin(escape + min(group, ESCAPE_CONTEXTS - 1), 0)
        for shift in range(group - 1, -1, -1):
            if labeller:
                labeller("mantissa (bypass)")
            encoder.encode_bypass((remainder >> shift) & 1)
    if magnitude:
        if labeller:
            labeller("sign (bypass)")
        encoder.encode_bypass(1 if value < 0 else 0)


def decode_residual(decoder, activity: int) -> int:
    base = activity * CLASS_SPAN
    decode_bin = decoder.decode_bin
    magnitude = 0
    while magnitude < UNARY_CUTOFF and decode_bin(base + magnitude):
        magnitude += 1
    if magnitude == UNARY_CUTOFF:
        escape = base + UNARY_CUTOFF
        group = 0
        while decode_bin(escape + min(group, ESCAPE_CONTEXTS - 1)):
            group += 1
        remainder = 1
        for _ in range(group):
            remainder = (remainder << 1) | decoder.decode_bypass()
        magnitude = remainder - 1 + UNARY_CUTOFF
    if magnitude and decoder.decode_bypass():
        return -magnitude
    return magnitude


def encode_raw(encoder, value: int, depth: int, neighbour_class: int, labeller=None) -> None:
    """MSB-first bit tree. The context is the path so far, split by neighbour class."""
    base = neighbour_class << depth
    node = 1
    for shift in range(depth - 1, -1, -1):
        bit = (value >> shift) & 1
        if labeller:
            labeller("tree bit")
        encoder.encode_bin(base + node, bit)
        node = (node << 1) | bit


def decode_raw(decoder, depth: int, neighbour_class: int) -> int:
    base = neighbour_class << depth
    node = 1
    for _ in range(depth):
        node = (node << 1) | decoder.decode_bin(base + node)
    return node - (1 << depth)


def _context_count(mode: int, depth: int) -> int:
    if mode == MODE_RESIDUAL:
        return MAGNITUDE_CLASSES * CLASS_SPAN
    return TREE_CLASSES << depth


def _encode_pixels(encoder, rows: list[list[int]], mode: int, depth: int, offset: int,
                   limit: int | None = None, labeller=None, on_symbol=None) -> None:
    """Scan in raster order. `limit` and the callbacks exist for the GUI trace."""
    width = len(rows[0])
    previous_row = [0] * width
    coded = 0
    for row in rows:
        left = 0
        for column in range(width):
            value = row[column] - offset
            top = previous_row[column]
            if on_symbol:
                on_symbol(coded, row[column])
            if mode == MODE_RESIDUAL:
                encode_residual(encoder, value, _magnitude_class(left, top), labeller)
            else:
                encode_raw(encoder, value, depth, _tree_class(left, top, depth), labeller)
            left = value
            coded += 1
            if limit is not None and coded >= limit:
                return
        previous_row = [value - offset for value in row]


def _analyse(image: np.ndarray) -> tuple[int, int, int]:
    """Negative values mean prediction residuals; otherwise treat values as raw."""
    minimum = int(image.min())
    if minimum < 0:
        return MODE_RESIDUAL, 0, 0
    depth = max(1, int(image.max() - minimum).bit_length())
    # Raw values are shifted to start at zero; the tree then needs fewer levels.
    return MODE_RAW, depth, minimum


def encode(image: np.ndarray, mode: int | None = None) -> Encoded:
    """Encode an image. `mode` says what the values are; without it, signed means residual."""
    image = check_image(image)
    header = bytearray()
    write_shape(header, image.shape)
    if image.size == 0:
        return Encoded(bytes(header), len(header), {})

    detected_mode, depth, offset = _analyse(image)
    mode = detected_mode if mode is None else mode
    if mode == MODE_RESIDUAL:
        depth, offset = 0, 0
    header.append(mode)
    write_varint(header, depth)
    write_varint(header, zigzag(offset))

    encoder = ArithmeticEncoder(_context_count(mode, depth))
    _encode_pixels(encoder, image.tolist(), mode, depth, offset)
    payload = encoder.finish()

    context_bins = sum(encoder.zeros) + sum(encoder.ones)
    used = sum(1 for zeros, ones in zip(encoder.zeros, encoder.ones) if zeros + ones)
    stats = {
        "mode": "residual (UEG0)" if mode == MODE_RESIDUAL else f"raw (bit tree, depth {depth})",
        "context_bins": context_bins,
        "bypass_bins": encoder.bypass_bins,
        "contexts_used": used,
        "contexts_available": len(encoder.state),
        "bins_per_symbol": (context_bins + encoder.bypass_bins) / image.size,
        "context_table": _context_table(encoder, mode, depth),
    }
    return Encoded(bytes(header) + payload, len(header), stats)


def decode(stream: bytes) -> np.ndarray:
    shape, position = read_shape(stream, 0)
    height, width = shape
    if height * width == 0:
        return np.zeros(shape, dtype=np.int64)

    mode = stream[position]
    position += 1
    depth, position = read_varint(stream, position)
    encoded_offset, position = read_varint(stream, position)
    offset = unzigzag(encoded_offset)

    decoder = ArithmeticDecoder(_context_count(mode, depth), stream, position)
    output = np.zeros(shape, dtype=np.int64)
    previous_row = [0] * width
    for row_index in range(height):
        row = [0] * width
        left = 0
        for column in range(width):
            top = previous_row[column]
            if mode == MODE_RESIDUAL:
                value = decode_residual(decoder, _magnitude_class(left, top))
            else:
                value = decode_raw(decoder, depth, _tree_class(left, top, depth))
            row[column] = value
            left = value
        previous_row = row
        output[row_index] = row
    return output + offset


def _describe_context(index: int, mode: int, depth: int) -> str:
    if mode == MODE_RESIDUAL:
        activity, position = divmod(index, CLASS_SPAN)
        if position == 0:
            kind = "significance"
        elif position < UNARY_CUTOFF:
            kind = f"unary {position}"
        else:
            kind = f"escape {position - UNARY_CUTOFF}"
        return f"activity {activity} / {kind}"
    neighbour_class, node = index >> depth, index & ((1 << depth) - 1)
    return f"neighbours {neighbour_class} / node {node}"


# State 63 is reserved for termination, so adaptive contexts stop at state 62.
LOWEST_ADAPTIVE_STATE = 62


def lps_probability(state: int) -> float:
    """H.264 defines p(state) = 0.5 * alpha^state, with alpha = (0.01875 / 0.5)^(1/63)."""
    alpha = (0.01875 / 0.5) ** (1 / 63)
    return 0.5 * alpha ** state


def _context_table(encoder: ArithmeticEncoder, mode: int, depth: int) -> list[dict]:
    rows = []
    for index, (zeros, ones) in enumerate(zip(encoder.zeros, encoder.ones)):
        if zeros + ones:
            rows.append(
                {
                    "context": _describe_context(index, mode, depth),
                    "bins": zeros + ones,
                    "share of 1": ones / (zeros + ones),
                    "final state": encoder.state[index],
                    "final MPS": encoder.mps[index],
                    "final p(LPS)": lps_probability(encoder.state[index]),
                }
            )
    rows.sort(key=lambda row: -row["bins"])
    return rows


class _TracingEncoder(ArithmeticEncoder):
    """Logs each bin. Kept apart so that the real encoder pays nothing for this."""

    __slots__ = ("rows", "symbol_index", "symbol_value", "label", "mode_", "depth_", "cost")

    def __init__(self, contexts: int, mode: int, depth: int):
        super().__init__(contexts)
        self.rows: list[dict] = []
        self.symbol_index = 0
        self.symbol_value = 0
        self.label = ""
        self.mode_ = mode
        self.depth_ = depth
        self.cost = 0.0

    def encode_bin(self, context: int, bin_value: int) -> None:
        state = self.state[context]
        mps = self.mps[context]
        range_before = self.range
        lps_range = RANGE_TAB_LPS[state][(range_before >> 6) & 3]
        p_lps = lps_range / range_before
        probability = 1 - p_lps if bin_value == mps else p_lps
        self.cost -= float(np.log2(probability))
        self.rows.append(
            {
                "symbol #": self.symbol_index,
                "symbol": self.symbol_value,
                "bin type": self.label,
                "bin": bin_value,
                "context": _describe_context(context, self.mode_, self.depth_),
                "context id": context,
                "state": state,
                "MPS": mps,
                "p(LPS)": p_lps,
                "range": range_before,
                "cost (bits)": -float(np.log2(probability)),
                "total bits": self.cost,
            }
        )
        super().encode_bin(context, bin_value)

    def encode_bypass(self, bin_value: int) -> None:
        self.cost += 1.0
        self.rows.append(
            {
                "symbol #": self.symbol_index,
                "symbol": self.symbol_value,
                "bin type": self.label,
                "bin": bin_value,
                "context": "bypass",
                "context id": -1,
                "state": None,
                "MPS": None,
                "p(LPS)": 0.5,
                "range": self.range,
                "cost (bits)": 1.0,
                "total bits": self.cost,
            }
        )
        super().encode_bypass(bin_value)


def trace(image: np.ndarray, limit: int, mode: int | None = None) -> list[dict]:
    """Encode the first `limit` symbols, logging every bin and its real cost."""
    if image.size == 0:
        return []
    detected_mode, depth, offset = _analyse(image)
    mode = detected_mode if mode is None else mode
    if mode == MODE_RESIDUAL:
        depth, offset = 0, 0
    encoder = _TracingEncoder(_context_count(mode, depth), mode, depth)

    def labeller(text: str) -> None:
        encoder.label = text

    def on_symbol(index: int, value: int) -> None:
        encoder.symbol_index = index
        encoder.symbol_value = value

    _encode_pixels(encoder, image.tolist(), mode, depth, offset, limit, labeller, on_symbol)
    return encoder.rows
