"""Coders must be lossless on every input class, including the awkward edge cases."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark  # noqa: E402
import prediction  # noqa: E402
import sources  # noqa: E402
from coders import CODERS, arithmetic, cabac, huffman  # noqa: E402

ALL_CODERS = list(CODERS)
ALL_PREDICTORS = list(prediction.PREDICTORS.values())
RNG = np.random.default_rng(7)

EDGE_CASES = {
    "single_pixel": np.array([[5]]),
    "single_zero": np.array([[0]]),
    "constant": np.full((20, 20), 7),
    "constant_zero": np.zeros((16, 16), dtype=np.int64),
    "binary": (RNG.random((30, 30)) < 0.1).astype(np.int64),
    "two_values_far_apart": np.array([[0, 4095], [4095, 0]]),
    "one_row": RNG.integers(0, 256, (1, 300)),
    "one_column": RNG.integers(0, 256, (300, 1)),
    "uniform_noise": RNG.integers(0, 256, (40, 40)),
    "twelve_bit": RNG.integers(0, 4096, (32, 32)),
    "sixteen_bit": RNG.integers(0, 65536, (24, 24)),
    "offset_values": RNG.integers(1000, 1016, (32, 32)),
    "signed": RNG.integers(-300, 300, (32, 32)),
    "large_negative": np.array([[-70000, 70000], [3, -1]]),
    "all_negative": np.full((8, 8), -3),
}


@pytest.mark.parametrize("coder", ALL_CODERS)
@pytest.mark.parametrize("case", EDGE_CASES)
def test_coder_round_trip(coder, case):
    image = EDGE_CASES[case]
    decoded = CODERS[coder].decode(CODERS[coder].encode(image).stream)
    assert decoded.shape == image.shape
    assert np.array_equal(decoded, image)


@pytest.mark.parametrize("coder", ALL_CODERS)
def test_empty_image(coder):
    image = np.zeros((0, 0), dtype=np.int64)
    decoded = CODERS[coder].decode(CODERS[coder].encode(image).stream)
    assert decoded.shape == (0, 0)


@pytest.mark.parametrize("predictor", ALL_PREDICTORS)
@pytest.mark.parametrize("case", ["single_pixel", "binary", "one_row", "one_column", "twelve_bit", "sixteen_bit"])
def test_prediction_is_reversible(predictor, case):
    image = EDGE_CASES[case]
    assert np.array_equal(prediction.inverse(prediction.forward(image, predictor), predictor), image)


@pytest.mark.parametrize("name", sources.GENERATORS)
@pytest.mark.parametrize("predictor", ALL_PREDICTORS)
def test_pipeline_on_generated_data(name, predictor):
    source = sources.generated(name, 48, seed=3)
    for coder in ALL_CODERS:
        assert benchmark.run_coder(coder, source.pixels, predictor, source.nominal_bits).lossless


def test_huffman_bound_and_prefix_property():
    image = RNG.integers(0, 40, (64, 64))
    table = huffman.code_table(image)
    codes = sorted(table.values())
    assert not any(b.startswith(a) for a, b in zip(codes, codes[1:]))
    entropy = benchmark.entropy_bits(image)
    mean = huffman.encode(image).stats["mean_code_length"]
    assert entropy <= mean < entropy + 1


def test_arithmetic_beats_huffman_on_skewed_source():
    """Huffman cannot go below 1 bit per symbol; arithmetic coding can."""
    image = (RNG.random((128, 128)) < 0.02).astype(np.int64)
    huffman_payload = huffman.encode(image).payload_bytes
    arithmetic_payload = arithmetic.encode(image).payload_bytes
    assert huffman_payload == math.ceil(image.size / 8)
    ideal_bytes = benchmark.entropy_bits(image) * image.size / 8
    assert arithmetic_payload < 1.02 * ideal_bytes + 8


def test_arithmetic_payload_tracks_entropy():
    image = np.clip(RNG.normal(128, 12, (128, 128)).round(), 0, 255).astype(np.int64)
    ideal_bytes = benchmark.entropy_bits(image) * image.size / 8
    assert arithmetic.encode(image).payload_bytes < ideal_bytes + 8


def test_cabac_exploits_context_that_order_zero_coders_cannot():
    """Half the image is ones and half zeros: order-0 entropy is 1 bit, context is cheap."""
    image = np.zeros((128, 128), dtype=np.int64)
    image[:, 64:] = 1
    order_zero_bytes = benchmark.entropy_bits(image) * image.size / 8
    assert cabac.encode(image).payload_bytes < 0.2 * order_zero_bytes


def test_cabac_tables_match_h264_definition():
    """Spot-check the transcribed tables against the formula they were derived from."""
    assert len(cabac.RANGE_TAB_LPS) == len(cabac.NEXT_STATE_LPS) == len(cabac.NEXT_STATE_MPS) == 64
    for state in range(63):
        # Each table row uses the midpoint of its quantized range: 9-bit values 256-511.
        for quarter, midpoint in enumerate((288, 352, 416, 480)):
            ideal = midpoint * cabac.lps_probability(state)
            assert abs(cabac.RANGE_TAB_LPS[state][quarter] - ideal) <= max(4, 0.12 * ideal) or state == 0


def test_cabac_trace_matches_real_encoder_cost():
    """The GUI's per-bin log must describe the same stream the encoder writes."""
    image = np.clip(RNG.normal(0, 3, (64, 64)).round(), -40, 40).astype(np.int64)
    rows = cabac.trace(image, image.size)
    payload_bits = cabac.encode(image).payload_bytes * 8
    assert rows[-1]["total bits"] == pytest.approx(payload_bits, rel=0.02)


def test_arithmetic_trace_matches_real_encoder():
    image = RNG.integers(0, 8, (32, 32))
    rows = arithmetic.trace(image, image.size)
    assert rows[-1]["bits sent"] == pytest.approx(arithmetic.encode(image).stats["bits_emitted"], abs=40)


@pytest.mark.parametrize("mode", [cabac.MODE_RAW, cabac.MODE_RESIDUAL])
def test_cabac_forced_mode_on_non_negative_values(mode):
    """A smooth image's residuals can all be >= 0; they must still use residual bins."""
    image = RNG.integers(0, 9, (24, 24))
    assert np.array_equal(cabac.decode(cabac.encode(image, mode).stream), image)
    rows = cabac.trace(image, 20, mode)
    assert rows and ("tree bit" == rows[0]["bin type"]) == (mode == cabac.MODE_RAW)


def test_raw_source_parsing():
    data = bytes(range(10))
    source = sources.raw_source(data, "x", 8, 3)
    assert source.pixels.shape == (3, 3)
    assert "1 trailing" in source.note
    assert sources.raw_source(data, "x", 16, 0).pixels.shape == (1, 5)
    big = sources.raw_source(b"\x01\x02", "x", 16, 0, big_endian=True)
    assert big.pixels[0, 0] == 0x0102


def test_harnverhalt_when_available():
    path = sources.find_harnverhalt_zip()
    if path is None:
        pytest.skip("Harnverhalt2.zip is not in the checkout")
    archive = path.read_bytes()
    member = sources.list_dicom_members(archive)[100]
    slice_bytes = sources.read_dicom_member(archive, member)
    for representation, bits in (("stored", 12), ("windowed", 8)):
        source = sources.dicom_source(slice_bytes, member, representation)
        assert source.nominal_bits == bits
        crop = sources.center_crop(source.pixels, 64)
        for coder in ALL_CODERS:
            assert benchmark.run_coder(coder, crop, "med", bits).lossless
