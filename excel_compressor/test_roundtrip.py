"""Regression checks for every coder and the notebook's expected sizes."""

from pathlib import Path

from coders import CODERS


ROOT = Path(__file__).parent
DATA_CANDIDATES = (
    ROOT / "insurance.csv",
    ROOT.parent / "Tarjei/smarttechbigdata/SmartTech-1-2b-HuffmanGUI/insurance.csv",
    ROOT.parent / "smarttechbigdata/SmartTech-1-2b-HuffmanGUI/insurance.csv",
)
EXPECTED_SIZES = {
    "Simple RLE": 108624,
    "PackBits": 56028,
    "Huffman": 31973,
    "Arithmetic": 31690,
    "CABAC (tree)": 31614,
}


def find_default_dataset():
    """Find the example dataset whether it is beside this folder or in the repository."""
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in DATA_CANDIDATES)
    raise FileNotFoundError(f"Could not find insurance.csv. Searched:\n{searched}")


def main():
    """Run roundtrips on small examples and the supplied insurance dataset."""
    insurance = find_default_dataset().read_bytes()
    examples = [
        ("BAC", b"BAC"),
        ("ABRACADABRA", b"ABRACADABRA"),
        ("1000 A", b"A" * 1000),
        ("insurance.csv", insurance),
    ]
    for name, (encode, decode) in CODERS.items():
        for example_name, data in examples:
            compressed = encode(data)
            decoded = decode(compressed)
            assert decoded == data, f"{name} failed roundtrip for {example_name}"
            if example_name == "insurance.csv":
                expected = EXPECTED_SIZES[name]
                assert len(compressed) == expected, (
                    f"{name} size changed: expected {expected}, got {len(compressed)}"
                )
                print(f"{name}: {len(compressed)} bytes")


if __name__ == "__main__":
    main()
