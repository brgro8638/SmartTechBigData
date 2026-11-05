"""Load the Harnverhalt DICOM series for the GUI and image-processing tasks."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pydicom


def _safe_member_path(output_dir: Path, member_name: str) -> Path:
	"""Return a ZIP member path that cannot escape output_dir."""
	destination = (output_dir / member_name).resolve()
	if output_dir.resolve() not in destination.parents:
		raise ValueError(f"Unsafe ZIP member: {member_name}")
	return destination


def extract_dicom_files(zip_path: Path, output_dir: Path) -> list[Path]:
	"""Extract DICOM files from a dataset ZIP and return them in filename order."""
	output_dir.mkdir(parents=True, exist_ok=True)
	with zipfile.ZipFile(zip_path) as archive:
		members = [
			member
			for member in archive.infolist()
			if not member.is_dir() and member.filename.lower().endswith(".dcm")
		]
		for member in members:
			destination = _safe_member_path(output_dir, member.filename)
			destination.parent.mkdir(parents=True, exist_ok=True)
			with archive.open(member) as source, destination.open("wb") as target:
				target.write(source.read())

	return sorted(
		(path for path in output_dir.rglob("*.dcm")),
		key=lambda path: path.name,
	)


def load_dicom_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
	"""Read a DICOM image and return its raw and OpenCV-normalized pixels."""
	dataset = pydicom.dcmread(path)
	raw_pixels = dataset.pixel_array.astype(np.float32)

	if raw_pixels.ndim != 2:
		raise ValueError(f"Expected a 2D image in {path.name}, got {raw_pixels.shape}")

	normalized = cv2.normalize(
		raw_pixels,
		None,
		alpha=0,
		beta=255,
		norm_type=cv2.NORM_MINMAX,
		dtype=cv2.CV_8U,
	)
	return raw_pixels, normalized


def enhance_image(normalized: np.ndarray) -> np.ndarray:
	"""Apply a small morphological cleanup that is visible in the GUI preview."""
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	opened = cv2.morphologyEx(normalized, cv2.MORPH_OPEN, kernel)
	return cv2.equalizeHist(opened)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("zip_path", type=Path)
	parser.add_argument("--output", type=Path, default=Path(".harnverhalt-cache"))
	parser.add_argument("--index", type=int, default=0)
	args = parser.parse_args()

	with tempfile.TemporaryDirectory(prefix="harnverhalt-") as extracted:
		files = extract_dicom_files(args.zip_path, Path(extracted))
		if not files:
			raise ValueError("The ZIP archive does not contain any DICOM files")
		if not 0 <= args.index < len(files):
			raise IndexError(f"Image index must be between 0 and {len(files) - 1}")

		_, normalized = load_dicom_image(files[args.index])
		enhanced = enhance_image(normalized)
		args.output.parent.mkdir(parents=True, exist_ok=True)
		cv2.imwrite(str(args.output), enhanced)
		print(f"{files[args.index].name}\t{normalized.shape[1]}x{normalized.shape[0]}\t{args.output}")


if __name__ == "__main__":
	main()
