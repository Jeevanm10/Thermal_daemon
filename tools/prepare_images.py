#!/usr/bin/env python3
"""Convert imgs/ pictures into filtered MobileNetV2 float32 NCHW tensors."""

import argparse
from array import array
import json
from pathlib import Path
import sys

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
except ImportError as exc:
    raise SystemExit(
        "Pillow is required. Install it with: python -m pip install Pillow"
    ) from exc


SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def variants(image):
    yield "original", image
    yield "grayscale", ImageOps.grayscale(image).convert("RGB")
    yield "brightness_0_65", ImageEnhance.Brightness(image).enhance(0.65)
    yield "contrast_1_35", ImageEnhance.Contrast(image).enhance(1.35)
    yield "sharpen", image.filter(ImageFilter.SHARPEN)


def tensor_values(image):
    resized = ImageOps.fit(
        image, SIZE, method=Image.Resampling.BILINEAR, centering=(0.5, 0.5)
    )
    channels = [array("f"), array("f"), array("f")]
    for red, green, blue in resized.getdata():
        channels[0].append(red / 127.5 - 1.0)
        channels[1].append(green / 127.5 - 1.0)
        channels[2].append(blue / 127.5 - 1.0)
    values = array("f")
    for channel in channels:
        values.extend(channel)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "imgs",
        help="folder containing PNG/JPEG images (default: repository imgs/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build") / "thermal_inputs",
        help="directory for generated .tensor files",
    )
    args = parser.parse_args()

    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        parser.error("no supported images found in {}".format(args.images))

    args.output.mkdir(parents=True, exist_ok=True)
    filtered_images_dir = args.output / "filtered_images"
    filtered_images_dir.mkdir(exist_ok=True)
    manifest = []
    for image_index, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            for filter_name, filtered in variants(image):
                tensor_name = "{:03d}_{}_{}.tensor".format(
                    image_index, image_path.stem, filter_name
                )
                tensor_path = args.output / tensor_name
                filtered_image_name = tensor_path.stem + ".png"
                filtered_image_path = filtered_images_dir / filtered_image_name
                values = tensor_values(filtered)
                if len(values) != 3 * SIZE[0] * SIZE[1]:
                    raise RuntimeError("unexpected tensor size for {}".format(image_path))
                filtered.save(filtered_image_path, format="PNG")
                with tensor_path.open("wb") as output_file:
                    values.tofile(output_file)
                manifest.append(
                    {
                        "tensor": tensor_name,
                        "filtered_image": str(
                            Path("filtered_images") / filtered_image_name
                        ),
                        "image": str(image_path),
                        "filter": filter_name,
                    }
                )

    with (args.output / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)
    print("Created {} filtered PNGs and tensors in {}".format(len(manifest), args.output))
    print("Input: RGB resized to 224x224, NCHW float32, normalized to [-1, 1].")
    print("Verify this preprocessing matches the converted TensorFlow Lite model.")


if __name__ == "__main__":
    main()
