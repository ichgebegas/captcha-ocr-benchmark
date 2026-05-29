from __future__ import annotations

import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from easyocr_background_suppress_eval import read_easyocr_with_config
from easyocr_lowpass_eval import GROUND_TRUTH, edit_distance


ROOT = Path(r"C:\xxx\captcha")
SOURCE_DIR = ROOT / "captcha_pairs_60"
OUT_DIR = ROOT / "light_blur_easyocr_pairs60"


def _gray(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("L"))


def _to_rgb(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).convert("RGB")


def apply_gauss_k3(image: Image.Image) -> Image.Image:
    return _to_rgb(cv2.GaussianBlur(_gray(image), (3, 3), 0))


def apply_gauss_k5(image: Image.Image) -> Image.Image:
    return _to_rgb(cv2.GaussianBlur(_gray(image), (5, 5), 0))


def apply_box_k3(image: Image.Image) -> Image.Image:
    return _to_rgb(cv2.blur(_gray(image), (3, 3)))


def apply_box_k4(image: Image.Image) -> Image.Image:
    return _to_rgb(cv2.blur(_gray(image), (4, 4)))


def apply_box_k5(image: Image.Image) -> Image.Image:
    return _to_rgb(cv2.blur(_gray(image), (5, 5)))


def apply_box_k6(image: Image.Image) -> Image.Image:
    return _to_rgb(cv2.blur(_gray(image), (6, 6)))


def darken_large_text(gray: np.ndarray, strength: float = 0.28) -> np.ndarray:
    source = gray.astype(np.float32)
    text_mask = cv2.GaussianBlur(source, (0, 0), 1.6) < 125
    text_mask = cv2.morphologyEx(text_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)) > 0
    result = source.copy()
    result[text_mask] = source[text_mask] * (1.0 - strength)
    return np.clip(result, 0, 255).astype(np.uint8)


def soft_local_contrast(gray: np.ndarray) -> np.ndarray:
    source = gray.astype(np.float32)
    smooth = cv2.GaussianBlur(source, (0, 0), 2.0)
    result = source * 1.18 - smooth * 0.18
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_box_k5_dark_text(image: Image.Image) -> Image.Image:
    blurred = cv2.blur(_gray(image), (5, 5))
    return _to_rgb(darken_large_text(blurred))


def apply_box_k5_soft_contrast(image: Image.Image) -> Image.Image:
    blurred = cv2.blur(_gray(image), (5, 5))
    return _to_rgb(soft_local_contrast(blurred))


def preserve_large_dark_regions(gray: np.ndarray, blurred: np.ndarray) -> np.ndarray:
    source = gray.astype(np.float32)
    base = blurred.astype(np.float32)

    large_dark = cv2.GaussianBlur(source, (0, 0), 2.2) < 135
    large_dark = cv2.morphologyEx(large_dark.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0

    result = base.copy()
    result[large_dark] = source[large_dark] * 0.78 + base[large_dark] * 0.22
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_bg_blend_gauss3(image: Image.Image) -> Image.Image:
    gray = _gray(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return _to_rgb(preserve_large_dark_regions(gray, blurred))


def apply_bg_blend_gauss5(image: Image.Image) -> Image.Image:
    gray = _gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return _to_rgb(preserve_large_dark_regions(gray, blurred))


def light_blur_variants() -> dict[str, Callable[[Image.Image], Image.Image]]:
    return {
        "gauss_k3": apply_gauss_k3,
        "gauss_k5": apply_gauss_k5,
        "box_k3": apply_box_k3,
        "box_k4": apply_box_k4,
        "box_k5": apply_box_k5,
        "box_k6": apply_box_k6,
        "box_k5_dark_text": apply_box_k5_dark_text,
        "box_k5_soft_contrast": apply_box_k5_soft_contrast,
        "bg_blend_gauss3": apply_bg_blend_gauss3,
        "bg_blend_gauss5": apply_bg_blend_gauss5,
    }


def easyocr_configs() -> list[dict[str, Any]]:
    allowlist = "0123456789abcdefghijklmnopqrstuvwxyz"
    return [
        {"name": "margin", "allowlist": allowlist, "decoder": "greedy", "add_margin": 0.25},
    ]


def source_paths(source_dir: Path) -> list[Path]:
    paths = []
    for captcha_id in GROUND_TRUTH:
        path = source_dir / f"captcha_{captcha_id}_original.jpg"
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def generate(source_dir: Path, out_dir: Path) -> None:
    for variant in light_blur_variants():
        (out_dir / variant).mkdir(parents=True, exist_ok=True)

    for path in source_paths(source_dir):
        captcha_id = path.stem.split("_")[1]
        image = Image.open(path).convert("RGB")
        for variant, processor in light_blur_variants().items():
            processor(image).save(out_dir / variant / f"captcha_{captcha_id}_{variant}.png")


def variant_paths(out_dir: Path) -> dict[str, Callable[[str], Path]]:
    paths: dict[str, Callable[[str], Path]] = {
        "original": lambda cid: SOURCE_DIR / f"captcha_{cid}_original.jpg",
    }
    for variant in light_blur_variants():
        paths[variant] = lambda cid, variant=variant: out_dir / variant / f"captcha_{cid}_{variant}.png"
    return paths


def evaluate(reader: Any, out_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for captcha_id, expected in GROUND_TRUTH.items():
        for variant, path_for in variant_paths(out_dir).items():
            image_path = path_for(captcha_id)
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            for config in easyocr_configs():
                raw, normalized, confidence = read_easyocr_with_config(reader, image_path, config)
                distance = edit_distance(expected, normalized)
                denominator = max(len(expected), len(normalized), 1)
                rows.append(
                    {
                        "captcha_id": captcha_id,
                        "variant": variant,
                        "config": config["name"],
                        "expected": expected,
                        "raw_text": raw,
                        "normalized_text": normalized,
                        "confidence": f"{confidence:.6f}",
                        "exact_match": str(normalized == expected).lower(),
                        "edit_distance": str(distance),
                        "char_accuracy": f"{max(0.0, 1.0 - distance / denominator):.6f}",
                        "image_path": str(image_path),
                    }
                )
    return rows


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["config"])].append(row)

    summary: list[dict[str, str]] = []
    for (variant, config), variant_rows in grouped.items():
        exact = sum(row["exact_match"] == "true" for row in variant_rows)
        distances = [int(row["edit_distance"]) for row in variant_rows]
        accuracies = [float(row["char_accuracy"]) for row in variant_rows]
        confidences = [float(row["confidence"]) for row in variant_rows]
        length_5 = sum(len(row["normalized_text"]) == 5 for row in variant_rows)
        empty = sum(not row["normalized_text"] for row in variant_rows)
        summary.append(
            {
                "variant": variant,
                "config": config,
                "rows": str(len(variant_rows)),
                "exact_matches": str(exact),
                "exact_match_rate": f"{exact / len(variant_rows):.6f}",
                "avg_edit_distance": f"{mean(distances):.6f}",
                "avg_char_accuracy": f"{mean(accuracies):.6f}",
                "avg_confidence": f"{mean(confidences):.6f}",
                "median_confidence": f"{median(confidences):.6f}",
                "length_5_count": str(length_5),
                "empty_count": str(empty),
            }
        )
    summary.sort(
        key=lambda row: (
            int(row["exact_matches"]),
            float(row["avg_char_accuracy"]),
            int(row["length_5_count"]),
            -float(row["avg_edit_distance"]),
        ),
        reverse=True,
    )
    return summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_contact_sheet(rows: list[dict[str, str]], out_dir: Path, path: Path) -> None:
    by_key = {(row["captcha_id"], row["variant"], row["config"]): row for row in rows}
    columns = [("original", lambda cid: SOURCE_DIR / f"captcha_{cid}_original.jpg")]
    for variant in light_blur_variants():
        columns.append((variant, lambda cid, variant=variant: out_dir / variant / f"captcha_{cid}_{variant}.png"))

    thumb_w, thumb_h = 190, 84
    pad, label_h, row_h = 8, 34, 134
    sheet = Image.new("RGB", (len(columns) * thumb_w + (len(columns) + 1) * pad, 32 * row_h + 33 * pad), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        small = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
        small = font

    for index in range(1, 33):
        captcha_id = f"{index:03d}"
        y = pad + (index - 1) * (row_h + pad)
        draw.text((pad, y), f"captcha {captcha_id} expected={GROUND_TRUTH[captcha_id]}", fill=(0, 0, 0), font=font)
        for col, (variant, path_for) in enumerate(columns):
            x = pad + col * (thumb_w + pad)
            image_path = path_for(captcha_id)
            footer = ""
            if variant != "original":
                row = by_key[(captcha_id, variant, "margin")]
                footer = f"{row['normalized_text'] or '-'} c={float(row['confidence']):.2f} d={row['edit_distance']}"
            image = ImageOps.contain(Image.open(image_path).convert("RGB"), (thumb_w, thumb_h), Image.Resampling.LANCZOS)
            image_y = y + label_h + 14
            image_x = x + (thumb_w - image.width) // 2
            draw.text((x, y + label_h), variant, fill=(60, 60, 60), font=small)
            if footer:
                draw.text((x, y + label_h + 12), footer, fill=(150, 0, 0), font=small)
            sheet.paste(image, (image_x, image_y))
            draw.rectangle((x, image_y, x + thumb_w - 1, image_y + thumb_h - 1), outline=(220, 220, 220))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate light background blur variants with EasyOCR.")
    parser.add_argument("--source-dir", default=str(SOURCE_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    generate(source_dir, out_dir)

    warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true.*")
    import easyocr

    reader = easyocr.Reader(["en"], gpu=args.gpu, verbose=False)
    rows = evaluate(reader, out_dir)
    summary = summarize_rows(rows)
    write_csv(out_dir / "light_blur_easyocr_report.csv", rows)
    write_csv(out_dir / "light_blur_easyocr_summary.csv", summary)
    build_contact_sheet(rows, out_dir, out_dir / "light_blur_easyocr_contact_sheet.png")
    print(f"report={out_dir / 'light_blur_easyocr_report.csv'}")
    print(f"summary={out_dir / 'light_blur_easyocr_summary.csv'}")
    print(f"sheet={out_dir / 'light_blur_easyocr_contact_sheet.png'}")
    for row in summary:
        print(
            "{variant}/{config}: exact={exact_matches}/{rows} avg_dist={avg_edit_distance} "
            "char_acc={avg_char_accuracy} len5={length_5_count} empty={empty_count}".format(**row)
        )


if __name__ == "__main__":
    main()
