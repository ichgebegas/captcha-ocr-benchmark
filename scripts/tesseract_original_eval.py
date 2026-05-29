from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from easyocr_lowpass_eval import GROUND_TRUTH, edit_distance


ROOT = Path(r"C:\xxx\captcha")
SOURCE_DIR = ROOT / "captcha_pairs_60"
OUT_DIR = ROOT / "tesseract_original_eval_pairs60"


@dataclass(frozen=True)
class TesseractConfig:
    name: str
    psm: int
    variables: tuple[tuple[str, str], ...] = ()
    preprocess: str = "original"
    lang: str = "eng"
    dpi: int | None = 300


def normalize_tesseract_text(value: str) -> str:
    return re.sub(r"[^0-9a-z]", "", value.lower())


def build_tesseract_configs() -> list[TesseractConfig]:
    whitelist = ("tessedit_char_whitelist", "0123456789abcdefghijklmnopqrstuvwxyz")
    return [
        TesseractConfig("psm8_whitelist", 8, (whitelist,)),
        TesseractConfig("psm7_whitelist", 7, (whitelist,)),
        TesseractConfig("psm13_whitelist", 13, (whitelist,)),
        TesseractConfig("psm6_whitelist", 6, (whitelist,)),
        TesseractConfig("psm8_no_dict", 8, (whitelist, ("load_system_dawg", "0"), ("load_freq_dawg", "0"))),
        TesseractConfig("psm8_no_invert", 8, (whitelist, ("tessedit_do_invert", "0"))),
        TesseractConfig("psm8_script_latin", 8, (whitelist,), lang="script/Latin"),
        TesseractConfig("psm8_upscaled", 8, (whitelist,), preprocess="upscaled"),
        TesseractConfig("psm8_gray_otsu", 8, (whitelist,), preprocess="gray_otsu"),
        TesseractConfig("psm8_adaptive", 8, (whitelist,), preprocess="adaptive"),
        TesseractConfig("psm13_upscaled", 13, (whitelist,), preprocess="upscaled"),
        TesseractConfig("psm13_no_dict_upscaled", 13, (whitelist, ("load_system_dawg", "0"), ("load_freq_dawg", "0")), preprocess="upscaled"),
    ]


def preprocess_image(path: Path, mode: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if mode == "original":
        return image
    if mode == "upscaled":
        return image.resize((image.width * 4, image.height * 4), Image.Resampling.LANCZOS)

    gray = np.array(image.convert("L"))
    if mode == "gray_otsu":
        _, prepared = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "adaptive":
        prepared = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")
    prepared_image = Image.fromarray(prepared).convert("RGB")
    return prepared_image.resize((prepared_image.width * 4, prepared_image.height * 4), Image.Resampling.LANCZOS)


def run_tesseract(image_path: Path, config: TesseractConfig) -> str:
    if shutil.which("tesseract") is None:
        return ""
    command = ["tesseract", str(image_path), "stdout", "-l", config.lang, "--psm", str(config.psm)]
    if config.dpi:
        command.extend(["--dpi", str(config.dpi)])
    for key, value in config.variables:
        command.extend(["-c", f"{key}={value}"])
    completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    return completed.stdout.strip()


def source_paths(source_dir: Path) -> list[Path]:
    return [source_dir / f"captcha_{captcha_id}_original.jpg" for captcha_id in GROUND_TRUTH]


def evaluate(source_dir: Path, out_dir: Path) -> list[dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    configs = build_tesseract_configs()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        for captcha_id, expected in GROUND_TRUTH.items():
            original_path = source_dir / f"captcha_{captcha_id}_original.jpg"
            if not original_path.exists():
                raise FileNotFoundError(original_path)
            for config in configs:
                prepared = preprocess_image(original_path, config.preprocess)
                prepared_path = temp_root / f"{captcha_id}_{config.name}.png"
                prepared.save(prepared_path)
                raw = run_tesseract(prepared_path, config)
                normalized = normalize_tesseract_text(raw)
                distance = edit_distance(expected, normalized)
                denominator = max(len(expected), len(normalized), 1)
                rows.append(
                    {
                        "captcha_id": captcha_id,
                        "config": config.name,
                        "expected": expected,
                        "raw_text": raw,
                        "normalized_text": normalized,
                        "exact_match": str(normalized == expected).lower(),
                        "edit_distance": str(distance),
                        "char_accuracy": f"{max(0.0, 1.0 - distance / denominator):.6f}",
                        "preprocess": config.preprocess,
                        "psm": str(config.psm),
                        "lang": config.lang,
                        "image_path": str(original_path),
                    }
                )
    return rows


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)

    summary: list[dict[str, str]] = []
    ordered_configs = [config.name for config in build_tesseract_configs()]
    for config in grouped:
        if config not in ordered_configs:
            ordered_configs.append(config)

    for config in ordered_configs:
        config_rows = grouped.get(config, [])
        if not config_rows:
            continue
        exact = sum(row["exact_match"] == "true" for row in config_rows)
        distances = [int(row["edit_distance"]) for row in config_rows]
        accuracies = [float(row["char_accuracy"]) for row in config_rows]
        length_5 = sum(len(row["normalized_text"]) == 5 for row in config_rows)
        empty = sum(not row["normalized_text"] for row in config_rows)
        summary.append(
            {
                "config": config,
                "rows": str(len(config_rows)),
                "exact_matches": str(exact),
                "exact_match_rate": f"{exact / len(config_rows):.6f}",
                "avg_edit_distance": f"{mean(distances):.6f}",
                "avg_char_accuracy": f"{mean(accuracies):.6f}",
                "length_5_count": str(length_5),
                "empty_count": str(empty),
            }
        )
    summary.sort(key=lambda row: (int(row["exact_matches"]), float(row["avg_char_accuracy"]), -int(row["empty_count"])), reverse=True)
    return summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_contact_sheet(rows: list[dict[str, str]], path: Path, top_n: int = 4) -> None:
    summary = summarize_rows(rows)[:top_n]
    config_names = [row["config"] for row in summary]
    by_id_config = {(row["captcha_id"], row["config"]): row for row in rows}
    columns = ["original", *config_names]

    thumb_w, thumb_h = 230, 96
    pad, label_h, row_h = 10, 18, 144
    sheet = Image.new("RGB", (len(columns) * thumb_w + (len(columns) + 1) * pad, 32 * row_h + 33 * pad), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        small = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
        small = font

    config_map = {config.name: config for config in build_tesseract_configs()}
    for index in range(1, 33):
        captcha_id = f"{index:03d}"
        y = pad + (index - 1) * (row_h + pad)
        draw.text((pad, y), f"captcha {captcha_id} expected={GROUND_TRUTH[captcha_id]}", fill=(0, 0, 0), font=font)
        original_path = SOURCE_DIR / f"captcha_{captcha_id}_original.jpg"
        for col, label in enumerate(columns):
            x = pad + col * (thumb_w + pad)
            if label == "original":
                image = Image.open(original_path).convert("RGB")
                footer = ""
            else:
                image = preprocess_image(original_path, config_map[label].preprocess)
                row = by_id_config[(captcha_id, label)]
                footer = f"{row['normalized_text'] or '-'} d={row['edit_distance']}"
            image = ImageOps.contain(image, (thumb_w, thumb_h), Image.Resampling.LANCZOS)
            image_y = y + label_h + 14
            image_x = x + (thumb_w - image.width) // 2
            draw.text((x, y + label_h), label, fill=(60, 60, 60), font=small)
            sheet.paste(image, (image_x, image_y))
            draw.rectangle((x, image_y, x + thumb_w - 1, image_y + thumb_h - 1), outline=(220, 220, 220))
            if footer:
                draw.text((x, image_y + thumb_h + 3), footer, fill=(150, 0, 0), font=small)

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tesseract configurations on original images only.")
    parser.add_argument("--source-dir", default=str(SOURCE_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    rows = evaluate(source_dir, out_dir)
    summary = summarize_rows(rows)
    write_csv(out_dir / "tesseract_original_report.csv", rows)
    write_csv(out_dir / "tesseract_original_summary.csv", summary)
    build_contact_sheet(rows, out_dir / "tesseract_original_contact_sheet.png")
    print(f"report={out_dir / 'tesseract_original_report.csv'}")
    print(f"summary={out_dir / 'tesseract_original_summary.csv'}")
    print(f"sheet={out_dir / 'tesseract_original_contact_sheet.png'}")
    for row in summary:
        print(
            "{config}: exact={exact_matches}/{rows} avg_dist={avg_edit_distance} "
            "char_acc={avg_char_accuracy} len5={length_5_count} empty={empty_count}".format(**row)
        )


if __name__ == "__main__":
    main()
