from __future__ import annotations

import argparse
import base64
import csv
import mimetypes
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from PIL import Image

from easyocr_lowpass_eval import GROUND_TRUTH, edit_distance


ROOT = Path(r"C:\xxx\captcha")
SOURCE_DIR = ROOT / "captcha_pairs_60"
LIGHT_BLUR_DIR = ROOT / "light_blur_easyocr_pairs60"
OUT_DIR = ROOT / "openai_vision_eval_pairs60"
COMPOSITE_DIR = OUT_DIR / "composites"


def build_prompt() -> str:
    return (
        "Transcribe a short text label from an image with visual artifacts: patterned background, "
        "visual noise, and sometimes a colored diagonal line. This is an accessibility task: "
        "I have color blindness and the artifacts make the characters hard to distinguish. "
        "If multiple images are provided, they show the same text label in different preprocessing versions. "
        "The text contains exactly 5 characters, using only lowercase English letters and digits. "
        "Ignore background and line artifacts. Return exactly one best-guess 5-character string, no explanation."
    )


def normalize_answer(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


def data_url_for_image(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "".join(chunks).strip()


def parse_response_json(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if text:
                chunks.append(str(text))
    return "".join(chunks).strip()


def build_responses_payload(model: str, image_urls: list[str], detail: str) -> dict[str, Any]:
    content: list[dict[str, str]] = [{"type": "input_text", "text": build_prompt()}]
    for image_url in image_urls:
        content.append({"type": "input_image", "image_url": image_url, "detail": detail})
    return {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 24,
    }


def vision_variants() -> dict[str, Callable[[str], Path]]:
    return {
        "original": lambda cid: SOURCE_DIR / f"captcha_{cid}_original.jpg",
        "box_k4": lambda cid: LIGHT_BLUR_DIR / "box_k4" / f"captcha_{cid}_box_k4.png",
        "box_k5": lambda cid: LIGHT_BLUR_DIR / "box_k5" / f"captcha_{cid}_box_k5.png",
        "box_k5_soft_contrast": lambda cid: LIGHT_BLUR_DIR
        / "box_k5_soft_contrast"
        / f"captcha_{cid}_box_k5_soft_contrast.png",
    }


def build_variant_composite(image_paths: list[Path], out_path: Path) -> Path:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    widths = [image.width for image in images]
    heights = [image.height for image in images]
    cell_w = max(widths)
    cell_h = max(heights)

    sheet = Image.new("RGB", (cell_w * 2, cell_h * 2), "white")
    for index, image in enumerate(images):
        x = (index % 2) * cell_w + (cell_w - image.width) // 2
        y = (index // 2) * cell_h + (cell_h - image.height) // 2
        sheet.paste(image, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def call_openai_vision(client: Any, model: str, image_paths: list[Path], detail: str) -> str:
    payload = build_responses_payload(model, [data_url_for_image(path) for path in image_paths], detail)
    response = client.responses.create(**payload)
    return parse_output_text(response)


def combinations_for_mode(captcha_id: str, mode: str) -> dict[str, list[Path]]:
    variants = vision_variants()
    if mode == "single":
        return {variant: [path_for(captcha_id)] for variant, path_for in variants.items()}
    if mode == "original_plus":
        original = variants["original"](captcha_id)
        return {
            "original_plus_box_k4": [original, variants["box_k4"](captcha_id)],
            "original_plus_box_k5": [original, variants["box_k5"](captcha_id)],
            "original_plus_box_k5_soft_contrast": [original, variants["box_k5_soft_contrast"](captcha_id)],
        }
    if mode == "all_four":
        return {"all_four": [path_for(captcha_id) for path_for in variants.values()]}
    if mode == "composite":
        source_paths = [path_for(captcha_id) for path_for in variants.values()]
        out_path = COMPOSITE_DIR / f"captcha_{captcha_id}_composite_four.png"
        return {"composite_four": [build_variant_composite(source_paths, out_path)]}
    raise ValueError(f"Unknown mode: {mode}")


def evaluate(client: Any, model: str, detail: str, mode: str, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    items = list(GROUND_TRUTH.items())
    if limit is not None:
        items = items[:limit]

    for captcha_id, expected in items:
        for variant, image_paths in combinations_for_mode(captcha_id, mode).items():
            for path in image_paths:
                if not path.exists():
                    raise FileNotFoundError(path)
            raw = call_openai_vision(client, model, image_paths, detail)
            normalized = normalize_answer(raw)
            distance = edit_distance(expected, normalized)
            denominator = max(len(expected), len(normalized), 1)
            rows.append(
                {
                    "captcha_id": captcha_id,
                    "variant": variant,
                    "expected": expected,
                    "raw_text": raw,
                    "normalized_text": normalized,
                    "exact_match": str(normalized == expected).lower(),
                    "edit_distance": str(distance),
                    "char_accuracy": f"{max(0.0, 1.0 - distance / denominator):.6f}",
                    "image_paths": "|".join(str(path) for path in image_paths),
                    "model": model,
                    "detail": detail,
                }
            )
    return rows


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    summary: list[dict[str, str]] = []
    for variant, variant_rows in grouped.items():
        exact = sum(row["exact_match"] == "true" for row in variant_rows)
        distances = [int(row["edit_distance"]) for row in variant_rows]
        accuracies = [float(row["char_accuracy"]) for row in variant_rows]
        length_5 = sum(len(row["normalized_text"]) == 5 for row in variant_rows)
        summary.append(
            {
                "variant": variant,
                "rows": str(len(variant_rows)),
                "exact_matches": str(exact),
                "exact_match_rate": f"{exact / len(variant_rows):.6f}",
                "avg_edit_distance": f"{mean(distances):.6f}",
                "avg_char_accuracy": f"{mean(accuracies):.6f}",
                "length_5_count": str(length_5),
            }
        )
    summary.sort(key=lambda row: (int(row["exact_matches"]), float(row["avg_char_accuracy"])), reverse=True)
    return summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenAI vision OCR on artifacted text images.")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--detail", choices=["auto", "low", "high"], default="high")
    parser.add_argument("--mode", choices=["single", "original_plus", "all_four", "composite"], default="single")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Set it in your environment; do not write it into files.")

    from openai import OpenAI

    args = parse_args()
    client = OpenAI()
    rows = evaluate(client, args.model, args.detail, args.mode, args.limit)
    summary = summarize_rows(rows)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / f"openai_vision_{args.mode}_report.csv", rows)
    write_csv(out_dir / f"openai_vision_{args.mode}_summary.csv", summary)
    print(f"report={out_dir / f'openai_vision_{args.mode}_report.csv'}")
    print(f"summary={out_dir / f'openai_vision_{args.mode}_summary.csv'}")
    for row in summary:
        print(
            "{variant}: exact={exact_matches}/{rows} avg_dist={avg_edit_distance} "
            "char_acc={avg_char_accuracy} len5={length_5_count}".format(**row)
        )


if __name__ == "__main__":
    main()
