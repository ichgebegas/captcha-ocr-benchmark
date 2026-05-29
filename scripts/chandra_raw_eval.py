from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from PIL import Image


GROUND_TRUTH = {
    "001": "ae6kx",
    "002": "gk2bc",
    "003": "a8gge",
    "004": "4pb4k",
    "005": "2d3kx",
    "006": "mk8k6",
    "007": "ya8r8",
    "008": "bpaye",
    "009": "bh62r",
    "010": "b3xcc",
}


def normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def collect_images(input_dir: Path, limit: int) -> list[Path]:
    images = []
    for idx in range(1, limit + 1):
        captcha_id = f"{idx:03d}"
        images.append(input_dir / f"captcha_{captcha_id}_original.jpg")
        images.append(input_dir / f"captcha_{captcha_id}_box_k5_soft_contrast.png")
    missing = [str(path) for path in images if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    return images


def parse_id_variant(path: Path) -> tuple[str, str]:
    match = re.search(r"captcha_(\d{3})_(.+?)\.(?:jpg|jpeg|png)$", path.name, re.I)
    if not match:
        raise ValueError(f"Unexpected image name: {path.name}")
    return match.group(1), match.group(2)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=r"C:\xxx\captcha\chandra_experiment_inputs_10")
    parser.add_argument("--output-dir", default=r"C:\xxx\captcha\chandra_raw_eval")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-image-dim", default="512")
    parser.add_argument("--max-output-tokens", type=int, default=64)
    args = parser.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ["TORCH_ATTN"] = "eager"
    os.environ["MIN_IMAGE_DIM"] = str(args.min_image_dim)

    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = (
        "Read the short text in this image. It is exactly five characters, "
        "using only lowercase latin letters and digits. Return only those five characters."
    )

    model = InferenceManager(method="hf")
    rows = []
    for image_path in collect_images(input_dir, args.limit):
        captcha_id, variant = parse_id_variant(image_path)
        expected = GROUND_TRUTH[captcha_id]
        image = Image.open(image_path).convert("RGB")
        result = model.generate(
            [BatchInputItem(image=image, prompt=prompt, prompt_type="ocr")],
            max_output_tokens=args.max_output_tokens,
            include_images=False,
            include_headers_footers=True,
        )[0]
        norm = normalize(result.raw)
        best5 = norm[:5]
        dist = edit_distance(expected, best5)
        row = {
            "captcha_id": captcha_id,
            "variant": variant,
            "expected": expected,
            "raw": result.raw,
            "normalized": norm,
            "best5": best5,
            "exact_match": best5 == expected,
            "edit_distance": dist,
            "char_accuracy": 1 - dist / max(len(expected), len(best5), 1),
            "token_count": result.token_count,
            "image_path": str(image_path),
        }
        rows.append(row)
        print(f"{captcha_id} {variant}: expected={expected} raw={result.raw!r} best5={best5}")

    report = output_dir / "chandra_raw_report.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for variant in sorted({row["variant"] for row in rows}):
        items = [row for row in rows if row["variant"] == variant]
        summary_rows.append(
            {
                "variant": variant,
                "total": len(items),
                "exact": sum(1 for row in items if row["exact_match"]),
                "exact_rate": sum(1 for row in items if row["exact_match"]) / len(items),
                "avg_edit_distance": sum(row["edit_distance"] for row in items) / len(items),
                "avg_char_accuracy": sum(row["char_accuracy"] for row in items) / len(items),
            }
        )

    summary = output_dir / "chandra_raw_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    (output_dir / "chandra_raw_report.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"report={report}")
    print(f"summary={summary}")


if __name__ == "__main__":
    main()
