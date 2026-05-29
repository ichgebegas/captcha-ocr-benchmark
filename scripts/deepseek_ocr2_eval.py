from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


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
    "011": "kgn6p",
    "012": "y3d84",
    "013": "dhb44",
    "014": "gacyp",
    "015": "46f75",
    "016": "88e5y",
    "017": "p24gy",
    "018": "n85gw",
    "019": "d6y2a",
    "020": "d7b2f",
    "021": "27ber",
    "022": "537nd",
    "023": "6cwmf",
    "024": "32exc",
    "025": "xmkmr",
    "026": "ygpp6",
    "027": "7kcwa",
    "028": "4xhdk",
    "029": "7bb48",
    "030": "6nyhr",
    "031": "b45hf",
    "032": "824f7",
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


def parse_id_variant(path: Path) -> tuple[str, str]:
    match = re.search(r"captcha_(\d{3})_(.+?)\.(?:jpg|jpeg|png)$", path.name, re.I)
    if not match:
        raise ValueError(f"Unexpected image name: {path.name}")
    return match.group(1), match.group(2)


def collect_images(limit: int) -> list[Path]:
    image_dir = Path(r"C:\xxx\captcha\glmocr_experiment_inputs_10")
    images = []
    for idx in range(1, limit + 1):
        captcha_id = f"{idx:03d}"
        images.append(image_dir / f"captcha_{captcha_id}_original.jpg")
        images.append(image_dir / f"captcha_{captcha_id}_box_k5_soft_contrast.png")
    missing = [str(path) for path in images if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    return images


def load_model(model_name: str, attn: str, gpu_memory: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs = {
        "trust_remote_code": True,
        "use_safetensors": True,
        "_attn_implementation": attn,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available() and gpu_memory.lower() != "all":
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: gpu_memory, "cpu": "32GiB"}
    model = AutoModel.from_pretrained(model_name, **kwargs).eval()
    if torch.cuda.is_available() and gpu_memory.lower() == "all":
        model = model.cuda().to(torch.bfloat16)
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR-2")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--gpu-memory", default="6GiB")
    parser.add_argument("--base-size", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--crop-mode", action="store_true")
    parser.add_argument(
        "--prompt",
        default="<image>\nFree OCR. ",
        help="Prompt passed to DeepSeek-OCR-2",
    )
    args = parser.parse_args()

    out_dir = Path(r"C:\xxx\captcha\deepseek_ocr2_eval")
    run_dir = out_dir / "model_outputs"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args.model, args.attn, args.gpu_memory)
    prompt = args.prompt

    rows = []
    for image_path in collect_images(args.limit):
        captcha_id, variant = parse_id_variant(image_path)
        expected = GROUND_TRUTH[captcha_id]
        item_dir = run_dir / f"{captcha_id}_{variant}"
        item_dir.mkdir(parents=True, exist_ok=True)
        print(f"{captcha_id} {variant} {image_path}")
        try:
            raw = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                output_path=str(item_dir),
                base_size=args.base_size,
                image_size=args.image_size,
                crop_mode=args.crop_mode,
                save_results=True,
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise SystemExit(f"CUDA OOM on {image_path.name}: {exc}") from exc

        result_file = item_dir / "result.mmd"
        raw_text = result_file.read_text(encoding="utf-8") if result_file.exists() else ("" if raw is None else str(raw))
        norm = normalize(raw_text)
        dist = edit_distance(expected, norm)
        rows.append(
            {
                "captcha_id": captcha_id,
                "variant": variant,
                "expected": expected,
                "raw_text": raw_text,
                "normalized_text": norm,
                "exact_match": norm == expected,
                "edit_distance": dist,
                "char_accuracy": 1 - dist / max(len(expected), len(norm), 1),
                "image_path": str(image_path),
            }
        )
        print(f"  expected={expected} raw={raw_text!r} norm={norm}")

    report = out_dir / "deepseek_ocr2_report.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    summary = out_dir / "deepseek_ocr2_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        fields = ["variant", "total", "exact", "exact_rate", "avg_edit_distance", "avg_char_accuracy"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant, items in sorted(by_variant.items()):
            writer.writerow(
                {
                    "variant": variant,
                    "total": len(items),
                    "exact": sum(1 for item in items if item["exact_match"]),
                    "exact_rate": sum(1 for item in items if item["exact_match"]) / len(items),
                    "avg_edit_distance": sum(item["edit_distance"] for item in items) / len(items),
                    "avg_char_accuracy": sum(item["char_accuracy"] for item in items) / len(items),
                }
            )
    print(f"report={report}")
    print(f"summary={summary}")


if __name__ == "__main__":
    main()
