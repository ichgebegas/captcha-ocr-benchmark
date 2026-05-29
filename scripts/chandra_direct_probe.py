from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-image-dim", default="512")
    parser.add_argument("--max-output-tokens", type=int, default=64)
    args = parser.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ["TORCH_ATTN"] = "eager"
    os.environ["MIN_IMAGE_DIM"] = str(args.min_image_dim)

    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem

    image = Image.open(args.image).convert("RGB")
    prompt = (
        "Read the short text in this image. It is exactly five characters, "
        "using only lowercase latin letters and digits. Return only those five characters."
    )

    model = InferenceManager(method="hf")
    result = model.generate(
        [BatchInputItem(image=image, prompt=prompt, prompt_type="ocr")],
        max_output_tokens=args.max_output_tokens,
        include_images=False,
        include_headers_footers=True,
    )[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "raw": result.raw,
                "markdown": result.markdown,
                "html": result.html,
                "token_count": result.token_count,
                "chunks": result.chunks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)
    print("raw=", result.raw)
    print("markdown=", result.markdown)


if __name__ == "__main__":
    main()
