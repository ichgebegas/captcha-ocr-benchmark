# Experiment Summary

## Goal

Recognize short 5-character strings from artifacted text images. The images contain strong patterned background noise and a red diagonal line crossing the characters. The objective was to find a practical preprocessing and OCR strategy without having access to the original recognizer code.

## Ground Truth

32 examples were manually labeled and used as the benchmark set. Exact match was the primary metric. Character accuracy and edit distance were used as secondary metrics.

## Preprocessing Results

The core observation was that aggressive background removal damages the characters. Many operations remove both the pattern and parts of the text because the artifacts and strokes overlap spatially.

Methods tried:

- grayscale conversion and 2x upscale;
- sharpening;
- color 2x upscale;
- median blur, Gaussian blur, light blur;
- low-pass smoothing;
- FFT / notch pattern suppression;
- red-line masking and inpainting;
- template/background residual suppression;
- morphology close/open;
- text mask generation;
- dark-text enhancement;
- Real-ESRGAN;
- Clarity upscaler.

Median blur was rejected visually because it destroys character shape. Generative upscalers were also rejected because they can invent or alter characters.

The best practical preprocessing was `box_k5_soft_contrast`. It does not fully remove the background; instead, it suppresses the pattern enough for a vision model while keeping character structure relatively intact.

## OCR Results

### OpenAI Vision

OpenAI Vision was the strongest tested recognizer.

| Variant | Exact | Exact rate | Avg edit distance | Avg char accuracy |
|---|---:|---:|---:|---:|
| `box_k5_soft_contrast` | 13 / 32 | 40.6% | 1.34 | 73.1% |
| `box_k5` | 9 / 32 | 28.1% | 1.44 | 71.3% |
| `box_k4` | 6 / 32 | 18.8% | 1.81 | 63.8% |
| `original` | 1 / 32 | 3.1% | 3.28 | 34.4% |

Two-image prompting with `original + box_k5_soft_contrast` matched the best exact result:

| Variant | Exact | Exact rate | Avg edit distance | Avg char accuracy |
|---|---:|---:|---:|---:|
| `original_plus_box_k5_soft_contrast` | 13 / 32 | 40.6% | 1.31 | 73.8% |
| `original_plus_box_k5` | 12 / 32 | 37.5% | 1.22 | 75.6% |

Composite four-image prompting had better average character accuracy but lower exact-match count:

| Variant | Exact | Exact rate | Avg edit distance | Avg char accuracy |
|---|---:|---:|---:|---:|
| `composite_four` | 12 / 32 | 37.5% | 1.16 | 77.3% |

### EasyOCR

EasyOCR did not perform well on the selected variants. Even the best local preprocessing was not usable for exact recognition.

Top observed EasyOCR preprocessing summary:

| Variant | Exact | Exact rate | Avg edit distance | Avg char accuracy |
|---|---:|---:|---:|---:|
| `box_k5_soft_contrast` | 0 / 32 | 0.0% | 3.16 | 38.5% |
| `box_k4` | 3 / 32 | 9.4% | 3.22 | 37.9% |
| `box_k5` | 0 / 32 | 0.0% | 3.19 | 37.6% |

### Tesseract

Tesseract recognized clean synthetic digit text in sanity checks, but failed on the artifacted images. The issue was not that Tesseract cannot read digits; the patterned background and line artifacts dominated.

### DeepSeek-OCR-2

DeepSeek-OCR-2 was installed and run locally. The model loaded on RTX 4060 after using the eager attention path and UTF-8 console handling. It was not useful for this task:

- original images were often reported as having no text;
- processed images produced hallucinated document-like descriptions;
- custom "return exactly five characters" prompts did not fix this.

The model appears optimized for document/layout OCR rather than short noisy labels.

### Chandra OCR 2

Chandra OCR 2 was installed and run locally. The default image scaling caused CUDA OOM on RTX 4060 8GB. Reducing `MIN_IMAGE_DIM` made inference possible.

Raw-output probing on 10 pairs showed:

| Variant | Exact | Exact rate | Avg edit distance | Avg char accuracy |
|---|---:|---:|---:|---:|
| `box_k5_soft_contrast` | 1 / 10 | 10.0% | 3.0 | 40.0% |
| `original` | 0 / 10 | 0.0% | 3.6 | 28.0% |

Chandra sometimes captured the prefix, for example `bh62r` was read correctly once, but it was not competitive with OpenAI Vision.

## Final Assessment

Best current pipeline:

1. Generate `box_k5_soft_contrast`.
2. Send to OpenAI Vision with a strict 5-character alphanumeric prompt.
3. Optionally include the original image as a second reference image, but do not rely on composite grids for final selection.

Expected performance on the current 32-example set:

- exact match: about 40-50%;
- character-level similarity: about 73-77%;
- not sufficient for fully automated high-confidence production use.

## Recommended Next Steps

The main bottleneck is data, not another generic OCR package.

Recommended next work:

1. Request 200-300 additional labeled images from the client.
2. Train a tiny character-level model or fine-tune a lightweight recognizer.
3. Keep `box_k5_soft_contrast` as the preprocessing baseline.
4. Use OpenAI Vision output as a benchmark and fallback.
5. Track all future experiments against the same exact-match and edit-distance metrics.
