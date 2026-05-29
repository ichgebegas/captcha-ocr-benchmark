# Captcha-like OCR Benchmark

[Русская версия](README.md)

This repository summarizes a short OCR benchmark for small 5-character text images with strong visual artifacts: patterned background noise, a red diagonal line crossing the text, low contrast, and tightly spaced characters.

The task was treated as legitimate OCR of ordinary artifacted images. The word "captcha" is used only because the images look captcha-like.

## Dataset

- 32 manually labeled examples were used as ground truth.
- Each target string has exactly 5 characters.
- Character set: lowercase Latin letters and digits.
- A few illustrative examples are included below; the full dataset is not committed.

## Examples

| ID | Original | `box_k5_soft_contrast` |
|---|---|---|
| 001 | ![001 original](assets/examples/captcha_001_original.jpg) | ![001 processed](assets/examples/captcha_001_box_k5_soft_contrast.png) |
| 009 | ![009 original](assets/examples/captcha_009_original.jpg) | ![009 processed](assets/examples/captcha_009_box_k5_soft_contrast.png) |

## Best Result

The best practical result was:

| Recognizer | Preprocessing | Exact matches | Exact rate | Avg char accuracy |
|---|---:|---:|---:|---:|
| OpenAI Vision | `box_k5_soft_contrast` | 13 / 32 | 40.6% | 73.1% |
| OpenAI Vision | `original + box_k5_soft_contrast` | 13 / 32 | 40.6% | 73.8% |
| OpenAI Vision | 4-image composite | 12 / 32 | 37.5% | 77.3% |
| EasyOCR | `box_k5_soft_contrast` | 0 / 32 | 0.0% | 38.5% |

Conclusion: **OpenAI Vision with `box_k5_soft_contrast` was the best stable option**, reaching roughly **40-50% exact-match accuracy** depending on prompt/run framing. Composite images improved average character similarity but did not improve exact-match count.

## Preprocessing Tried

The main families tested:

- Original image without preprocessing.
- Grayscale upscale and sharpen variants.
- Color upscale variants.
- Low-pass / light blur variants to suppress the patterned background without fully deleting it.
- FFT / notch-style periodic pattern suppression.
- Template residual / background suppression attempts.
- Red-line masking / geometry-based line removal.
- Morphological close/open operations.
- Text-mask attempts.
- Thin / dark-text enhancement variants.
- Real-ESRGAN upscale.
- Clarity upscaler attempt.

The best-looking and best-performing local preprocessing was:

`box_k5_soft_contrast`

It lightly suppresses the background and improves character contrast without destroying the letter shapes as aggressively as stronger denoising or median blur.

## OCR Engines Tried

| OCR / Vision system | Result |
|---|---|
| EasyOCR | Poor on this dataset; often failed digits/letters under patterned background. |
| Tesseract | Poor on originals and noisy variants; digit sanity check worked, but artifacted images failed. |
| CapMonster | Not evaluated fully due missing usable recognizer logs/API access during this stage. |
| OpenAI Vision | Best observed option; around 40.6% exact match on 32 examples. |
| DeepSeek-OCR-2 | Local run worked, but the model was document/layout oriented and hallucinated or reported no text. |
| Chandra OCR 2 | Local run worked only after reducing input scaling; raw output sometimes captured prefixes, but exact rate was low in probe. |
| Real-ESRGAN / Clarity upscalers | Not useful; generative enhancement risks changing characters. |
