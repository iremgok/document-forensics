"""
Gerçek üretken YZ (difüzyon inpainting) sahtecilikleri üretir.

``generate_synthetic``'teki ``inpaint_sim`` yalnızca ``cv2.inpaint`` ile yapılan bir **benzetimdir**. Bu betik
orijinal belgelerdeki tutar / tarih / numara alanlarını açık kaynak bir difüzyon modeliyle yeniden üretir;
maske, modele verilen inpainting maskesinin kendisidir. İki deney için kullanılır:

1. **Zero-shot dayanıklılık:** eğitilmiş model (ör. Stage E) bu sahtecilikleri hiç görmeden ne yapıyor?
2. **Az örnekli ince ayar:** 100-200 YZ örneğini eğitime ekleyip kalanlarla ve AIForge-Doc ile test.

Sızıntıyı önlemek için girdi belgeler bir ``splits.csv``'den **bölüm seçilerek** alınır (varsayılan:
yalnızca ``test`` bölümündeki orijinaller) ve her çıktı kaynağının ``group_id``'sini korur. Böylece
zero-shot deneyde eğitimde görülmüş bir belgenin YZ varyantı test edilmez.

Akış (her belge için):
  1. EasyOCR kelime kutuları; tutar / tarih / numara içeren biri seçilir.
  2. Kutunun 1-2 rakamı değiştirilerek yeni değer üretilir (``forge_real.mutate_digits``).
  3. Kutuyu içeren 512x512 pencere kırpılır; kutu (+ ``--pad``) maskelenip difüzyon modeline
     ``'printed text "<yeni değer>" ...'`` istemiyle verilir.
  4. Model çıktısı yalnızca **maskeli piksellerde** belgeye geri yapıştırılır (maske = gerçek sahte bölge).
  5. Orijinal ve sahte aynı biçimde (PNG) kaydedilir: "yeniden kaydedilmiş = sahte" kestirmesi oluşmaz.

Gereksinim: CUDA'lı GPU ve ``pip install diffusers transformers accelerate``. Model ``--model`` ile seçilir;
``AutoPipelineForInpainting`` ile yüklenir (SD 2 inpainting varsayılan; SDXL inpainting ya da sürümünüz
destekliyorsa FLUX Fill da denenebilir — metin yazımında genellikle daha başarılıdır).
``--dry-run`` difüzyon yerine ``cv2.inpaint`` + metin çizimi kullanır (akışı GPU'suz denemek için).

Çıktı (``src/dataset.py`` → ``diffusion_forged`` kaynağı)::

    data/10_diffusion_forged/images/*.png   masks/*.png   metadata.csv

Kullanım:
    python tools/diffusion_inpaint.py --splits checkpoints/stageE/splits.csv --sources real_forged synthetic \\
        --split test --limit 300 --out data/10_diffusion_forged
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import image_key, read_image_rgb  # noqa: E402
from src.forge_real import mutate_digits  # noqa: E402

FIELD_RE = re.compile(r"\d[\d.,/:\s-]{2,}\d")
PROMPT = 'printed text "{text}" on a white document, sharp black sans-serif font, scanned invoice'
NEGATIVE = "blurry, handwriting, distorted letters, watermark, color"


def pick_field(image: np.ndarray, reader, rng: random.Random):
    """Değiştirilecek kelime kutusu: (x0, y0, x1, y1, metin) ya da None."""
    words = reader.readtext(image, detail=1, paragraph=False)
    candidates = []
    for quad, text, conf in words:
        if conf < 0.4 or not FIELD_RE.search(text):
            continue
        pts = np.asarray(quad, dtype=float)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        if x1 - x0 < 12 or y1 - y0 < 6 or x1 - x0 > 480 or y1 - y0 > 120:
            continue
        candidates.append((int(x0), int(y0), int(x1), int(y1), text))
    return rng.choice(candidates) if candidates else None


def crop_window(shape: tuple[int, int], box: tuple[int, int, int, int], size: int = 512) -> tuple[int, int]:
    h, w = shape
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    x = int(np.clip(cx - size // 2, 0, max(0, w - size)))
    y = int(np.clip(cy - size // 2, 0, max(0, h - size)))
    return y, x


def dry_run_inpaint(crop: np.ndarray, mask: np.ndarray, text: str) -> np.ndarray:
    """GPU'suz deneme: klasik inpaint + yeni metin (difüzyonun yerine)."""
    out = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
    ys, xs = np.nonzero(mask)
    scale = max(0.4, (ys.max() - ys.min()) / 30)
    cv2.putText(out, text, (int(xs.min()) + 2, int(ys.max()) - 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (25, 25, 25), max(1, int(scale * 2)), cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Difüzyon inpainting ile YZ sahteciliği üret")
    p.add_argument("--splits", required=True, help="Girdi orijinallerinin seçileceği splits.csv")
    p.add_argument("--sources", nargs="+", default=["real_forged", "synthetic"])
    p.add_argument("--split", default="test", help="Yalnızca bu bölümdeki orijinaller (sızıntı önleme)")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--model", default="stabilityai/stable-diffusion-2-inpainting")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--pad", type=int, default=4, help="Kutu çevresine eklenen maske payı (piksel)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Difüzyon yerine cv2.inpaint (GPU'suz akış testi)")
    p.add_argument("--out", default="data/10_diffusion_forged")
    args = p.parse_args()

    rng = random.Random(args.seed)
    splits = pd.read_csv(args.splits, keep_default_na=False)
    pool = splits[(splits["label"].astype(int) == 0) & splits["source"].isin(args.sources)
                  & (splits["split"] == args.split)]
    pool = pool.sample(min(len(pool), args.limit), random_state=args.seed)
    print(f"{len(pool)} orijinal belge ({args.split} bölümü, kaynaklar: {', '.join(args.sources)})")

    import easyocr
    import torch

    reader = easyocr.Reader(["tr", "en"], gpu=torch.cuda.is_available(), verbose=False)
    pipe = None
    if not args.dry_run:
        from diffusers import AutoPipelineForInpainting
        from PIL import Image

        if not torch.cuda.is_available():
            raise SystemExit("Difüzyon için CUDA gerekli (GPU'suz akış testi: --dry-run).")
        pipe = AutoPipelineForInpainting.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
        pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator("cuda" if pipe is not None else "cpu").manual_seed(args.seed)

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for row in pool.itertuples():
        image = read_image_rgb(row.image_path)
        field = pick_field(image, reader, rng)
        if field is None:
            continue
        x0, y0, x1, y1, text = field
        new_text, _ = mutate_digits(text, rng, rng.choice([1, 2]))
        if new_text == text:
            continue
        h, w = image.shape[:2]
        pad_h, pad_w = max(0, 512 - h), max(0, 512 - w)
        canvas = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE) if pad_h or pad_w else image
        cy, cx = crop_window(canvas.shape[:2], (x0, y0, x1, y1))
        crop = canvas[cy:cy + 512, cx:cx + 512].copy()
        local = np.zeros((512, 512), np.uint8)
        local[max(0, y0 - cy - args.pad):y1 - cy + args.pad, max(0, x0 - cx - args.pad):x1 - cx + args.pad] = 255
        if args.dry_run:
            result = dry_run_inpaint(crop, local, new_text)
        else:
            result = np.asarray(pipe(prompt=PROMPT.format(text=new_text), negative_prompt=NEGATIVE,
                                     image=Image.fromarray(crop), mask_image=Image.fromarray(local),
                                     height=512, width=512, num_inference_steps=args.steps,
                                     guidance_scale=args.guidance, generator=generator).images[0].convert("RGB"))
            if result.shape[:2] != (512, 512):
                result = cv2.resize(result, (512, 512), interpolation=cv2.INTER_AREA)
        sel = local > 0
        forged = canvas.copy()
        forged[cy:cy + 512, cx:cx + 512][sel] = result[sel]  # yalnızca maskeli pikseller değişir
        forged = forged[:h, :w]
        mask = np.zeros((canvas.shape[0], canvas.shape[1]), np.uint8)
        mask[cy:cy + 512, cx:cx + 512] = local
        mask = mask[:h, :w]

        stem = image_key(row.image_path)
        for suffix, img, msk, label in (("00_orig", image, None, 0), ("01_diffusion_inpaint", forged, mask, 1)):
            name = f"{stem}__{suffix}"
            cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(str(out / "images" / f"{name}.png"))
            if msk is not None:
                cv2.imencode(".png", msk)[1].tofile(str(out / "masks" / f"{name}.png"))
            rows.append({"image": f"images/{name}.png", "mask": f"masks/{name}.png" if msk is not None else "",
                         "group_id": row.group_id, "label": label,
                         "tamper_type": "orig" if label == 0 else "diffusion_inpaint",
                         "lang": getattr(row, "lang", "") or "", "doc_type": getattr(row, "doc_type", "") or "",
                         "old_text": text if label else "", "new_text": new_text if label else "",
                         "model": "dry-run(cv2.inpaint)" if args.dry_run else args.model})
        print(f"  {stem}: '{text}' → '{new_text}'")
    fields = ["image", "mask", "group_id", "label", "tamper_type", "lang", "doc_type", "old_text", "new_text", "model"]
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows) // 2} YZ sahteciliği (+ orijinalleri) → {out}")


if __name__ == "__main__":
    main()
