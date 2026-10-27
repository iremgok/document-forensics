"""
Dağıtım ortamı (Hugging Face Spaces, Streamlit Community Cloud, Docker) için çalışma zamanı yardımcıları.

* **Model ağırlıkları:** ``*.pth`` dosyaları git'e girmez (``.gitignore``). ``checkpoints/`` altında
  yoksa şu ortam değişkenlerinden biriyle ilk açılışta indirilir:

  - ``FORENSICS_WEIGHTS_REPO`` — Hugging Face model deposu (ör. ``kullanici/document-forensics``);
    özel depo için ``HF_TOKEN`` da tanımlanmalıdır.
  - ``FORENSICS_WEIGHTS_URL``  — dosyaların bulunduğu temel URL (``<url>/stageE/best.pth``,
    ``<url>/finetuned_multilingual_stageB.pth``).

  Depodaki/URL'deki dosya düzeni ``WEIGHT_FILES`` ile aynıdır; alt klasörlü yollar (``stageE/...``)
  ``checkpoints/`` altında aynı alt klasöre indirilir. Stage E'nin kalibrasyonu kendi klasöründedir
  (``predict.load_calibration`` önce checkpoint klasörüne bakar).

  İndirme başarısız olursa uygulama çökmez; model seçimi boş kalır ve yalnızca OCR analizi sunulur.
* **Opsiyonel bağımlılıklar:** EasyOCR kurulu değilse OCR seçeneği devre dışı bırakılır; CUDA yoksa
  her şey CPU'da çalışır.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import urllib.request
from pathlib import Path

from src.model_registry import DEPLOY_FILES

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")
#: İndirilen dosyalar: varsayılan Stage E + karşılaştırma için gerçek Stage B
#: (``finetuned_multilingual_stageB.pth``) ve Stage A. ``finetuned_multilingual.pth`` bir Stage C ara modeliyle
#: ezildiği için artık dağıtılmaz (bkz. ``src/model_registry.py``).
WEIGHT_FILES = DEPLOY_FILES


def ensure_weights(dest: Path = CHECKPOINT_DIR) -> dict[str, str]:
    """Eksik ağırlık dosyalarını yapılandırılmış kaynaktan indirir.

    Returns:
        {dosya adı: durum} — "mevcut", "indirildi" ya da hata mesajı. Kaynak tanımlı değilse boş sözlük.
    """
    repo = os.environ.get("FORENSICS_WEIGHTS_REPO", "").strip()
    base_url = os.environ.get("FORENSICS_WEIGHTS_URL", "").strip().rstrip("/")
    if not repo and not base_url:
        return {}
    dest.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    for name in WEIGHT_FILES:
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            status[name] = "mevcut"
            continue
        try:
            if repo:
                from huggingface_hub import hf_hub_download

                hf_hub_download(repo_id=repo, filename=name, local_dir=str(dest), token=os.environ.get("HF_TOKEN"))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".part")
                urllib.request.urlretrieve(f"{base_url}/{name}", tmp)
                tmp.replace(target)
            status[name] = "indirildi"
            logger.info("Ağırlık indirildi: %s", target)
        except Exception as exc:  # Ağ/izin/eksik dosya: uygulama çalışmaya devam etmeli
            status[name] = f"indirilemedi: {exc.__class__.__name__}: {exc}"
            logger.warning("Ağırlık indirilemedi (%s): %s", name, exc)
    return status


def easyocr_available() -> bool:
    """EasyOCR import edilebilir mi (modeli yüklemeden kontrol eder)."""
    return importlib.util.find_spec("easyocr") is not None


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # Bozuk/CPU-only kurulumlarda CUDA sorgusu da hata verebilir
        return False
