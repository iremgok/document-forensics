"""
Model kaydı: uygulamada, dashboard'da ve dağıtımda gösterilen ağırlık dosyalarının **tek kaynağı**.

Neden gerekli? ``checkpoints/`` altındaki dosya adları içeriklerini yansıtmıyor:
``finetuned_multilingual.pth`` 21.10'da bir Stage C ara modeliyle (epoch 4) ezilmiş, gerçek Stage B
``finetuned_multilingual_stageB.pth`` adıyla kalmıştır; ``stageD/best.pth`` aslında yalnız sentetikle
eğitilmiş bir Stage C çalışmasıdır; ``runs/fast_run`` 3 epoch'luk bir denemedir. Bu modellerin
validasyon yanlış alarm oranı ≈ 1.0'dır (her belgeye "sahte" derler). Etiketler burada içeriğe göre
verilir ve bilinmeyen dosyalar checkpoint içindeki ``config`` / ``metrics`` sözlüğünden otomatik sınıflanır.

Durumlar:
* ``recommended``  — uygulamanın varsayılanı (Stage E),
* ``comparison``   — rapor/karşılaştırma için tutulan önceki aşamalar,
* ``experimental`` — ara/deneme modelleri; varsayılan olarak listede gösterilmez.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

RECOMMENDED, COMPARISON, EXPERIMENTAL = "recommended", "comparison", "experimental"

#: Validasyonda orijinallerin bu orandan fazlasına "sahte" diyen (ham belge skoru, eşik 0.5) ya da piksel
#: F1'i bu değerin altında kalan modeller "güvenilmez" olarak işaretlenir.
UNRELIABLE_FPR = 0.9
UNRELIABLE_PIXEL_F1 = 0.1


@dataclass(frozen=True)
class ModelInfo:
    path: str
    label: str
    status: str
    note: str = ""


#: Bilinen ağırlıklar (yol → etiket). Sıra, uygulamadaki listeleme sırasıdır.
KNOWN_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("checkpoints/stageE/best.pth", "Stage E — önerilen (sentetik v4 + gerçek belgeler)", RECOMMENDED,
              "Kalibre (T ve eşik val'de seçildi). Gerçek tabanlı sahte testinde TPR 0.57 / FPR 0.05."),
    ModelInfo("checkpoints/finetuned_multilingual_stageB.pth", "Stage B — karşılaştırma (sentetik + SROIE + DocTamper)",
              COMPARISON, "Val'de orijinallerin %38'ine sahte diyor; dijital belgelerde yanlış alarm verir."),
    ModelInfo("checkpoints/base_doctamper.pth", "Stage A — karşılaştırma (yalnız DocTamper)", COMPARISON,
              "Eğitimde hiç orijinal belge görmedi: belge düzeyinde karar için uygun değil, piksel haritası referansı."),
    ModelInfo("checkpoints/finetuned_multilingual.pth", "Deneysel — Stage C ara model (epoch 4; eski adı yanıltıcı)",
              EXPERIMENTAL, "Val FPR 1.00, piksel F1 0.07: her belgeye sahte der. Kullanmayın."),
    ModelInfo("checkpoints/stageD/best.pth", "Deneysel — Stage C/D (yalnız sentetik)", EXPERIMENTAL,
              "Val FPR 0.995: 'alan konumu = sahte' ezberi. Kullanmayın."),
    ModelInfo("runs/fast_run/best.pth", "Deneysel — 3 epoch deneme (best)", EXPERIMENTAL, "Kullanmayın."),
    ModelInfo("runs/fast_run/last.pth", "Deneysel — 3 epoch deneme (last)", EXPERIMENTAL, "Kullanmayın."),
)
_BY_PATH = {m.path: m for m in KNOWN_MODELS}

#: Dağıtımda (HF Spaces) indirilen dosyalar: önerilen model + karşılaştırma modelleri ve kalibrasyonları.
DEPLOY_FILES: tuple[str, ...] = (
    "stageE/best.pth", "stageE/calibration.json",
    "finetuned_multilingual_stageB.pth", "base_doctamper.pth", "calibration.json",
)


def _posix(path: str | Path) -> str:
    return Path(path).as_posix()


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    """Checkpoint içindeki aşama, epoch ve son validasyon metrikleri (yükleme maliyeti: tüm dosya)."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config, metrics = ckpt.get("config", {}) or {}, ckpt.get("metrics", {}) or {}
    return {"epoch": ckpt.get("epoch"), "stage": config.get("stage"), "sources": config.get("sources"),
            "streams": config.get("streams", "rgb+srm"), "crop_size": config.get("crop_size"),
            "val_pixel_f1": metrics.get("val/pixel_f1"), "val_image_auc": metrics.get("val/image_auc"),
            "val_image_fpr": metrics.get("val/image_fpr")}


def reliability_warning(summary: dict[str, Any]) -> Optional[str]:
    """Validasyon metriklerine göre model güvenilmezse açıklayıcı uyarı, değilse None."""
    fpr, f1 = summary.get("val_image_fpr"), summary.get("val_pixel_f1")
    reasons = []
    if fpr is not None and fpr == fpr and fpr >= UNRELIABLE_FPR:
        reasons.append(f"validasyonda orijinallerin %{fpr * 100:.0f}'ine sahte demiş")
    if f1 is not None and f1 == f1 and f1 < UNRELIABLE_PIXEL_F1:
        reasons.append(f"validasyon piksel F1 {f1:.3f}")
    if summary.get("sources") == ["doctamper"]:
        reasons.append("yalnız DocTamper ile eğitildi, hiç orijinal belge görmedi")
    return "; ".join(reasons) if reasons else None


def describe(path: str | Path, summary: Optional[dict[str, Any]] = None) -> ModelInfo:
    """Bilinen modelse kayıttaki bilgi; değilse checkpoint içeriğinden türetilmiş etiket."""
    key = _posix(path)
    if key in _BY_PATH:
        return _BY_PATH[key]
    summary = summary or {}
    stage = summary.get("stage") or "aşama yok"
    status = EXPERIMENTAL if reliability_warning(summary) else COMPARISON
    return ModelInfo(key, f"{key} ({stage}, epoch {summary.get('epoch', '?')})", status)


def list_models(dirs: tuple[str, ...] = ("checkpoints", "runs"), include_experimental: bool = False
                ) -> list[ModelInfo]:
    """Diskteki modeller: önce kayıttakiler (kayıt sırasıyla), sonra diğer .pth dosyaları (en yeni önce)."""
    found = [m for m in KNOWN_MODELS if Path(m.path).is_file()]
    known = {m.path for m in found}
    others = sorted((p for d in dirs if Path(d).is_dir() for p in Path(d).rglob("*.pth")
                     if p.stat().st_size > 0 and _posix(p) not in known and not p.name.endswith(".tmp")),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    models = found + [describe(p, _safe_summary(p)) for p in others]
    return [m for m in models if include_experimental or m.status != EXPERIMENTAL]


def _safe_summary(path: Path) -> dict[str, Any]:
    try:
        return checkpoint_summary(path)
    except Exception:  # Bozuk/eksik dosya: listede yine görünsün, yüklenirken hata verilir.
        return {}
