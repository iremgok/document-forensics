"""
Streamlit web arayüzü: Türkçe finansal belgelerde hibrit sahtecilik analizi.

Çalıştırma:
    streamlit run app.py

Sayfa üç analiz üretir:
  1. **Görsel analiz** — Derin öğrenme modelinin piksel olasılık haritası,
     orijinal belgenin üzerine kırmızı ısı haritası olarak bindirilir.
  2. **İçerik analizi** — EasyOCR çıktısı ve kural motoru raporu
     (IBAN Mod-97, TCKN, Matrah + KDV = Genel Toplam, KDV oranı).
  3. **Nihai karar** — iki kanıt kaynağını birleştiren hibrit güven skoru.
"""

from __future__ import annotations

import logging
import warnings

import torch

# Streamlit'in dosya izleyicisi modülleri tararken ``torch.classes.__path__``'e erişir;
# bu özel modül ``__getattr__`` ile C++ sınıfı aramaya çalıştığı için şu uyarı düşer:
# "Examining the path of torch.classes raised: RuntimeError: Tried to instantiate class '__path__._path'".
# Boş bir ``__path__`` vermek izleyicinin modülü sessizce atlamasını sağlar.
torch.classes.__path__ = []
warnings.filterwarnings("ignore", message=r".*torch\.classes.*")
warnings.filterwarnings("ignore", message=r".*Tried to instantiate class '__path__\._path'.*")
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, limit_long_side
from src.ocr_checker import FAIL, PASS, ContentReport, check_document_text, run_ocr

CHECKPOINT_DIRS = (Path("checkpoints"), Path("runs"))
PDF_DPI = 200


# ---------------------------------------------------------------------------
# Yükleme yardımcıları
# ---------------------------------------------------------------------------

def find_checkpoints() -> list[str]:
    """``checkpoints/`` ve ``runs/`` altındaki boş olmayan .pth dosyalarını listeler (en yeni önce)."""
    paths = [p for d in CHECKPOINT_DIRS if d.is_dir() for p in d.rglob("*.pth") if p.stat().st_size > 0]
    return [str(p) for p in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)]


@st.cache_resource(show_spinner="Model yükleniyor...")
def load_detector(path: str) -> tuple[torch.nn.Module, dict, float]:
    """Modeli, yapılandırmasını ve (varsa) val kümesinde öğrenilmiş sıcaklığı yükler."""
    from src.models import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(path, device)
    calib = Path(path).parent / "calibration.json"
    temperature = json.loads(calib.read_text(encoding="utf-8"))["temperature"] if calib.exists() else 1.0
    return model, config, float(temperature)


def decode_upload(data: bytes, filename: str, page: int = 0) -> tuple[np.ndarray, int]:
    """Yüklenen dosyayı RGB görüntüye çevirir. PDF ise seçilen sayfa ``PDF_DPI`` ile rasterleştirilir.

    Returns:
        (RGB görüntü, toplam sayfa sayısı)
    """
    if filename.lower().endswith(".pdf"):
        import pymupdf

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pix = doc[page].get_pixmap(dpi=PDF_DPI)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            return np.ascontiguousarray(image[:, :, :3]), doc.page_count
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Görüntü çözülemedi.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1


# ---------------------------------------------------------------------------
# Analiz
# ---------------------------------------------------------------------------

def run_detector(model: torch.nn.Module, config: dict, image: np.ndarray, temperature: float
                 ) -> tuple[np.ndarray, float]:
    """Eğitimle aynı ön işlem (boyut sınırı + ImageNet normalizasyonu) ve kayan pencere çıkarımı.

    Returns:
        (orijinal boyuta ölçeklenmiş olasılık haritası, kalibre belge olasılığı)
    """
    from src.models import sliding_window_predict

    small, _ = limit_long_side(image, None, config.get("max_side", 1600))
    tensor = (torch.from_numpy(small).float().permute(2, 0, 1) / 255.0
              - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    prob_map, doc_logit = sliding_window_predict(model, tensor, tile=config.get("crop_size", 512))
    prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return prob_map, float(torch.sigmoid(torch.tensor(doc_logit / temperature)))


def heatmap_overlay(image: np.ndarray, prob_map: np.ndarray, threshold: float) -> np.ndarray:
    """Sahte olasılığı yüksek pikselleri kırmızı ile boyar; saydamlık olasılıkla orantılıdır."""
    alpha = np.where(prob_map >= threshold, 0.25 + 0.5 * prob_map, 0.0)[..., None]
    red = np.zeros_like(image)
    red[..., 0] = 255
    out = (image * (1 - alpha) + red * alpha).astype(np.uint8)
    contours, _ = cv2.findContours((prob_map >= threshold).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (200, 0, 0), max(1, image.shape[1] // 600))
    return out


def draw_findings(image: np.ndarray, report: ContentReport) -> np.ndarray:
    """Başarısız kuralların satır kutularını turuncu, başarılı olanları yeşil çerçeveler."""
    out = image.copy()
    for f in report.findings:
        if f.bbox and f.bbox[2] > f.bbox[0] and f.status in (PASS, FAIL):
            color = (255, 140, 0) if f.status == FAIL else (0, 170, 0)
            cv2.rectangle(out, f.bbox[:2], f.bbox[2:], color, max(2, image.shape[1] // 500))
    return out


def hybrid_score(dl_score: Optional[float], report: Optional[ContentReport]) -> Optional[float]:
    """
    Hibrit güven skoru — "noisy-OR" birleşimi:

        P(sahte) = 1 − (1 − p_DL) · (1 − r_OCR)

    * p_DL: derin öğrenme modelinin kalibre belge olasılığı.
    * r_OCR: kural motorunun risk skoru (yalnızca *başarısız* kurallardan oluşur).

    Gerekçe: İki kaynak farklı sahtecilik türlerini yakalar. Kural ihlali (ör. geçersiz
    IBAN) piksel izi zayıf olsa bile tek başına güçlü kanıttır; tersine, matematiği
    koruyan "akıllı" bir sahtecilikte kuralların geçmesi pikselde görülen izi
    **azaltmamalıdır** — bu yüzden geçen kurallar skoru düşürmez (r_OCR = 0 → skor = p_DL).
    """
    risk = report.risk if report is not None and report.has_evidence else None
    if dl_score is None:
        return risk
    if risk is None:
        return dl_score
    return 1.0 - (1.0 - dl_score) * (1.0 - risk)


def verdict(score: float) -> tuple[str, str]:
    """Skoru insan tarafından okunabilir bir karara çevirir."""
    if score >= 0.7:
        return "🔴 Yüksek sahtecilik olasılığı", "error"
    if score >= 0.3:
        return "🟠 Şüpheli — manuel inceleme önerilir", "warning"
    return "🟢 Belirgin sahtecilik izi bulunamadı", "success"


# ---------------------------------------------------------------------------
# Arayüz
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Belge Sahtecilik Analizi", page_icon="🔍", layout="wide")
    st.title("🔍 Finansal Belge Sahtecilik Tespiti")
    st.caption("Dekont · Fatura · Fiş — Derin öğrenme (RGB + SRM gürültü) ve OCR içerik tutarlılığı ile hibrit analiz")

    with st.sidebar:
        st.header("⚙️ Ayarlar")
        checkpoints = find_checkpoints()
        ckpt = st.selectbox("Model ağırlığı", checkpoints) if checkpoints else None
        if not checkpoints:
            st.warning("`checkpoints/` veya `runs/` altında eğitilmiş model yok. Yalnızca OCR analizi yapılacak.\n\n"
                       "Eğitim: `python train.py --sources synthetic`")
        threshold = st.slider("Piksel eşiği", 0.05, 0.95, 0.5, 0.05,
                              help="Bu olasılığın üzerindeki pikseller sahte bölge olarak boyanır.")
        use_ocr = st.toggle("OCR içerik analizi", value=True)
        ocr_gpu = st.toggle("OCR için GPU", value=torch.cuda.is_available(), disabled=not torch.cuda.is_available())

    upload = st.file_uploader("Belge yükleyin (PDF, PNG, JPG)", type=["pdf", "png", "jpg", "jpeg"])
    if upload is None:
        st.info("Analize başlamak için bir dekont, fatura veya fiş yükleyin.")
        return

    data = upload.getvalue()
    page = 0
    if upload.name.lower().endswith(".pdf"):
        _, n_pages = decode_upload(data, upload.name, 0)
        if n_pages > 1:
            page = st.number_input("PDF sayfası", 1, n_pages, 1) - 1
        st.caption("Not: PDF rasterleştirildiğinde dosya içi sıkıştırma izleri değişir; "
                   "mümkünse belgenin orijinal görüntü dosyasını yükleyin.")
    try:
        image, _ = decode_upload(data, upload.name, page)
    except Exception as exc:  # Bozuk/şifreli dosya
        st.error(f"Dosya okunamadı: {exc}")
        return

    dl_score: Optional[float] = None
    prob_map: Optional[np.ndarray] = None
    if ckpt:
        model, config, temperature = load_detector(ckpt)
        with st.spinner("Derin öğrenme analizi..."):
            prob_map, dl_score = run_detector(model, config, image, temperature)

    report: Optional[ContentReport] = None
    lines = []
    if use_ocr:
        with st.spinner("OCR ve kural motoru çalışıyor (ilk çalıştırmada model indirilir)..."):
            lines = run_ocr(image, gpu=ocr_gpu)
            report = check_document_text(lines)

    # --- Nihai karar ----------------------------------------------------------------
    final = hybrid_score(dl_score, report)
    st.subheader("🧭 Nihai Karar")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hibrit Sahtecilik Olasılığı", f"%{final * 100:.1f}" if final is not None else "—")
    c2.metric("Derin Öğrenme Skoru", f"%{dl_score * 100:.1f}" if dl_score is not None else "model yok")
    c3.metric("İçerik (OCR) Riski",
              f"%{report.risk * 100:.1f}" if report is not None and report.has_evidence else "kanıt yok")
    if final is not None:
        text, kind = verdict(final)
        getattr(st, kind)(text)
    else:
        st.info("Karar üretmek için yeterli kanıt yok (model yüklü değil ve OCR doğrulanabilir alan bulamadı).")

    # --- Görsel analiz --------------------------------------------------------------
    st.subheader("🖼️ Görsel Analiz")
    if prob_map is not None:
        col_a, col_b = st.columns(2)
        col_a.image(image, caption="Orijinal belge", use_container_width=True)
        col_b.image(heatmap_overlay(image, prob_map, threshold), caption="Sahte bölge ısı haritası",
                    use_container_width=True)
        area = float((prob_map >= threshold).mean() * 100)
        st.caption(f"Eşik üstü alan: %{area:.2f} · En yüksek piksel olasılığı: {prob_map.max():.2f}")
    else:
        st.image(image, caption="Yüklenen belge", use_container_width=True)

    # --- İçerik analizi -------------------------------------------------------------
    if report is not None:
        st.subheader("📑 İçerik Analizi")
        col_a, col_b = st.columns([3, 2])
        with col_a:
            if report.findings:
                for f in report.findings:
                    {"pass": st.success, "fail": st.error}.get(f.status, st.warning)(f.message)
            else:
                st.info("Belgede IBAN, TCKN veya Matrah/KDV/Toplam alanı bulunamadı.")
            if report.fields:
                st.json({k: v for k, v in report.fields.items()}, expanded=False)
        with col_b:
            st.image(draw_findings(image, report), caption="Kural kutuları (turuncu = ihlal)",
                     use_container_width=True)
        with st.expander("OCR satırları"):
            st.dataframe(pd.DataFrame([{"metin": l.text, "güven": round(l.confidence, 2), "kutu": l.bbox}
                                       for l in lines]), use_container_width=True)


if __name__ == "__main__":
    main()
