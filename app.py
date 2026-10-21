"""
Streamlit web arayüzü: Türkçe finansal belgelerde sahtecilik ve manipülasyon tespiti.

Çalıştırma:
    streamlit run app.py

Çıkarım altyapısı ``predict.py`` ile ortaktır (aynı ön işlem, kayan pencere / 512x512
modları ve ``checkpoints/calibration.json`` sıcaklık + eşik kalibrasyonu). Sayfa:
  1. **Görsel analiz** — orijinal belge | jet olasılık haritası | %60/%40 ısı haritası
     bindirmesi (eşik üstü bölgeler beyaz konturla).
  2. **Özet** — belge olasılığı, maksimum piksel olasılığı, eşik üstü alan ve karar kutusu.
  3. **İçerik analizi (opsiyonel)** — EasyOCR + kural motoru (IBAN Mod-97, TCKN,
     Matrah + KDV = Genel Toplam, KDV oranı) ve hibrit güven skoru.
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

import hashlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from predict import CLEAN_PIXEL_LIMIT, fuse_doc_score, load_calibration, make_overlay, predict
from src.ocr_checker import FAIL, PASS, ContentReport, check_document_text, run_ocr

CHECKPOINT_DIRS = (Path("checkpoints"), Path("runs"))
PRESET_CHECKPOINTS = {
    "checkpoints/finetuned_multilingual.pth": "Stage B — finetuned_multilingual (varsayılan)",
    "checkpoints/base_doctamper.pth": "Stage A — base_doctamper",
}
MODES = {"sliding": "Kayan Pencere (varsayılan)", "resize": "512x512 Hızlı Önizleme"}
PDF_DPI = 200


# ---------------------------------------------------------------------------
# Yükleme yardımcıları
# ---------------------------------------------------------------------------

def find_checkpoints() -> list[str]:
    """Önce Stage B / Stage A ağırlıkları, ardından ``checkpoints/`` ve ``runs/`` altındaki
    diğer boş olmayan .pth dosyaları (en yeni önce)."""
    presets = [p for p in PRESET_CHECKPOINTS if Path(p).is_file()]
    others = [p for d in CHECKPOINT_DIRS if d.is_dir() for p in d.rglob("*.pth") if p.stat().st_size > 0]
    others = [p.as_posix() for p in sorted(others, key=lambda p: p.stat().st_mtime, reverse=True)
              if p.as_posix() not in presets]
    return presets + others


@st.cache_resource(show_spinner="Model yükleniyor...")
def load_detector(path: str) -> tuple[torch.nn.Module, dict, str]:
    """Modeli, yapılandırmasını ve kullanılan cihazı yükler."""
    from src.models import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(path, device)
    return model, config, device


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
# Görselleştirme
# ---------------------------------------------------------------------------

def jet_map(prob_map: np.ndarray) -> np.ndarray:
    """[0, 1] olasılık haritasını sabit ölçekli jet renk haritasına (RGB uint8) çevirir."""
    heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(prob_map, 0.0, 1.0)), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def overlay_with_contours(image: np.ndarray, prob_map: np.ndarray, threshold: float) -> np.ndarray:
    """%60 orijinal + %40 ısı haritası; eşiği geçen bölgeler beyaz konturla çizilir."""
    out = make_overlay(image, prob_map)
    contours, _ = cv2.findContours((prob_map >= threshold).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), max(2, image.shape[1] // 400))
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


# ---------------------------------------------------------------------------
# Arayüz
# ---------------------------------------------------------------------------

def sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        checkpoints = find_checkpoints()
        ckpt = st.selectbox("Model", checkpoints, format_func=lambda p: PRESET_CHECKPOINTS.get(p, p)) \
            if checkpoints else None
        if not checkpoints:
            st.warning("`checkpoints/` veya `runs/` altında eğitilmiş model yok. Yalnızca OCR analizi yapılabilir.")
        mode = st.radio("Çıkarım modu", list(MODES), format_func=MODES.get,
                        help="Kayan pencere SRM gürültü istatistiğini korur; 512x512 mod daha hızlıdır "
                             "ama büyük belgelerde ayrıntı kaybeder.")

        st.subheader("🎚️ Kalibrasyon")
        try:
            file_t, file_thr, calib_path = load_calibration()
            st.caption(f"Kaynak: `{calib_path.as_posix()}` (T={file_t:.3f}, eşik={file_thr:.2f})")
        except (OSError, ValueError) as exc:
            st.error(f"Kalibrasyon dosyası okunamadı: {exc}")
            file_t, file_thr = 1.0, 0.5
        temperature = st.number_input("Temperature (T)", min_value=0.1, max_value=20.0, value=file_t,
                                      step=0.1, format="%.3f",
                                      help="Logit'ler sigmoid'den önce T'ye bölünür: p = sigmoid(logit / T).")
        threshold = st.slider("Optimal Threshold", 0.05, 0.95, float(file_thr), 0.01,
                              help="Bu olasılığın üzerindeki pikseller ve belgeler sahte kabul edilir.")

        st.subheader("📑 İçerik analizi")
        use_ocr = st.toggle("OCR + kural motoru", value=False,
                            help="IBAN, TCKN ve Matrah/KDV/Toplam tutarlılığını denetler (yavaş).")
        ocr_gpu = st.toggle("OCR için GPU", value=torch.cuda.is_available(),
                            disabled=not torch.cuda.is_available() or not use_ocr)
    return {"ckpt": ckpt, "mode": mode, "temperature": float(temperature), "threshold": float(threshold),
            "use_ocr": use_ocr, "ocr_gpu": ocr_gpu}


def run_analysis(image: np.ndarray, settings: dict) -> dict:
    result: dict = {"prob_map": None, "doc_prob": None, "raw_doc_prob": None, "report": None, "lines": []}
    if settings["ckpt"]:
        model, config, device = load_detector(settings["ckpt"])
        with st.spinner("Derin öğrenme analizi..."):
            result["prob_map"], result["raw_doc_prob"] = predict(model, image, config, settings["temperature"],
                                                                 device, settings["mode"])
        result["doc_prob"] = fuse_doc_score(result["raw_doc_prob"], float(result["prob_map"].max()))
    if settings["use_ocr"]:
        with st.spinner("OCR ve kural motoru çalışıyor (ilk çalıştırmada model indirilir)..."):
            result["lines"] = run_ocr(image, gpu=settings["ocr_gpu"])
            result["report"] = check_document_text(result["lines"])
    return result


def show_visual(image: np.ndarray, result: dict, threshold: float) -> None:
    prob_map, doc_prob = result["prob_map"], result["doc_prob"]
    if prob_map is None:
        st.image(image, caption="Yüklenen belge", use_container_width=True)
        return

    st.subheader("🖼️ Görsel Analiz")
    c1, c2, c3 = st.columns(3)
    c1.image(image, caption="1. Orijinal Belge", use_container_width=True)
    c2.image(jet_map(prob_map), caption="2. Sahtecilik Olasılık Haritası (Jet, 0 mavi → 1 kırmızı)",
             use_container_width=True)
    c3.image(overlay_with_contours(image, prob_map, threshold),
             caption="3. Çakıştırılmış Görsel (%60 orijinal / %40 ısı haritası)", use_container_width=True)

    max_pixel = float(prob_map.max())
    area = float((prob_map >= threshold).mean() * 100)
    m1, m2, m3 = st.columns(3)
    raw = result["raw_doc_prob"]
    m1.metric("Belge Sahtecilik Olasılığı", f"%{doc_prob * 100:.1f}",
              delta=f"{(doc_prob - raw) * 100:+.1f} puan (ham %{raw * 100:.1f})" if doc_prob != raw else None,
              delta_color="off",
              help=f"Maks. piksel olasılığı {CLEAN_PIXEL_LIMIT} altındaysa: 0.7 · belge + 0.3 · maks. piksel")
    m2.metric("Maksimum Piksel Olasılığı", f"{max_pixel:.3f}")
    m3.metric("Sahte/Manipüle Edilmiş Alan", f"%{area:.2f}")

    if doc_prob >= threshold or max_pixel >= threshold:
        st.error("⚠️ Belgede manipülasyon/sahtecilik tespiti yapıldı!")
    else:
        st.success("✅ Belgede belirgin bir sahtecilik izine rastlanmadı.")


def show_content(image: np.ndarray, result: dict) -> None:
    report: Optional[ContentReport] = result["report"]
    if report is None:
        return
    st.subheader("📑 İçerik Analizi")
    final = hybrid_score(result["doc_prob"], report)
    h1, h2 = st.columns(2)
    h1.metric("İçerik (OCR) Riski", f"%{report.risk * 100:.1f}" if report.has_evidence else "kanıt yok")
    h2.metric("Hibrit Sahtecilik Olasılığı", f"%{final * 100:.1f}" if final is not None else "—",
              help="1 − (1 − p_model) · (1 − r_OCR)")
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
                                   for l in result["lines"]]), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Document Forensics", page_icon="🔍", layout="wide")
    st.title("🔍 Document Forensics - Sahtecilik ve Manipülasyon Tespit Sistemi")
    st.caption("Dekont · Fatura · Fiş — RGB + SRM gürültü akışlı derin öğrenme modeli, "
               "kalibre olasılıklar ve opsiyonel OCR içerik tutarlılığı")

    settings = sidebar()

    upload = st.file_uploader("Belge yükleyin (JPG, JPEG, PNG, PDF)", type=["jpg", "jpeg", "png", "pdf"])
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

    # Eşik çıkarım sonucunu değiştirmez (yalnızca görselleştirme/karar); diğer ayarlar değişirse
    # önceki sonuç geçersiz sayılır ve yeniden "Analiz Et" gerekir.
    key = (hashlib.sha1(data).hexdigest(), page, settings["ckpt"], settings["mode"], settings["temperature"],
           settings["use_ocr"])
    if st.button("🔎 Analiz Et", type="primary"):
        st.session_state["analysis"] = (key, run_analysis(image, settings))

    stored = st.session_state.get("analysis")
    if stored is None or stored[0] != key:
        if stored is not None:
            st.warning("Ayarlar veya belge değişti; güncel sonuç için yeniden **Analiz Et**'e basın.")
        st.image(image, caption=f"{upload.name} ({image.shape[1]}x{image.shape[0]})", width=420)
        return

    result = stored[1]
    show_visual(image, result, settings["threshold"])
    show_content(image, result)


if __name__ == "__main__":
    main()
