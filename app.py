"""
Streamlit web arayüzü: Türkçe finansal belgelerde sahtecilik ve manipülasyon tespiti.

Çalıştırma:
    streamlit run app.py

Çıkarım altyapısı ``predict.py`` ile ortaktır (aynı ön işlem, kayan pencere / 512x512
modları ve modelin ``calibration.json`` dosyasındaki ayrı belge / piksel kalibrasyonu). Sayfa:
  1. **Görsel analiz** — orijinal belge | jet olasılık haritası | %60/%40 ısı haritası
     bindirmesi (eşik üstü bölgeler beyaz konturla).
  2. **Özet** — belge olasılığı, maksimum piksel olasılığı, eşik üstü alan ve karar kutusu.
  3. **İçerik analizi (opsiyonel)** — EasyOCR + kural motoru (IBAN Mod-97, TCKN,
     Matrah + KDV = Genel Toplam, KDV oranı).
  4. **PDF yapı analizi (PDF yüklendiğinde)** — düzenleme aracı izi, artımlı kayıt, dijital imza
     kapsamı, gizlenmiş/üst üste yazılmış metin, satır içi font tutarsızlığı (``src/pdf_checker.py``).
  5. **Hibrit karar** — model + OCR + PDF (+ isteğe bağlı kopyala-taşı) kanıtlarının noisy-OR birleşimi.
  6. **Açıklanabilirlik (opsiyonel)** — sınıflandırma kafası için Grad-CAM ve gerçek maske yüklenirse
     TP/FP/FN hata haritası (``src/explain.py``).

**Geliştirici modu** (kenar çubuğu, varsayılan kapalı): kapalıyken yalnızca önerilen model (Stage E) kayan
pencere moduyla ve val'de doğrulanmış kalibrasyonla çalışır; sıcaklık/eşik değerleri salt okunur özet olarak
gösterilir. Açıkken karşılaştırma/deneysel modeller, sıcaklık/eşik kutuları ve 512x512 hızlı önizleme görünür;
değerler kalibrasyondan farklıysa "doğrulanmış çalışma noktası dışında" uyarısı verilir. Kalibrasyonu val'de
öğrenilmemiş modeller "kalibre edilmemiş" olarak işaretlenir (``src/model_registry.py``).
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
from dataclasses import replace
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from predict import (CLEAN_PIXEL_LIMIT, Calibration, decide_with, explain_cam, fuse_doc_score,
                     load_calibration_info, make_overlay, predict)
from src.copymove import CopyMoveReport, detect_copy_move, draw_pairs
from src.dataset import limit_long_side
from src.explain import cam_overlay, error_counts, error_map
from src.model_registry import (EXPERIMENTAL, RECOMMENDED, UNCALIBRATED_LABEL, checkpoint_summary, is_calibrated,
                                list_models, reliability_warning)
from src.ocr_checker import FAIL, PASS, ContentReport, check_document_text, run_ocr
from src.pdf_checker import PdfReport, check_pdf
from src.runtime import cuda_available, easyocr_available, ensure_weights

MODES = {"sliding": "Kayan Pencere (varsayılan)", "resize": "512x512 Hızlı Önizleme"}
PDF_DPI = 200


# ---------------------------------------------------------------------------
# Yükleme yardımcıları
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Model ağırlıkları kontrol ediliyor / indiriliyor...")
def fetch_weights() -> dict[str, str]:
    """Dağıtım ortamında eksik ağırlıkları bir kez indirir (bkz. ``src/runtime.py``)."""
    return ensure_weights()


@st.cache_data(show_spinner=False)
def find_models(include_experimental: bool, include_archived: bool = False) -> dict[str, tuple[str, str, str, bool]]:
    """Diskteki modeller: {yol: (etiket, durum, not, kalibre mi)} — sıra ve etiketler ``src/model_registry.py``'den."""
    return {m.path: (m.label, m.status, m.note, is_calibrated(m.path))
            for m in list_models(include_experimental=include_experimental, include_archived=include_archived)}


@st.cache_data(show_spinner=False)
def model_warning(path: str) -> Optional[str]:
    """Seçili modelin validasyon metriklerine göre güvenilirlik uyarısı (checkpoint içinden okunur)."""
    try:
        return reliability_warning(checkpoint_summary(path))
    except Exception:
        return None


@st.cache_resource(show_spinner="Model yükleniyor...")
def load_detector(path: str) -> tuple[torch.nn.Module, dict, str]:
    """Modeli, yapılandırmasını ve kullanılan cihazı yükler."""
    from src.models import load_model

    device = "cuda" if cuda_available() else "cpu"
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


def draw_findings(image: np.ndarray, report: ContentReport, findings: Optional[list] = None) -> np.ndarray:
    """Başarısız kuralların satır kutularını turuncu, başarılı olanları yeşil çerçeveler."""
    out = image.copy()
    for f in report.findings if findings is None else findings:
        if f.bbox and f.bbox[2] > f.bbox[0] and f.status in (PASS, FAIL):
            color = (255, 140, 0) if f.status == FAIL else (0, 170, 0)
            cv2.rectangle(out, f.bbox[:2], f.bbox[2:], color, max(2, image.shape[1] // 500))
    return out


def hybrid_score(dl_score: Optional[float], *reports: Optional[ContentReport | PdfReport | CopyMoveReport]
                 ) -> Optional[float]:
    """
    Hibrit güven skoru — "noisy-OR" birleşimi:

        P(sahte) = 1 − (1 − p_DL) · (1 − r_OCR) · (1 − r_PDF) · (1 − r_CM)

    * p_DL: derin öğrenme modelinin kalibre belge olasılığı.
    * r_OCR: kural motorunun risk skoru (yalnızca *başarısız* kurallardan oluşur).
    * r_PDF: PDF yapı analizinin risk skoru (yalnızca PDF yüklendiğinde).
    * r_CM: kopyala-taşı modülünün risk skoru (isteğe bağlı, varsayılan kapalı).

    Gerekçe: İki kaynak farklı sahtecilik türlerini yakalar. Kural ihlali (ör. geçersiz
    IBAN) piksel izi zayıf olsa bile tek başına güçlü kanıttır; tersine, matematiği
    koruyan "akıllı" bir sahtecilikte kuralların geçmesi pikselde görülen izi
    **azaltmamalıdır** — bu yüzden geçen kurallar skoru düşürmez (r_OCR = 0 → skor = p_DL).
    """
    risks = [r.risk for r in reports if r is not None and r.has_evidence]
    if dl_score is None and not risks:
        return None
    survive = 1.0 - (dl_score or 0.0)
    for risk in risks:
        survive *= 1.0 - risk
    return 1.0 - survive


# ---------------------------------------------------------------------------
# Arayüz
# ---------------------------------------------------------------------------

def calibration_inputs(file_cal: Calibration) -> Calibration:
    """Geliştirici modunda belge/piksel sıcaklığı ve eşik kutuları; değerler kalibrasyondan başlar."""
    c1, c2 = st.columns(2)
    doc_t = c1.number_input("Belge T", 0.1, 20.0, float(file_cal.doc_temperature), 0.1, format="%.3f",
                            help="Belge logit'i sigmoid'den önce T'ye bölünür: p = sigmoid(logit / T).")
    pix_t = c2.number_input("Piksel T", 0.1, 20.0, float(file_cal.pixel_temperature), 0.1, format="%.3f",
                            help="Piksel logit'lerinin sıcaklığı (belge T'sinden ayrı kalibre edilir).")
    doc_thr = st.slider("Belge eşiği", 0.05, 0.95, float(file_cal.doc_threshold), 0.01,
                        help="Hibrit koşul: belge skoru bu eşiği geçmeli (ve haritada hafif iz olmalı).")
    pix_thr = st.slider("Piksel (bölge) eşiği", 0.05, 0.95, float(file_cal.pixel_threshold), 0.01,
                        help="Bölge koşulu: bu olasılığı geçen bağlantılı bölge sayfanın en az "
                             f"%{file_cal.min_region_fraction * 100:.2f}'i olmalı.")
    return replace(file_cal, doc_temperature=float(doc_t), pixel_temperature=float(pix_t),
                   doc_threshold=float(doc_thr), pixel_threshold=float(pix_thr))


def off_operating_point(cal: Calibration, file_cal: Calibration) -> bool:
    """Ayarlar dosyadaki (val'de doğrulanmış) çalışma noktasından farklı mı?"""
    pairs = ((cal.doc_temperature, file_cal.doc_temperature), (cal.pixel_temperature, file_cal.pixel_temperature),
             (cal.doc_threshold, file_cal.doc_threshold), (cal.pixel_threshold, file_cal.pixel_threshold))
    return any(abs(a - b) > 1e-6 for a, b in pairs)


def sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        failed = {k: v for k, v in fetch_weights().items() if v.startswith("indirilemedi")}
        if failed:
            st.warning("Bazı ağırlıklar indirilemedi:\n\n" + "\n".join(f"- `{k}`: {v}" for k, v in failed.items()))
        dev_mode = st.toggle("Geliştirici modu", value=False,
                             help="Karşılaştırma/deneysel modeller, sıcaklık ve eşik ayarları ve 512x512 hızlı "
                                  "önizleme. Kapalıyken yalnızca önerilen model, val'de doğrulanmış çalışma "
                                  "noktasıyla kullanılır.")
        show_experimental = show_archived = False
        if dev_mode:
            show_experimental = st.toggle("Deneysel / eski modelleri de göster", value=False,
                                          help="Ara ve deneme ağırlıkları (Stage C/D, runs/fast_run). Validasyonda "
                                               "neredeyse her belgeye 'sahte' demişlerdir; karar için kullanmayın.")
            show_archived = show_experimental and st.toggle("Arşivdeki modelleri de göster (_arsiv/)", value=False)
        models = find_models(show_experimental, show_archived)
        if not dev_mode:
            models = {k: v for k, v in models.items() if v[1] == RECOMMENDED}

        def label(path: str) -> str:
            text = models[path][0]
            return text if models[path][3] else f"{text} · {UNCALIBRATED_LABEL}"

        ckpt = st.selectbox("Model", list(models), format_func=label, disabled=not dev_mode and len(models) <= 1) \
            if models else None
        if not models:
            st.warning("Önerilen model (Stage E) bulunamadı. Yalnızca OCR analizi yapılabilir."
                       + ("" if dev_mode else " Diğer modeller için geliştirici modunu açın."))
        elif ckpt:
            text, status, note, calibrated = models[ckpt]
            warning = model_warning(ckpt)
            if status == EXPERIMENTAL or (warning and status != RECOMMENDED):
                st.error(f"⚠️ Bu model karar için güvenilir değil: {warning or note}")
            elif note:
                st.caption(note)
            if not calibrated:
                st.warning("Bu model **kalibre edilmemiş**: kalibrasyonu val kümesinde `evaluate.py` ile "
                           "öğrenilmemiş (genel/elle girilmiş değerler). Kararlar doğrulanmış bir çalışma "
                           "noktasına dayanmaz.")
        if dev_mode:
            mode = st.radio("Çıkarım modu", list(MODES), format_func=MODES.get,
                            help="Kayan pencere SRM gürültü istatistiğini korur; 512x512 mod daha hızlıdır "
                                 "ama büyük belgelerde ayrıntı kaybeder ve doğrulanmış çalışma noktasının dışındadır.")
        else:
            mode = "sliding"

        st.subheader("🎚️ Kalibrasyon")
        try:
            file_cal = load_calibration_info(checkpoint=ckpt)
        except (OSError, ValueError) as exc:
            st.error(f"Kalibrasyon dosyası okunamadı: {exc}")
            file_cal = Calibration()
        source = file_cal.path.as_posix() if file_cal.path else "—"
        if dev_mode:
            st.caption(f"Kaynak: `{source}`" + (" (eski tek-T şema)" if file_cal.legacy else ""))
            cal = calibration_inputs(file_cal)
            if off_operating_point(cal, file_cal) or mode != "sliding":
                st.warning("⚠️ Doğrulanmış çalışma noktası dışında: ayarlar val kümesinde seçilen değerlerden "
                           "farklı; yanlış alarm ve tespit oranları raporlanan değerleri tutmayabilir.")
        else:
            cal = file_cal
            st.markdown(
                f"| | Sıcaklık | Eşik |\n|---|---|---|\n"
                f"| Belge | {file_cal.doc_temperature:.3f} | {file_cal.doc_threshold:.2f} |\n"
                f"| Piksel | {file_cal.pixel_temperature:.3f} | {file_cal.pixel_threshold:.2f} |")
            val = (f"val: FPR {file_cal.val_fpr:.3f}, TPR {file_cal.val_tpr:.3f} ({file_cal.val_images} görüntü)"
                   if file_cal.calibrated and file_cal.val_fpr is not None else UNCALIBRATED_LABEL)
            st.caption(f"Salt okunur · `{source}`" + (" · eski tek-T şema" if file_cal.legacy else "") + f" · {val}")

        # OCR ve Grad-CAM her analizde çalışır; OCR, CUDA varsa GPU'yu kendiliğinden kullanır.
        ocr_ok = easyocr_available()
        use_ocr, use_cam, use_copymove = ocr_ok, bool(models), False
        if not ocr_ok:
            st.warning("EasyOCR kurulu değil (`pip install easyocr`); içerik kuralları çalışmayacak.")
        if dev_mode:
            st.subheader("🧩 Analiz katmanları")
            use_ocr = st.toggle("OCR + kural motoru", value=ocr_ok, disabled=not ocr_ok)
            use_cam = st.toggle("Grad-CAM (belge skoru)", value=bool(models), disabled=not models,
                                help="CPU'da analizi uzatır; hızlı denemelerde kapatılabilir.")
            use_copymove = st.toggle("Kopyala-taşı kanıtı (deneysel)", value=False,
                                     help="Belge içinde birebir aynı glif bloklarını arar. Doğrulanmış bir çalışma "
                                          "noktası yoktur; düz dijital zeminde kanıt üretmez. Hibrit skora katılır.")
    return {"ckpt": ckpt, "mode": mode, "cal": cal, "dev_mode": dev_mode, "use_ocr": use_ocr,
            "use_cam": use_cam, "use_copymove": use_copymove}


def run_analysis(image: np.ndarray, settings: dict, pdf_data: Optional[bytes] = None) -> dict:
    result: dict = {"prob_map": None, "doc_prob": None, "raw_doc_prob": None, "report": None, "lines": [],
                    "pdf": None, "cam": None, "copymove": None, "errors": []}
    cal: Calibration = settings["cal"]
    if pdf_data is not None:
        try:
            result["pdf"] = check_pdf(pdf_data, dpi=PDF_DPI)
        except Exception as exc:  # Beklenmedik PDF yapısı: diğer analizler yine de çalışsın
            result["errors"].append(f"PDF yapı analizi başarısız: {exc.__class__.__name__}: {exc}")
    if settings["ckpt"]:
        try:
            model, config, device = load_detector(settings["ckpt"])
            with st.spinner("Derin öğrenme analizi..."):
                result["prob_map"], result["raw_doc_prob"] = predict(model, image, config, cal.doc_temperature,
                                                                     device, settings["mode"],
                                                                     pixel_temperature=cal.pixel_temperature)
            result["doc_prob"] = fuse_doc_score(result["raw_doc_prob"], float(result["prob_map"].max()))
            if settings.get("use_cam"):
                with st.spinner("Grad-CAM hesaplanıyor..."):
                    result["cam"] = explain_cam(model, image, config)
        except Exception as exc:  # Bozuk/eksik ağırlık, bellek yetersizliği: OCR yine de çalışabilsin
            result["prob_map"] = None
            result["errors"].append(f"Derin öğrenme analizi başarısız: {exc.__class__.__name__}: {exc}")
    if settings.get("use_copymove"):
        try:
            with st.spinner("Kopyala-taşı eşleştirmesi..."):
                small, _ = limit_long_side(image, None, 1600)  # modelle aynı ölçek
                result["copymove"] = (detect_copy_move(small), small)
        except Exception as exc:  # Beklenmedik görüntü yapısı: diğer analizler yine de gösterilsin
            result["errors"].append(f"Kopyala-taşı analizi başarısız: {exc.__class__.__name__}: {exc}")
    if settings["use_ocr"]:
        try:
            with st.spinner("OCR ve kural motoru çalışıyor (ilk çalıştırmada model indirilir)..."):
                result["lines"] = run_ocr(image, gpu=cuda_available())
                result["report"] = check_document_text(result["lines"])
        except Exception as exc:  # EasyOCR modeli indirilemedi / kurulu değil
            result["errors"].append(f"OCR çalıştırılamadı: {exc.__class__.__name__}: {exc}")
    return result


def show_visual(image: np.ndarray, result: dict, cal: Calibration) -> None:
    prob_map, doc_prob = result["prob_map"], result["doc_prob"]
    threshold = cal.pixel_threshold  # kontur ve alan piksel eşiğiyle
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

    if decide_with(cal, doc_prob, prob_map):
        st.error("⚠️ Görüntü analizi: belgede manipülasyon/sahtecilik izi bulundu.")
    else:
        st.success("✅ Görüntü analizi: belirgin bir düzenleme izine rastlanmadı. (Baştan sona sahte üretilmiş "
                   "belgelerde piksel izi olmayabilir; içerik ve PDF kontrollerine de bakın.)")
        if doc_prob >= cal.doc_threshold and max_pixel < CLEAN_PIXEL_LIMIT:
            st.info("ℹ️ Belge skoru sınırda/yüksek olsa da piksel seviyesinde sahte bölge izine rastlanmadı.")


def show_explain(image: np.ndarray, result: dict, threshold: float) -> None:
    """Grad-CAM ve (gerçek maske yüklenirse) TP/FP/FN hata haritası."""
    prob_map = result["prob_map"]
    if prob_map is None:
        return
    with st.expander("🧠 Açıklanabilirlik: Grad-CAM ve hata haritası", expanded=result.get("cam") is not None):
        gt_upload = st.file_uploader("Gerçek sahtecilik maskesi (opsiyonel, PNG; beyaz = sahte)", type=["png"],
                                     key="gt_mask")
        cols = st.columns(2)
        if result.get("cam") is not None:
            cols[0].image(cam_overlay(image, result["cam"]), use_container_width=True,
                          caption="Grad-CAM: belge skorunu en çok etkileyen bölgeler (kırmızı = yüksek etki)")
        else:
            cols[0].info("Bu analizde Grad-CAM üretilmedi.")
        if gt_upload is not None:
            gt = cv2.imdecode(np.frombuffer(gt_upload.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
            counts = error_counts(prob_map, gt, threshold)
            cols[1].image(error_map(image, prob_map, gt, threshold), use_container_width=True,
                          caption=f"Hata haritası — yeşil: doğru tespit ({counts['tp']} px), kırmızı: yanlış "
                                  f"alarm ({counts['fp']} px), mavi: kaçırılan ({counts['fn']} px)")
        else:
            cols[1].info("Gerçek maske yüklenirse doğru tespit / yanlış alarm / kaçırılan pikseller gösterilir.")


def show_content(image: np.ndarray, result: dict) -> None:
    report: Optional[ContentReport] = result["report"]
    if report is None:
        return
    st.subheader("📑 İçerik Analizi")
    st.metric("İçerik (OCR) Riski", f"%{report.risk * 100:.1f}" if report.has_evidence else "kanıt yok")
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


def show_pdf(image: np.ndarray, result: dict, page: int) -> None:
    report: Optional[PdfReport] = result.get("pdf")
    if report is None:
        return
    st.subheader("📄 PDF Yapı Analizi")
    info = report.info
    p1, p2, p3 = st.columns(3)
    p1.metric("PDF Yapı Riski", f"%{report.risk * 100:.1f}" if report.has_evidence else "analiz edilemedi")
    p2.metric("Revizyon", str(info.get("revizyon", "—")), help="1 = dosya oluşturulduktan sonra yeniden kaydedilmemiş")
    p3.metric("Dijital İmza", str(info.get("imza_sayisi", 0)))
    col_a, col_b = st.columns([3, 2])
    with col_a:
        order = {FAIL: 0, "warn": 1, PASS: 2}
        for f in sorted(report.findings, key=lambda f: order.get(f.status, 1)):
            {"pass": st.success, "fail": st.error}.get(f.status, st.warning)(f.message)
        st.json({k: info.get(k) for k in ("producer", "creator", "olusturma", "degistirme", "sayfa", "fontlar")},
                expanded=False)
    with col_b:
        marked = report.page_findings(page)
        if marked:
            st.image(draw_findings(image, report, marked),
                     caption=f"Sayfa {page + 1}: PDF bulguları (turuncu = ihlal)", use_container_width=True)


def show_copymove(result: dict) -> None:
    """Kopyala-taşı modülünün şüpheli blok çiftleri (isteğe bağlı)."""
    stored = result.get("copymove")
    if stored is None:
        return
    report, small = stored
    st.subheader("🧬 Kopyala-taşı kanıtı (deneysel)")
    c1, c2 = st.columns([2, 3])
    c1.metric("Kopyala-taşı riski", f"%{report.risk * 100:.1f}",
              help="Belge içinde gürültüsüyle birlikte birebir aynı olan en şüpheli glif çiftinin skoru.")
    c1.caption(f"{report.info.get('glif', 0)} glif bloğu, {report.info.get('karsilastirma', 0)} ayrıntılı karşılaştırma. "
               "Düz (gürültüsüz) dijital zeminde doğal tekrar ile kopya ayırt edilemez; bu durumda risk 0'dır.")
    if report.pairs:
        c1.dataframe(pd.DataFrame([{"kutu_a": p.box_a, "kutu_b": p.box_b, "skor": round(p.score, 2),
                                    "gürültü_korelasyonu": round(p.noise_corr, 2), "zncc": round(p.zncc, 3)}
                                   for p in report.pairs[:10]]), hide_index=True, use_container_width=True)
        c2.image(draw_pairs(small, report), caption="Şüpheli blok çiftleri (aynı renk = eşleşen çift)",
                 use_container_width=True)


def show_verdict(result: dict) -> None:
    """Model, OCR, PDF (ve isteğe bağlı kopyala-taşı) kanıtlarının birleşik skoru (en az iki kaynak varsa)."""
    copymove = result["copymove"][0] if result.get("copymove") else None
    sources = [result["doc_prob"] is not None, result["report"] is not None, result.get("pdf") is not None,
               copymove is not None]
    if sum(sources) < 2:
        return
    final = hybrid_score(result["doc_prob"], result["report"], result.get("pdf"), copymove)
    st.subheader("⚖️ Hibrit Karar")
    st.metric("Hibrit Sahtecilik Olasılığı", f"%{final * 100:.1f}" if final is not None else "—",
              help="1 − (1 − p_model) · (1 − r_OCR) · (1 − r_PDF) · (1 − r_kopyala-taşı); mevcut olmayan kaynak 0 sayılır")


def analysis_tab(settings: dict) -> None:
    upload = st.file_uploader("Belge yükleyin (JPG, JPEG, PNG, PDF)", type=["jpg", "jpeg", "png", "pdf"])
    if upload is None:
        st.info("Analize başlamak için bir dekont, fatura veya fiş yükleyin.")
        return

    data = upload.getvalue()
    page = 0
    is_pdf = upload.name.lower().endswith(".pdf")
    if is_pdf:
        _, n_pages = decode_upload(data, upload.name, 0)
        if n_pages > 1:
            page = st.number_input("PDF sayfası", 1, n_pages, 1) - 1
        st.caption("Not: PDF'in dosya yapısı (üretici yazılım, revizyonlar, fontlar, gizlenmiş metin) ayrıca "
                   "analiz edilir; görüntü modeli sayfanın rasterleştirilmiş hâlini inceler.")
    try:
        image, _ = decode_upload(data, upload.name, page)
    except Exception as exc:  # Bozuk/şifreli dosya
        st.error(f"Dosya okunamadı: {exc}")
        return

    # Eşik çıkarım sonucunu değiştirmez (yalnızca görselleştirme/karar); diğer ayarlar değişirse
    # önceki sonuç geçersiz sayılır ve yeniden "Analiz Et" gerekir.
    cal: Calibration = settings["cal"]
    key = (hashlib.sha1(data).hexdigest(), page, settings["ckpt"], settings["mode"], cal.doc_temperature,
           cal.pixel_temperature, settings["use_ocr"], settings["use_cam"], settings["use_copymove"])
    if st.button("🔎 Analiz Et", type="primary"):
        st.session_state["analysis"] = (key, run_analysis(image, settings, data if is_pdf else None))

    stored = st.session_state.get("analysis")
    if stored is None or stored[0] != key:
        if stored is not None:
            st.warning("Ayarlar veya belge değişti; güncel sonuç için yeniden **Analiz Et**'e basın.")
        st.image(image, caption=f"{upload.name} ({image.shape[1]}x{image.shape[0]})", width=420)
        return

    result = stored[1]
    for message in result.get("errors", []):
        st.warning(message)
    show_visual(image, result, cal)
    show_explain(image, result, cal.pixel_threshold)
    show_content(image, result)
    show_pdf(image, result, page)
    show_copymove(result)
    show_verdict(result)



def main() -> None:
    st.set_page_config(page_title="Document Forensics", page_icon="🔍", layout="wide")
    st.title("🔍 Document Forensics - Sahtecilik ve Manipülasyon Tespit Sistemi")
    st.caption("Dekont · Fatura · Fiş — RGB + SRM gürültü akışlı derin öğrenme modeli, "
               "kalibre olasılıklar ve OCR içerik tutarlılığı")

    settings = sidebar()
    tab_analysis, tab_metrics = st.tabs(["🔎 Belge Analizi", "📊 Model Başarımı ve Metrik Analizi"])
    with tab_analysis:
        analysis_tab(settings)
    with tab_metrics:
        from src.dashboard import render_dashboard

        render_dashboard(settings["cal"], settings["ckpt"], settings["dev_mode"])


if __name__ == "__main__":
    main()
