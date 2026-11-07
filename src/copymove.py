"""
Belge içi kopyala-taşı (copy-move) kanıt modülü — model ağırlığı gerektirmez, CPU'da çalışır.

Neden ayrı bir modül?
---------------------
Kopyala-taşı sahteciliğinde (``forge_real.op_copy_move``) belgenin **kendi** glifi kopyalanır; yapıştırılan
bölge çevresiyle aynı gürültü/sıkıştırma istatistiğini taşıdığı için SRM akışı iz bulamaz (Stage E kırılımında
tespit oranı ≈ %33). Bunun yerine belgede **birebir aynı** iki bölge aranır.

Yöntem (glif bloğu tabanlı eşleştirme)
--------------------------------------
1. **Glif blokları:** Otsu ile mürekkep maskesi çıkarılır; 8-bağlantılı bileşenler (karakter/rakam) metin
   boyutu süzgecinden geçirilir. Her bileşen, ``margin`` piksellik **bağlam halkasıyla** birlikte bir blok olur.
2. **Aday çiftler:** Yalnızca mürekkep kutusu boyutu (en × boy) en fazla ±1 px farklı bloklar karşılaştırılır
   (yeniden boyutlandırılmadan kopyalanan bir glifin kutusu değişmez). İki blok arasındaki ofset
   ``min_offset_ratio · glif boyu``'ndan kısaysa (komşu/örtüşen bloklar) çift atlanır — **en küçük uzaklık kuralı**.
3. **Benzerlik:** bağlamlı blokta ZNCC (sıfır ortalamalı normalize çapraz korelasyon) ≥ ``zncc_min``.
4. **Doğal tekrarların elenmesi:** aynı rakamın belgede normal olarak birkaç kez yazılması da yüksek ZNCC
   verir. İki ek kanıt aranır:

   * **Piksel düzeyinde birebirlik (bağlam halkası):** Kopyalanan blok, glifin çevresindeki kâğıt/sensör/
     sıkıştırma gürültüsünü de taşır; doğal tekrarda bu gürültü bağımsızdır. Halkanın zemin piksellerinde
     (mürekkepten ≥ 2 px uzak) bant geçiren gürültü kalıntılarının korelasyonu (``noise_corr``; kopyada ≈ 1,
     bağımsız gürültüde ≈ 0, genlikten bağımsız, JPEG q80-90 yeniden kaydına görece dayanıklı) ve
     |fark| ≤ ``identity_tol`` oranı (``ring_identity``) ölçülür.
   * **Alt piksel kenar profili:** glifin karanlık ağırlıklı ağırlık merkezi blok içinde alt piksel
     hassasiyetle hesaplanır; rasterleştirmede farklı alt piksel konumuna düşen doğal tekrarlarda kenar
     profili (dolayısıyla ağırlık merkezi) kayar. Fark ``max_phase`` pikselden büyükse çift elenir.

   Gürültüsüz (düz zeminli dijital) belgede halka gürültüsü yoktur; aynı fontla aynı alt piksel konumuna
   yazılmış doğal tekrar ile kopya **ayırt edilemez**. Bu durumda skor ``texture`` (halka gürültü düzeyi)
   ile çarpılarak sıfıra çekilir: modül yanlış alarm üretmek yerine "kanıt yok" der. Ofsetin her iki ekseni
   8'in katıysa (JPEG blok ızgarası hizalı) doğal tekrarın sıkıştırma izi de aynı olabileceği için skor
   ``JPEG_ALIGNED_DISCOUNT`` ile çarpılır.
5. **Ofset kümeleme:** aynı ofseti (±1 px) paylaşan birden çok şüpheli çift (bölge kopyası: birkaç rakam
   birlikte taşınmış) skoru artırır.

Çıktı: şüpheli blok çiftleri (``CopyMovePair``) ve ``risk`` = en yüksek çift skoru ([0, 1]).
``app.py`` hibrit skoruna isteğe bağlı kanıt olarak (noisy-OR) girer; karar eşiği
``COPYMOVE_DECISION_RISK`` sabittir (test kümesine bakılarak seçilmemiştir).

Örnek:
    from src.copymove import detect_copy_move
    report = detect_copy_move(image_rgb)
    print(report.risk, [(p.box_a, p.box_b, round(p.score, 2)) for p in report.pairs])
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

Box = tuple[int, int, int, int]

#: Bu riskin üzerindeki eşleşme kararda tek başına "sahte" için yeterli sayılır (ablasyon / app).
COPYMOVE_DECISION_RISK = 0.8
#: Ofseti JPEG 8x8 ızgarasına hizalı çiftlerde skor çarpanı (doğal tekrar da aynı sıkıştırma izini taşıyabilir).
JPEG_ALIGNED_DISCOUNT = 0.5
#: Kova başına aday sayısı sınırı (aynı boyutlu çok sayıda glif; çift sayısı n² büyür).
MAX_BUCKET = 120


@dataclass
class CopyMovePair:
    """Şüpheli blok çifti; kutular (x0, y0, x1, y1) mürekkep kutusudur (bağlam halkası hariç)."""

    box_a: Box
    box_b: Box
    offset: tuple[int, int]
    zncc: float
    ring_identity: float
    noise_corr: float
    phase_diff: float
    texture: float
    score: float


@dataclass
class CopyMoveReport:
    """Kopyala-taşı raporu (``ContentReport`` / ``PdfReport`` ile aynı ``risk`` / ``has_evidence`` arayüzü)."""

    pairs: list[CopyMovePair] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    analyzed: bool = True

    @property
    def has_evidence(self) -> bool:
        return self.analyzed

    @property
    def risk(self) -> float:
        return max((p.score for p in self.pairs), default=0.0)


def _components(gray: np.ndarray, min_height: int, max_height: int) -> tuple[list[Box], np.ndarray]:
    """Metin boyutundaki mürekkep bileşenlerinin kutuları ve **zemin** maskesi.

    Zemin: mürekkepten en az 2 piksel uzak pikseller. Halka gürültüsü ve birebirlik yalnızca burada
    ölçülür; aksi hâlde komşu glif kenarlarının yumuşatması (anti-aliasing) gürültü sanılır.
    """
    gray8 = np.clip(gray, 0, 255).astype(np.uint8)
    thr, ink = cv2.threshold(gray8, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if ink.mean() > 0.4:  # Koyu zeminli/çok yoğun görüntü: metin bileşeni varsayımı geçersiz
        return [], np.zeros_like(ink, bool)
    # Yumuşatılmış kenarlar Otsu eşiğinin üstünde kalabilir: zemin için daha açık bir eşik + genişletme.
    light = gray8 > (int(thr) + int(np.median(gray8))) / 2
    background = cv2.erode(light.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes = []
    for x, y, w, h, area in stats[1:]:
        if min_height <= h <= max_height and 2 <= w <= 2.5 * h and area >= 12:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes, background


def _centroid(patch: np.ndarray) -> tuple[float, float]:
    """Karanlık ağırlıklı ağırlık merkezi (alt piksel; kenar profilinin konumu)."""
    weight = np.clip(255.0 - patch, 0, None)
    weight = weight - weight.min()
    total = weight.sum()
    if total <= 0:
        return patch.shape[1] / 2, patch.shape[0] / 2
    ys, xs = np.mgrid[:patch.shape[0], :patch.shape[1]]
    return float((weight * xs).sum() / total), float((weight * ys).sum() / total)


def detect_copy_move(image: np.ndarray, min_height: int = 6, max_height: int = 100, margin: Optional[int] = None,
                     zncc_min: float = 0.97, identity_tol: float = 2.0, max_phase: float = 0.15,
                     min_offset_ratio: float = 1.5, texture_floor: float = 0.6, texture_full: float = 2.0
                     ) -> CopyMoveReport:
    """Belge içi birebir benzer glif bloklarını arar (bkz. modül açıklaması).

    Args:
        image: RGB (H, W, 3) ya da gri (H, W) uint8.
        margin: bağlam halkası genişliği (piksel); verilmezse glif boyunun ~%25'i (en az 2).
        texture_floor / texture_full: halka gürültüsü (yüksek geçiren kalıntının std'si) bu aralıkta 0 → 1
            ağırlığa çevrilir; düz zeminde (dijital) kanıt sıfırlanır.
    """
    gray = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image).astype(np.float32)
    h_img, w_img = gray.shape
    boxes, background = _components(gray, min_height, max_height)
    # Gürültü düzeyi: 3x3 yüksek geçiren kalıntı (yalnız zeminde okunur).
    residual = gray - cv2.blur(gray, (3, 3))
    # Korelasyon için bant geçiren kalıntı (σ=1 − σ=3): JPEG yeniden sıkıştırmasından sonra da gürültünün orta
    # frekansları korunur (q90'da kopya ≈ 0.9, bağımsız ≈ 0.1). Mürekkep pikselleri önce zemin medyanıyla
    # doldurulur; aksi hâlde aynı glifin bulanık gölgesi doğal tekrarlarda da yüksek korelasyon verirdi.
    filled = np.where(background, gray, float(np.median(gray[background])) if background.any() else 0.0)
    band = cv2.GaussianBlur(filled, (0, 0), 1.0) - cv2.GaussianBlur(filled, (0, 0), 3.0)
    report = CopyMoveReport(info={"glif": len(boxes), "karsilastirma": 0})
    # Kova: kutu boyutu (en, boy) // 2. JPEG kenar eşiklemesini ±1 px oynatabildiği için komşu kovalar da
    # karşılaştırılır; yamalar kutu merkezine hizalanır ve iki kutunun büyük boyutuyla kesilir.
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k, b in enumerate(boxes):
        buckets[((b[2] - b[0]) // 2, (b[3] - b[1]) // 2)].append(k)
    for key in buckets:
        buckets[key] = buckets[key][:MAX_BUCKET]

    def crop(arr: np.ndarray, center: tuple[int, int], size: tuple[int, int]) -> Optional[np.ndarray]:
        (cx, cy), (pw, ph) = center, size
        x0, y0 = cx - pw // 2, cy - ph // 2
        if x0 < 0 or y0 < 0 or x0 + pw > w_img or y0 + ph > h_img:
            return None
        return arr[y0:y0 + ph, x0:x0 + pw]

    # Hızlı ön eleme: her glifin 8x8 gri tanımlayıcısı (sıfır ortalama, birim norm); ayrıntılı karşılaştırma
    # yalnızca tanımlayıcı benzerliği ≥ 0.9 çiftlerde yapılır.
    desc = np.zeros((len(boxes), 64), np.float32)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        d = cv2.resize(gray[y0:y1, x0:x1], (8, 8), interpolation=cv2.INTER_AREA).ravel()
        d = d - d.mean()
        desc[k] = d / (np.linalg.norm(d) + 1e-6)

    candidates: list[CopyMovePair] = []
    for (kw, kh), members in buckets.items():
        neighbours = np.array([j for dw in (-1, 0, 1) for dh in (-1, 0, 1) for j in buckets.get((kw + dw, kh + dh), [])])
        if len(neighbours) < 2:
            continue
        sims = desc[members] @ desc[neighbours].T
        for mi, i in enumerate(members):
            for j in neighbours[(sims[mi] >= 0.9) & (neighbours > i)]:
                a, b = boxes[i], boxes[int(j)]
                wa, ha, wb, hb = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
                if abs(wa - wb) > 1 or abs(ha - hb) > 1:
                    continue
                gw, gh = max(wa, wb), max(ha, hb)
                ca, cb = ((a[0] + a[2]) // 2, (a[1] + a[3]) // 2), ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
                dx, dy = cb[0] - ca[0], cb[1] - ca[1]
                if np.hypot(dx, dy) < min_offset_ratio * max(gw, gh):
                    continue
                m = margin if margin is not None else max(2, gh // 4)
                size = (gw + 2 * m, gh + 2 * m)
                pa = crop(gray, ca, size)
                if pa is None:
                    continue
                report.info["karsilastirma"] += 1
                # Kutu merkezleri JPEG/eşikleme yüzünden 1 px kayabilir: ±1 px tamsayı hizalama araması.
                fa = pa - pa.mean()
                zncc, cb_best = -1.0, None
                for sx in (-1, 0, 1):
                    for sy in (-1, 0, 1):
                        cand = crop(gray, (cb[0] + sx, cb[1] + sy), size)
                        if cand is None:
                            continue
                        fb = cand - cand.mean()
                        value = float((fa * fb).sum() / (np.sqrt((fa * fa).sum() * (fb * fb).sum()) + 1e-6))
                        if value > zncc:
                            zncc, cb_best = value, (cb[0] + sx, cb[1] + sy)
                if cb_best is None or zncc < zncc_min:
                    continue
                cb = cb_best
                dx, dy = cb[0] - ca[0], cb[1] - ca[1]
                pb = crop(gray, cb, size)
                inner = (slice(m, -m), slice(m, -m))
                (xa, ya), (xb, yb) = _centroid(pa[inner]), _centroid(pb[inner])
                phase = float(np.hypot(xa - xb, ya - yb))
                if phase > max_phase:
                    continue
                ring = np.ones(pa.shape, bool)
                ring[inner] = False
                ring_a, ring_b = ring & crop(background, ca, size), ring & crop(background, cb, size)
                both = ring_a & ring_b
                if both.sum() < 20:
                    continue
                ring_identity = float((np.abs(pa - pb)[both] <= identity_tol).mean())
                # Gürültü kalıntılarının korelasyonu: kopyada ≈ 1, bağımsız gürültüde ≈ 0. Genlikten
                # bağımsızdır; düşük genlikli JPEG izinde |fark| toleransı doğal tekrarları da "aynı" sayardı.
                ra, rb = crop(band, ca, size)[both], crop(band, cb, size)[both]
                ra, rb = ra - ra.mean(), rb - rb.mean()
                noise_corr = float((ra * rb).sum() / (np.sqrt((ra * ra).sum() * (rb * rb).sum()) + 1e-6))
                texture = float(min(crop(residual, ca, size)[ring_a].std(), crop(residual, cb, size)[ring_b].std()))
                noise_weight = float(np.clip((texture - texture_floor) / (texture_full - texture_floor), 0, 1))
                copy_weight = float(np.clip((noise_corr - 0.5) / 0.4, 0, 1))
                score = noise_weight * copy_weight
                if dx % 8 == 0 and dy % 8 == 0:
                    score *= JPEG_ALIGNED_DISCOUNT
                candidates.append(CopyMovePair(a, b, (int(dx), int(dy)), zncc, ring_identity, noise_corr, phase,
                                               texture, score))

    # Ofset kümeleme: aynı ofseti paylaşan şüpheli çiftler birbirini destekler (bölge kopyası).
    suspicious = [p for p in candidates if p.score > 0.3]
    for p in suspicious:
        support = sum(1 for q in suspicious if abs(q.offset[0] - p.offset[0]) <= 1 and abs(q.offset[1] - p.offset[1]) <= 1)
        if support > 1:
            p.score = float(min(1.0, p.score + 0.15 * (support - 1)))
    report.pairs = sorted((p for p in candidates if p.score > 0), key=lambda p: -p.score)
    report.info["aday_cift"] = len(candidates)
    return report


def draw_pairs(image: np.ndarray, report: CopyMoveReport, min_score: float = 0.3, top: int = 10) -> np.ndarray:
    """Şüpheli çiftleri aynı renkte kutu ve bağlantı çizgisiyle işaretler (RGB uint8)."""
    out = image.copy()
    palette = [(255, 140, 0), (220, 20, 60), (30, 144, 255), (50, 205, 50), (186, 85, 211)]
    width = max(2, image.shape[1] // 600)
    for k, p in enumerate([p for p in report.pairs if p.score >= min_score][:top]):
        color = palette[k % len(palette)]
        for box in (p.box_a, p.box_b):
            cv2.rectangle(out, box[:2], box[2:], color, width)
        ca = ((p.box_a[0] + p.box_a[2]) // 2, (p.box_a[1] + p.box_a[3]) // 2)
        cb = ((p.box_b[0] + p.box_b[2]) // 2, (p.box_b[1] + p.box_b[3]) // 2)
        cv2.line(out, ca, cb, color, max(1, width // 2))
    return out
