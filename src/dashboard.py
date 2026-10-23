"""
``app.py`` → "📊 Model Başarımı ve Metrik Analizi" sekmesi.

Tüm grafikler gerçek değerlendirme çıktılarından çizilir; uydurma/örnek veri kullanılmaz:

* ``results/summary.csv``, ``summary.json``, ``breakdown_*.csv`` — ``evaluate.py`` çıktıları (val'de
  seçilen eşikle test/OOD kararları ve sahtecilik türü / kurum / kanal kırılımı). Sekmenin en üstünde.
* ``results/stageB/``  — ``benchmark.py`` çıktıları (görsel başına skorlar, piksel sayımları,
  piksel örneklemi, JPEG dayanıklılığı, opsiyonel OCR).
* ``runs/**/history.csv``, ``checkpoints/**/history.csv`` — ``train.py``'nin epoch günlükleri.
* Checkpoint içindeki son epoch metrikleri ve ``checkpoints/synthetic/splits.csv``,
  ``data/4_synthetic_tr_en/metadata.csv`` — veri seti dağılımı.

Bir kaynak yoksa ilgili grafik yerine onu üretecek komut gösterilir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predict import CLEAN_PIXEL_LIMIT, DOC_WEIGHT, MIN_REGION_FRACTION
from src.metrics import expected_calibration_error, fit_temperature, roc_auc, sigmoid

RESULTS_DIR = Path("results/stageB")
EVAL_DIR = Path("results")
#: Bu sayının altında sahte örnek içeren kırılım satırları "küçük örneklem" diye işaretlenir.
SMALL_SAMPLE = 10
HISTORY_GLOBS = ("runs/**/history.csv", "checkpoints/**/history.csv")
SYN_META = Path("data/4_synthetic_tr_en/metadata.csv")
STAGEB_SPLITS = Path("checkpoints/synthetic/splits.csv")

# Kategorik sıra sabittir (referans palet, ışık modu): 1 mavi, 2 turuncu, 3 su yeşili, 4 sarı
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MUTED = "#8a8984"
LABEL_NAMES = {0: "Orijinal", 1: "Sahte"}
LABEL_COLORS = {"Orijinal": SERIES[0], "Sahte": SERIES[1]}
SUBSET_NAMES = {"synthetic_test": "Sentetik test", "synthetic_unseen": "Sentetik (görülmemiş)",
                "aiforgedoc_ood": "AIForge-Doc (OOD)"}
SUBSET_COLORS = dict(zip(SUBSET_NAMES.values(), SERIES))
SEQUENTIAL = "Blues"
DIVERGING = [[0.0, SERIES[1]], [0.5, "#d9d8d3"], [1.0, SERIES[0]]]


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------

def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_results(directory: str, stamp: float) -> Optional[dict]:
    """``benchmark.py`` çıktıları; ``stamp`` dosya değişince önbelleği geçersiz kılar."""
    d = Path(directory)
    if not (d / "per_image.csv").exists():
        return None
    counts = np.load(d / "pixel_counts.npz")
    samples = np.load(d / "pixel_samples.npz")
    out = {
        "per_image": pd.read_csv(d / "per_image.csv"),
        "counts": counts["counts"], "thresholds": counts["thresholds"],
        # Eski çıktılarda yok: ``decisions`` bu durumda maks. piksel kuralına düşer
        "max_region": counts["max_region"] if "max_region" in counts.files else None,
        "px_prob": samples["prob"].astype(np.float32), "px_label": samples["label"], "px_image": samples["image"],
        "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
        "robust": pd.read_csv(d / "robustness.csv") if (d / "robustness.csv").exists() else None,
    }
    return out


@st.cache_data(show_spinner=False)
def load_evaluation(directory: str, stamp: float) -> Optional[dict]:
    """``evaluate.py`` çıktıları (özet + kırılımlar); ``stamp`` dosya değişince önbelleği geçersiz kılar."""
    d = Path(directory)
    if not (d / "summary.csv").exists():
        return None
    meta = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else {}
    return {
        "summary": pd.read_csv(d / "summary.csv", index_col=0),
        "meta": meta,
        "breakdowns": {p.stem.removeprefix("breakdown_"): pd.read_csv(p) for p in sorted(d.glob("breakdown_*.csv"))},
    }


@st.cache_data(show_spinner=False)
def load_checkpoint_metrics(paths: tuple[str, ...]) -> pd.DataFrame:
    import torch

    rows = {}
    for p in paths:
        if Path(p).exists():
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            rows[Path(p).stem] = {"epoch": ckpt.get("epoch"), **ckpt.get("metrics", {})}
    return pd.DataFrame(rows)


def find_histories() -> list[Path]:
    return sorted({p for g in HISTORY_GLOBS for p in Path(".").glob(g)})


# ---------------------------------------------------------------------------
# Metrik yardımcıları
# ---------------------------------------------------------------------------

def doc_scores(df: pd.DataFrame, temperature: float) -> pd.Series:
    """``app.py`` ile aynı belge skoru: sigmoid(logit/T), piksel haritası temizse birleştirilir."""
    p = sigmoid(df["doc_logit"].to_numpy() / temperature)
    fused = np.where(df["max_pixel"] < CLEAN_PIXEL_LIMIT, DOC_WEIGHT * p + (1 - DOC_WEIGHT) * df["max_pixel"], p)
    return pd.Series(fused, index=df.index)


def region_at(res: dict, sel: np.ndarray, ti: int) -> Optional[np.ndarray]:
    """Seçili görsellerin ``thresholds[ti]`` eşiğindeki en büyük bölge oranı (eski çıktılarda None)."""
    return None if res["max_region"] is None else res["max_region"][sel, ti]


def decisions(score: pd.Series, max_pixel: pd.Series, region: Optional[np.ndarray], threshold: float) -> pd.Series:
    """``predict.is_forged`` kuralının vektörel hâli.

    ``region``: eşiği geçen en büyük bağlantılı bölgenin sayfa oranı (``benchmark.py`` → ``max_region``).
    Eşik ızgarası 0.05 adımlıdır; ızgara dışı eşiklerde en yakın ızgara eşiğinin bölge oranı kullanılır.
    ``None`` ise (eski çıktılar) piksel koşulu eski ``max_pixel >= threshold`` kuralına düşer.
    """
    hybrid = (score >= threshold) & (max_pixel >= CLEAN_PIXEL_LIMIT)
    pixel = max_pixel >= threshold if region is None else np.asarray(region) >= MIN_REGION_FRACTION
    return hybrid | pixel


def f1_iou(tp, fp, fn):
    tp, fp, fn = (np.asarray(x, float) for x in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)


def roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr


def pr_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(labels, scores)
    return recall, precision, float(average_precision_score(labels, scores))


def _layout(fig: go.Figure, title: str, x: str, y: str, height: int = 360) -> go.Figure:
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=height + 60,
                      margin=dict(l=10, r=10, t=50, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0, title_text=""))
    return fig


def _line(fig: go.Figure, x, y, name: str, color: str, dash: Optional[str] = None, hover: str = "") -> None:
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=2, dash=dash),
                             hovertemplate=hover or f"{name}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>"))


def _diagonal(fig: go.Figure) -> None:
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Rastgele / ideal",
                             line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip"))


def _show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def _heatmap(z: np.ndarray, x: list[str], y: list[str], text: np.ndarray, title: str, colorscale,
             zmin=None, zmax=None, zmid=None, hover: str = "%{y} → %{x}<br>%{text}<extra></extra>") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, text=text, colorscale=colorscale, zmin=zmin, zmax=zmax, zmid=zmid,
                               hovertemplate=hover, xgap=2, ygap=2))
    # Hücre metni anotasyonla yazılır: texttemplate, metindeki "%" karakterlerini biçim belirteci sanıp bozar.
    lo = np.nanmin(z) if zmin is None else zmin
    hi = np.nanmax(z) if zmax is None else zmax
    for i, row_name in enumerate(y):
        for j, col_name in enumerate(x):
            strength = abs(z[i, j]) if zmid is not None else (z[i, j] - lo) / max(hi - lo, 1e-9)
            fig.add_annotation(x=col_name, y=row_name, text=str(text[i, j]), showarrow=False,
                               font=dict(color="#ffffff" if strength > 0.55 else "#0b0b0b", size=12))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, "", "", height=340)


# ---------------------------------------------------------------------------
# Bölümler
# ---------------------------------------------------------------------------

def section_evaluation(ev: dict) -> None:
    """``evaluate.py`` özeti: küme başına uygulama kararının başarımı ve kırılım tabloları."""
    meta, summary = ev["meta"], ev["summary"]
    st.subheader("🧪 Değerlendirme sonuçları (evaluate.py)")
    if meta:
        st.caption(f"Kaynak: `{EVAL_DIR.as_posix()}/summary.csv` · T={meta.get('temperature', float('nan')):.3f} · "
                   f"karar eşiği={meta.get('threshold', float('nan')):.2f} (val'de hedef FPR ≤ "
                   f"{meta.get('target_fpr', float('nan')):.2f}; val FPR={meta.get('val_fpr', float('nan')):.3f}, "
                   f"TPR={meta.get('val_tpr', float('nan')):.3f}). Kararlar uygulamadaki `is_forged` kuralıdır; "
                   "yan paneldeki eşik bu bölümü değiştirmez.")
    for key, row in summary.iterrows():
        st.markdown(f"**{key}**")
        k = st.columns(5)
        k[0].metric("Görsel (sahte / orijinal)",
                    f"{int(row['n_images'])} ({int(row['n_forged'])} / {int(row['n_images'] - row['n_forged'])})")
        k[1].metric("Belge AUC", f"{row['image_auc']:.3f}" if pd.notna(row.get("image_auc")) else "—")
        k[2].metric("Tespit oranı (TPR)", f"%{row['decision_tpr'] * 100:.1f}" if pd.notna(row.get("decision_tpr")) else "—")
        k[3].metric("Yanlış alarm (FPR)", f"%{row['decision_fpr'] * 100:.1f}" if pd.notna(row.get("decision_fpr")) else "—")
        k[4].metric("Piksel F1", f"{row['pixel_f1']:.3f}")

    if not ev["breakdowns"]:
        st.info("Sahtecilik türü kırılımı yok. Üretmek için:\n\n```\npython evaluate.py --breakdown-only --out results\n```")
        return
    c1, c2 = st.columns(2)
    key = c1.selectbox("Kırılım kümesi", list(ev["breakdowns"]))
    table = ev["breakdowns"][key]
    dim = c2.selectbox("Boyut", list(table["boyut"].unique()))
    view = table[table["boyut"] == dim].drop(columns="boyut").copy()
    view["örneklem"] = np.where(view["n_sahte"].between(1, SMALL_SAMPLE - 1), "küçük", "")
    st.dataframe(view, hide_index=True, use_container_width=True, column_config={
        "deger": st.column_config.TextColumn(dim),
        "tespit_orani": st.column_config.ProgressColumn("Tespit oranı", min_value=0.0, max_value=1.0, format="%.2f"),
        "yanlis_alarm": st.column_config.ProgressColumn("Yanlış alarm", min_value=0.0, max_value=1.0, format="%.2f"),
        "auc": st.column_config.NumberColumn("AUC", format="%.3f"),
        "ort_belge_olasiligi": st.column_config.NumberColumn("Ort. belge olasılığı", format="%.3f"),
    })
    weak = view[(view["n_sahte"] >= 1) & (view["tespit_orani"] < 0.5)]
    if not weak.empty:
        st.warning("Kör nokta (tespit oranı < %50): " + ", ".join(
            f"{r.deger} (%{r.tespit_orani * 100:.0f}, n={r.n_sahte})" for r in weak.itertuples()))
    st.caption(f"n_sahte < {SMALL_SAMPLE} olan satırlar 'küçük örneklem'dir; oranlar büyük belirsizlik taşır.")


def _no_results() -> None:
    st.info("Değerlendirme sonuçları bulunamadı (`results/stageB/`). Grafikleri üretmek için:\n\n"
            "```\npython benchmark.py            # ~15 dk (CPU)\npython benchmark.py --ocr      # + OCR kural motoru\n```")


def section_performance(df: pd.DataFrame, res: dict, sel: np.ndarray, threshold: float, temperature: float) -> None:
    st.markdown("### A. Model Performansı ve Kalibrasyon")
    y = df["label"].to_numpy()
    score = df["doc_score"].to_numpy()
    px_mask = np.isin(res["px_image"], sel)
    px_p, px_y = res["px_prob"][px_mask], res["px_label"][px_mask]
    has_px_pos = px_y.any() and not px_y.all()

    # 1. ROC ve PR -----------------------------------------------------------------------
    c1, c2 = st.columns(2)
    fig = go.Figure()
    if len(np.unique(y)) == 2:
        fpr, tpr = roc_points(score, y)
        _line(fig, fpr, tpr, f"Belge (AUC={roc_auc(score, y):.3f})", SERIES[0])
    if has_px_pos:
        fpr, tpr = roc_points(px_p, px_y)
        _line(fig, fpr, tpr, f"Piksel (AUC={roc_auc(px_p, px_y):.3f})", SERIES[1])
    _diagonal(fig)
    with c1:
        _show(_layout(fig, "1a · ROC eğrisi", "Yanlış pozitif oranı", "Doğru pozitif oranı"))
    fig = go.Figure()
    if len(np.unique(y)) == 2:
        r, p, ap = pr_points(score, y)
        _line(fig, r, p, f"Belge (AP={ap:.3f})", SERIES[0])
        fig.add_hline(y=y.mean(), line=dict(color=SERIES[0], width=1, dash="dot"))
    if has_px_pos:
        r, p, ap = pr_points(px_p, px_y)
        _line(fig, r, p, f"Piksel (AP={ap:.3f})", SERIES[1])
        fig.add_hline(y=px_y.mean(), line=dict(color=SERIES[1], width=1, dash="dot"))
    with c2:
        _show(_layout(fig, "1b · Precision-Recall eğrisi", "Recall", "Precision"))
    st.caption(f"Piksel eğrileri, görsel başına düzgün rastgele örneklenen {len(px_p):,} piksel üzerinden "
               "(noktalı yatay çizgiler: rastgele sınıflandırıcının PR düzeyi = pozitif oranı).")

    # 2. Eşik ayarı ------------------------------------------------------------------------
    c1, c2 = st.columns(2)
    thr = res["thresholds"]
    counts = res["counts"][sel].sum(axis=0)  # (T, 4)
    pf1, piou = f1_iou(counts[:, 0], counts[:, 1], counts[:, 2])
    doc_f1 = []
    for i, t in enumerate(thr):
        pred = decisions(df["doc_score"], df["max_pixel"], region_at(res, sel, i), t).to_numpy()
        tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
        doc_f1.append(f1_iou(tp, fp, fn)[0])
    fig = go.Figure()
    _line(fig, thr, pf1, "Piksel F1", SERIES[0])
    _line(fig, thr, piou, "Piksel IoU", SERIES[1])
    _line(fig, thr, doc_f1, "Belge F1 (karar kuralı)", SERIES[2])
    fig.add_vline(x=threshold, line=dict(color=MUTED, width=1, dash="dash"))
    fig.add_annotation(x=threshold, y=0.97, yref="paper", text=f"seçili eşik {threshold:.2f}", showarrow=False,
                       xanchor="left", xshift=4, font=dict(size=11))
    if np.isfinite(pf1).any():
        best = thr[int(np.nanargmax(pf1))]
        fig.add_vline(x=best, line=dict(color=SERIES[0], width=1, dash="dot"))
        fig.add_annotation(x=best, y=0.87, yref="paper", text=f"en iyi piksel F1 @ {best:.2f}", showarrow=False,
                           xanchor="left", xshift=4, font=dict(size=11))
    fig.update_yaxes(range=[0, 1])
    with c1:
        _show(_layout(fig, "2 · Eşik ayarı: F1 / IoU – eşik", "Eşik", "Skor"))

    # 3. Kalibrasyon -----------------------------------------------------------------------
    logits = df["doc_logit"].to_numpy()
    temps = {"T = 1.0 (kalibrasyonsuz)": 1.0, "T = 6.389": 6.389}
    if not any(np.isclose(temperature, v) for v in temps.values()):
        temps[f"T = {temperature:.3f} (seçili)"] = temperature
    fitted = fit_temperature(logits, y) if len(np.unique(y)) == 2 else None
    if fitted is not None:
        temps[f"T = {fitted:.3f} (bu kümede uydurulmuş*)"] = fitted
    fig = go.Figure()
    _diagonal(fig)
    for (name, t), color in zip(temps.items(), SERIES):
        probs = sigmoid(logits / t)
        ece, table = expected_calibration_error(probs, y, n_bins=10)
        table = table[table["count"] > 0]
        fig.add_trace(go.Scatter(x=table["mean_confidence"], y=table["fraction_forged"], name=f"{name} · ECE={ece:.3f}",
                                 mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=8),
                                 customdata=table["count"],
                                 hovertemplate="güven=%{x:.2f}<br>sahte oranı=%{y:.2f}<br>n=%{customdata}<extra></extra>"))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    with c2:
        _show(_layout(fig, "3 · Kalibrasyon (reliability) eğrisi — belge skoru", "Ortalama güven",
                      "Gerçek sahte oranı"))
    st.caption("*Uydurulmuş sıcaklık aynı küme üzerinde hem öğrenilip hem ölçüldüğü için iyimserdir; "
               "üretimde validasyon kümesinden `evaluate.py` ile öğrenin. Sıcaklık sıralamayı değiştirmez "
               "(ROC/AUC aynı kalır), yalnızca olasılıkların güvenilirliğini etkiler.")

    # 4. Skor dağılımı ---------------------------------------------------------------------
    long = pd.concat([
        pd.DataFrame({"Skor": df["doc_score"], "Sınıf": df["label"].map(LABEL_NAMES), "Ölçüt": "Belge skoru"}),
        pd.DataFrame({"Skor": df["max_pixel"], "Sınıf": df["label"].map(LABEL_NAMES), "Ölçüt": "Maks. piksel"}),
    ])
    fig = px.violin(long, x="Ölçüt", y="Skor", color="Sınıf", box=True, points="all", color_discrete_map=LABEL_COLORS,
                    category_orders={"Sınıf": ["Orijinal", "Sahte"]})
    fig.add_hline(y=threshold, line=dict(color=MUTED, width=1, dash="dash"), annotation_text=f"eşik {threshold:.2f}")
    fig.update_traces(marker=dict(size=8), line=dict(width=2))
    _show(_layout(fig, "4 · Orijinal ve sahte belgelerin skor dağılımı", "", "Olasılık", height=400))


def section_matrices(df: pd.DataFrame, res: dict, sel: np.ndarray, threshold: float) -> None:
    st.markdown("### B. Karar Matrisleri ve Korelasyonlar")
    y = df["label"].to_numpy()
    c1, c2 = st.columns(2)

    # 5a. Belge düzeyi karmaşıklık matrisi -------------------------------------------------
    ti = int(np.argmin(np.abs(res["thresholds"] - threshold)))
    pred = decisions(df["doc_score"], df["max_pixel"], region_at(res, sel, ti), threshold).to_numpy()
    cm = np.array([[((y == a) & (pred == b)).sum() for b in (0, 1)] for a in (0, 1)])
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    text = np.array([[f"{cm[i, j]}<br>(%{pct[i, j] * 100:.0f})" for j in range(2)] for i in range(2)])
    with c1:
        _show(_heatmap(pct, ["Tahmin: Orijinal", "Tahmin: Sahte"], ["Gerçek: Orijinal", "Gerçek: Sahte"], text,
                       f"5a · Belge düzeyi karmaşıklık matrisi (karar kuralı, eşik {threshold:.2f})", SEQUENTIAL, 0, 1))

    # 5b. Piksel düzeyi --------------------------------------------------------------------
    tp, fp, fn, tn = res["counts"][sel, ti].sum(axis=0)
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    text = np.array([[f"{int(cm[i, j]):,}<br>(%{pct[i, j] * 100:.2f})" for j in range(2)] for i in range(2)])
    with c2:
        _show(_heatmap(pct, ["Tahmin: Temiz", "Tahmin: Sahte"], ["Gerçek: Temiz", "Gerçek: Sahte"], text,
                       f"5b · Piksel düzeyi karmaşıklık matrisi (eşik {res['thresholds'][ti]:.2f})", SEQUENTIAL, 0, 1))
    st.caption("Renk satır yüzdesidir (her gerçek sınıf kendi içinde %100); hücrelerde sayı ve satır yüzdesi.")

    # 6. Korelasyon ------------------------------------------------------------------------
    c1, c2 = st.columns(2)
    cols = {"doc_score": "Belge skoru", "max_pixel": "Maks. piksel", "pred_area": "Tahmini sahte alan",
            "gt_ratio": "Gerçek sahte alan", "megapixels": "Çözünürlük (MP)", "jpeg_quality": "JPEG kalitesi",
            "infer_s": "Çıkarım süresi", "label": "Etiket (sahte=1)"}
    usable = [c for c in cols if c in df and df[c].nunique() > 1]
    corr = df[usable].corr(method="spearman").to_numpy()
    names = [cols[c] for c in usable]
    with c1:
        _show(_heatmap(corr, names, names, np.round(corr, 2).astype(str), "6 · Metrik korelasyon matrisi (Spearman ρ)",
                       DIVERGING, -1, 1, 0, hover="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>"))
    dropped = [cols[c] for c in cols if c not in usable]
    if dropped:
        st.caption(f"Sabit olduğu için çıkarılan değişkenler: {', '.join(dropped)}.")

    # 7. OCR -----------------------------------------------------------------------------
    with c2:
        if "ocr_fail" in df:
            ocr = df[df["ocr_evidence"].astype(bool)]
            flag = (ocr["ocr_fail"] > 0).to_numpy()
            yy = ocr["label"].to_numpy()
            cm = np.array([[((yy == a) & (flag == b)).sum() for b in (False, True)] for a in (0, 1)])
            pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            text = np.array([[f"{cm[i, j]}<br>(%{pct[i, j] * 100:.0f})" for j in range(2)] for i in range(2)])
            _show(_heatmap(pct, ["Kurallar geçti", "Kural ihlali"], ["Gerçek: Orijinal", "Gerçek: Sahte"], text,
                           "7 · OCR kural motoru karmaşıklık matrisi", SEQUENTIAL, 0, 1))
            st.caption(f"Doğrulanabilir alan (IBAN/TCKN/Matrah-KDV-Toplam) bulunan {len(ocr)} / {len(df)} belge. "
                       "Karakter/kelime düzeyinde OCR karışıklık matrisi için referans (ground-truth) metin "
                       "gerekir; veri setinde bulunmadığından belge düzeyi kural kararı gösterilir.")
            by_type = ocr.assign(ihlal=flag).groupby("tamper_type")["ihlal"].agg(["mean", "size"]).reset_index()
            st.dataframe(by_type.rename(columns={"tamper_type": "Tahrifat türü", "mean": "İhlal oranı", "size": "n"}),
                         hide_index=True, use_container_width=True)
        else:
            st.info("7 · OCR karmaşıklık matrisi için değerlendirmeyi OCR ile çalıştırın: `python benchmark.py --ocr`")


def section_robustness(res: dict, sel_paths: set[str], temperature: float, ckpt_paths: tuple[str, ...]) -> None:
    st.markdown("### C. Dayanıklılık ve Eğitim Dinamikleri")
    c1, c2 = st.columns(2)

    # 8. JPEG dayanıklılığı ----------------------------------------------------------------
    rob = res["robust"]
    with c1:
        if rob is not None and len(rob):
            rob = rob[rob["image_path"].isin(sel_paths)]
            g = rob.groupby("quality")
            f1, iou = f1_iou(g["tp"].sum(), g["fp"].sum(), g["fn"].sum())
            q = g.size().index.to_numpy()
            auc = np.array([roc_auc(d["doc_logit"].to_numpy(), d["label"].to_numpy()) if d["label"].nunique() == 2
                            else np.nan for _, d in g])
            fig = go.Figure()
            _line(fig, q, f1, "Piksel F1", SERIES[0])
            _line(fig, q, iou, "Piksel IoU", SERIES[1])
            _line(fig, q, auc, "Belge AUC (ham logit)", SERIES[2])
            fig.update_traces(mode="lines+markers", marker=dict(size=8))
            fig.update_xaxes(autorange="reversed", dtick=10)
            fig.update_yaxes(range=[0, 1])
            _show(_layout(fig, f"8 · JPEG sıkıştırma dayanıklılığı (piksel eşiği {res['meta']['robustness_threshold']:.2f})",
                          "JPEG kalite faktörü (Q)", "Skor"))
            st.caption(f"{rob['image_path'].nunique()} görsellik alt küme her Q değerinde yeniden sıkıştırıldı.")
        else:
            st.info("Dayanıklılık sonucu yok (`robustness.csv`).")

    # 9–10. Eğitim geçmişi ------------------------------------------------------------------
    histories = find_histories()
    with c2:
        if histories:
            path = st.selectbox("Eğitim geçmişi", histories, format_func=lambda p: p.as_posix())
            hist = pd.read_csv(path)
            fig = go.Figure()
            for col, name, color in (("train/loss", "Eğitim kaybı", SERIES[0]), ("val/loss", "Validasyon kaybı", SERIES[1]),
                                     ("train/dice", "Eğitim Dice kaybı", SERIES[2])):
                if col in hist:
                    _line(fig, hist["epoch"], hist[col], name, color)
            _show(_layout(fig, "9 · Eğitim ve validasyon kaybı", "Epoch", "Kayıp"))
        else:
            st.info("9–10 · Epoch bazlı eğitim geçmişi bulunamadı. Stage A/B eğitimleri Kaggle'da yapıldığı için "
                    "epoch günlükleri bu makinede yok; checkpoint yalnızca son epoch metriklerini içerir (aşağıda). "
                    "`train.py` artık her epoch'u `<run_dir>/history.csv` dosyasına yazıyor — Kaggle çıktısındaki "
                    "bu dosyayı `runs/<ad>/history.csv` altına koyduğunuzda grafikler otomatik çizilir.")
    if histories:
        cols = [("val/pixel_f1", "Val piksel F1", SERIES[0]), ("val/pixel_iou", "Val piksel IoU", SERIES[1]),
                ("val/score", "Val skor", SERIES[2])]
        fig = go.Figure()
        for col, name, color in cols:
            if col in hist:
                _line(fig, hist["epoch"], hist[col], name, color)
        fig.update_yaxes(range=[0, 1])
        _show(_layout(fig, "10 · Epoch bazlı piksel F1 / IoU", "Epoch", "Skor"))

    metrics = load_checkpoint_metrics(ckpt_paths)
    if not metrics.empty:
        keep = [r for r in metrics.index if r == "epoch" or r.startswith("val/")]
        st.markdown("**Checkpoint'lerde kayıtlı son epoch validasyon metrikleri** (eğitim sırasındaki val bölümü)")
        st.dataframe(metrics.loc[keep].astype(float).round(4), use_container_width=True)


def section_dataset(df: pd.DataFrame, res: dict, sel: np.ndarray, threshold: float) -> None:
    st.markdown("### D. Veri Seti ve Tahrifat Karakteristikleri")
    ti = int(np.argmin(np.abs(res["thresholds"] - threshold)))
    tp, fp, fn, _ = res["counts"][sel, ti].T
    df = df.assign(pixel_f1=f1_iou(tp, fp, fn)[0])
    forged = df[df["label"] == 1]

    c1, c2 = st.columns(2)
    order = forged.groupby("tamper_type")["pixel_f1"].median().sort_values(ascending=False).index.tolist()
    fig = px.box(forged, x="tamper_type", y="pixel_f1", points="all", category_orders={"tamper_type": order},
                 color_discrete_sequence=[SERIES[0]], hover_data=["subset"])
    fig.update_traces(marker=dict(size=8), line=dict(width=2), showlegend=False)
    fig.update_yaxes(range=[-0.02, 1.02])
    with c1:
        _show(_layout(fig, f"11 · Tahrifat türüne göre piksel F1 (eşik {res['thresholds'][ti]:.2f})", "Tahrifat türü",
                      "Piksel F1"))
    order = {"subset_name": [n for n in SUBSET_NAMES.values() if n in set(df["subset_name"])]}
    fig = px.box(forged, x="subset_name", y=forged["gt_ratio"] * 100, points="all", color="subset_name",
                 color_discrete_map=SUBSET_COLORS, category_orders=order, labels={"y": "Sahte alan (%)"})
    fig.update_traces(marker=dict(size=8), line=dict(width=2))
    with c2:
        _show(_layout(fig, "12 · Sahte alan kaplama oranı (gerçek maske)", "", "Belge alanının %'si"))

    c1, c2 = st.columns(2)
    regions = forged.groupby(["gt_regions", "subset_name"]).size().reset_index(name="Belge sayısı")
    fig = px.bar(regions, x="gt_regions", y="Belge sayısı", color="subset_name", color_discrete_map=SUBSET_COLORS,
                 category_orders=order, barmode="group")
    fig.update_layout(bargap=0.15, bargroupgap=0.05)
    fig.update_xaxes(dtick=1)
    fig.update_traces(marker_line=dict(width=2, color="rgba(0,0,0,0)"))
    with c1:
        _show(_layout(fig, "13 · Belge başına tahrif edilmiş bölge sayısı", "Bağlantılı bölge sayısı", "Belge sayısı"))

    with c2:
        if SYN_META.exists():
            meta = pd.read_csv(SYN_META)
            if STAGEB_SPLITS.exists():
                splits = pd.read_csv(STAGEB_SPLITS)
                splits = splits[splits["source"] == "synthetic"]
                split_of = dict(zip(splits["image_path"].str.split("/").str[-1], splits["split"]))
                meta["split"] = meta["image"].str.split("/").str[-1].map(split_of).fillna("—")
            choice = st.radio("Bölüm", ["Tümü", "train", "val", "test"], horizontal=True, key="dist_split")
            m = meta if choice == "Tümü" else meta[meta.get("split") == choice]
            dist = m.groupby(["doc_type", "lang"]).size().reset_index(name="Görsel sayısı")
            fig = px.bar(dist, x="doc_type", y="Görsel sayısı", color="lang",
                         color_discrete_map={"tr": SERIES[0], "en": SERIES[1]}, barmode="relative",
                         category_orders={"lang": ["tr", "en"]}, labels={"doc_type": "Belge tipi", "lang": "Dil"})
            fig.update_layout(bargap=0.35)
            # 2px yüzey aralığı: yığılmış segmentler arasında arka plan rengi çizgi
            fig.update_traces(marker_line=dict(width=2, color="rgba(255,255,255,1)"))
            _show(_layout(fig, "14 · Stage B sentetik veri: belge tipi × dil", "Belge tipi", "Görsel sayısı"))
            st.caption("Veri setinde dekont, fatura ve fiş bulunur; öğrenci belgesi örneği yoktur.")

    c1, c2 = st.columns(2)
    hover = {"subset_name": True, "width": True, "height": True, "pixel_f1": ":.3f", "infer_s": ":.2f"}
    fig = px.scatter(df, x="megapixels", y="infer_s", color="subset_name", color_discrete_map=SUBSET_COLORS,
                     hover_data=hover, category_orders=order, labels={"megapixels": "Çözünürlük (MP)", "infer_s": "Çıkarım süresi (s)"})
    fig.update_traces(marker=dict(size=9, line=dict(width=2, color="rgba(255,255,255,0.8)")))
    with c1:
        _show(_layout(fig, "15a · Çözünürlük – çıkarım süresi", "Çözünürlük (megapiksel)", "Süre (s)"))
    fig = px.scatter(forged.assign(pixel_f1=forged["pixel_f1"]), x="megapixels", y="pixel_f1", color="subset_name",
                     color_discrete_map=SUBSET_COLORS, hover_data=hover, category_orders=order,
                     labels={"megapixels": "Çözünürlük (MP)", "pixel_f1": "Piksel F1"})
    fig.update_traces(marker=dict(size=9, line=dict(width=2, color="rgba(255,255,255,0.8)")))
    with c2:
        _show(_layout(fig, "15b · Çözünürlük – piksel F1 (sahte belgeler)", "Çözünürlük (megapiksel)", "Piksel F1"))


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

def render_dashboard(threshold: float, temperature: float, ckpt_paths: tuple[str, ...]) -> None:
    ev = load_evaluation(str(EVAL_DIR), max(_mtime(p) for p in [EVAL_DIR / "summary.csv", *EVAL_DIR.glob("breakdown_*.csv")]))
    if ev is not None:
        section_evaluation(ev)
        st.divider()
    st.subheader("📈 Ayrıntılı analiz (benchmark.py)")
    res = load_results(str(RESULTS_DIR), _mtime(RESULTS_DIR / "meta.json"))
    if res is None:
        _no_results()
        section_robustness({"robust": None}, set(), temperature, ckpt_paths)
        return

    meta = res["meta"]
    df = res["per_image"].copy()
    df["subset_name"] = df["subset"].map(SUBSET_NAMES).fillna(df["subset"])
    st.caption(f"Kaynak: `{RESULTS_DIR.as_posix()}` · {meta['created']} · checkpoint `{meta['checkpoint']}` · "
               f"mod `{meta['mode']}` · piksel olasılıkları T={meta['temperature']:.3f} ile üretildi. "
               "Eşik ve sıcaklık yan paneldeki değerlerden alınır.")
    default_ckpt = next((p for p in ckpt_paths if Path(p).is_file()), None)
    if default_ckpt and Path(meta["checkpoint"]).as_posix() != Path(default_ckpt).as_posix():
        st.warning(f"Bu bölüm `{meta['checkpoint']}` modelinin benchmark sonuçlarıdır; uygulamanın varsayılan modeli "
                   f"`{default_ckpt}`. Güncel model için: `python benchmark.py --checkpoint {default_ckpt}`")
    if res["max_region"] is None:
        st.warning("Bu `benchmark.py` çıktısında bölge alanı (`max_region`) yok; belge kararları eski "
                   "`maks. piksel ≥ eşik` kuralıyla hesaplanıyor ve arayüzdeki karardan farklı olabilir. "
                   "Güncel kural için `python benchmark.py` komutunu yeniden çalıştırın.")

    subsets = st.multiselect("Değerlendirme alt kümeleri", list(df["subset_name"].unique()),
                             default=list(df["subset_name"].unique()))
    keep = df["subset_name"].isin(subsets).to_numpy()
    if not keep.any():
        st.warning("En az bir alt küme seçin.")
        return
    sel = np.flatnonzero(keep)
    df = df.iloc[sel].copy()
    df["doc_score"] = doc_scores(df, temperature)
    ti = int(np.argmin(np.abs(res["thresholds"] - threshold)))
    tp, fp, fn, tn = res["counts"][sel, ti].T
    df["pred_area"] = (tp + fp) / (tp + fp + fn + tn)

    # Özet kartları
    y = df["label"].to_numpy()
    pred = decisions(df["doc_score"], df["max_pixel"], region_at(res, sel, ti), threshold).to_numpy()
    pf1, piou = f1_iou(tp.sum(), fp.sum(), fn.sum())
    ece, _ = expected_calibration_error(sigmoid(df["doc_logit"].to_numpy() / temperature), y, n_bins=10)
    k = st.columns(5)
    k[0].metric("Görsel (sahte / orijinal)", f"{len(df)} ({int(y.sum())} / {int((1 - y).sum())})")
    k[1].metric("Belge AUC", f"{roc_auc(df['doc_score'].to_numpy(), y):.3f}" if len(np.unique(y)) == 2 else "—")
    k[2].metric("Belge doğruluğu (karar kuralı)", f"%{(pred == y).mean() * 100:.1f}")
    k[3].metric(f"Piksel F1 @ {res['thresholds'][ti]:.2f}", f"{pf1:.3f}", delta=f"IoU {piou:.3f}", delta_color="off")
    k[4].metric(f"ECE (T={temperature:.2f})", f"{ece:.3f}")

    section_performance(df, res, sel, threshold, temperature)
    section_matrices(df, res, sel, threshold)
    section_robustness(res, set(df["image_path"]), temperature, ckpt_paths)
    section_dataset(df, res, sel, threshold)
