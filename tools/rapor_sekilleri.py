"""Rapor ekleri için statik şekiller (Word'e girecek PNG'ler). Proje kökünden çalıştırılır."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("docs/rapor_ekleri"); OUT.mkdir(parents=True, exist_ok=True)
# Referans palet (açık mod) + metin/ızgara tonları
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def bars_h(ax, labels, values, color, fmt="{:.2f}", xmax=1.0, err=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.6, color=color, edgecolor=SURF, linewidth=2,
            xerr=None if err is None else np.abs(np.array(err).T - np.array(values)), ecolor=INK2, capsize=3)
    ax.set_yticks(y, labels); ax.set_xlim(0, xmax); ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.012, yi, fmt.format(v), va="center", color=INK, fontsize=9)


def learning_curves():
    h = pd.read_csv("checkpoints/stageE_v2/history.csv")
    best = int(h.loc[h["val/decision_tpr_at_fpr02"].idxmax(), "epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(h.epoch, h["train/loss"], color=S1, lw=2, marker="o", ms=4, label="Eğitim kaybı")
    ax.plot(h.epoch, h["val/loss"], color=S2, lw=2, marker="o", ms=4, label="Doğrulama kaybı")
    ax.set_title("Kayıp"); ax.set_xlabel("Epoch")
    ax = axes[1]
    for col, c, lab in [("val/image_auc", S1, "Belge AUC"), ("val/pixel_f1", S2, "Piksel F1"),
                        ("val/decision_tpr_at_fpr02", S3, "Karar TPR @ %2 FPR (izlenen)")]:
        ax.plot(h.epoch, h[col], color=c, lw=2, marker="o", ms=4, label=lab)
    ax.set_ylim(0, 1); ax.set_title("Doğrulama metrikleri"); ax.set_xlabel("Epoch")
    for ax in axes:
        ax.axvline(best, color=INK2, lw=1, ls="--"); ax.legend(fontsize=8.5)
    axes[1].text(best + 0.2, 0.04, f"en iyi epoch: {best}", color=INK2, fontsize=8.5)
    fig.suptitle("Şekil: Stage E v2 öğrenme eğrileri", x=0.01, ha="left", color=INK, fontsize=11)
    fig.savefig(OUT / "sekil_ogrenme_egrileri.png"); plt.close(fig)
    return best, h


def type_breakdown():
    b = pd.read_csv("results/stageE_v2/breakdown_test_real_forged.csv")
    b = b[(b.boyut == "sahtecilik_turu") & (b.deger != "orig")].sort_values("tespit_orani", ascending=False)
    names = {"inpaint_rewrite": "YZ benzeri doldurma", "field_rewrite": "Alan yeniden yazma", "multi": "Çoklu düzenleme",
             "name_swap": "İsim değiştirme", "delete": "Silme", "digit_change_2+": "Rakam (2+ hane)",
             "digit_change_1": "Rakam (tek hane)", "copy_move": "Kopyala-taşı"}
    labels = [f"{names.get(v, v)} (n={n})" for v, n in zip(b.deger, b.n_sahte)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars_h(ax, labels, b.tespit_orani.to_list(), S1)
    ax.set_xlabel("Tespit oranı (seçilen çalışma noktasında, orijinallerde FPR = 0)")
    ax.set_title("Şekil: Stage E v2 — sahtecilik türüne göre tespit (gerçek tabanlı test, 27 belge)", loc="left")
    fig.savefig(OUT / "sekil_tur_kirilimi.png"); plt.close(fig)
    return b


def canva_channels():
    d = pd.read_csv("results/stageE_v2_human/kural_katmani.csv", keep_default_na=False)
    d["model"] = d.model_flag.astype(str).eq("True")
    d["gizli"] = d.pdf_fail.str.contains("hidden_text")
    d["editor"] = d.pdf_fail.str.contains("pdf_editor")
    order = [("banka_orijinal", "Banka\norijinali\n(n=46, orijinal)"),
             ("canva_temiz", "Canva'dan geçmiş,\ndeğişmemiş\n(n=46, orijinal)"),
             ("metin_duzenleme", "Canva'da metin\ndüzenleme\n(n=23, sahte)"),
             ("beyaz_katman", "Canva'da beyaz\nkutu + yazı\n(n=23, sahte)")]
    g = d.groupby("channel")[["model", "editor", "gizli"]].mean()
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (col, c, lab) in enumerate([("model", S1, "Derin öğrenme modeli (Stage E v2)"),
                                       ("editor", S2, "PDF kuralı: düzenleme aracı izi"),
                                       ("gizli", S3, "PDF kuralı: gizlenmiş / üst üste metin")]):
        vals = [g.loc[k, col] for k, _ in order]
        ax.bar(x + (i - 1) * w, vals, width=w, color=c, edgecolor=SURF, linewidth=2, label=lab)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", color=INK, fontsize=8)
    ax.set_xticks(x, [l for _, l in order], fontsize=8.5); ax.set_ylim(0, 1.15); ax.grid(axis="x", visible=False)
    ax.set_ylabel("\"Sahte\" deme oranı"); ax.legend(fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3)
    ax.set_title("Şekil: İnsan yapımı Canva test kümesi — katman bazında \"sahte\" deme oranı", loc="left", pad=30)
    fig.savefig(OUT / "sekil_canva_kanallar.png"); plt.close(fig)
    return g


if __name__ == "__main__":
    which = sys.argv[1:] or ["learning", "types", "canva"]
    if "learning" in which:
        best, h = learning_curves(); print("en iyi epoch", best)
    if "types" in which:
        print(type_breakdown()[["deger", "n_sahte", "tespit_orani"]].to_string(index=False))
    if "canva" in which:
        print(canva_channels().round(2).to_string())
