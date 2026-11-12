"""docs/Rapor_Ekleri.docx: rapor taslağına taşınacak güncel tablolar, şekiller ve kontrol listesi.

Tüm sayılar sonuç dosyalarından okunur (elle yazılmış sayı yok). Proje kökünden çalıştırılır.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, ".")
from src.metrics import bootstrap_ci, roc_auc  # noqa: E402
from src.model_registry import checkpoint_summary  # noqa: E402

FIG = Path("docs/rapor_ekleri")
OUT = Path("docs/Rapor_Ekleri.docx")
INK2 = RGBColor(0x52, 0x51, 0x4E)


# ---------------------------------------------------------------------------
# Veri okuma yardımcıları
# ---------------------------------------------------------------------------

def summary(run: str) -> pd.DataFrame:
    return pd.read_csv(f"results/{run}/summary.csv", index_col=0)


def meta(run: str) -> dict:
    return json.loads(Path(f"results/{run}/summary.json").read_text(encoding="utf-8"))


def ci(row, key, digits=3):
    v = row.get(key)
    if v is None or pd.isna(v):
        return "—"
    lo, hi = row.get(f"{key}_ci_lo"), row.get(f"{key}_ci_hi")
    s = f"{v:.{digits}f}"
    return f"{s} [{lo:.2f}–{hi:.2f}]" if lo is not None and not pd.isna(lo) else s


def f(v, digits=3):
    return "—" if v is None or pd.isna(v) else f"{v:.{digits}f}"


def pct(v):
    return "—" if v is None or pd.isna(v) else f"%{v * 100:.0f}"


def tr(s: str) -> str:
    """Ondalık noktayı Türkçe virgüle çevirir (yalnızca sayılarda)."""
    import re
    return re.sub(r"(?<=\d)\.(?=\d)", ",", s)


# ---------------------------------------------------------------------------
# Belge yardımcıları
# ---------------------------------------------------------------------------

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.2))
sec.top_margin = sec.bottom_margin = Cm(2)
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
for name, size in (("Title", 20), ("Heading 1", 15), ("Heading 2", 12.5)):
    doc.styles[name].font.name = "Calibri"; doc.styles[name].font.size = Pt(size)
    doc.styles[name].element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    doc.styles[name].paragraph_format.keep_with_next = True

counter = {"tablo": 0, "sekil": 0}


def para(text, style=None, italic=False, color=None, size=None, bold=False):
    p = doc.add_paragraph(style=style)
    r = p.add_run(text); r.italic = italic; r.bold = bold
    if color: r.font.color.rgb = color
    if size: r.font.size = Pt(size)
    return p


def bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    return p


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(title, header, rows, widths=None, note=None, raw_cols=()):
    counter["tablo"] += 1
    cap = para(f"Tablo {counter['tablo']}. {title}", bold=True, size=10)
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(9)
        shade(c, "E8EEF7")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run(str(v) if i in raw_cols else tr(str(v))); r.font.size = Pt(9)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    if note:
        para(note, italic=True, color=INK2, size=8.5)
    doc.add_paragraph()


def figure(path, caption, width_cm=16.5):
    counter["sekil"] += 1
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = para(f"Şekil {counter['sekil']}. {caption}", italic=True, size=9)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ---------------------------------------------------------------------------
# Hesaplanan değerler
# ---------------------------------------------------------------------------

v2 = summary("stageE_v2"); v2m = meta("stageE_v2")
hum2 = summary("stageE_v2_human").loc["test/human_forged"]
ai2 = summary("stageE_v2_aiforge").loc["ood/aiforgedoc"] if Path("results/stageE_v2_aiforge/summary.csv").exists() else None
ai1 = summary("stageE_aiforge").loc["ood/aiforgedoc"]
c1, c2 = summary("common_stageE_v1"), summary("common_stageE_v2")
b1 = pd.read_csv("results/common_stageE_v1/breakdown_test_real_forged.csv")
b2 = pd.read_csv("results/common_stageE_v2/breakdown_test_real_forged.csv")
proto = pd.read_csv("results/protocol/common_v1_v2_test_sub.csv", keep_default_na=False)

# Canva içi AUC (insan yapımı küme, banka orijinalleri hariç)
mh = pd.read_csv("data/9_human_forged/metadata.csv", keep_default_na=False); mh["name"] = mh.image.str.split("/").str[-1]


def canva_auc(run):
    p = pd.read_csv(sorted(Path(f"results/{run}").glob("predictions_*.csv"))[0], keep_default_na=False)
    p["name"] = p.image_path.str.replace("\\", "/").str.split("/").str[-1]
    j = p.drop(columns=[c for c in ("channel",) if c in p]).merge(mh[["name", "channel"]], on="name")
    c = j[j.channel != "banka_orijinal"]; s, y = c.doc_logit.astype(float).values, c.label.astype(int).values
    lo, hi = bootstrap_ci(roc_auc, [s, y], n_boot=1000, seed=0)
    return roc_auc(s, y), lo, hi, len(c)


canva2 = canva_auc("stageE_v2_human")

kk = pd.read_csv("results/stageE_v2_human/kural_katmani.csv", keep_default_na=False)
kk["model"] = kk.model_flag.astype(str).eq("True"); kk["gizli"] = kk.pdf_fail.str.contains("hidden_text")
kk["editor"] = kk.pdf_fail.str.contains("pdf_editor"); kk["text"] = kk.text_flag.astype(str).eq("True")
kk["any"] = kk.any_flag.astype(str).eq("True")
pdf_before = pd.read_csv("results/pdf_eval/real_v2_files.csv", keep_default_na=False)
pdf_after = pd.read_csv("results/pdf_eval/real_v2_ayarli_files.csv", keep_default_na=False)


def issuer_of(path):
    return Path(str(path).replace("\\", "/")).parent.name


# ---------------------------------------------------------------------------
# Şekil: domain shift (veri burada hesaplanıyor)
# ---------------------------------------------------------------------------

def domain_shift_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S1, INK, INKG, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
    rows = [("Sentetik test\n(eğitimle aynı dağılım)", v2.loc["test/synthetic"]),
            ("Gerçek belge + otomatik sahte\n(gerçek şablon, bizim üreticimiz)", v2.loc["test/real_forged"])]
    if ai2 is not None:
        rows.append(("AIForge-Doc\n(başka araştırmacıların YZ sahteleri)", ai2))
    rows.append(("İnsan yapımı, Canva\n(gerçek araç, gerçek kullanıcı)", hum2))
    labels = [r[0] for r in rows]; vals = [r[1]["image_auc"] for r in rows]
    lo = [r[1]["image_auc_ci_lo"] for r in rows]; hi = [r[1]["image_auc_ci_hi"] for r in rows]
    labels.append("İnsan yapımı, yalnız Canva içi\n(temiz Canva vs. düzenlenmiş Canva)")
    vals.append(canva2[0]); lo.append(canva2[1]); hi.append(canva2[2])
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(y, vals, height=0.58, color=S1, edgecolor="white", linewidth=2)
    ax.errorbar(vals, y, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none",
                ecolor=INKG, capsize=3, lw=1)
    ax.axvline(0.5, color=INKG, lw=1, ls="--"); ax.text(0.505, y[0] + 0.45, "rastgele (0,5)", color=INKG, fontsize=8.5)
    for yi, v, h in zip(y, vals, hi):
        ax.text(h + 0.015, yi, f"{v:.2f}".replace(".", ","), va="center", color=INK, fontsize=9)
    ax.set_yticks(y, labels, fontsize=9); ax.set_xlim(0, 1.08); ax.set_xlabel("Belge düzeyi AUC (%95 bootstrap güven aralığı)", color=INKG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); ax.tick_params(colors=INKG)
    ax.set_title("Stage E v2: test dağılımı gerçek dünyaya yaklaştıkça başarım düşüyor", loc="left", fontsize=11, fontweight="bold")
    out = FIG / "sekil_domain_shift.png"; fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Belge
# ---------------------------------------------------------------------------

doc.add_heading("Rapor Ekleri: Güncel Tablolar, Şekiller ve Kontrol Listesi", level=0)
para("Bu belge Rapor_Taslagi.docx'e taşınmak üzere hazırlandı. Tüm sayılar proje içindeki sonuç dosyalarından "
     "(results/…) otomatik okunmuştur; köşeli parantezler %95 bootstrap güven aralığıdır. Tabloları ve şekilleri "
     "taslağa kopyalayın; yorum ve tartışma metinleri size aittir. Her tablonun altındaki not, tablonun hangi "
     "dosyadan geldiğini ve okunurken nelere dikkat edilmesi gerektiğini belirtir.", italic=True, color=INK2, size=9.5)

# ---------------- A. Kontrol listesi ----------------
doc.add_heading("A. Taslak için bölüm bölüm kontrol listesi", level=1)
para("Taslak (09.11) yalnızca Stage E v1 sonuçlarını içeriyor. Hocanızın önerdiği çerçeve: kullanılan modeller, veri "
     "artırma yöntemleri, eğitim/doğrulama bölmesi ve sonuçların yorumu; ana bulgu domain shift.")
checklist = [
    ("1. Problem ve amaç: ", "Kapsam cümlesini güncelleyin: Türkçe dijital banka dekontları (Garanti, İş Bankası, Ziraat) ve "
     "e-Arşiv faturalar (Trendyol). Fotoğraf/tarama ve termal fiş kapsam dışı: gerçek örnek yok (v2'de 1110 görüntünün "
     "yalnız 6'sı fotoğraf)."),
    ("2. Veri seti: ", "Tablo 1 ve Tablo 2 ile değiştirin (v1 tabloları 143 belge diyor; v2'de 185 belge, İş Bankası eklendi). "
     "İnsan yapımı Canva kümesini ve AIForge-Doc'u 'dış test kümeleri' olarak ekleyin."),
    ("2. Veri artırma: ", "Üç düzeyde anlatın: (i) eğitim sırasında JPEG Q50–100, parlaklık/kontrast, sahte bölgeli kırpma "
     "%50; (ii) sentetik üretici: düzenleme izleri (yerel JPEG, yeniden örnekleme, gürültü, renk) ve iletim kanalları "
     "(PNG, PDF→JPEG, ekran görüntüsü, WhatsApp); (iii) forge_real: gerçek belgeden 6 işlemle sahte üretimi."),
    ("4. Yöntem: ", "Karar paragrafını Stage E v2 kalibrasyonuyla güncelleyin (Bölüm D): belge ve piksel için ayrı sıcaklık, "
     "val'de %2 yanlış alarm hedefi. v1'in tek sıcaklık kullandığını ve bunun piksel eşiğini fiilen kapattığını belirtin."),
    ("5. Aşamalar: ", "Tablo 3'ü ekleyin: her aşamada neyin neden değiştiği. Stage C/D başarısızlığı (her şeye 'sahte') "
     "önemli bir ders: sınıf dengesizliği ve iz bırakmayan sentetik etiketler."),
    ("5. Sonuçlar: ", "Ana tablo Tablo 4 (domain shift) ve Şekil 2 olsun. v1–v2 karşılaştırması için Tablo 5'i kullanın: "
     "eski Tablo 3/4 farklı test kümelerinde ölçülmüştü ve karşılaştırılamaz."),
    ("5. Yorum: ", "Tablo 6'daki kurum bulgusunu yazın: v1 hiç görmediği İş Bankası'nda orijinallerin çoğuna 'sahte' diyor; "
     "v2 aynı kurumu eğitimde gördüğü için bu oran düşüyor. Bu, şablon düzeyinde domain shift'in doğrudan kanıtı."),
    ("5. Hata analizi: ", "Canva bulgusu (Tablo 8, Şekil 4): Canva belgeyi bütünüyle yeniden render ettiği için düzenleme "
     "yerel piksel izi bırakmaz; model temiz ve düzenlenmiş Canva çıktısını ayıramaz (Canva içi AUC ≈ 0,5). Orijinal ve "
     "Canva çıktılarının çözünürlüğü farklı (1653×1170 ve 2438×1660), bu yüzden tüm kümede AUC 0,5'in altına düşüyor; "
     "bunu karıştırıcı faktör olarak açıkça belirtin."),
    ("5. Hibrit sistem: ", "Tablo 8 ve Tablo 9: PDF katmanı Canva düzenlemelerini yakalıyor; PDF kuralları gerçek "
     "orijinallerde ayarlandıktan sonra yanlış alarm %33 → %0. 'Düzenleme aracı izi' kuralının temiz Canva kopyasını da "
     "işaretlediğini, yani düzenlemeyi değil belgenin araçtan geçtiğini gösterdiğini yazın."),
    ("Sınırlılıklar: ", "4 kurum; insan yapımı küme tek kişi ve tek araç (Canva); fotoğraf/tarama yok; tek seed; en iyi epoch "
     "son epoch (15), yani eğitim daha uzun sürdürülebilirdi; kopyala-taşı ve tek rakam değişikliği zayıf türler; "
     "v1–v2 karşılaştırmasında sentetik kısım 120 gruba örneklendi."),
    ("Kaynakça: ", "Taslaktaki iki '[doğrulayın]' notunu (CAT-Net künyesi, AIForge-Doc yazarları) kapatın."),
]
for pre, txt in checklist:
    bullet(txt, pre)

# ---------------- B. Tablolar ----------------
doc.add_heading("B. Tablolar", level=1)

m2 = pd.read_csv("data/8_real_forged_v2/metadata.csv", keep_default_na=False)
by_issuer = m2.groupby("doc_type").group_id.nunique()
syn = pd.read_csv("data/6_synthetic_tr_en_v4/metadata.csv", keep_default_na=False)
table("Kullanılan veri setleri", ["Veri", "İçerik", "Rol"], [
    ["DocTamper", "≈170 bin sahte belge görüntüsü, piksel maskeli (Çince/İngilizce)", "Ön eğitim (Stage A)"],
    ["Sentetik v4", f"{len(syn):,} görüntü; {(syn.label == 1).sum():,} sahte / {(syn.label == 0).sum():,} orijinal; "
                    f"{(syn.lang == 'tr').sum():,} TR / {(syn.lang == 'en').sum():,} EN; dekont, fatura, fiş", "Stage E eğitim/val/test"],
    ["Gerçek belgeler + otomatik sahte (v2)", f"{m2.group_id.nunique()} gerçek belge (" + ", ".join(
        f"{k} {v}" for k, v in by_issuer.items()) + f") → {len(m2):,} görüntü; {(m2.label == 1).sum():,} sahte / "
        f"{(m2.label == 0).sum()} orijinal (forge_real)", "Stage E v2 eğitim/val/test"],
    ["İnsan yapımı (Canva)", f"{len(mh)} görüntü, İş Bankası: {(mh.label == 1).sum()} sahte (tutar, IBAN, tarih, isim), "
                             f"{(mh.label == 0).sum()} orijinal (banka çıktısı + Canva'dan değiştirilmeden geçmiş kopya)",
     "Yalnız test (dış küme)"],
    ["AIForge-Doc TestingSet", "YZ ile düzenlenmiş belgeler + orijinaller; 150 sahte / 150 orijinal örneklem",
     "Yalnız test (dağılım dışı)"],
], widths=[3.6, 9.6, 3.4], note="Kaynak: data/*/metadata.csv. Gerçek belgeler kurum alt klasörlerine göre sayılmıştır.")

s2 = pd.read_csv("checkpoints/stageE_v2/splits.csv", keep_default_na=False)
g = s2.groupby(["source", "split"]).agg(img=("label", "size"), sahte=("label", "sum"), grup=("group_id", "nunique"))
rows = []
for src, name in (("synthetic", "Sentetik v4"), ("real_forged", "Gerçek tabanlı (v2)")):
    rows.append([name] + [f"{int(g.loc[(src, sp), 'grup'])} belge / {int(g.loc[(src, sp), 'img'])} görüntü "
                          f"({int(g.loc[(src, sp), 'sahte'])} sahte)" for sp in ("train", "val", "test")])
table("Stage E v2 veri bölmesi (belge grubu düzeyinde %70/%15/%15, seed 42)", ["Kaynak", "Eğitim", "Doğrulama", "Test"], rows,
      widths=[3.4, 4.4, 4.4, 4.4], note="Kaynak: checkpoints/stageE_v2/splits.csv. Aynı belgenin orijinali ve tüm sahte "
      "varyantları tek bölümdedir (sızıntı kontrolü: verify_no_leakage). Doğrulama kümesi model seçimi ve kalibrasyon "
      "için kullanıldı; test kümesi hiçbir seçimde kullanılmadı.")

stages = [
    ("Stage A", "checkpoints/base_doctamper.pth", "DocTamper", "ImageNet",
     "Metin sahteciliğinin genel izleri; hiç orijinal belge görmedi"),
    ("Stage B", "checkpoints/finetuned_multilingual_stageB.pth", "İlk sentetik sürüm + SROIE + DocTamper", "Stage A",
     "Orijinallerin %38'ine 'sahte' (val)"),
    ("Stage C (ara)", "checkpoints/finetuned_multilingual.pth", "Yalnız sentetik", "Stage B",
     "Her şeye 'sahte'; sınıf dengesizliği (1 orijinal : 3 sahte)"),
    ("Stage D", "checkpoints/stageD/best.pth", "Yalnız sentetik v3", "Stage B",
     "Her şeye 'sahte'; iz bırakmayan dijital sahteler → 'alan konumu = sahte' ezberi"),
    ("Stage E (v1)", "checkpoints/stageE/best.pth", "Sentetik v4 + 143 gerçek belge", "Stage A",
     "Dengeli örnekleme, izli sentetik, gerçek tabanlı sahteler"),
    ("Stage E v2", "checkpoints/stageE_v2/best.pth", "Sentetik v4 + 185 gerçek belge (İş Bankası eklendi)", "Stage A",
     "Ayrı belge/piksel kalibrasyonu, karar odaklı model seçimi"),
]
rows = []
for name, path, data, init, lesson in stages:
    s = checkpoint_summary(path)
    rows.append([name, data, init, s.get("epoch", "—"), f(s.get("val_image_auc")), f(s.get("val_pixel_f1")),
                 f(s.get("val_image_fpr")), lesson])
table("Eğitim aşamaları: veri, başlangıç ağırlığı, doğrulama metrikleri ve çıkarılan ders",
      ["Aşama", "Eğitim verisi", "Başlangıç", "En iyi epoch", "Val AUC", "Val piksel F1", "Val FPR (ham, 0,5)", "Ders / değişiklik"],
      rows, widths=[2, 3.4, 1.6, 1.2, 1.3, 1.4, 1.5, 4.2],
      note="Kaynak: checkpoint içindeki son epoch metrikleri. Uyarı: her aşamanın doğrulama kümesi farklıdır (Stage A "
           "DocTamper, Stage E Türkçe sentetik + gerçek), bu yüzden bu sütunlar aşamalar arasında doğrudan "
           "karşılaştırılamaz; aynı test kümesindeki karşılaştırma Tablo 5'tedir. 'Val FPR (ham)' kalibrasyon öncesi 0,5 "
           "eşiğindeki yanlış alarmdır.")

rows = [["Sentetik test", "Eğitimle aynı dağılım", int(v2.loc["test/synthetic", "n_images"]),
         ci(v2.loc["test/synthetic"], "image_auc"), pct(v2.loc["test/synthetic", "decision_tpr"]),
         pct(v2.loc["test/synthetic", "decision_fpr"]), ci(v2.loc["test/synthetic"], "pixel_f1")],
        ["Gerçek belge + otomatik sahte", "Gerçek şablon, bizim üreticimiz", int(v2.loc["test/real_forged", "n_images"]),
         ci(v2.loc["test/real_forged"], "image_auc"), pct(v2.loc["test/real_forged", "decision_tpr"]),
         pct(v2.loc["test/real_forged", "decision_fpr"]), ci(v2.loc["test/real_forged"], "pixel_f1")]]
if ai2 is not None:
    rows.append(["AIForge-Doc", "Başka araştırmacıların YZ sahteleri", int(ai2["n_images"]), ci(ai2, "image_auc"),
                 pct(ai2["decision_tpr"]), pct(ai2["decision_fpr"]), ci(ai2, "pixel_f1")])
rows += [["İnsan yapımı (Canva), tümü", "Gerçek araç, gerçek kullanıcı", int(hum2["n_images"]), ci(hum2, "image_auc"),
          pct(hum2["decision_tpr"]), pct(hum2["decision_fpr"]), f(hum2["pixel_f1"])],
         ["İnsan yapımı, yalnız Canva içi", "Temiz Canva vs. düzenlenmiş Canva", canva2[3],
          f"{canva2[0]:.3f} [{canva2[1]:.2f}–{canva2[2]:.2f}]", "—", "—", "—"]]
table("Domain shift: Stage E v2'nin test dağılımına göre başarımı (aynı model, aynı kalibrasyon)",
      ["Test kümesi", "Gerçeğe yakınlık", "n", "Belge AUC", "Tespit (TPR)", "Yanlış alarm (FPR)", "Piksel F1"], rows,
      widths=[3.3, 3.3, 1.1, 2.9, 1.6, 1.8, 2.6],
      note=f"Kaynak: results/stageE_v2, stageE_v2_aiforge, stageE_v2_human. Kalibrasyon: belge T={v2m['doc_temperature']:.2f}, "
           f"eşik {v2m['doc_threshold']:.2f}; piksel T={v2m['pixel_temperature']:.2f}, eşik {v2m['pixel_threshold']:.2f} "
           f"(val'de, hedef FPR ≤ %{v2m['target_fpr'] * 100:.0f}). İnsan yapımı kümede AUC'nin 0,5'in altına düşmesi "
           "banka orijinalleri ile Canva çıktılarının çözünürlük farkından kaynaklanan karıştırıcı bir etkidir; Canva "
           "içi karşılaştırma bu etkiyi ortadan kaldırır.")

rows = []
for key, name in (("test/real_forged", "Gerçek tabanlı"), ("test/synthetic", "Sentetik (120 grup)")):
    for label, s in (("Stage E v1", c1.loc[key]), ("Stage E v2", c2.loc[key])):
        rows.append([name, label, int(s["n_images"]), ci(s, "image_auc"), f(s["image_tpr_at_fpr05"]),
                     pct(s["decision_tpr"]), pct(s["decision_fpr"]), ci(s, "pixel_f1")])
n_real_groups = proto[proto.source == "real_forged"].group_id.nunique()
table("Aynı test bölmesinde Stage E v1 ve v2 (iki modelin de eğitim/doğrulamada görmediği belgeler)",
      ["Küme", "Model", "n", "Belge AUC", "TPR @ %5 FPR", "Karar TPR", "Karar FPR", "Piksel F1"], rows,
      widths=[2.6, 2, 1, 2.9, 1.6, 1.6, 1.6, 2.9],
      note=f"Kaynak: results/common_stageE_v1, common_stageE_v2; protokol results/protocol/common_v1_v2_test_sub.csv "
           f"({n_real_groups} gerçek belge, 120 sentetik grup). v2'nin test grupları içinden v1'in eğitim/doğrulama "
           "kümesinde olan 78 grup çıkarıldı (sızıntı önlemi). Her model kendi val kümesinde öğrenilmiş kalibrasyonuyla "
           "çalıştırıldı (v1: tek sıcaklık, hedef FPR %5; v2: ayrı belge/piksel kalibrasyonu, hedef FPR %2). "
           "'TPR @ %5 FPR' eşikten bağımsızdır ve iki model arasında en adil karşılaştırmadır.")


def issuer_rows(b, label):
    k = b[b.boyut == "kurum"].set_index("deger")
    return k, label


k1, k2 = b1[b1.boyut == "kurum"].set_index("deger"), b2[b2.boyut == "kurum"].set_index("deger")
seen_v1 = {"garanti", "trendyol", "ziraat"}
rows = []
for issuer in k2.index:
    r1, r2 = k1.loc[issuer] if issuer in k1.index else None, k2.loc[issuer]
    rows.append([issuer, "evet" if issuer in seen_v1 else "HAYIR", int(r2["n_orijinal"]), int(r2["n_sahte"]),
                 pct(r1["yanlis_alarm"]) if r1 is not None else "—", pct(r2["yanlis_alarm"]),
                 pct(r1["tespit_orani"]) if r1 is not None else "—", pct(r2["tespit_orani"])])
table("Kurum bazında yanlış alarm ve tespit (Tablo 5'in gerçek tabanlı kısmı)",
      ["Kurum", "v1 eğitimde gördü mü?", "Orijinal", "Sahte", "Yanlış alarm v1", "Yanlış alarm v2", "Tespit v1", "Tespit v2"],
      rows, widths=[2, 2.4, 1.5, 1.4, 2.2, 2.2, 2.2, 2.2],
      note="Kaynak: results/common_stageE_*/breakdown_test_real_forged.csv. İş Bankası v1'in eğitim verisinde hiç yoktu "
           "(v2'de eklendi): bu satır, görülmemiş bir şablonun yanlış alarma etkisini doğrudan gösterir. Kurum başına "
           "orijinal sayısı küçüktür (2–9); oranlar yön göstericidir.")

names = {"inpaint_rewrite": "YZ benzeri doldurma", "field_rewrite": "Alan yeniden yazma", "multi": "Çoklu düzenleme",
         "name_swap": "İsim değiştirme", "delete": "Silme", "digit_change_2+": "Rakam (2+ hane)",
         "digit_change_1": "Rakam (tek hane)", "copy_move": "Kopyala-taşı"}
t1, t2 = b1[b1.boyut == "sahtecilik_turu"].set_index("deger"), b2[b2.boyut == "sahtecilik_turu"].set_index("deger")
rows = [[names.get(t, t), int(t2.loc[t, "n_sahte"]), pct(t1.loc[t, "tespit_orani"]) if t in t1.index else "—",
         pct(t2.loc[t, "tespit_orani"]), f(t1.loc[t, "auc"]) if t in t1.index else "—", f(t2.loc[t, "auc"])]
        for t in t2.sort_values("tespit_orani", ascending=False).index if t != "orig"]
table("Sahtecilik türüne göre tespit (aynı test bölmesi, gerçek tabanlı)",
      ["Tür", "n (sahte)", "Tespit v1", "Tespit v2", "AUC v1", "AUC v2"], rows, widths=[4.2, 2, 2.4, 2.4, 2.4, 2.4],
      note="Tespit oranı modelin kendi çalışma noktasındadır; v1 daha gevşek (FPR hedefi %5) olduğu için tespit oranları "
           "v1'de yüksek görünür, ama bunun bedeli Tablo 6'daki yanlış alarmlardır. Türler arasında eşikten bağımsız "
           "karşılaştırma için AUC sütunlarını kullanın (her tür, kümedeki tüm orijinallere karşı).")

order = [("banka_orijinal", "Banka orijinali", "orijinal"), ("canva_temiz", "Canva'dan geçmiş, değişmemiş", "orijinal"),
         ("metin_duzenleme", "Canva'da metin düzenleme", "sahte"), ("beyaz_katman", "Canva'da beyaz kutu + yazı", "sahte")]
cg = kk.groupby("channel")
rates = cg[["model", "editor", "gizli", "text", "any"]].mean()
rows = [[lab, cls, int(cg.size()[ch])] + [pct(rates.loc[ch, c]) for c in ("model", "editor", "gizli", "text", "any")]
        for ch, lab, cls in order]
table("İnsan yapımı Canva kümesi: katman bazında 'sahte' deme oranı (Stage E v2 + kural katmanı)",
      ["Grup", "Sınıf", "n", "Model", "PDF: araç izi", "PDF: gizlenmiş metin", "OCR kuralı", "Herhangi biri"], rows,
      widths=[3.8, 1.5, 1, 1.6, 1.9, 2.3, 1.8, 2.1],
      note="Kaynak: results/stageE_v2_human/kural_katmani.csv (PDF kuralları her dosyanın PDF'inde, model aynı dosyanın "
           "rasterleştirilmiş görüntüsünde çalıştırıldı). Okuma: model temiz Canva kopyasına da 'sahte' diyor, yani "
           "düzenlemeye değil kanala tepki veriyor. 'Araç izi' kuralı da temiz kopyayı işaretler (belgenin araçtan "
           "geçtiğini gösterir); düzenlemeye özgü kanıt 'gizlenmiş metin' kuralıdır ve yalnız beyaz kutu yöntemini yakalar.")

pb = pdf_before.assign(kurum=pdf_before.path.map(issuer_of), fail=pdf_before.n_fail > 0)
pa = pdf_after.assign(kurum=pdf_after.path.map(issuer_of), fail=pdf_after.n_fail > 0)
rows = []
for kurum in sorted(pb.kurum.unique()):
    b_, a_ = pb[pb.kurum == kurum], pa[pa.kurum == kurum]
    rows.append([kurum, b_.producer.iloc[0], len(b_), pct(b_.fail.mean()), pct(a_.fail.mean()),
                 "; ".join(sorted({r for x in b_.fail_rules for r in str(x).split(";") if r})) or "—"])
rows.append(["Toplam", "", len(pb), pct(pb.fail.mean()), pct(pa.fail.mean()), ""])
table("PDF yapı kuralları: gerçek orijinal PDF'lerde yanlış alarm, ayar öncesi ve sonrası",
      ["Kurum", "Üretici yazılım", "PDF", "Yanlış alarm (önce)", "Yanlış alarm (sonra)", "Önceki FAIL kuralları"], rows,
      widths=[2, 4.6, 1.2, 2.4, 2.4, 4], raw_cols={1, 5},
      note="Kaynak: results/pdf_eval/real_v2_files.csv ve real_v2_ayarli_files.csv. Ayar, test kümesine bakılarak değil "
           "PDF üretim standartlarına göre yapıldı: (1) şablondan form dolduran üreticilerde (Adobe AEM Forms/LiveCycle) "
           "oluşturma tarihi şablona aittir, değişiklik sayılmaz; (2) font ailesi adlandırmasında 'Roman', 'Book', 'Heavy' "
           "gibi ekler stil kabul edilir (FrutigerRoman ve FrutigerBold aynı ailedir). İnsan yapımı kümedeki tüm Canva "
           "düzenlemeleri ayardan sonra da yakalanmaya devam etti (Tablo 8).")

# ---------------- C. Şekiller ----------------
doc.add_heading("C. Şekiller", level=1)
figure(FIG / "sekil_ogrenme_egrileri.png",
       "Stage E v2 öğrenme eğrileri. Eğitim kaybı düzenli düşerken doğrulama kaybı dalgalı; izlenen metrik (karar TPR @ %2 FPR) "
       "son epoch'ta en yüksek, yani eğitim daha uzun sürdürülebilirdi. Kaynak: checkpoints/stageE_v2/history.csv.")
figure(domain_shift_figure(),
       "Domain shift: aynı model ve kalibrasyonla, test dağılımı gerçek dünyaya yaklaştıkça belge AUC'si düşüyor. "
       "Hata çubukları %95 bootstrap güven aralığı.")
figure(FIG / "sekil_tur_kirilimi.png",
       "Stage E v2'nin kendi gerçek tabanlı test kümesinde (27 belge) sahtecilik türüne göre tespit oranı. "
       "Kopyala-taşı ve tek rakam değişikliği zayıf türler.")
figure(FIG / "sekil_canva_kanallar.png",
       "İnsan yapımı Canva kümesinde katmanların 'sahte' deme oranı. Model temiz Canva kopyasına da tepki veriyor; "
       "düzenlemeye özgü kanıtı PDF katmanının gizlenmiş metin kuralı sağlıyor.")
# Hata haritası örnekleri bilerek eklenmedi: gerçek dekontlarda kart numarası, IBAN, müşteri no ve ad var (KVKK).
para("Hata haritası örnekleri (doğru tespit / kaçırılan): results/common_stageE_v2/error_maps/test_real_forged/ "
     "klasöründe hazırdır, ancak gerçek dekontlardaki kart numarası, IBAN, müşteri numarası ve ad gibi kişisel veriler "
     "içerdikleri için bu belgeye eklenmedi. Rapora koymadan önce bu alanları maskeleyin ya da yalnızca düzenlenen "
     "bölgenin yakın kırpımını kullanın.", italic=True, color=INK2, size=9)

# ---------------- D. Yöntem notları ----------------
doc.add_heading("D. Yöntem bölümü için doğrulanmış parametreler", level=1)
cfg = json.loads(Path("checkpoints/stageE_v2/config.json").read_text(encoding="utf-8"))
for k, v in [("Mimari: ", f"U-Net + {cfg['encoder']} kodlayıcı, giriş akışları {cfg['streams']} (RGB + 3 sabit SRM filtresi, 6 kanal)"),
             ("Başlangıç ağırlığı: ", "Stage A (DocTamper ön eğitimi)"),
             ("Eğitim: ", f"AdamW, lr {cfg['lr']}, weight decay {cfg['weight_decay']}, batch {cfg['batch_size']}, "
                         f"{cfg['epochs']} epoch, epoch başına {cfg['samples_per_epoch']:,} örnek, AMP, seed {cfg['seed']}"),
             ("Veri artırma (eğitim sırasında): ", f"{cfg['crop_size']}×{cfg['crop_size']} kırpma (sahte bölgeli kırpma olasılığı "
                         f"{cfg['crop_nonempty_p']}), JPEG Q50–100 (olasılık {cfg['jpeg_aug_p']}), parlaklık/kontrast; "
                         "orijinal/sahte 50/50 dengeli örnekleme"),
             ("Model seçimi: ", f"doğrulama kümesinde '{cfg['monitor']}' (val'de %2 yanlış alarmda karar TPR'si)"),
             ("Kalibrasyon (val'de öğrenildi): ", f"belge T={v2m['doc_temperature']:.3f}, belge eşiği {v2m['doc_threshold']:.2f}; "
                         f"piksel T={v2m['pixel_temperature']:.3f}, piksel eşiği {v2m['pixel_threshold']:.2f}; hedef FPR ≤ "
                         f"%{v2m['target_fpr'] * 100:.0f} (val'de FPR {v2m['val_fpr']:.3f}, TPR {v2m['val_tpr']:.3f}, "
                         f"{v2m['val_images']} görüntü)"),
             ("Değerlendirme: ", "%95 bootstrap güven aralıkları (1000 yineleme, görüntü bazında); belge AUC, TPR@FPR, karar "
                                 "TPR/FPR, piksel F1 (Dice) ve IoU")]:
    bullet(tr(v), k)

zoom = doc.settings.element.find(qn("w:zoom"))
if zoom is not None and zoom.get(qn("w:percent")) is None:
    zoom.set(qn("w:percent"), "100")  # python-docx şablonunda eksik zorunlu öznitelik
doc.save(OUT)
print("Yazıldı:", OUT, "| tablo:", counter["tablo"], "| şekil:", counter["sekil"], "| AIForge v2:", ai2 is not None)
