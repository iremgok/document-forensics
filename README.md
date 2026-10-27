---
title: Document Forensics
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Türkçe finansal belgelerde sahtecilik tespiti ve konumlandırma
---

# Türkçe Finansal Belgelerde Geleneksel ve Üretken YZ Kaynaklı Sahteciliğin Tespiti ve Konumlandırılması

> **🌐 Canlı demo:** _yakında_ — `https://huggingface.co/spaces/<kullanici>/document-forensics` <!-- Space yayınlanınca bu satırı güncelleyin -->
>
> Teknik rapor: [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md)

Banka dekontu, fatura ve fişlerde sahteciliği **piksel düzeyinde konumlandıran** (ısı haritası / maske)
ve **belge düzeyinde sahtecilik olasılığı** üreten hibrit bir sistem:

| Bileşen | Ne yakalar? |
|---|---|
| **RGB akışı** (U-Net, EfficientNet-B4) | Font, tipografi, hizalama, renk tutarsızlıkları |
| **SRM gürültü akışı** (sabit 5×5 yüksek geçiren filtreler) | Düz boyama, kopyala-taşı, farklı sıkıştırma geçmişi, difüzyon (inpainting) dolgusu |
| **OCR kural motoru** (EasyOCR + Regex) | Geçersiz IBAN (Mod-97), geçersiz TCKN, Matrah + KDV ≠ Genel Toplam, KDV oranı hatası |
| **Hibrit karar** | `P = 1 − (1 − p_DL)·(1 − r_OCR)` (noisy-OR) |

---

## Klasör yapısı

```text
document_forensics/
├── data/                        # .gitignore içinde
│   ├── 1_base_doctamper/        # DocTamper (LMDB'den dışa aktarılmış: <Küme>/images, <Küme>/masks)
│   ├── 2_finetune_aiforgedoc/   # AIForge-Doc (mevcut "finetune_aiforgedoc" klasörü de otomatik tanınır)
│   ├── 3_sroie2019/             # SROIE2019 orijinal fişler (yalnızca OOD yanlış-alarm testi)
│   ├── 4_synthetic_tr_en/       # Önceki sentetik sürüm (Stage B)
│   ├── 5_synthetic_tr_en_v3/    # Sentetik v3 (~8000 görüntü, Stage C)
│   ├── 6_synthetic_tr_en_v4/    # Sentetik v4: yarı taranmış / yarı dijital, düzenleme izli (12 000, Stage E)
│   ├── 7_real_originals/        # Gerçek orijinal belgeler (<kurum>/...)
│   ├── 8_real_forged/           # src/forge_real.py: gerçek belgelerin sahte varyantları (Stage E)
│   ├── 9_human_forged/          # tools/make_human_masks.py: gönüllü sahtecilikleri (yalnızca test)
│   ├── 10_diffusion_forged/     # tools/diffusion_inpaint.py: difüzyon inpainting sahtecilikleri
│   └── splits/                  # Önbellek: AIForge-Doc kaynak-grup eşlemesi
├── src/
│   ├── dataset.py               # Manifest, sızıntısız grup ayırıcı, PyTorch Dataset
│   ├── models.py                # SRM + çift kafalı U-Net, kayan pencere çıkarımı
│   ├── losses.py                # Dice + 2·Focal (+ belge BCE)
│   ├── metrics.py               # AUC, Piksel-F1/IoU, ECE, Brier, sıcaklık ölçekleme
│   ├── generate_synthetic.py    # TR/EN dekont-fatura-fiş + sahtecilik + maske üretici
│   ├── ocr_checker.py           # EasyOCR + IBAN Mod-97 + TCKN/VKN + KDV kural motoru
│   ├── explain.py               # Hata haritası (TP/FP/FN) ve Grad-CAM
│   ├── dashboard.py             # Model başarımı sekmesi (Plotly)
│   └── runtime.py               # Dağıtım: ağırlık indirme, EasyOCR/CUDA kontrolleri
├── checkpoints/<run>/           # best.pth, last.pth, splits.csv, config.json, calibration.json
├── train.py · evaluate.py · predict.py · benchmark.py · app.py
├── Dockerfile · .dockerignore · packages.txt · .streamlit/config.toml
└── requirements.txt
```

## Kurulum

```bash
python -m venv .venv && .venv\Scripts\activate         # Linux/macOS: source .venv/bin/activate
# GPU için önce https://pytorch.org adresindeki CUDA'lı torch komutunu çalıştırın
pip install -r requirements.txt
```

## Kullanım

```bash
# 1) Sentetik Türkçe/İngilizce veri (varsayılan 2000 belge × (1 orijinal + 3 sahte) ≈ 8000 görüntü)
python -m src.generate_synthetic                       # → data/5_synthetic_tr_en_v3
python -m src.generate_synthetic --num-images 10000    # hedef görüntü sayısıyla

# 2) Eğitim — yalnızca sentetik veri
python train.py --sources synthetic --run-dir checkpoints/synthetic --epochs 50 --batch-size 8

#    DocTamper ile temel eğitim → AIForge-Doc + sentetik ile ince ayar
python train.py --sources doctamper --run-dir checkpoints/base
python train.py --sources synthetic aiforgedoc --init checkpoints/base/best.pth --lr 1e-4 \
                --run-dir checkpoints/finetuned

# 3) Değerlendirme (test + OOD kümeleri, val üzerinde kalibrasyon)
python evaluate.py --checkpoint checkpoints/synthetic/best.pth --splits checkpoints/synthetic/splits.csv

# 4) Web arayüzü
streamlit run app.py
```

TensorBoard kuruluysa: `tensorboard --logdir checkpoints`.

### Kaggle: aşamalı eğitim ve kaldığı yerden devam

`/kaggle/working` varsa çıktılar varsayılan olarak oraya yazılır (`last.pth`, `best.pth`, `<stage>_best.pth`).
`last.pth` optimizer, scheduler, AMP ölçekleyici ve erken durdurma durumunu da içerir.

```bash
# Stage A — DocTamper temel eğitim (epoch başına 20 000 örnek, val için sabit 2000 görüntü)
python train.py --stage stageA --samples-per-epoch 20000
# Oturum kesildiyse kaldığı epoch'tan devam
python train.py --stage stageA --resume /kaggle/working/last.pth
# Stage B — stageA_best.pth'den Türkçe sentetik + SROIE2019 ile ince ayar (lr=1e-5)
python train.py --stage stageB
# Stage C — Stage B'den yalnızca TR/EN sentetik ince ayar, TR val ile model seçimi
python train.py --stage stageC --init /kaggle/input/<ağırlık-veri-seti>/finetuned_multilingual.pth                 --data-root /kaggle/input/<sentetik-v3-veri-seti>
```

Stage B'de SROIE2019 eğitime girdiği için OOD yanlış-alarm testi olarak kullanılamaz.
`--samples-per-epoch 0` / `--val-samples 0` tüm kümeyi kullanır; karışık hassasiyet (AMP) GPU'da varsayılan olarak açıktır (`--no-amp` ile kapatılır).

DocTamper LMDB'yi klasöre aktarmak için (`pip install lmdb`):

```bash
python -c "from src.dataset import export_doctamper_lmdb as e; e('DocTamperV1-TrainingSet', 'data/1_base_doctamper/TrainingSet')"
```

---

## Son geliştirmeler: Adım A · B · C

```text
             Stage A                      Stage B                          Stage C (yeni)
  DocTamper ───────────► base_doctamper ───────────► finetuned_multilingual ───────────► stageC/best.pth
  lr 3e-4                .pth             TR/EN sent.  .pth                 yalnız TR/EN     (TR val ile seçilir)
                                          + SROIE                           sentetik v3
                                          lr 1e-5                           lr 1e-5, 12 ep.
  ───────────────────────────────────────────────────────────────────────────────────────────────
  Tüm aşamalarda eğitim dönüşümü:  kırpma → parlaklık/kontrast → JPEG(Q∈[50,100], p=0.5) → normalize
```

**Adım A — Türkçe transfer öğrenme (Stage C).**
* `src/generate_synthetic.py` varsayılan olarak 2000 belge × 4 ≈ **8000 görüntü** üretir
  (`--num-images 5000…10000`), çıktı `data/5_synthetic_tr_en_v3`'e yazılır ve mevcut veri seti
  `--overwrite` verilmeden ezilmez. `src/dataset.py` bu klasörü `4_synthetic_tr_en`'den önce arar.
* `train.py --stage stageC`: başlangıç `checkpoints/finetuned_multilingual.pth`, kaynak yalnızca `synthetic`
  (DocTamper/SROIE eğitime girmez), AdamW (lr 1e-5, weight decay 1e-4), 12 epoch, çıktı `checkpoints/stageC/`.
* **TR validasyon kümesi (`val_tr`).** `--val-lang tr` ile model seçimi, erken durdurma ve LR planlaması yalnızca
  Türkçe sentetik val belgeleri üzerinden yapılır; kullanılan liste `<run_dir>/val_tr.csv` olarak kaydedilir.
  Gerekçe: Stage B'nin `val/score=0.891`'i 882 DocTamper'a karşı 24 sentetik val görüntüsünden hesaplanmıştı ve
  Türkçe belgelerdeki başarımı yansıtmıyordu (`results/stageB/`: sentetik testte belge AUC ≈ 0.47).
* Epoch geçmişi `<run_dir>/history.csv`'ye yazılır; uygulamadaki **Model Başarımı** sekmesi bunu otomatik çizer.

**Adım B — JPEG sıkıştırma dayanıklılığı.** Eğitim dönüşümlerine
`A.ImageCompression(quality_range=(50, 100), p=0.5)` eklendi (Albumentations 1.x için `quality_lower/upper`
yedeği var; `--jpeg-aug-p 0` ile kapatılır). WhatsApp/e-posta/ekran görüntüsü yeniden sıkıştırması SRM
kalıntısını zayıflattığından model bu bozulmayı eğitimde görür. Etkisi `benchmark.py`'nin Q=100→50
dayanıklılık eğrisiyle ölçülür.

**Adım C — Canlı yayın.** Depo Hugging Face Spaces (Docker SDK) için hazırdır; ayrıntılar aşağıda.

---

## Dağıtım (Hugging Face Spaces / Docker / Streamlit Cloud)

Bu README'nin başındaki YAML bloğu Space ayarlarıdır (`sdk: docker`, `app_port: 7860`).

1. Ağırlıkları bir **HF model deposuna** yükleyin: `stageE/best.pth`, `stageE/calibration.json` (varsayılan model),
   `finetuned_multilingual_stageB.pth`, `base_doctamper.pth`, `calibration.json` (karşılaştırma). Liste
   `src/model_registry.py::DEPLOY_FILES`'tadır. `*.pth` dosyaları `.gitignore` nedeniyle Space deposuna girmez.
   **Not:** `finetuned_multilingual.pth` 21.10'da bir Stage C ara modeliyle (val FPR 1.00) ezildiği için artık
   dağıtılmaz; gerçek Stage B `finetuned_multilingual_stageB.pth`'tir.
2. Space → *Settings → Variables*: `FORENSICS_WEIGHTS_REPO=<kullanici>/<model-deposu>`
   (özel depo ise *Secrets*'a `HF_TOKEN`). Alternatif: `FORENSICS_WEIGHTS_URL=<dosyaların temel URL'si>`.
3. Depoyu Space'e gönderin; `Dockerfile` CPU-only PyTorch kurar ve Streamlit'i 7860'ta başlatır.

```bash
docker build -t document-forensics .
docker run -p 7860:7860 -e FORENSICS_WEIGHTS_REPO=<kullanici>/<model-deposu> document-forensics
```

**Esnek çalışma (fallback):**

| Durum | Davranış |
|---|---|
| CUDA yok | Model ve OCR CPU'da çalışır; "OCR için GPU" seçeneği devre dışı |
| EasyOCR kurulu değil | OCR seçeneği devre dışı, açıklaması gösterilir |
| EasyOCR modeli indirilemedi / OCR hata verdi | Uyarı gösterilir, görsel analiz sonuçları yine sunulur |
| Ağırlık yok / indirilemedi | Uyarı gösterilir; model seçimi boş, yalnızca OCR analizi |
| Ağırlık bozuk / bellek yetersiz | Derin öğrenme adımı uyarıyla atlanır, uygulama çökmez |
| Seçili modelin sonucu yok | Model Başarımı sekmesi `python evaluate.py` / `python benchmark.py` komutunu gösterir |

**Streamlit Community Cloud:** `packages.txt` OpenCV'nin sistem kütüphanelerini kurar. Ancak ücretsiz katmanın
~1 GB belleği EfficientNet-B4 + EasyOCR için genellikle yetersizdir ve `requirements.txt`'teki `torch` Linux'ta
CUDA tekerleğini indirir; bu nedenle HF Spaces (16 GB CPU) önerilir. `results/` `.gitignore` içinde olduğundan
değerlendirme panelinin canlıda görünmesi için Stage E'nin `results/` çıktıları (`summary.json`, `breakdown_*.csv`,
`results/benchmark_stageE/`; birkaç MB) depoya ayrıca eklenmelidir.

**Model listesi:** uygulama yalnızca `src/model_registry.py`'deki önerilen (Stage E) ve karşılaştırma (Stage B, A)
modellerini gösterir. Deneysel / ara ağırlıklar (Stage C/D, `runs/fast_run`) kenar çubuğundaki
"Deneysel / eski modelleri de göster" anahtarıyla listelenir ve seçildiklerinde güvenilmezlik uyarısı çıkar.

---

## Veri sızıntısını önleme

1. **Grup düzeyinde bölme (70/15/15).** Aynı orijinal belgeden türeyen tüm görüntüler aynı `group_id`'yi
   taşır ve tek bir bölüme gider. `verify_no_leakage()` her çalıştırmada bunu doğrular, ihlalde hata fırlatır.
   Resmi test kümesi olan kaynaklarda (AIForge-Doc, DocTamper) ikinci bir test kümesi ayrılmaz; serbest
   gruplar 70:15 oranıyla yalnızca train/val'e bölünür.
2. **Resmi test kümeleri korunur.** Bir grubun tek bir üyesi bile resmi test kümesindeyse grubun tamamı
   test'e gider (`ood > test > val > train` önceliği).
3. **AIForge-Doc kaynak eşlemesi.** `metadata.jsonl` kaynak görüntü kimliği içermez ve resmi
   Training/Testing bölümleri kaynak paylaşır: 254 orijinal fiş iki bölümde birebir aynıdır (MD5) ve
   piksel eşlemesine göre **TrainingSet'teki 1208 görüntü (606 sahte + 602 orijinal) TestingSet ile aynı
   kaynak fişten türemiştir**. `link_aiforge_groups()` her görüntüyü piksel benzerliğiyle (64×64 küçük
   resim, yalnızca mürekkep piksellerinde |fark|'ın 75. yüzdeliği < 1.5) kaynağına bağlar ve Union-Find
   ile 8122 görüntüden 4095 kaynak grup oluşturur. Bu gruplar test'e taşınır. Sonuç
   `data/splits/aiforgedoc_groups.csv` olarak önbelleğe alınır (ilk çalıştırma ~6 dk).
4. **Kaynaklar arası tekrar.** AIForge-Doc'un bir kısmı SROIE fişlerinden üretilmiştir. AIForge-Doc eğitimde
   kullanılırsa, aynı adlı SROIE2019 fişleri OOD değerlendirmesinden otomatik çıkarılır.
5. **Diskte ön işleme yok.** Boyut sınırlama, kırpma ve normalizasyon yalnızca `ForgeryDataset.__getitem__`
   içinde bellekte yapılır. Diske yazılan tek şey dosya listeleridir (`splits.csv`).
6. **Değerlendirme izolasyonu.** `train.py` yalnızca `train` ve `val` satırlarını okur; erken durdurma ve
   öğrenme oranı planlaması val ile yapılır. Kalibrasyon sıcaklığı da val'de öğrenilir. `--sources` listesinde
   olmayan kaynaklar (varsayılan: AIForge-Doc, SROIE2019) `ood` olarak işaretlenir ve yalnızca
   `evaluate.py` tarafından görülür.
7. **Kestirme ipucu yok.** Sentetik üreticide orijinal ve sahte varyantlar aynı edinim zincirinden ve aynı son
   JPEG kalitesinden geçer; model "kaç kez sıkıştırıldı" bilgisinden değil, yerel düzenleme izlerinden öğrenir.

**Bilinen sınırlama:** DocTamper kaynak belge kimliği yayınlamadığı için `TrainingSet` içinde her görüntü kendi
grubudur. Resmi `TestingSet`/`FCD`/`SCD` kümeleri ise daima test olarak ayrılır.

---

## Yöntem notları

**SRM filtreleri.** Üç sabit 5×5 çekirdek (KB, KV ve yatay 2. derece fark) katsayı toplamı sıfır olan
yüksek geçiren filtrelerdir: düz bölgelerde 0 üretir, geriye komşu piksellerden tahmin edilemeyen
**gürültü kalıntısı** kalır. Çıktı TLU ile `[−3, 3]` aralığına kırpılır, böylece ağ metin kenarlarından
çok gürültü istatistiğine odaklanır. Filtreler `register_buffer` ile tutulur, eğitilmez.

**Kayıp.** `L = Dice + 2·Focal + 0.5·BCE(yama etiketi)`. Dice küçük sahte bölgelerde sınıf dengesizliğini
telafi eder; Focal (γ=2) zor kenar piksellerine odaklanır. Belge kafası yama düzeyinde etiketle eğitilir
("bu kırpıntıda sahte piksel var mı?"); çıkarımda pencerelerin maksimumu alınır.

**Çıkarım.** Eğitim yerel kırpıntılarla yapıldığından çıkarım da yeniden ölçeklemeden, %25 örtüşmeli
512×512 kayan pencerelerle yapılır (SRM gürültü istatistiği korunur).

**Hibrit skor.** `P = 1 − (1 − p_DL)·(1 − r_OCR)`. `r_OCR` yalnızca *ihlal edilen* kurallardan oluşur
(`1 − Π(1 − w_i)`; IBAN 0.90, Toplam 0.85, TCKN 0.70, KDV oranı 0.60). Geçen kurallar skoru düşürmez:
matematiği koruyan bir sahtecilik (`amount_consistent`) yalnızca piksel izleriyle yakalanabilir.

**Sentetik sahtecilik türleri.** `copy_move`, `text_rewrite`, `amount_tamper` (matematik bozulur),
`amount_consistent` (matematik korunur), `iban_tamper` (Mod-97 bozulur), `inpaint_sim`
(üretken YZ inpainting benzetimi: `cv2.inpaint` + yumuşatma).

## Metrikler (`evaluate.py`)

| Düzey | Metrik |
|---|---|
| Belge | ROC-AUC (Mann-Whitney), AP, doğruluk, TPR, FPR, **TPR@FPR = %1 / %2 / %5** |
| Karar (`is_forged`) | TP / FP / FN / TN, precision, recall, specificity, F1, accuracy, dengeli doğruluk |
| Piksel | F1 (= Dice), IoU, precision, recall — küresel ve sahte görüntü başına; örneklenmiş piksellerden ROC-AUC ve AP |
| Kalibrasyon | ECE (15 kutu), Brier, güvenilirlik tablosu (`results/reliability_*.csv`) |
| Belirsizlik | %95 **bootstrap güven aralıkları** (`*_ci_lo` / `*_ci_hi`): belge AUC, TPR@FPR=%2, karar TPR/FPR/F1, piksel F1 |

Kaynak başına ayrı raporlanır (`results/summary.csv`). Yalnızca orijinal belgeden oluşan kümelerde
(SROIE2019) AUC tanımsızdır; FPR (yanlış alarm oranı) yorumlanmalıdır. `--error-maps N` her küme için
TP/FP/FN hata haritaları, `--ocr` OCR kural motoru riskini (`ocr_risk`) de kaydeder.

---

## Değerlendirme, ablasyon ve baseline araçları

| Araç | Ne yapar? |
|---|---|
| `evaluate.py` | Yukarıdaki metrikler + güven aralıkları; `arrays_*.npz` (ablasyon için), `--error-maps`, `--ocr`, `--external-maps` (dış model) |
| `ablation_decision.py` | Yeniden çıkarım yapmadan: belge havuzlaması (max / top-%5 / %10 / %25 / ortalama), `fuse_doc_score` açık/kapalı, karar kuralı, en küçük bölge alanı, sıcaklık, OCR hibrit |
| `predict.py --gt-mask m.png --gradcam` | Tek belge: hata haritası (yeşil = TP, kırmızı = FP, mavi = FN) ve sınıflandırma kafası için Grad-CAM |
| `app.py` | Kenar çubuğunda **Grad-CAM** anahtarı; "Açıklanabilirlik" bölümünde gerçek maske yüklenirse hata haritası |
| `tools/build_test_protocol.py` | Tüm modeller için tek, dondurulmuş test dosyası (`results/protocol/test_v1.csv`); DocTamper TestingSet / FCD / SCD ayrı satırlarda |
| `tools/export_eval_images.py` | Test görüntülerini (LMDB dahil) ham baytlarıyla TruFor / DTD / CAT-Net için dışa aktarır |
| `tools/convert_baseline_outputs.py` | Dış model çıktılarını (`.npz` / `.npy` / `.png`) `evaluate.py --external-maps` biçimine çevirir |
| `tools/compare_models.py` | Birden fazla modelin özetini güven aralıklı tek karşılaştırma tablosunda birleştirir (`.csv` + `.md`) |
| `tools/plot_history.py` | `history.csv` dosyalarından train/val kayıp ve metrik eğrileri (overfitting / underfitting analizi) |
| `tools/make_human_masks.py` | Gönüllü (insan eliyle yapılmış) sahtecilik çiftlerinden maske → `data/9_human_forged` (**yalnızca test**) |
| `tools/diffusion_inpaint.py` | Gerçek difüzyon inpainting ile YZ sahteciliği → `data/10_diffusion_forged` (test grupları korunur) |
| `tools/ocr_eval.py` | OCR alan çıkarma doğruluğu, kural bazında yanlış alarm / tetiklenme oranı, EasyOCR ↔ PaddleOCR |
| `tools/splits_from_predictions.py` | Kaggle'da üretilmiş `predictions_*.csv`'den yerel test bölmesi (eğitimde görülen belgeler teste karışmaz) |
| `src/model_registry.py` | Ağırlık dosyalarının gerçek içeriğine göre etiketleri; önerilen / karşılaştırma / deneysel ayrımı |

**Ablasyon ve iyileştirme seçenekleri (`train.py`):**

```bash
# Akış ablasyonu (aynı veri, aynı bütçe, aynı seed)
python train.py --stage stageE --streams rgb         --run-dir checkpoints/abl_rgb
python train.py --stage stageE --streams srm         --run-dir checkpoints/abl_srm
python train.py --stage stageE --streams rgb+srm     --run-dir checkpoints/abl_rgbsrm
# DCT (JPEG) akışı + kayıpsız çevirme + ayrımsal LR + JPEG müfredatı + çalışma noktasına göre model seçimi
python train.py --stage stageE --streams rgb+srm+dct --flip-aug-p 0.5 --encoder-lr-mult 0.1     --jpeg-curriculum 3 --monitor image_tpr_at_fpr02 --init checkpoints/stageE/best.pth --run-dir checkpoints/stageF
# Zayıf sahtecilik türlerini daha sık örnekle (sahte sınıfın toplam payı değişmez)
python train.py --stage stageE --init checkpoints/stageE/best.pth --lr 3e-5     --type-weights copy_move=3 digit_change=3 delete=2 --run-dir checkpoints/stageF
# Görülmemiş kurum / belge tipine genelleme
python train.py --stage stageE --holdout-issuers garanti   --run-dir checkpoints/holdout_garanti
python train.py --stage stageE --holdout-doc-types fis     --run-dir checkpoints/holdout_fis
# Tekrarlanabilirlik: ana sonuçlar için 3 seed (ortalama ± std)
for s in 42 43 44; do python train.py --stage stageE --seed $s --run-dir checkpoints/stageE_s$s; done
```

* `--streams rgb+srm+dct`: DCT katmanı **sıfırla başlatılır**; mevcut bir RGB+SRM ağırlığından (`--init`) başlayan
  model ilk adımda birebir aynı çıktıyı verir. Eğitim kırpmaları JPEG 8x8 ızgarasına hizalanır. Uzun kenarı
  `max_side`'dan büyük olup küçültülen görüntülerde ve PNG ekran görüntülerinde DCT akışı katkı vermez.
* `--init` farklı akışlı bir ağırlıktan uyumlu yüklenir (6 → 3 kanal ilk konvolüsyon dilimlenir).

**Uygulamadaki kararları yerelde yeniden üretmek (Kaggle'daki test bölmesiyle):**

```bash
python tools/splits_from_predictions.py --predictions "results/predictions_test_*.csv"     --out results/protocol/stageE_kaggle_test.csv
python evaluate.py --checkpoint checkpoints/stageE/best.pth --splits results/protocol/stageE_kaggle_test.csv     --no-calibration --temperature 4.1956 --threshold 0.77 --error-maps 5 --out results/stageE_local
python ablation_decision.py --results results/stageE_local
```

**Baseline akışı (TruFor örneği):**

```bash
python tools/build_test_protocol.py --from-splits checkpoints/stageE/splits.csv --out results/protocol/test_v1.csv
python tools/export_eval_images.py --splits results/protocol/test_v1.csv --out baselines/eval_images
# TruFor deposunun çıkarım betiğiyle (ör. trufor_test.py -in <girdi> -out <çıktı>; komut için deponun README'sine bakın):
#   girdi = baselines/eval_images/images, çıktı = baselines/trufor_raw
python tools/convert_baseline_outputs.py --inputs baselines/trufor_raw --out baselines/trufor/maps
python evaluate.py --external-maps baselines/trufor/maps --external-scores baselines/trufor/maps/scores.csv     --splits results/protocol/test_v1.csv --out results/baselines/trufor
python evaluate.py --checkpoint checkpoints/stageE/best.pth --splits results/protocol/test_v1.csv --out results/stageE
python tools/compare_models.py --run "Stage E=results/stageE" --run "TruFor=results/baselines/trufor"
```

Dış modelin çıktı anahtar adları sürüme göre değişebilir: `convert_baseline_outputs.py --list-keys` ile kontrol
edin (`--map-key`, `--score-key`).
