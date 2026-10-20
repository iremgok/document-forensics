# Türkçe Finansal Belgelerde Geleneksel ve Üretken YZ Kaynaklı Sahteciliğin Tespiti ve Konumlandırılması

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
│   ├── 4_synthetic_tr_en/       # src/generate_synthetic.py çıktısı
│   └── splits/                  # Önbellek: AIForge-Doc kaynak-grup eşlemesi
├── src/
│   ├── dataset.py               # Manifest, sızıntısız grup ayırıcı, PyTorch Dataset
│   ├── models.py                # SRM + çift kafalı U-Net, kayan pencere çıkarımı
│   ├── losses.py                # Dice + 2·Focal (+ belge BCE)
│   ├── metrics.py               # AUC, Piksel-F1/IoU, ECE, Brier, sıcaklık ölçekleme
│   ├── generate_synthetic.py    # TR/EN dekont-fatura-fiş + sahtecilik + maske üretici
│   └── ocr_checker.py           # EasyOCR + IBAN Mod-97 + TCKN + KDV kural motoru
├── checkpoints/<run>/           # best.pth, last.pth, splits.csv, config.json, calibration.json
├── train.py · evaluate.py · app.py
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
# 1) Sentetik Türkçe/İngilizce veri (1000 belge × (1 orijinal + 3 sahte) = 4000 görüntü)
python -m src.generate_synthetic --num-docs 1000 --variants 3

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
```

Stage B'de SROIE2019 eğitime girdiği için OOD yanlış-alarm testi olarak kullanılamaz.
`--samples-per-epoch 0` / `--val-samples 0` tüm kümeyi kullanır; karışık hassasiyet (AMP) GPU'da varsayılan olarak açıktır (`--no-amp` ile kapatılır).

DocTamper LMDB'yi klasöre aktarmak için (`pip install lmdb`):

```bash
python -c "from src.dataset import export_doctamper_lmdb as e; e('DocTamperV1-TrainingSet', 'data/1_base_doctamper/TrainingSet')"
```

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
| Belge | ROC-AUC (Mann-Whitney), doğruluk, TPR, FPR |
| Piksel | F1 ve IoU — küresel ve sahte görüntü başına ortalama |
| Kalibrasyon | ECE (15 kutu), Brier, güvenilirlik tablosu (`results/reliability_*.csv`) |

Kaynak başına ayrı raporlanır (`results/summary.csv`). Yalnızca orijinal belgeden oluşan kümelerde
(SROIE2019) AUC tanımsızdır; FPR (yanlış alarm oranı) yorumlanmalıdır.
