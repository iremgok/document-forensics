# Model Kartı — Stage E (`checkpoints/stageE/best.pth`)

> Bu karttaki her sayı, yanında belirtilen dosyadan **okunmuştur**. Dosya yolları depo köküne göredir.
> Son güncelleme: 2026-11-07.

## 1. Kapsam

**Amaç:** Türkçe dijital banka dekontları ve e-Arşiv faturalarında (PDF, PNG, ekran görüntüsü) yapılmış
sahteciliği **belge düzeyinde** tespit etmek ve **piksel düzeyinde** konumlandırmak. Model bir karar destek
aracıdır; tek başına hukuki ya da finansal karar için kullanılmamalıdır.

| | Belge / edinim türü |
|---|---|
| **Desteklenen** | Bankanın / GİB'in ürettiği PDF ve onun 200 DPI PNG dışa aktarımı; ekran görüntüsü; WhatsApp üzerinden iletilmiş kopya (uzun kenar ≤ 1600, JPEG) |
| **Deneysel / kapsam dışı** | Telefonla çekilmiş fotoğraf, tarayıcıdan geçmiş kâğıt belge, termal fiş, İngilizce belgeler, el yazısı içeren belgeler |

Kapsam dışı belgelerde model çalışır, ancak bu koşullar için doğrulanmış bir başarım yoktur (bkz. §6).

## 2. Mimari

U-Net + EfficientNet-B4 kodlayıcı; girdi RGB + sabit SRM gürültü kalıntısı (6 kanal, `src/models.py`).
İki çıktı: piksel sahtecilik haritası ve belge logit'i. Çıkarım: uzun kenar ≤ 1600 px, 512 px'lik %25 örtüşen
kayan pencere; belge skoru en şüpheli %10 pencerenin ortalaması. Checkpoint yapılandırması
(`checkpoints/stageE/best.pth` → `config`): `encoder=efficientnet-b4`, `crop_size=512`, `max_side=1600`,
`jpeg_aug_p=0.5`, `balance=True`, `seed=42`.

Uygulamada model; OCR kural motoru (`src/ocr_checker.py`), PDF yapı analizi (`src/pdf_checker.py`) ve isteğe
bağlı kopyala-taşı modülüyle (`src/copymove.py`) noisy-OR hibrit skorda birleşir. Aşağıdaki sayılar
**yalnızca görüntü modelinin** kararına aittir.

## 3. Eğitim verisi ve aşamalar

| Aşama | Başlangıç | Veri | Not |
|---|---|---|---|
| **A** (`checkpoints/base_doctamper.pth`) | ImageNet | DocTamper (Çince belgeler) | Hiç orijinal belge görmedi |
| **B** (`checkpoints/finetuned_multilingual_stageB.pth`) | Stage A | Sentetik TR/EN + SROIE2019 | — |
| **E** (`checkpoints/stageE/best.pth`) | **Stage A** | Sentetik v4 (`data/6_synthetic_tr_en_v4`, yarı dijital / yarı taranmış, düzenleme izli) + gerçek belgelerden üretilmiş sahteler (`data/8_real_forged`) | Checkpoint `config.sources = ['synthetic', 'real_forged']`, en iyi epoch 12 |

Stage C/D (yalnız sentetik) Stage E'ye taşınmadı: "alan konumu = sahte" ezberi nedeniyle val FPR ≈ 1.0'dı
(`src/model_registry.py`). Gerçek belgeler: 143 orijinal (Garanti 37, Trendyol 72, Ziraat 34; hepsi PNG, biri JPEG
taranmış belge — `data/8_real_forged/metadata.csv`), her birinden `src/forge_real.py` ile 3 sahte varyant.
Örnekleme orijinal/sahte 50/50; model seçimi TR sentetik + gerçek belge val kümesinde yapıldı. İzlenen metrik
checkpoint'te kayıtlı değil (`config`'te `monitor` yok); o sürümün varsayılanı `val/score` = (piksel F1 + belge
AUC) / 2'dir. Yeni `stageE`/`stageF` ön ayarları karar kuralıyla TPR@FPR=%2'yi izler.

**Bilinen veri hatası:** `data/8_real_forged` (v1), 200 DPI'da PNG'ye çevrilmiş 50 dijital belgeyi (37 Trendyol,
13 Ziraat) gürültü ölçümüne dayanarak `acquisition=photo` diye etiketledi ve sahte bölgelerine fotoğraf
bulanıklığı/gürültüsü ekledi (`python tools/audit_acquisition.py`). Stage E bu veriyle eğitildi. Düzeltilmiş
üretici `data/8_real_forged_v2`'ye yazar; yeniden eğitim henüz yapılmadı.

## 4. Kalibrasyon ve çalışma noktası

Kaynak: `checkpoints/stageE/calibration.json` (eski tek-T şema; val'de `evaluate.py` ile öğrenildi).

| Alan | Değer |
|---|---|
| Sıcaklık (belge **ve** piksel; tek T) | 4.196 |
| Karar eşiği (belge **ve** piksel) | 0.77 |
| Hedef val FPR | 0.05 |
| Val FPR / TPR | 0.0488 / 0.787 |
| Val görüntü sayısı | 1884 |

Karar kuralı (`predict.is_forged`): *belge skoru ≥ eşik ve haritada en az hafif iz* **veya** *eşiği geçen
bağlantılı bölge ≥ sayfanın %0.05'i*. Tek T piksel haritasına da uygulandığından piksel eşiği 0.77, ham
olasılıkta σ(4.196 · logit 0.77) ≈ 0.994'e denk gelir. Kod artık belge ve piksel tarafını ayrı kalibre ediyor
(yeni şema: `doc_temperature/doc_threshold/pixel_temperature/pixel_threshold`), fakat Stage E'nin val bölmesi
yerelde olmadığı için yeni kalibrasyon **henüz üretilmedi**. Bilgi amaçlı test analizi için §5.3'e bakın.

Not: val görüntü sayısı (1884), test protokolünün görüntü sayısına (`results/protocol/stageE_kaggle_test.csv`,
1884) eşittir. Bu, tüm grupların 4 görüntülü olması ve val/test oranlarının (%15/%15) aynı olmasıyla
açıklanabilir; ancak val ve test grup listeleri yerelde karşılaştırılamadı (Stage E `splits.csv` yerelde yok).

## 5. Değerlendirme

### 5.1 Gerçek tabanlı sahte testi (84 görüntü: 21 orijinal, 63 sahte)

Kaynak: `results/compare_real/stageE/summary.json` (bölme `results/protocol/stageE_kaggle_test_real.csv`;
`results/stageE_local/summary.json` ile birebir aynı). Köşeli parantez: %95 bootstrap güven aralığı (1000 yineleme,
sınıf katmanlı).

| Metrik | Değer |
|---|---|
| Belge AUC | 0.828 [0.727–0.904] |
| Karar TPR (eşik 0.77) | 0.571 [0.460–0.698] |
| Karar FPR | 0.048 [0.000–0.143] (21 orijinalden 1'i) |
| Karar F1 | 0.720 [0.623–0.811] |
| TPR @ FPR=%2 (yalnız belge skoru) | 0.397 [0.286–0.730] |
| Piksel F1 (genel) | 0.604 [0.464–0.720] |

Aynı bölmede önceki aşamalar (`results/comparison.md`, kaynaklar `results/compare_real/stage{A,B}/summary.json`):

| Model | Kalibrasyon | Belge AUC | Karar TPR | Karar FPR | Piksel F1 |
|---|---|---|---|---|---|
| Stage A | kalibre edilmemiş (genel T=1, eşik 0.60) | 0.501 [0.40–0.61] | 1.000 | 1.000 | 0.022 |
| Stage B | kalibre edilmemiş (genel T=1, eşik 0.60) | 0.499 [0.36–0.64] | 0.857 | 0.857 | 0.004 |
| Stage E | kalibre | 0.828 [0.73–0.90] | 0.571 | 0.048 | 0.604 |

Stage A/B'nin karar metrikleri doğrulanmış bir çalışma noktasına dayanmaz; eşikten bağımsız AUC
karşılaştırılabilir ve şansa eşittir.

### 5.2 Sentetik test (1800 görüntü: 450 orijinal, 1350 sahte)

Kaynak: `results/summary.json` → `test/synthetic` (Kaggle'da aynı kalibrasyonla).

| Metrik | Değer |
|---|---|
| Belge AUC | 0.912 |
| Karar TPR / FPR | 0.819 / 0.067 |
| Piksel F1 | 0.912 |

### 5.3 Ayrı kalibrasyon — yalnızca bilgi amaçlı (test kümesinde)

Kaynak: `results/compare_real/stageE/ablation_decision.csv` ve `kalibrasyon_bilgi_amacli.csv`. **Bu değerler test
kümesinde hesaplanmıştır; hiçbir eşik/sıcaklık bunlara göre seçilmemiştir ve seçilmemelidir.**

| Yapılandırma (eşikler val'deki 0.77) | TPR | FPR |
|---|---|---|
| Tek T (eski, varsayılan) | 0.571 | 0.048 |
| Piksel T = 1, piksel eşiği 0.77 | 0.619 | 0.048 |
| Belge T = 1 (piksel tarafı değişmeden) | 0.857 | 0.524 |
| Test kümesinde seçilmiş en iyi çift, FPR ≤ 0.05 (iyimser üst sınır) | 0.667 | 0.048 |

Eski ablasyondaki "T=1 → TPR 0.86 / FPR 0.52" satırı yalnızca **belge** sıcaklığını değiştirir; artış belge
koşulundan gelir, bölge koşulundan değil. Bölge koşulu tek başına 0.77 eşikte sahtelerin %44'ünü yakalar
(FPR 0.048). Ayrı kalibrasyonun gerçek etkisi ancak val kümesinde `evaluate.py` ile ölçülebilir.

## 6. Bilinen kör noktalar

Kaynak: `results/compare_real/stageE/breakdown_test_real_forged.csv` (gerçek tabanlı test) ve
`results/breakdown_test_synthetic.csv` (sentetik test). Gerçek tabanlı testte tür başına sahte sayısı 4-13'tür;
oranlar çok belirsizdir.

| Kör nokta | Gerçek tabanlı test tespit oranı | Sentetik test tespit oranı | Açıklama |
|---|---|---|---|
| **Kopyala-taşı** | 0.333 (n=9) | 0.328 (n=253) | Belgenin kendi glifi kopyalandığı için gürültü izi yok. Ayrı modül (`src/copymove.py`) yalnızca gürültülü zeminde ve yeniden örneklenmemiş kopyada kanıt üretir; `forge_real` kopyaları yeniden boyutlandırdığı için testte 0/9 yakaladı, bir orijinalde yanlış eşleşti. Modül varsayılan olarak kapalı. |
| **Tek rakam değişikliği** | 0.25 (n=4, `digit_change_1`) | — | İki ve daha fazla rakamda 0.50 (n=4) |
| **Sahte isim** | 0.583 (n=12, `name_swap`) | — | — |
| **Silme** | 0.429 (n=7) | — | — |
| **Temiz PDF düzenlemesi** | ölçülmedi | ölçülmedi | PDF'i bir editörle düzenleyip yeniden rasterleştirmek gürültü izi bırakmaz. PDF yapı analizi (`src/pdf_checker.py`) bunun için tasarlandı, ancak gerçek PDF verisiyle henüz değerlendirilmedi (`data/7_real_originals` içinde PDF yok; araç `tools/pdf_eval.py`). |

Taranmış belgelerde (sentetik) FPR 0.103'tür, dijitalde 0.031 (`results/breakdown_test_synthetic.csv` → edinim).

## 7. Test edilmemiş koşullar ve sınırlamalar

* **Test kümesi küçük:** gerçek tabanlı test 84 görüntü / **21 belge** (21 orijinal + 63 sahte varyant). FPR'nin
  güven aralığı 0–0.14'tür; tek bir ek yanlış alarm FPR'yi 0.048'den 0.095'e çıkarır.
* **İyimser sonuçlar:** test sahteleri eğitimdeki üreticiyle (`src/forge_real.py`) üretildi. Model üreticinin
  izlerini öğrenmiş olabilir; gerçek sahtecilerin araç ve tekniklerine genelleme ölçülmedi.
* **Elle yapılmış test kümesi yok:** gönüllülerin Photoshop/GIMP/telefonla yaptığı sahtecilikler için altyapı
  hazır (`tools/make_human_masks.py`, `data/9_human_forged`), fakat veri henüz toplanmadı.
* **Kurum çeşitliliği:** yalnızca Garanti, Trendyol ve Ziraat belgeleri var. Görülmemiş bir bankanın şablonuna
  genelleme ölçülmedi (`train.py --holdout-issuers` ile ölçülebilir).
* **Edinim etiketi hatası:** gerçek tabanlı testteki "photo" kırılımı (6 orijinal / 18 sahte) aslında dijital
  PNG'lerdir (§3); fotoğraf/tarama üzerindeki başarım için gerçek fotoğraf verisi yoktur.
* **Telefon fotoğrafı, tarama, termal fiş, İngilizce belge:** kapsam dışı; gerçek veride ölçülmedi.
* **Dayanıklılık:** JPEG Q=100→50 ve WhatsApp benzetimi `benchmark.py`'de ölçülür; Stage E için güncel test
  protokolüyle benchmark sonucu henüz üretilmedi (`results/benchmark_stageE` yok).
* **Yalnızca tek tohum:** sonuçlar tek eğitim çalışmasına (seed 42) aittir; tohumlar arası değişkenlik bilinmiyor.
