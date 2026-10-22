# Document Forensics — Sahtecilik ve Manipülasyon Tespit Sistemi

**Teknik Dokümantasyon ve Proje Raporu**

| | |
|---|---|
| **Kapsam** | Banka dekontu, fatura, perakende fişi ve resmi belgelerde piksel/içerik düzeyinde sahtecilik tespiti ve konumlandırma |
| **Model** | `DualStreamForensicNet` — RGB + SRM çift akışlı, çift kafalı (segmentasyon + sınıflandırma) U-Net / EfficientNet-B4 |
| **Ağırlıklar** | `checkpoints/base_doctamper.pth` (Stage A, epoch 25) · `checkpoints/finetuned_multilingual.pth` (Stage B, epoch 13) |
| **Kalibrasyon** | `checkpoints/calibration.json` → `temperature = 1.0`, `optimal_threshold = 0.60` |
| **Arayüz** | Streamlit (`app.py`) + Plotly değerlendirme paneli (`src/dashboard.py`) |
| **Sürüm / tarih** | `main` @ `1de3c9e` · 21.10.2026 |

> Bu belgedeki tüm sayısal değerler depodaki kontrol noktalarından (`*.pth` içindeki `metrics` sözlüğü) ve
> `results/stageB/` altındaki `benchmark.py` çıktılarından okunmuş ya da bu çıktılardan yeniden hesaplanmıştır.
> Kaynak her tabloda belirtilmiştir.

---

## İçindekiler

1. [Proje Özeti ve Amacı](#1-proje-özeti-ve-amacı)
2. [Derin Öğrenme Mimarisi ve Model Detayları](#2-derin-öğrenme-mimarisi-ve-model-detayları)
3. [Kalibrasyon ve Skorlama Mantığı](#3-kalibrasyon-ve-skorlama-mantığı)
4. [Kritik Eşik Değerleri ve Parametreler](#4-kritik-eşik-değerleri-ve-parametreler)
5. [Deneysel Sonuçlar ve Dersler](#5-deneysel-sonuçlar-ve-dersler)
6. [Sistem Mimarisi ve Arayüz](#6-sistem-mimarisi-ve-arayüz)
7. [Ek A — Dosya ve Parametre Referansı](#ek-a--dosya-ve-parametre-referansı)

---

## 1. Proje Özeti ve Amacı

### 1.1 Temel amaç

Finansal ve resmi belgeler (banka dekontu, e-fatura, perakende fişi, resmi yazı) dijital ortamda kolayca
düzenlenebilir: bir tutarın rakamı değiştirilebilir, IBAN'ın birkaç hanesi yeniden yazılabilir, bir alan
silinip üzerine farklı bir değer basılabilir ya da üretken yapay zekâ (difüzyon tabanlı *inpainting*) ile
bölge yeniden üretilebilir. Bu düzenlemeler çıplak gözle çoğunlukla fark edilmez.

Projenin amacı, bu tür **piksel düzeyindeki (görsel) ve içerik düzeyindeki (anlamsal/matematiksel)**
manipülasyonları otomatik olarak tespit etmek ve **nerede** yapıldıklarını göstermektir.

### 1.2 Çözülen problem ve sunulan çıktılar

Sistem tek bir belge görüntüsü (JPG/PNG) veya PDF sayfası için üç tamamlayıcı çıktı üretir:

| Çıktı | Düzey | Açıklama |
|---|---|---|
| **Sahtecilik olasılık haritası** | Piksel | Her piksel için $p \in [0, 1]$; jet ısı haritası ve bindirme olarak gösterilir (konumlandırma / *localization*) |
| **Belge sahtecilik olasılığı** | Görüntü | Kalibre edilmiş ve hibrit kuralla yumuşatılmış tek skor (tespit / *detection*) |
| **İçerik tutarlılık raporu** (opsiyonel) | Anlamsal | OCR ile okunan IBAN (Mod-97), TCKN ve Matrah + KDV = Genel Toplam kurallarının sonucu |

Bu çıktılar bir **karar kuralı** (`is_forged`, Bölüm 3.4) ile "manipülasyon tespit edildi / edilmedi"
sonucuna bağlanır. Böylece sistem hem bir sınıflandırıcı hem de bir adli inceleme (*forensic*) aracı olarak
çalışır: kullanıcı yalnızca kararı değil, kararın dayandığı bölgeyi de görür.

### 1.3 Hibrit yaklaşımın gerekçesi

İki kanıt kaynağı farklı sahtecilik türlerini yakalar:

* **Piksel izleri** (derin öğrenme): matematiği koruyan "akıllı" bir sahtecilikte (ör. Matrah, KDV ve
  Toplam birlikte değiştirilmiş — `amount_consistent`) kurallar geçer; geriye yalnızca düzenleme izi kalır.
* **İçerik kuralları** (OCR): piksel izi zayıf ama anlamsal olarak hatalı bir belgede (ör. geçersiz IBAN
  kontrol hanesi) kural ihlali tek başına güçlü kanıttır.

---

## 2. Derin Öğrenme Mimarisi ve Model Detayları

### 2.1 Problem neden derin öğrenmeye uygundur?

1. **İzler insan algısının altındadır.** Sahte bölge, çevresinden çoğu zaman yalnızca yüksek frekanslı
   gürültü istatistiği, JPEG niceleme ızgarası hizası veya birkaç piksellik kenar yumuşaklığı ile ayrılır.
   Bu farkların el ile tasarlanmış tek bir öznitelikle (ör. ELA, tek bir gürültü kestirimi) güvenilir
   biçimde yakalanması zordur; öznitelik birleşimleri veriden öğrenilmelidir.
2. **Yoğun (dense) tahmin gerekir.** Çıktı yalnızca "sahte/orijinal" değil, piksel düzeyinde bir maskedir.
   Bu, kodlayıcı–kod çözücü (encoder–decoder) segmentasyon ağlarının doğal problem tanımıdır.
3. **Büyük etiketli veri mevcuttur.** DocTamper (≈120 bin belge, piksel maskeli), AIForge-Doc ve
   projede üretilen sentetik TR/EN küme, denetimli öğrenmeyi mümkün kılar.
4. **Transfer öğrenme.** ImageNet ile önceden eğitilmiş kodlayıcı, düşük seviyeli kenar/doku
   filtrelerini hazır getirir; belge-özel izler ince ayar ile öğrenilir.
5. **Çeşitlilik.** Font, düzen, tarayıcı/kamera zinciri ve sıkıştırma kalitesi belgeden belgeye değişir;
   sabit kurallar bu çeşitliliğe genellenemezken öğrenilmiş temsiller (yeterli veriyle) genellenebilir.

### 2.2 Mimari: `DualStreamForensicNet`

Kaynak: [src/models.py](../src/models.py)

```text
girdi (3, H, W) normalize RGB ──┬────────────────────────────────► RGB akışı (3 kanal)
                                │                                         │
                                └─► SRMNoiseExtractor (sabit, eğitilmez) ─► gürültü akışı (3 kanal)
                                                                          │
                                birleştirme (concat, 6 kanal) ◄───────────┘
                                              │
                     smp.Unet(encoder = efficientnet-b4, in_channels = 6)
                       ├── kod çözücü ──► mask_logits (1, H, W)   ← SEGMENTASYON KAFASI
                       └── en derin kodlayıcı özniteliği
                             └─► global max-pool → dropout(0.2) → Linear ──► cls_logit (1)
                                                                    ← SINIFLANDIRMA KAFASI
```

| Bileşen | Seçim | Gerekçe |
|---|---|---|
| Omurga | U-Net + EfficientNet-B4 (ImageNet) | Atlamalı bağlantılar (skip connections) ince sahte bölgelerin sınırlarını korur; B4, parametre/başarım dengesi |
| Füzyon | **Erken birleştirme** (early fusion), 6 kanal | Tek kodlayıcı RGB ve gürültü özniteliklerini ilk katmandan itibaren birlikte işler; ImageNet ağırlıklı ilk katman `smp` tarafından 6 kanala genişletilir |
| Segmentasyon kafası | U-Net kod çözücü, 1 kanal logit | Piksel sahtecilik haritası |
| Sınıflandırma kafası | `aux_params = {pooling: "max", dropout: 0.2}` | **Max-pool**, küçük bir sahte bölgenin ortalama havuzlamada "erimesini" önler |
| Girdi kısıtı | $H, W$ 32'nin katı | Kodlayıcı 5 kez yarıya iner; kayan pencere bunu otomatik sağlar |

### 2.3 Modelin incelediği öznitelikler

| Öznitelik ailesi | Hangi akış? | Neyi yakalar? |
|---|---|---|
| **Görsel pikseller (RGB)** | RGB akışı | Font/tipografi farkı, hizalama kayması, renk/ton tutarsızlığı, karakter aralığı bozukluğu |
| **SRM steganaliz gürültü kalıntıları** | SRM akışı | Komşu piksellerden tahmin edilemeyen gürültü; düz boyama (silme), kopyala-taşı, difüzyon dolgusu bu istatistiği bozar |
| **JPEG sıkıştırma artıkları** | Her iki akış (dolaylı) | Farklı sıkıştırma geçmişine sahip bölgelerdeki 8×8 blok ızgarası ve niceleme gürültüsü uyumsuzluğu (çift sıkıştırma) |
| **Yüksek frekanslı metin kenarları** | Her iki akış | Yeniden yazılan metnin kenar keskinliği/yumuşaklığı (anti-aliasing) çevre metinden farklıdır; *inpainting* bölgeleri ise anormal derecede düşük yüksek-frekans enerjisi taşır |

#### SRM (Spatial Rich Model) gürültü çıkarıcısı

Fridrich & Kodovský (2012) SRM ailesinden, RGB-N (Zhou ve ark., 2018) ile yaygınlaşan üç sabit 5×5 yüksek
geçiren filtre kullanılır. Katsayılarının toplamı sıfırdır; bu nedenle düz bölgelerde çıktı 0 olur ve
yalnızca **gürültü kalıntısı** kalır.

$$
K_{KB} = \tfrac{1}{4}\begin{bmatrix}0&0&0&0&0\\0&-1&2&-1&0\\0&2&-4&2&0\\0&-1&2&-1&0\\0&0&0&0&0\end{bmatrix},\quad
K_{KV} = \tfrac{1}{12}\begin{bmatrix}-1&2&-2&2&-1\\2&-6&8&-6&2\\-2&8&-12&8&-2\\2&-6&8&-6&2\\-1&2&-2&2&-1\end{bmatrix},\quad
K_{H} = \tfrac{1}{2}\,[\,1\;\;-2\;\;1\,]
$$

Hesap adımları (`SRMNoiseExtractor.forward`):

1. Normalize girdi piksel ölçeğine geri çevrilir: $x_{255} = (x\cdot\sigma + \mu)\cdot 255$.
2. Her çekirdek kanal ortalamasına uygulanır: $r_i = K_i * \operatorname{mean}_c(x_{255})$.
3. **TLU (Truncated Linear Unit):** $r_i \leftarrow \operatorname{clip}(r_i, -3, 3)/3$ → çıktı $[-1, 1]$.
   Metin kenarlarındaki büyük yanıtlar kırpılır; ağın dikkati içerikten çok gürültü istatistiğine yönelir.

Ağırlıklar `register_buffer` ile tutulur: optimizasyona girmez, `state_dict` ile kaydedilir.

### 2.4 Kayıp fonksiyonu

Kaynak: [src/losses.py](../src/losses.py). Belgelerde sahte pikseller çoğunlukla alanın **%1'inden azdır**;
düz BCE bu dengesizlikte "her şey orijinal" tahminiyle düşük kayba ulaşır.

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{Dice}} + 2\cdot\mathcal{L}_{\text{Focal}}}_{\text{segmentasyon}} \;+\; 0.5\cdot \mathrm{BCE}\big(\text{cls\_logit},\, y_{\text{yama}}\big)
$$

* **Soft Dice** (batch bazında toplam): bölge örtüşmesini doğrudan optimize eder; arka plan piksel
  sayısından bağımsızdır. Batch bazlı hesap, maskesi boş yamalarda gradyan patlamasını önler.
* **Focal Loss** ($\gamma = 2$, $\alpha = 0.25$): $\mathrm{FL} = -\alpha_t (1-p_t)^\gamma \log p_t$; kolay
  pikselleri bastırıp zor kenar piksellerine odaklanır.
* **Yama etiketi** $y_{\text{yama}} = \mathbb{1}[\max(\text{maske}) > 0.5]$: rastgele kırpma sahte bölgeyi
  dışarıda bırakabileceği için belge etiketi yerine "bu yamada sahte piksel var mı?" kullanılır
  (çoklu örnek öğrenme — *Multiple Instance Learning* — mantığı).

### 2.5 Eğitim aşamaları

Kaynak: [train.py](../train.py) (`STAGE_PRESETS`). Eğitim Kaggle GPU ortamında, karışık hassasiyet (AMP),
`spawn` tabanlı DataLoader ve doğrudan LMDB okuma ile yapılmıştır.

| | **Stage A — Base DocTamper** | **Stage B — Multilingual Fine-Tuning** |
|---|---|---|
| Amaç | Genel belge sahtecilik izlerini öğrenmek | Türkçe/İngilizce finansal belge alanına uyarlama |
| Başlangıç | ImageNet ağırlıkları | `stageA_best.pth` |
| Veri | DocTamper TrainingSet | Sentetik TR/EN + SROIE2019 (orijinal fişler) + DocTamper (≤ 5 000, unutmayı önlemek için *replay*) |
| Öğrenme oranı | $3\times10^{-4}$ | $1\times10^{-5}$ (önceki bilgiyi korumak için 30× düşük) |
| Kırpıntı | 512×512 | 512×512 |
| Epoch başına örnek | 20 000 (her epoch yeni örneklem) | 4 000 |
| Validasyon | Sabit 2 000 görüntü | Sabit 500 görüntü |
| Seçilen epoch | **25** | **13** (en fazla 15) |
| İzlenen metrik | `val/score` $= (\text{pixel\_F1} + \text{image\_AUC})/2$ | aynı |
| Erken durdurma | sabır = 8 epoch | sabır = 8 epoch |
| Çıktı | `base_doctamper.pth` | `finetuned_multilingual.pth` |

Ortak ayarlar: AdamW, `weight_decay = 1e-4`, batch = 8, uzun kenar sınırı `max_side = 1600` (yalnızca
küçültme), artırma: `CropNonEmptyMaskIfExists` (%70) / `RandomCrop` (%30), hafif parlaklık-kontrast.

**Veri sızıntısı önlemleri** ([src/dataset.py](../src/dataset.py)): grup düzeyinde 70/15/15 bölme (aynı
kaynak belgeden türeyen tüm varyantlar tek bölümde), resmi test kümelerinin korunması
(`ood > test > val > train` önceliği), AIForge-Doc için piksel benzerliği + Union-Find ile kaynak grup
eşlemesi (8 122 görüntü → 4 095 grup), `verify_no_leakage()` doğrulaması ve sentetik üreticide orijinal ile
sahte varyantın **aynı son JPEG kalitesinden** geçirilmesi (model "kaç kez sıkıştırıldı" kestirmesini
öğrenemez).

**Sentetik tahrifat türleri** ([src/generate_synthetic.py](../src/generate_synthetic.py)): `copy_move`,
`text_rewrite`, `amount_tamper` (matematik bozulur), `amount_consistent` (matematik korunur),
`iban_tamper` (Mod-97 bozulur), `inpaint_sim` (üretken YZ *inpainting* benzetimi).

### 2.6 Çıkarım (inference) modları

Kaynak: [predict.py](../predict.py) `predict_raw`, [src/models.py](../src/models.py) `sliding_window_predict`.

| | **Kayan Pencere (`sliding`, varsayılan)** | **Resize 512×512 (`resize`)** |
|---|---|---|
| Ön işlem | Uzun kenar ≤ 1600 px'e sınırlanır, **yeniden ölçekleme yok** | Tüm belge 512×512'ye `INTER_AREA` ile küçültülür |
| Tarama | 512×512 pencereler, %25 örtüşme (adım 384 px), kenara hizalı son pencere | Tek ileri geçiş |
| Piksel haritası | Örtüşen bölgelerde olasılıkların ortalaması | Doğrudan çıktı, orijinal boyuta bilineer büyütme |
| Belge logit'i | Pencere `cls_logit`'lerinin **Top-K ortalaması** (K = %10) | Tek `cls_logit` |
| Avantaj | Eğitimle aynı ölçek → **SRM gürültü istatistiği korunur**; küçük sahte bölgeler kaybolmaz | Hızlı önizleme |
| Dezavantaj | Daha yavaş (benchmark medyanı 2.93 s/belge, CPU) | Yeniden ölçekleme yüksek frekanslı izleri yok eder; büyük belgelerde ayrıntı kaybı |

Model kırpıntılar üzerinde eğitildiği için **kayan pencere, eğitim–çıkarım dağılımını eşleyen tek moddur**
ve tüm raporlanan sonuçlarda kullanılmıştır.

---

## 3. Kalibrasyon ve Skorlama Mantığı

Skorlama boru hattı:

```text
mask_logits ──► σ(z / T) ─────────────────────────────► prob_map ──► max_pixel_prob
cls_logits  ──► Top-K(%10) ort. ──► σ(z / T) ──► raw_doc_prob ─┐
                                                               ├─► fuse_doc_score ──► doc_score
                                         max_pixel_prob ───────┘
                       doc_score, max_pixel_prob, threshold ──► is_forged ──► KARAR
       (opsiyonel)     doc_score, r_OCR ──► 1 − (1 − p)(1 − r) ──► Hibrit (noisy-OR) olasılık
```

### 3.1 Temperature Scaling ($T$)

Guo ve ark. (2017) *temperature scaling* yöntemi, ağın logit'ini sigmoid'den önce tek bir skaler $T > 0$
ile böler:

$$
p = \sigma\!\left(\frac{z}{T}\right)
$$

* $T > 1$ → olasılıklar 0.5'e doğru çekilir (aşırı özgüvenin yumuşatılması).
* $T = 1$ → ham model çıktısı.
* **Sıralamayı değiştirmez** ($z/T$ monoton) → ROC-AUC etkilenmez; yalnızca kalibrasyon (ve sabit bir
  eşikte verilen kararlar) değişir.

$T$, `src/metrics.py::fit_temperature` ile **validasyon** kümesinde negatif log-olabilirliği (NLL) en aza
indiren değer olarak, $[0.05, 20]$ aralığında log-uzayda 400 noktalı ızgara aramasıyla bulunur. Test
kümesi bu adıma girmez. Sistemde aynı $T$ hem piksel hem belge logit'lerine uygulanır.

### 3.2 $T = 1.00$ ve $T = 6.389$ karşılaştırması — ECE analizi

**Expected Calibration Error (ECE)**, olasılıkları $B = 15$ eşit kutuya ayırıp her kutuda ortalama güven
ile gerçek sahte oranı arasındaki farkı ağırlıklandırır:

$$
\mathrm{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N}\,\Big|\,\overline{p}_b - \overline{y}_b\,\Big|
$$

`results/stageB/per_image.csv` (121 belge; ayrıntı Bölüm 5.2) üzerinden belge skoru için yeniden
hesaplanan değerler:

| Ölçüt | $T = 1.00$ | $T = 6.389$ | Yorum |
|---|---|---|---|
| ROC-AUC (ham belge skoru) | 0.573 | 0.573 | Değişmez (monoton dönüşüm) |
| **ECE** | **0.252** | **0.113** | Kalibrasyon hatası ≈ %55 azalır |
| Brier skoru | 0.297 | 0.241 | Olasılık kalitesi iyileşir |
| Eşik 0.60'ta FPR | 0.109 | 0.087 | Yanlış alarm azalır |
| Eşik 0.60'ta TPR | 0.107 | 0.067 | Duyarlılık da azalır |
| Bu kümede NLL-optimal $T$ | — | — | 5.58 (6.389'a yakın; aynı kümede öğrenilip ölçüldüğü için iyimser) |

**Yorum:**

* $T = 6.389$, kalibrasyon dosyası yokken kullanılan varsayılan değerdir (`predict.py`
  `DEFAULT_CALIBRATION`). Model **aşırı özgüvenlidir** (val/ECE = 0.170); büyük $T$ olasılıkları
  sıkıştırarak ECE ve Brier'i belirgin biçimde iyileştirir.
* Ancak büyük $T$ piksel olasılıklarını da sıkıştırır: $p = \sigma(z/6.389)$ ile bir pikselin 0.60'ı
  geçmesi için $z \geq 6.389\cdot\operatorname{logit}(0.6) \approx 2.59$ gerekir (T=1'de 0.41 yeterlidir).
  Bu, ısı haritasını "soluklaştırır" ve **Şart 2'yi (doğrudan piksel) neredeyse işlevsiz kılar**.
* Bu nedenle üretim yapılandırması `calibration.json` → **$T = 1.00$** ve eşik **0.60**'tır: piksel
  haritasının ayırt ediciliği korunur, yanlış alarm denetimi ise yüksek eşik ve hibrit kuralla
  sağlanır. $T$ arayüzde değiştirilebilir; olasılıkların "yorumlanabilir yüzde" olarak okunması
  gereken senaryolarda $T \approx 6.4$ tercih edilmelidir.

### 3.3 Hibrit skorlama: `fuse_doc_score`

Kaynak: [predict.py](../predict.py). Sınıflandırma kafası, segmentasyon haritası tamamen temizken bile
yüksek logit üretebilir (ör. temiz bir fişte %85). Bu, iki kafanın birbiriyle çelişmesidir ve neredeyse
her zaman bir yanlış alarmdır.

$$
\text{doc\_score} =
\begin{cases}
0.7\cdot p_{\text{doc}} + 0.3\cdot p_{\max}^{\text{pix}}, & p_{\max}^{\text{pix}} < 0.05 \quad\text{(sahte bölge izi yok)}\\[4pt]
p_{\text{doc}}, & \text{aksi hâlde}
\end{cases}
$$

* `CLEAN_PIXEL_LIMIT = 0.05`, `DOC_WEIGHT = 0.7`.
* Temiz haritada skorun üst sınırı $0.7 + 0.3\cdot 0.05 = \mathbf{0.715}$'tir; tipik bir temiz belgede
  ($p_{\max}^{\text{pix}} \approx 0.001$) skor %30 kadar düşer (ör. 0.85 → 0.60).
* Piksel kanıtı varsa belge skoru **değiştirilmez**; yani füzyon yalnızca "kafalar çelişiyor" durumunda
  devreye girer.
* Arayüz hem yumuşatılmış hem ham skoru gösterir (`delta` alanı: "ham %…").

### 3.4 Karar mantığı: `is_forged` ve VEYA (OR) kuralı

Kaynak: [app.py](../app.py) `is_forged`; [src/dashboard.py](../src/dashboard.py) `decisions` aynı kuralı
uygular.

```python
hybrid_condition = doc_prob >= threshold and prob_map.max() >= CLEAN_PIXEL_LIMIT     # Şart 1
num_labels, _, stats, _ = cv2.connectedComponentsWithStats((prob_map >= threshold).astype(np.uint8))
min_area = MIN_REGION_FRACTION * prob_map.size                                     # sayfanın %0.05'i
pixel_condition = (stats[1:, cv2.CC_STAT_AREA] >= min_area).any() if num_labels > 1 else False  # Şart 2
return hybrid_condition or pixel_condition
```

$$
\text{SAHTE} \iff \underbrace{\big(\text{doc\_score} \geq \tau \;\wedge\; p_{\max}^{\text{pix}} \geq 0.05\big)}_{\text{Şart 1: Hibrit}}
\;\vee\;
\underbrace{\big(\max_k \lvert R_k^{\tau} \rvert \geq 0.0005 \cdot N\big)}_{\text{Şart 2: Bölge alanı}}
$$

| Şart | Ne zaman tetiklenir? | Neyi engeller / neyi yakalar? |
|---|---|---|
| **1 — Hibrit** | Belge skoru eşiği geçer **ve** haritada en az hafif bir iz (≥ %5) vardır | Sınıflandırma kafasının "boş harita + yüksek skor" yanlış alarmlarını **engeller**; belge skoru, zayıf ama yaygın piksel izlerini kanıt olarak kullanır |
| **2 — Bölge alanı** | Eşiği geçen bağlantılı bir bölge sayfanın en az %0.05'i kadardır | Belge skoru düşük kalsa bile **belirgin yerel tahrifatı kaçırmaz** (ör. tek bir rakamın değiştirilmesi); dijital belgelerde kenar yumuşatma (anti-aliasing) kaynaklı tekil piksellerin yanlış alarmını **engeller** |

**Önemli gözlem:** $p_{\max}^{\text{pix}} < 0.05$ ise Şart 1 tanım gereği yanlış, Şart 2 de ($\tau > 0.05$
olduğundan) yanlıştır. Yani **temiz haritalı bir belge, belge skoru ne olursa olsun asla "sahte" olarak
işaretlenmez**. Bu durumda `fuse_doc_score` kararı değil, yalnızca raporlanan olasılığı ve sıralama
metriklerini (AUC) etkiler. Arayüz bu durumu ayrıca bildirir: *"Belge skoru sınırda/yüksek olsa da
piksel seviyesinde sahte bölge izine rastlanmadı."*

### 3.5 OCR ile hibrit güven skoru (noisy-OR)

Kaynak: [src/ocr_checker.py](../src/ocr_checker.py), [app.py](../app.py) `hybrid_score`.

$$
r_{\text{OCR}} = 1 - \prod_{i\,\in\,\text{ihlal}} (1 - w_i), \qquad
P(\text{sahte}) = 1 - (1 - \text{doc\_score})\,(1 - r_{\text{OCR}})
$$

| Kural | Ağırlık $w_i$ | Doğrulama |
|---|---|---|
| IBAN | 0.90 | ISO 13616 / ISO 7064 Mod-97-10, ülkeye göre uzunluk |
| Matrah + KDV = Genel Toplam | 0.85 | ±0.02 tolerans (kuruş yuvarlaması) |
| TCKN | 0.70 | Resmi 10. ve 11. hane algoritması |
| KDV oranı | 0.60 | KDV / Matrah oranının yasal oranlarla uyumu |

Yalnızca **ihlal edilen** kurallar riske katkı verir; geçen kurallar skoru düşürmez (matematiği koruyan
sahtecilik yalnızca piksel izleriyle yakalanabilir).

---

## 4. Kritik Eşik Değerleri ve Parametreler

### 4.1 Özet tablo

| Parametre | Değer | Konum | Kısa gerekçe |
|---|---|---|---|
| Optimal threshold $\tau$ | **0.60** (önerilen bant 0.60–0.65) | `checkpoints/calibration.json` | Yanlış alarm denetimi; 0.60–0.65 arasında karar değişmiyor |
| Temperature $T$ | **1.00** (alternatif 6.389) | `checkpoints/calibration.json` | Piksel haritasının ayırt ediciliğini korur |
| Piksel şüphe eşiği | **0.05** (%5) | `predict.py` `CLEAN_PIXEL_LIMIT` | "Sahte bölge izi yok" tanımı |
| Füzyon ağırlığı | **0.7 / 0.3** | `predict.py` `DOC_WEIGHT` | Temiz haritada skoru en çok 0.715'e sınırlar |
| Top-K average ratio | **%10** | `src/models.py` `top_k_ratio` | Tek aykırı pencereye karşı sağlamlık |
| Pencere / örtüşme | 512 px / %25 | `sliding_window_predict` | Eğitim kırpıntısıyla aynı ölçek |
| Uzun kenar sınırı | 1600 px | `config["max_side"]` | Eğitimle aynı ön işlem |
| SRM TLU eşiği | 3.0 | `SRMNoiseExtractor` | Kenar yanıtlarını kırpar |

### 4.2 Optimal threshold: neden 0.60–0.65?

`results/stageB` benchmark'ı (121 belge, $T = 1.00$, `is_forged` kuralı) üzerinde eşik taraması:

| Eşik $\tau$ | TP | FP | FN | TN | TPR | **FPR** |
|---|---|---|---|---|---|---|
| 0.50 | 10 | 7 | 65 | 39 | 0.133 | 0.152 |
| **0.60** | 8 | 5 | 67 | 41 | 0.107 | **0.109** |
| 0.65 | 8 | 5 | 67 | 41 | 0.107 | 0.109 |

Gerekçeler:

1. **Maliyet asimetrisi.** Finansal belge doğrulamada yanlış alarm (orijinal bir dekontu "sahte" ilan
   etmek) operasyonel olarak pahalıdır ve kullanıcı güvenini hızla aşındırır. 0.50 → 0.60 geçişi
   FPR'yi %15.2'den %10.9'a indirirken (~%28 göreli azalma) yalnızca 2 doğru pozitif kaybettirir.
2. **Kararlılık bölgesi.** 0.60 ile 0.65 arasında hiçbir karar değişmez; eşik bu bantta küçük
   oynamalara duyarsızdır. 0.65 üzerine çıkıldığında ise Şart 2'nin duyarlılığı azalmaya başlar.
3. **Füzyon tavanıyla uyum.** Temiz haritalı belgelerde skor ≤ 0.715 olduğu için eşik 0.715'in altında
   tutulmalıdır; aksi hâlde yumuşatılmış skor ile ham skor aynı karara varır ve füzyonun raporlamadaki
   ayrımı kaybolur.
4. **Kalibrasyon bağımlılığı.** Eşik $T$ ile birlikte anlam kazanır. $T = 6.389$ seçilirse olasılıklar
   sıkıştığından aynı yanlış alarm düzeyi için eşik daha düşük seçilmelidir
   (`DEFAULT_CALIBRATION` bu yüzden $T = 6.389$ ile $\tau = 0.35$ çiftini kullanır).

> **Not:** Bu benchmark'ta TPR düşüktür (Bölüm 5.2); eşik seçimi şu aşamada duyarlılığı değil, yanlış
> alarmı denetlemeyi hedefler. Türkçe veride model iyileştikçe eşik validasyon kümesinde (ör. FPR ≤ %5
> kısıtı altında TPR'yi en büyükleyen nokta olarak) yeniden belirlenmelidir.

### 4.3 Pikselsel şüphe eşiği: 0.05

* **Anlamı:** Haritadaki en yüksek piksel olasılığı %5'in altındaysa model, belgenin hiçbir yerinde
  sahte bölgeye işaret etmiyordur.
* **Neden 0.05?** Benchmark'ta temiz belgelerin önemli bir kısmında $p_{\max}^{\text{pix}}$ 0.0005–0.01
  aralığında kalır (ör. `syn000003__00_orig` → 0.0005, `syn000032__00_orig` → 0.0014). Yüksek
  çözünürlüklü belgelerde milyonlarca pikselin maksimumu alındığından tamamen sıfır beklenmez; 0.05 bu
  doğal gürültü tabanının bir büyüklük mertebesi üstündedir, ancak gerçek sahte bölgelerde görülen
  0.1–0.99 değerlerinin çok altındadır.
* **Rolü:** (i) `fuse_doc_score`'u tetikler, (ii) Şart 1'in "piksel izi var" önkoşuludur.

### 4.4 Top-K Average Ratio: %10

$$
k = \max\big(1,\; \lceil 0.10 \cdot N_{\text{pencere}} \rceil\big), \qquad
z_{\text{doc}} = \frac{1}{k}\sum_{j \in \text{top-}k} z_j^{\text{cls}}
$$

* **Neden maksimum değil?** Eğitimde yama etiketi "yamada sahte piksel var mı?" olduğundan belge skoru
  en şüpheli pencerelerden gelmelidir (MIL). Ancak saf maksimum, tek bir pencerenin aykırı logit'ine
  (kenar dolgusu, logo, damga, yoğun tablo çizgisi) karşı aşırı duyarlıdır ve gerçek belgelerde yanlış
  alarm üretir.
* **Neden ortalama değil?** Tüm pencerelerin ortalaması, belgenin küçük bir bölümündeki tahrifatı
  "seyreltir"; tek rakamlık bir değişiklik kaybolur.
* **%10** bu iki uç arasındadır: 1600 px uzun kenarlı tipik bir belgede (≈12–20 pencere) $k = 2$
  olur — tahrifatı taşıyan 1–2 pencere skoru belirlemeye devam ederken tek bir aykırı pencerenin etkisi
  yarıya iner. Tek pencereli (≤ 512 px) belgelerde davranış maksimumla aynıdır.

---

## 5. Deneysel Sonuçlar ve Dersler

### 5.1 Validasyon başarımı (eğitim sırasındaki val bölümü)

Kaynak: kontrol noktalarındaki `metrics` sözlüğü (seçilen epoch).

| Metrik | Stage A (`base_doctamper`, ep. 25) | Stage B (`finetuned_multilingual`, ep. 13) |
|---|---|---|
| **val/score** | 0.7085 | **0.8914** |
| val/pixel_F1 (küresel) | 0.9169 | 0.9201 |
| val/pixel_IoU (küresel) | 0.8466 | 0.8521 |
| val/pixel_F1 (görüntü başına) | 0.8672 | 0.8825 |
| val/pixel_IoU (görüntü başına) | 0.8048 | 0.8279 |
| val/image_AUC | tanımsız (val'de yalnızca sahte) | 0.8626 |
| val/image_acc | 0.838 | 0.836 |
| val/image_TPR | 0.838 | 0.875 |
| val/image_FPR | tanımsız | **0.377** |
| val/ECE | 0.163 | 0.170 |
| val/Brier | 0.161 | 0.153 |

**Yorum:**

* Stage A'nın `val/score` değeri düşük görünür çünkü val kümesinde orijinal belge olmadığından AUC
  tanımsızdır ve skora 0 olarak yansır; piksel başarımı zaten yüksektir (F1 = 0.917).
* Stage B, DocTamper bilgisini **unutmadan** (piksel F1 0.917 → 0.920) belge düzeyinde ayırt edicilik
  kazanmıştır (AUC 0.863). `val/score = 0.8914`, büyük ölçüde **DocTamper ağırlıklı** val kümesini
  yansıtır (Stage B val'inde DocTamper örnekleri piksel metriklerine baskındır).
* Belge düzeyindeki **FPR = 0.377** ve **ECE = 0.170**, sınıflandırma kafasının orijinal belgelerde aşırı
  özgüvenli olduğunu gösterir; Bölüm 3'teki temperature scaling, Top-K ve `fuse_doc_score` tasarımları
  doğrudan bu gözleme yanıttır.

### 5.2 Dış veri (OOD) ve Türkçe sentetik veri başarımı

Kaynak: `python benchmark.py --ocr` → `results/stageB/` (21.10.2026, kayan pencere, $T = 1.00$,
$\tau = 0.60$). Değerlendirme kümesi modelin eğitimde görmediği görsellerdir:

| Alt küme | Belge | Sahte / Orijinal | ROC-AUC (belge) | ROC-AUC ($p_{\max}^{\text{pix}}$) | ECE ($T$=1) | ECE ($T$=6.389) |
|---|---|---|---|---|---|---|
| Sentetik test (Stage B test bölümü) | 21 | 14 / 7 | **0.475** | 0.490 | 0.267 | 0.131 |
| Sentetik (görülmemiş gruplar) | 40 | 31 / 9 | **0.455** | 0.649 | 0.275 | 0.226 |
| AIForge-Doc TestingSet (OOD) | 60 | 30 / 30 | **0.476** | 0.510 | 0.316 | 0.186 |
| **Tümü** | 121 | 75 / 46 | 0.573 | 0.479 | 0.252 | 0.113 |

Piksel düzeyinde (eşik 0.60, küresel F1): sentetik alt kümelerde **0.00**, AIForge-Doc'ta **0.037**
(eşik 0.30'da 0.106).

**Alt küme başına belge AUC değerleri 0.455–0.476 aralığındadır, yani rastgele tahminin (0.50)
düzeyindedir.** (Havuzlanmış 0.573 değeri, alt kümeler arası skor ölçeği farkından kaynaklanan bir
Simpson etkisidir ve gerçek ayırt edicilik olarak yorumlanmamalıdır.)

#### Neden düşük?

1. **Alan kayması (domain shift).** Model, piksel izlerini ağırlıklı olarak DocTamper'dan (Çince belgeler,
   belirli bir tarama/sıkıştırma zinciri, belirli düzenleme araçları) öğrenmiştir. Türkçe dekont/fatura
   düzenleri, Latin karakter setindeki diakritikler (ç, ğ, ı, ö, ş, ü), font aileleri ve sentetik edinim
   zinciri farklı bir dağılımdır. Val'deki yüksek başarı (0.89) ile dış verideki rastgele düzey (≈0.47)
   arasındaki fark klasik bir **genelleme açığıdır**.
2. **Sentetik veri azlığı.** Stage B, epoch başına yalnızca 4 000 örnek ve en fazla 15 epoch ile, düşük
   öğrenme oranında ($10^{-5}$) eğitilmiştir. Bu oran DocTamper bilgisini korumak için seçilmiştir ancak
   Türkçe alana uyum için yetersiz kalmıştır.
3. **Veri dengesizliği.** Stage B karışımında DocTamper *replay* örnekleri (≤ 5 000) sentetik Türkçe
   örneklerle rekabet eder; val metriği de DocTamper'a baskın olduğundan **model seçimi Türkçe başarımı
   değil DocTamper başarımını ödüllendirmiştir**.
4. **Sentetik tahrifatların doğası.** Sentetik sahteler, orijinalle aynı son JPEG kaydından geçirilir
   (kestirme ipucunu engellemek için bilinçli tasarım). Bu, izleri gerçekçi biçimde zayıflatır;
   `amount_consistent` ve `text_rewrite` gibi türlerde maske alanı küçüktür (belge alanının ~%1'i).
5. **Küçük değerlendirme örneklemi.** 21–60 belgelik alt kümelerde AUC'nin standart hatası ±0.07–0.10
   mertebesindedir; sonuçlar yön göstericidir, kesin değildir.

**Tespit edilen ihtiyaç:** Türkçe belgeler için **hedefli transfer öğrenme** — daha büyük ve çeşitli
Türkçe sentetik/gerçek veri, Türkçe val kümesiyle model seçimi ve kodlayıcının üst bloklarında daha
yüksek öğrenme oranı.

### 5.3 JPEG sıkıştırma dayanıklılığı

Kaynak: `results/stageB/robustness.csv` (20 görsellik alt küme her kalite faktöründe yeniden sıkıştırıldı).

| JPEG kalitesi $Q$ | 100 | 90 | 80 | 70 | 60 | 50 |
|---|---|---|---|---|---|---|
| Belge AUC | 0.660 | 0.611 | 0.535 | 0.549 | 0.562 | 0.486 |
| $p_{\max}^{\text{pix}}$ AUC | 0.493 | 0.368 | 0.316 | 0.455 | 0.493 | 0.396 |

Yeniden sıkıştırma, yüksek frekanslı SRM izlerini yok ettiği için belge AUC'si $Q = 100$'den $Q = 50$'ye
0.66'dan 0.49'a düşer. Bu, gerçek dünyada (WhatsApp/e-posta ile iletilen, ekran görüntüsü alınan
belgeler) beklenen bozulmayı nicel olarak gösterir.

### 5.4 OCR kural motoru başarımı

Aynı benchmark'ta (`--ocr`): **sahte belgelerin %17.3'ünde** en az bir kural ihlali bulunurken
**orijinal belgelerde bu oran %2.2'dir**; ortalama OCR riski sahte belgelerde 0.222, orijinallerde
0.035'tir. Kural motoru düşük duyarlılık / yüksek kesinlik (*precision*) profiline sahiptir ve derin
öğrenme modelinin Türkçe belgelerdeki zayıflığını kısmen telafi eden, **alan kaymasından etkilenmeyen**
bir kanıt kaynağıdır.

### 5.5 Erken temel çizgi (referans)

`results/summary.csv`, yalnızca sentetik veriyle 3 epoch eğitilmiş hızlı bir deneme modeline
(`runs/fast_run`, kırpıntı 256, NLL-optimal $T = 9.02$) aittir: sentetik test AUC 0.469, AIForge-Doc
AUC 0.593, piksel F1 0.027 / 0.014. Bu, sıfırdan yalnızca küçük sentetik veriyle eğitimin de yeterli
olmadığını; DocTamper ön eğitiminin (Stage A) piksel düzeyinde gerekli temsil gücünü sağladığını gösterir.

### 5.6 Güçlü yönler

* **Sızıntısız değerlendirme protokolü:** grup düzeyinde bölme, AIForge-Doc kaynak eşlemesi, resmi test
  kümelerinin korunması ve otomatik doğrulama.
* **Kaynak dağılımında yüksek piksel başarımı:** val piksel F1 = 0.92, IoU = 0.85.
* **Yorumlanabilirlik:** her karar bir ısı haritası ve kontur ile desteklenir.
* **Katmanlı yanlış alarm denetimi:** Top-K, `fuse_doc_score`, piksel önkoşullu karar kuralı ve
  ayarlanabilir kalibrasyon.
* **Alan bilgisi entegrasyonu:** IBAN/TCKN/KDV kural motoru, öğrenilmiş modelden bağımsız kanıt sağlar.
* **Mühendislik:** kaldığı yerden devam eden aşamalı eğitim, doğrudan LMDB okuma, AMP, eğitim/çıkarım
  ön işleminin tek kaynaktan paylaşılması.

### 5.7 Gelişime açık yönler — ablasyon ve gelecek çalışmalar

| Başlık | Öneri |
|---|---|
| **Türkçe transfer öğrenme** | Sentetik küme 1 000 → ≥ 10 000 belge; gerçek Türkçe dekont/fatura örnekleri; Stage B'de Türkçe val ile model seçimi |
| **Aşamalı katman çözme** | Stage B'de kod çözücü + son kodlayıcı blokları için $10^{-4}$, alt bloklar için $10^{-5}$ (ayrımsal öğrenme oranı) |
| **Ablasyon çalışmaları** | (a) yalnız RGB vs. RGB+SRM, (b) erken vs. geç füzyon, (c) Top-K oranı {max, %5, %10, %25, ortalama}, (d) `fuse_doc_score` açık/kapalı, (e) $T$ ∈ {1, 6.389} — hepsi aynı Türkçe test kümesinde |
| **Sıkıştırma dayanıklılığı** | Eğitimde rastgele JPEG yeniden sıkıştırma ($Q \in [50, 100]$) ve yeniden ölçekleme artırması; DCT/niceleme tablosu akışı (CAT-Net benzeri) |
| **Kalibrasyon** | Piksel ve belge logit'leri için **ayrı** sıcaklıklar; Türkçe val kümesinde yeniden uydurma; eşiğin FPR kısıtı altında otomatik seçimi |
| **Üretken YZ tahrifatı** | Gerçek difüzyon *inpainting* örnekleri (SD-Inpaint, vb.) ile eğitim; `inpaint_sim` yalnızca bir benzetimdir |
| **Değerlendirme** | ≥ 500 belgelik Türkçe test kümesi, bootstrap güven aralıkları, belge tipi bazında raporlama |
| **OCR** | Alan bazlı düzen analizi (layout parsing), daha fazla kural (vergi no, tarih tutarlılığı, banka-IBAN eşleşmesi) |

---

## 6. Sistem Mimarisi ve Arayüz

### 6.1 Bileşen diyagramı

```text
┌──────────────────────────── app.py (Streamlit) ─────────────────────────────┐
│ Kenar çubuğu: Model seçimi · Çıkarım modu · T · Eşik · OCR aç/kapat · OCR GPU │
│                                                                              │
│  Sekme 1: 🔎 Belge Analizi              Sekme 2: 📊 Model Başarımı ve Metrik  │
│  ┌──────────────────────────────┐       ┌─────────────────────────────────┐  │
│  │ Yükleme (JPG/PNG/PDF@200dpi) │       │ src/dashboard.py (Plotly)       │  │
│  │      │                       │       │   ▲ results/stageB/*            │  │
│  │      ▼                       │       │   ▲ runs/**/history.csv         │  │
│  │ predict.py ── src/models.py  │       │   ▲ checkpoint metrics          │  │
│  │      │  (sliding / resize)   │       └─────────────────────────────────┘  │
│  │      ▼                       │                     ▲                      │
│  │ fuse_doc_score → is_forged   │          benchmark.py (çevrimdışı)         │
│  │      │                       │                                            │
│  │      ▼ (opsiyonel)           │                                            │
│  │ src/ocr_checker.py (EasyOCR) │                                            │
│  │  → kural motoru → noisy-OR   │                                            │
│  └──────────────────────────────┘                                            │
└──────────────────────────────────────────────────────────────────────────────┘
        ▲ checkpoints/finetuned_multilingual.pth · base_doctamper.pth · calibration.json
```

Önbellekleme: model `st.cache_resource` ile bir kez yüklenir; analiz sonucu (görüntü SHA-1 + sayfa +
model + mod + $T$ + OCR anahtarıyla) `session_state`'te tutulur. **Eşik değişikliği yeniden çıkarım
gerektirmez** — yalnızca görselleştirme ve karar güncellenir; diğer ayarlar değişirse kullanıcı
uyarılır.

### 6.2 Belge Analizi sekmesi

**3'lü görselleştirme:**

| Panel | İçerik |
|---|---|
| 1. Orijinal Belge | Yüklenen görüntü |
| 2. Sahtecilik Olasılık Haritası | Sabit ölçekli **jet** renk haritası (0 = mavi → 1 = kırmızı) |
| 3. Çakıştırılmış Görsel | **%60 orijinal + %40 ısı haritası** (`cv2.addWeighted`); eşiği geçen bölgeler **beyaz kontur** ile çizilir |

**Özet metrikler:** Belge Sahtecilik Olasılığı (füzyon sonrası, ham değer delta olarak), Maksimum Piksel
Olasılığı, Sahte/Manipüle Edilmiş Alan (% — eşik üstü piksel oranı) ve karar kutusu
(⚠️ tespit / ✅ iz yok / ℹ️ "skor yüksek ama piksel izi yok").

**PDF desteği:** PyMuPDF ile seçilen sayfa 200 DPI'da rasterleştirilir; çok sayfalı PDF'lerde sayfa
seçici çıkar. Arayüz, rasterleştirmenin dosya içi sıkıştırma izlerini değiştirdiği konusunda uyarır.

**Opsiyonel OCR ve kural motoru:** EasyOCR (TR + EN) kelime kutularını satırlara birleştirir; kural
motoru IBAN (Mod-97), TCKN, Matrah + KDV = Genel Toplam ve KDV oranını denetler. İhlal eden satırlar
**turuncu**, geçenler **yeşil** kutuyla işaretlenir; İçerik (OCR) Riski ve Hibrit Sahtecilik Olasılığı
($1 - (1-p)(1-r)$) gösterilir; ayıklanan alanlar JSON, ham OCR satırları tablo olarak sunulur.

### 6.3 "Model Başarımı ve Metrik Analizi" sekmesi

Kaynak: [src/dashboard.py](../src/dashboard.py). Tüm grafikler `benchmark.py` çıktılarından ve eğitim
günlüklerinden çizilir; uydurma veri kullanılmaz. Kenar çubuğundaki $T$ ve eşik değiştikçe grafikler
**canlı** olarak yeniden hesaplanır (belge skorları ham logit olarak saklandığı için).

| # | Grafik | Ne gösterir? |
|---|---|---|
| **A. Performans ve kalibrasyon** | | |
| 1a | ROC eğrisi | Belge ve piksel düzeyinde TPR–FPR; AUC |
| 1b | Precision-Recall eğrisi | Dengesiz piksel sınıfı için daha bilgilendirici; ortalama kesinlik |
| 2 | Eşik ayarı: F1 / IoU – eşik | 0.05–0.95 ızgarasında piksel F1/IoU; seçili eşik işaretli |
| 3 | Kalibrasyon (reliability) eğrisi | $T = 1$, $T = 6.389$, seçili $T$ ve NLL-uydurulmuş $T$ için güven–doğruluk; ECE lejantta |
| 4 | Violin plot | Orijinal ve sahte belgelerin belge skoru / maks. piksel dağılımları (kutu + noktalar) |
| **B. Karar matrisleri ve korelasyonlar** | | |
| 5 | Confusion matrix (belge + piksel) | `is_forged` kararı ve piksel temiz/sahte; satır yüzdesi + sayı |
| 6 | Metrik korelasyon matrisi | Spearman ρ (çözünürlük, JPEG kalitesi, skor, GT alanı, süre…) |
| 7 | OCR karar matrisi | Kural geçti/ihlal × gerçek sınıf |
| **C. Dayanıklılık ve eğitim dinamikleri** | | |
| 8 | JPEG sıkıştırma dayanıklılığı | $Q$ = 100…50 için AUC ve piksel F1 |
| 9 | Eğitim ve validasyon kaybı | Epoch bazlı (`history.csv` varsa) |
| 10 | Epoch bazlı piksel F1 / IoU | Öğrenme eğrileri; ayrıca checkpoint'lerdeki son val metrikleri tablosu |
| **D. Veri seti ve tahrifat karakteristikleri** | | |
| 11 | Tahrifat türüne göre piksel F1 | `copy_move`, `text_rewrite`, `amount_*`, `iban_tamper`, `inpaint_sim`, AIForge |
| 12 | Sahte alan kaplama oranı | Gerçek maskenin belge alanına oranı (%) |
| 13 | Bölge sayısı | Belge başına bağlantılı tahrif bölgesi sayısı |
| 14 | Belge tipi × dil | Stage B sentetik veri dağılımı (dekont/fatura/fiş × TR/EN) |
| 15a/b | Çözünürlük – süre / piksel F1 | Megapiksel'e göre çıkarım süresi ve başarım |

Bir veri kaynağı eksikse ilgili grafik yerine onu üretecek komut gösterilir.

---

## Ek A — Dosya ve Parametre Referansı

| Dosya | Sorumluluk |
|---|---|
| [src/models.py](../src/models.py) | `SRMNoiseExtractor`, `DualStreamForensicNet`, `sliding_window_predict` (Top-K), kayıt/yükleme |
| [src/losses.py](../src/losses.py) | `DiceLoss`, `FocalLoss`, `ForensicLoss` |
| [src/dataset.py](../src/dataset.py) | Manifest, sızıntısız grup bölme, LMDB okuma, artırmalar |
| [src/metrics.py](../src/metrics.py) | ROC-AUC (Mann-Whitney), piksel F1/IoU, ECE, Brier, `fit_temperature` |
| [src/generate_synthetic.py](../src/generate_synthetic.py) | TR/EN dekont-fatura-fiş + 6 tahrifat türü + maske |
| [src/ocr_checker.py](../src/ocr_checker.py) | EasyOCR, IBAN Mod-97, TCKN, KDV kural motoru |
| [src/dashboard.py](../src/dashboard.py) | Metrik paneli (Plotly) |
| [train.py](../train.py) | Aşamalı eğitim, erken durdurma, devam ettirme |
| [evaluate.py](../evaluate.py) | Test/OOD değerlendirmesi, val'de sıcaklık uydurma |
| [benchmark.py](../benchmark.py) | Stage B kapsamlı değerlendirme → `results/stageB/` |
| [predict.py](../predict.py) | Tek belge çıkarımı, `fuse_doc_score`, kalibrasyon yükleme |
| [app.py](../app.py) | Streamlit arayüzü, `is_forged`, `hybrid_score` |

**Yeniden üretim:**

```bash
python train.py --stage stageA --samples-per-epoch 20000
python train.py --stage stageB
python benchmark.py --ocr                 # results/stageB/ (~15 dk, CPU)
streamlit run app.py
```

**Kaynakça**

* Fridrich, J., Kodovský, J. (2012). *Rich Models for Steganalysis of Digital Images.* IEEE TIFS.
* Zhou, P., Han, X., Morariu, V., Davis, L. (2018). *Learning Rich Features for Image Manipulation Detection.* CVPR.
* Lin, T.-Y. ve ark. (2017). *Focal Loss for Dense Object Detection.* ICCV.
* Guo, C., Pleiss, G., Sun, Y., Weinberger, K. (2017). *On Calibration of Modern Neural Networks.* ICML.
* Ronneberger, O. ve ark. (2015). *U-Net.* MICCAI. · Tan, M., Le, Q. (2019). *EfficientNet.* ICML.
* Qu, C. ve ark. (2023). *Towards Robust Tampered Text Detection in Document Image (DocTamper).* CVPR.
