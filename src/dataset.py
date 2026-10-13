"""
Çift akışlı (RGB + SRM) model için PyTorch Dataset ve sızıntısız veri ayırıcı.

Veri sızıntısı (data leakage) önleme ilkeleri
---------------------------------------------
1. **Grup düzeyinde ayırma:** Her satırın bir ``group_id``'si vardır. Aynı orijinal
   belgeden türetilen tüm görüntüler (orijinal + tüm sahte varyantları) aynı gruptadır
   ve bir grup *yalnızca bir* bölüme (train / val / test) atanır.
2. **Sabit bölümler en kısıtlayıcı olanı kazanır:** Bir grubun herhangi bir üyesi resmi
   test kümesindeyse (ör. AIForge-Doc ``TestingSet``) grubun tamamı test'e gider.
   Sıralama: ``ood > test > val > train``.
3. **Diskte ön işleme yok:** Normalizasyon, kırpma ve boyut sınırlama yalnızca
   ``ForgeryDataset.__getitem__`` içinde, her örnek okunduğunda dinamik olarak yapılır.
4. **OOD izolasyonu:** Eğitimde kullanılmayan benchmark kümeleri ``ood`` olarak
   işaretlenir; ``train.py`` yalnızca ``train`` ve ``val`` satırlarını görür.

Manifest sütunları
------------------
``image_path, mask_path, label, group_id, source, fixed_split``
(``mask_path`` boşsa maske tamamen 0 kabul edilir; ``fixed_split`` boşsa grup
rastgele 70/15/15 oranında atanır.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

#: Her veri kaynağı için aranacak klasör adları (ilk bulunan kullanılır).
SOURCE_DIRS: dict[str, tuple[str, ...]] = {
    "doctamper": ("1_base_doctamper",),
    "aiforgedoc": ("2_finetune_aiforgedoc", "finetune_aiforgedoc"),
    "sroie2019": ("3_sroie2019",),
    "synthetic": ("4_synthetic_tr_en",),
}

SPLIT_PRIORITY = {"": 0, "train": 1, "val": 2, "test": 3, "ood": 4}
MANIFEST_COLUMNS = ["image_path", "mask_path", "label", "group_id", "source", "fixed_split"]


# ---------------------------------------------------------------------------
# Yardımcı G/Ç fonksiyonları
# ---------------------------------------------------------------------------

def read_image_rgb(path: str | Path) -> np.ndarray:
    """Görüntüyü RGB uint8 olarak okur.

    ``cv2.imread`` Windows'ta Türkçe karakter içeren yollarda başarısız olduğu için
    dosya önce bayt olarak okunup ``cv2.imdecode`` ile çözülür.
    """
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Görüntü okunamadı: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path | None, shape: tuple[int, int]) -> np.ndarray:
    """İkili maskeyi {0, 1} uint8 olarak okur; yol yoksa sıfır maske döndürür.

    Maske boyutu görüntüden farklıysa en yakın komşu ile yeniden ölçeklenir
    (ikili değerlerin bozulmaması için).
    """
    if not path or (isinstance(path, float) and np.isnan(path)):
        return np.zeros(shape, dtype=np.uint8)
    buffer = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Maske okunamadı: {path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def limit_long_side(image: np.ndarray, mask: Optional[np.ndarray], max_side: int
                    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Uzun kenarı ``max_side``'dan büyük görüntüleri orantılı olarak küçültür.

    Yalnızca küçültme yapılır (büyütme gürültü izlerini yapay olarak yumuşatır).
    Eğitim, doğrulama ve uygulama (app.py) aynı fonksiyonu kullanır; böylece
    eğitim/çıkarım ölçek uyumsuzluğu oluşmaz.
    """
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image, mask
    size = (round(w * scale), round(h * scale))
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return image, mask


def _list_images(folder: Path) -> list[Path]:
    """Klasördeki görüntü dosyalarını sıralı döndürür."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_source_dir(data_root: Path, source: str) -> Optional[Path]:
    """``SOURCE_DIRS`` içindeki aday klasörlerden ilk var olanı döndürür."""
    for name in SOURCE_DIRS[source]:
        candidate = data_root / name
        if candidate.is_dir():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Kaynak bazlı manifest üreticileri
# ---------------------------------------------------------------------------

def _manifest_synthetic(root: Path, for_training: bool) -> pd.DataFrame:
    """``generate_synthetic.py`` tarafından yazılan metadata.csv'yi okur.

    Grup kimliği, sentetik belgenin şablon örneğidir: orijinal ve tüm sahte
    varyantlar aynı ``group_id``'yi paylaşır.
    """
    meta_path = root / "metadata.csv"
    if not meta_path.exists():
        logger.warning("Sentetik metadata bulunamadı: %s", meta_path)
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    meta = pd.read_csv(meta_path)
    return pd.DataFrame({
        "image_path": [str(root / p) for p in meta["image"]],
        "mask_path": [str(root / p) for p in meta["mask"]],
        "label": meta["label"].astype(int),
        "group_id": "synthetic:" + meta["group_id"].astype(str),
        "source": "synthetic",
        "fixed_split": "" if for_training else "ood",
    })


def link_aiforge_groups(root: Path, cache_path: Path, thumb: int = 64,
                        threshold: float = 1.5, ink_level: float = 235.0) -> dict[str, str]:
    """
    AIForge-Doc görüntülerini kaynak (orijinal) belgeye göre gruplar.

    Neden gerekli? ``metadata.jsonl`` dosyasında kaynak görüntü kimliği yoktur ve
    resmi Training/Testing bölümleri aynı kaynak fişleri paylaşabilir (bu projede
    yapılan analizde 254 orijinal görüntünün her iki bölümde de birebir aynı olduğu
    görülmüştür). Bu nedenle grup bilgisi piksel benzerliğinden çıkarılır:

    1. Her görüntü gri tona çevrilip ``thumb × thumb`` boyutuna (INTER_AREA) indirgenir.
    2. Yalnızca aynı orijinal çözünürlüğe sahip görüntüler karşılaştırılır.
    3. Yalnızca "mürekkep" piksellerinde (iki küçük resimden en az birinde gri
       değer < ``ink_level``) mutlak farkın **75. yüzdelik dilimi** hesaplanır;
       ``threshold``'un altındaysa aynı kaynaktan geldikleri kabul edilir.
       * Yüzdelik istatistik, sahte bölgenin yarattığı yerel farka dayanıklıdır.
       * Mürekkep kısıtı olmadan, geniş beyaz kenarlı dar fişler (ör. SROIE) beyaz
         alanların sıfır farkı nedeniyle yanlışlıkla birleşir (analizde 201 görüntülük
         tek bir grup oluşmuştu; kısıtla 114 doğru gruba ayrıldı).
    4. Eşleşmeler Union-Find ile birleştirilir; aynı ``spec_id`` paylaşan sahteler de
       birleştirilir.

    Hatalı birleştirme (farklı iki belgeyi aynı gruba koymak) yalnızca grupları
    büyütür ve sızıntı açısından güvenli yöndedir. Sonuç ``cache_path``'e yazılır
    (yalnızca dosya adı → grup eşlemesi; piksel verisi değil).
    """
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        return dict(zip(cached["image_path"], cached["group_id"]))

    paths: list[str] = []
    for split_dir in ("TrainingSet", "TestingSet"):
        for sub in ("images", "authentic"):
            paths.extend(str(p) for p in _list_images(root / split_dir / sub))
    logger.info("AIForge-Doc grup bağlama: %d görüntü işleniyor (tek seferlik)...", len(paths))

    thumbs = np.zeros((len(paths), thumb * thumb), dtype=np.float32)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, path in enumerate(paths):
        gray = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        thumbs[i] = cv2.resize(gray, (thumb, thumb), interpolation=cv2.INTER_AREA).ravel()
        buckets.setdefault(gray.shape, []).append(i)
        if (i + 1) % 500 == 0:
            logger.info("  %d / %d", i + 1, len(paths))

    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for members in buckets.values():
        if len(members) < 2:
            continue
        block = thumbs[members]
        for row, i in enumerate(members[:-1]):
            others, current = block[row + 1:], block[row]
            ink = (others < ink_level) | (current < ink_level)
            diff = np.where(ink, np.abs(others - current), np.nan)
            with np.errstate(all="ignore"):  # Tamamen boş sayfa çiftleri NaN üretir → eşleşmez.
                dist = np.nanpercentile(diff, 75, axis=1)
            for offset in np.nonzero(dist < threshold)[0]:
                union(i, members[row + 1 + offset])

    meta_path = root / "metadata.jsonl"
    if meta_path.exists():
        index = {Path(p).as_posix(): i for i, p in enumerate(paths)}
        first_of_spec: dict[str, int] = {}
        with open(meta_path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                i = index.get((root / rec["image"]).as_posix())
                if i is None:
                    continue
                if rec["spec_id"] in first_of_spec:
                    union(first_of_spec[rec["spec_id"]], i)
                else:
                    first_of_spec[rec["spec_id"]] = i

    groups = {p: f"aiforgedoc:g{find(i)}" for i, p in enumerate(paths)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_path": list(groups), "group_id": list(groups.values())}).to_csv(cache_path, index=False)
    logger.info("AIForge-Doc: %d görüntü → %d kaynak grup", len(paths), len(set(groups.values())))
    return groups


def _manifest_aiforgedoc(root: Path, for_training: bool, cache_dir: Path) -> pd.DataFrame:
    """AIForge-Doc'u manifest'e çevirir.

    * Eğitimde kullanılacaksa: ``TrainingSet`` grup bazlı train/val/test'e bölünür,
      ``TestingSet`` (ve onunla grup paylaşan her şey) daima ``test`` olur.
    * Kullanılmayacaksa: tamamı ``ood`` (yalnızca evaluate.py görür).
    """
    groups = link_aiforge_groups(root, cache_dir / "aiforgedoc_groups.csv")
    rows = []
    for split_dir, fixed in (("TrainingSet", ""), ("TestingSet", "test")):
        fixed = fixed if for_training else "ood"
        for img in _list_images(root / split_dir / "images"):
            mask = root / split_dir / "masks" / f"{img.stem}.png"
            rows.append((str(img), str(mask) if mask.exists() else "", 1, groups[str(img)], fixed))
        for img in _list_images(root / split_dir / "authentic"):
            rows.append((str(img), "", 0, groups[str(img)], fixed))
    df = pd.DataFrame(rows, columns=["image_path", "mask_path", "label", "group_id", "fixed_split"])
    df["source"] = "aiforgedoc"
    return df[MANIFEST_COLUMNS]


def _manifest_doctamper(root: Path, for_training: bool) -> pd.DataFrame:
    """DocTamper'ı (``export_doctamper_lmdb`` ile dışa aktarılmış hâliyle) okur.

    Beklenen yapı: ``<root>/<KümeAdı>/images/*.jpg`` ve ``<root>/<KümeAdı>/masks/*.png``.
    ``TrainingSet`` adını içeren küme grup bazlı bölünür; ``TestingSet``, ``FCD``,
    ``SCD`` kümeleri resmi test kümeleridir ve daima ``test`` olarak kalır.

    Sınırlama: DocTamper, sahte görüntülerin kaynak belge kimliğini yayınlamaz. Bu
    nedenle TrainingSet içinde her görüntü kendi grubudur; train/val ayrımı resmi
    test kümelerinden bağımsızdır ancak TrainingSet içi kaynak tekrarı ölçülemez.
    """
    rows = []
    for set_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        is_train = "train" in set_dir.name.lower()
        fixed = ("" if is_train else "test") if for_training else "ood"
        for img in _list_images(set_dir / "images"):
            mask = set_dir / "masks" / f"{img.stem}.png"
            # DocTamper'daki tüm görüntüler sahtedir; maske dosyası varlığı etiket için yeterli
            # (170 bin maskeyi yalnızca etiket için okumak gereksiz G/Ç olurdu).
            rows.append((str(img), str(mask) if mask.exists() else "", int(mask.exists()),
                         f"doctamper:{set_dir.name}/{img.stem}", fixed))
    df = pd.DataFrame(rows, columns=["image_path", "mask_path", "label", "group_id", "fixed_split"])
    df["source"] = "doctamper"
    return df[MANIFEST_COLUMNS]


def _manifest_sroie(root: Path) -> pd.DataFrame:
    """SROIE2019 fişleri: tamamı orijinaldir (label=0) ve yalnızca OOD yanlış-alarm testi içindir."""
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return pd.DataFrame({
        "image_path": [str(p) for p in images],
        "mask_path": "",
        "label": 0,
        "group_id": [f"sroie2019:{p.stem}" for p in images],
        "source": "sroie2019",
        "fixed_split": "ood",
    })


def build_manifest(data_root: str | Path, train_sources: Sequence[str],
                   cache_dir: str | Path = "data/splits") -> pd.DataFrame:
    """
    Tüm veri kaynaklarını tek bir manifest'te toplar.

    Args:
        data_root: ``data/`` klasörü.
        train_sources: Eğitim/validasyonda kullanılacak kaynaklar
            (ör. ``["synthetic", "doctamper"]``). Listede olmayan kaynaklar
            ``ood`` olarak işaretlenir. ``sroie2019`` daima ``ood``'dur.
        cache_dir: AIForge-Doc grup önbelleğinin yazılacağı klasör.
    """
    data_root, cache_dir = Path(data_root), Path(cache_dir)
    frames = []
    for source in SOURCE_DIRS:
        root = resolve_source_dir(data_root, source)
        if root is None:
            logger.info("Kaynak bulunamadı, atlanıyor: %s", source)
            continue
        for_training = source in train_sources
        if source == "synthetic":
            frame = _manifest_synthetic(root, for_training)
        elif source == "aiforgedoc":
            frame = _manifest_aiforgedoc(root, for_training, cache_dir)
        elif source == "doctamper":
            frame = _manifest_doctamper(root, for_training)
        else:
            frame = _manifest_sroie(root)
        logger.info("%-11s %6d görüntü (%s)", source, len(frame), "eğitim" if for_training else "yalnız test/OOD")
        frames.append(frame)
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(f"{data_root} altında kullanılabilir veri bulunamadı.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Sızıntısız ayırıcı
# ---------------------------------------------------------------------------

def assign_splits(manifest: pd.DataFrame, ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
                  seed: int = 42) -> pd.DataFrame:
    """
    Manifest'e ``split`` sütunu ekler (grup düzeyinde, deterministik).

    Algoritma:
      1. Her grubun zorunlu bölümü, üyelerinin ``fixed_split`` değerlerinden en
         kısıtlayıcısıdır (``ood > test > val > train``).
      2. Zorunlu bölümü olmayan gruplar kaynak bazında ``seed`` ile karıştırılır ve
         görüntü sayısı oranları ``ratios``'a yaklaşacak şekilde sırayla
         train → val → test'e doldurulur (resmi test kümesi olan kaynaklarda yalnızca
         train → val).
      3. Son olarak her grubun tek bir bölümde olduğu doğrulanır (assert).
    """
    df = manifest.copy()
    df["fixed_split"] = df["fixed_split"].fillna("")
    forced = (df.assign(_p=df["fixed_split"].map(SPLIT_PRIORITY))
                .groupby("group_id")["_p"].max())
    inverse = {v: k for k, v in SPLIT_PRIORITY.items()}
    group_split = forced.map(inverse).to_dict()

    rng = np.random.default_rng(seed)
    group_source = df.groupby("group_id")["source"].first()
    for source in sorted(df["source"].unique()):
        free = sorted(g for g in group_source.index[group_source == source] if group_split[g] == "")
        if not free:
            continue
        # Resmi test kümesi olan kaynaklarda (AIForge-Doc TestingSet, DocTamper TestingSet)
        # serbest gruplardan ikinci bir test kümesi ayrılmaz; train/val oranı korunarak
        # (70:15 → ~%82 / %18) yalnızca bu ikisine bölünür.
        has_official_test = (df.loc[df["source"] == source, "fixed_split"] == "test").any()
        r_train, r_val = ratios[0], ratios[1]
        if has_official_test:
            r_train, r_val = r_train / (r_train + r_val), r_val / (r_train + r_val)
        sizes = df[df["group_id"].isin(free)].groupby("group_id").size()
        total = max(int(sizes.sum()), 1)
        bounds = (r_train * total, (r_train + r_val) * total)
        seen = 0
        for group in rng.permutation(np.array(free, dtype=object)):
            # Grup, başlangıç noktasına göre atanır: bölümler sınırda taşmaz, kayma ≤ 1 grup.
            group_split[group] = "train" if seen < bounds[0] else ("val" if seen < bounds[1] else "test")
            seen += int(sizes[group])

    df["split"] = df["group_id"].map(group_split)
    df = remove_cross_source_duplicates(df)
    verify_no_leakage(df)
    return df.drop(columns=["fixed_split"])


def remove_cross_source_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kaynaklar arası tekrarları temizler.

    AIForge-Doc'un bir kısmı SROIE fişlerinden üretilmiştir ve orijinal dosya adlarını
    korur (ör. ``X51009568881.png``). AIForge-Doc eğitimde kullanıldıysa, SROIE2019
    OOD kümesindeki aynı adlı fişler değerlendirmeden çıkarılır; aksi hâlde model
    "görmediği" sanılan bir belgeyle test edilmiş olur.
    """
    seen_stems = {Path(p).stem for p in df.loc[(df["source"] == "aiforgedoc")
                                               & df["split"].isin(["train", "val"]), "image_path"]}
    overlap = (df["source"] == "sroie2019") & df["image_path"].map(lambda p: Path(p).stem in seen_stems)
    if overlap.any():
        logger.warning("SROIE2019: AIForge-Doc eğitim kümesiyle örtüşen %d fiş OOD'dan çıkarıldı.", int(overlap.sum()))
    return df[~overlap].reset_index(drop=True)


def verify_no_leakage(df: pd.DataFrame) -> None:
    """Her ``group_id``'nin tam olarak bir bölümde olduğunu doğrular; aksi hâlde hata fırlatır."""
    per_group = df.groupby("group_id")["split"].nunique()
    leaking = per_group[per_group > 1]
    if not leaking.empty:
        raise AssertionError(f"Veri sızıntısı: {len(leaking)} grup birden fazla bölümde! Örn: {list(leaking.index[:5])}")
    counts = df.groupby(["source", "split"]).size().unstack(fill_value=0)
    logger.info("Bölüm dağılımı (görüntü sayısı):\n%s", counts.to_string())


# ---------------------------------------------------------------------------
# Dönüşümler (yalnızca bellekte, dinamik)
# ---------------------------------------------------------------------------

def build_transforms(train: bool, crop_size: int = 512) -> Callable:
    """
    Albumentations dönüşüm hattı.

    Eğitim:
      * ``PadIfNeeded`` → küçük belgeleri kırpma boyutuna tamamlar.
      * %70 ``CropNonEmptyMaskIfExists`` (sahte bölgeyi içeren kırpıntı) /
        %30 ``RandomCrop``. Küçük sahte alanlar rastgele kırpmada çoğu zaman
        kaçırılacağı için örnekleme sahte bölgeye yönlendirilir.
      * Hafif parlaklık/kontrast. Geometrik bozma, bulanıklaştırma veya JPEG
        sıkıştırma **kullanılmaz**: bunlar SRM akışının aradığı gürültü izlerini
        yok eder ya da sahte iz üretir.
    Doğrulama/Test: kırpma yok — tüm görüntü kayan pencere ile işlenir.

    Normalizasyon her iki durumda da ImageNet ortalama/std ile yapılır.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    tail = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    if not train:
        return A.Compose(tail)
    return A.Compose([
        A.PadIfNeeded(min_height=crop_size, min_width=crop_size, border_mode=cv2.BORDER_REFLECT_101),
        A.OneOf([
            A.CropNonEmptyMaskIfExists(height=crop_size, width=crop_size, p=0.7),
            A.RandomCrop(height=crop_size, width=crop_size, p=0.3),
        ], p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        *tail,
    ])


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ForgeryDataset(Dataset):
    """
    Belge sahteciliği veri kümesi.

    Her örnek bir sözlüktür:
        ``image``: (3, H, W) float32, ImageNet normalize edilmiş RGB.
                   SRM gürültü akışı modelin içinde bu tensörden hesaplanır
                   (bkz. ``src/models.py``), böylece iki akış piksel-hizalıdır.
        ``mask``:  (1, H, W) float32, {0, 1}.
        ``label``: () float32 — belge düzeyinde etiket (0 orijinal, 1 sahte).
        ``index``: manifest satır numarası.
    """

    def __init__(self, frame: pd.DataFrame, transform: Callable, max_side: int = 1600) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.max_side = max_side

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        image = read_image_rgb(row["image_path"])
        mask = read_mask(row["mask_path"], image.shape[:2])
        image, mask = limit_long_side(image, mask, self.max_side)
        out = self.transform(image=image, mask=mask)
        mask_t = torch.as_tensor(out["mask"], dtype=torch.float32).unsqueeze(0)
        return {
            "image": out["image"].float(),
            "mask": mask_t,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "index": torch.tensor(index),
        }


# ---------------------------------------------------------------------------
# Opsiyonel: DocTamper LMDB → klasör dışa aktarımı
# ---------------------------------------------------------------------------

def export_doctamper_lmdb(lmdb_dir: str | Path, out_dir: str | Path) -> int:
    """
    DocTamper LMDB veritabanını ham ``images/`` + ``masks/`` klasörlerine yazar.

    Görüntüler **değiştirilmeden** (yeniden sıkıştırma yapılmadan, orijinal JPEG
    baytları) yazılır; bu bir ön işleme değil, yalnızca biçim dönüşümüdür.
    LMDB anahtarları: ``num-samples``, ``image-%09d``, ``label-%09d``.

    Kullanım:
        python -c "from src.dataset import export_doctamper_lmdb as e; \
                   e('DocTamperV1-TrainingSet', 'data/1_base_doctamper/TrainingSet')"
    """
    import lmdb  # Opsiyonel bağımlılık.

    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n = int(txn.get(b"num-samples"))
        for i in range(n):
            img_bytes = txn.get(f"image-{i:09d}".encode())
            lbl_bytes = txn.get(f"label-{i:09d}".encode())
            if img_bytes is None:
                continue
            (out_dir / "images" / f"{i:09d}.jpg").write_bytes(img_bytes)
            if lbl_bytes is not None:
                (out_dir / "masks" / f"{i:09d}.png").write_bytes(lbl_bytes)
    return n
