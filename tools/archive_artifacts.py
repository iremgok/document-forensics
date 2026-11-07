"""
Kullanılmayan / yanıltıcı ağırlık ve veri klasörlerini **silmeden** ``_arsiv/`` altına taşır.

Varsayılan olarak yalnızca ne yapılacağını listeler (``--dry-run``); taşımak için ``--apply`` verin. Her taşıma
``_arsiv/manifest.csv`` dosyasına eklenir (kaynak, hedef, boyut, gerekçe, tarih). Hedef zaten varsa öğe atlanır
(üzerine yazılmaz). Arşivden geri almak için manifest'teki hedefi kaynağa taşımak yeterlidir.

Arşivlenen modeller ``src/model_registry.py``'de ``_arsiv/...`` yoluyla kayıtlıdır ve arayüzde geliştirici
modunda isteğe bağlı listelenebilir (``list_models(include_archived=True)``). Kod tabanında bu yollara başka
başvuru yoktur.

Kullanım:
    python tools/archive_artifacts.py            # kuru çalıştırma: liste + toplam boyut
    python tools/archive_artifacts.py --apply    # taşı ve manifest yaz
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

ARCHIVE = Path("_arsiv")
MANIFEST = ARCHIVE / "manifest.csv"

#: (yol, gerekçe)
ITEMS: tuple[tuple[str, str], ...] = (
    ("checkpoints/finetuned_multilingual.pth",
     "Stage C ara modeliyle (epoch 4) ezilmiş; val FPR 1.00. Gerçek Stage B: finetuned_multilingual_stageB.pth"),
    ("checkpoints/stageD", "Yalnız sentetikle eğitilmiş Stage C/D; val FPR 0.995 ('alan konumu = sahte' ezberi)"),
    ("runs/fast_run", "3 epoch'luk deneme çalışması"),
    ("find.zip", "0 baytlık boş dosya"),
    ("data/_backup_4_synthetic_tr_en_1000docs", "Sentetik v1'in eski yedeği; hiçbir kaynak tarafından okunmuyor"),
    ("data/synthetic_fast", "Hızlı deneme için küçük sentetik küme"),
    ("data/_forge_preview", "forge_real önizleme çıktısı"),
    ("data/5_synthetic_tr_en_v3", "Sentetik v3 (Stage C); v4 (6_synthetic_tr_en_v4) kullanılıyor"),
)


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def plan(root: Path) -> list[dict[str, object]]:
    rows = []
    for rel, reason in ITEMS:
        src = root / rel
        dst = root / ARCHIVE / rel
        if not src.exists():
            rows.append({"kaynak": rel, "hedef": (ARCHIVE / rel).as_posix(), "boyut": 0, "gerekce": reason,
                         "durum": "yok (zaten taşınmış ya da hiç olmamış)"})
        elif dst.exists():
            rows.append({"kaynak": rel, "hedef": (ARCHIVE / rel).as_posix(), "boyut": size_of(src), "gerekce": reason,
                         "durum": "atlandı: hedef zaten var"})
        else:
            rows.append({"kaynak": rel, "hedef": (ARCHIVE / rel).as_posix(), "boyut": size_of(src), "gerekce": reason,
                         "durum": "taşınacak"})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Kullanılmayan dosyaları _arsiv/ altına taşı (varsayılan: kuru çalıştırma)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="Yalnızca listele (varsayılan)")
    mode.add_argument("--apply", action="store_true", help="Taşı ve _arsiv/manifest.csv'ye yaz")
    p.add_argument("--root", default=".", help="Proje kökü")
    args = p.parse_args()

    root = Path(args.root)
    rows = plan(root)
    for r in rows:
        print(f"[{r['durum']}] {r['kaynak']} -> {r['hedef']} ({human(int(r['boyut']))}) — {r['gerekce']}")
    todo = [r for r in rows if r["durum"] == "taşınacak"]
    print(f"\nToplam: {len(todo)} öğe, {human(sum(int(r['boyut']) for r in todo))}")
    if not args.apply:
        print("Kuru çalıştırma: hiçbir dosya taşınmadı. Taşımak için --apply verin.")
        return

    manifest = root / MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    new_file = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kaynak", "hedef", "boyut", "gerekce", "tarih"])
        if new_file:
            writer.writeheader()
        for r in todo:
            src, dst = root / str(r["kaynak"]), root / str(r["hedef"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))  # Taşıma: silme yok
            writer.writerow({"kaynak": r["kaynak"], "hedef": r["hedef"], "boyut": r["boyut"], "gerekce": r["gerekce"],
                             "tarih": time.strftime("%Y-%m-%d %H:%M")})
            print(f"Taşındı: {r['kaynak']} -> {r['hedef']}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
