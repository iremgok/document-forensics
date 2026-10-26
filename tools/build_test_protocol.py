"""
Sabit test protokolü: tüm modellerin (kendi aşamalarınız ve baseline'lar) **aynı** görüntülerle
değerlendirilmesi için tek, dondurulmuş bir ``splits`` dosyası üretir.

İçerik:

* ``--from-splits`` (ör. ``checkpoints/stageE/splits.csv``) içindeki **test** satırları
  (sentetik v4 ve gerçek tabanlı sahte test grupları) ile ``val`` satırları (kalibrasyon için).
* DocTamper resmi test kümeleri — kaynak adı kümeye göre ayrılır (``doctamper_TestingSet``,
  ``doctamper_FCD``, ``doctamper_SCD``); böylece ``evaluate.py`` her birini ayrı satırda raporlar.
  ``--doctamper-per-set`` ile küme başına sabit (seed'li) alt küme alınabilir.
* AIForge-Doc TestingSet (``ood``), gönüllü sahtecilikler (``human_forged``, test) ve difüzyon
  sahtecilikleri (``diffusion_forged``, test) — ``--data-root`` altında varsa.

Protokol bir kez üretilip ``results/protocol/test_v1.csv`` olarak saklanır ve **değiştirilmez**; veri
eklenirse ``test_v2`` olarak yeni sürüm açılır ve raporda hangi sürümün kullanıldığı yazılır.

Uyarı: ``--from-splits`` modelin kendi bölmesidir. Başka bir modelin (ör. farklı seed'le eğitilmiş)
eğitim kümesi bu test gruplarını içerebilir; karşılaştırılan tüm modeller aynı bölme ile eğitilmiş
olmalı ya da yalnızca hiçbir modelin eğitmediği kaynaklar (DocTamper testleri, AIForge-Doc, gönüllü,
difüzyon) karşılaştırılmalıdır.

Kullanım:
    python tools/build_test_protocol.py --from-splits checkpoints/stageE/splits.csv --data-root data \\
        --doctamper-per-set 2000 --out results/protocol/test_v1.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import build_manifest  # noqa: E402

EXTRA_SOURCES = ("doctamper", "aiforgedoc", "human_forged", "diffusion_forged")


def main() -> None:
    p = argparse.ArgumentParser(description="Sabit test protokolü (tüm modeller için ortak splits dosyası)")
    p.add_argument("--from-splits", required=True, help="Referans modelin splits.csv'si (test + val satırları)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--doctamper-per-set", type=int, default=2000, help="0 = tamamı")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/protocol/test_v1.csv")
    args = p.parse_args()

    base = pd.read_csv(args.from_splits, keep_default_na=False)
    base = base[base["split"].isin(["test", "val"])]
    frames = [base]

    extra = build_manifest(args.data_root, [], cache_dir=Path(args.data_root) / "splits")
    extra = extra[extra["source"].isin(EXTRA_SOURCES)].copy()
    extra["split"] = extra["fixed_split"].replace("", "test")
    extra.loc[extra["source"].isin(["human_forged", "diffusion_forged"]), "split"] = "test"
    dt = extra["source"] == "doctamper"
    extra.loc[dt, "split"] = "test"
    set_name = extra.loc[dt, "group_id"].str.extract(r"doctamper:[^/]*?(TrainingSet|TestingSet|FCD|SCD)/",
                                                     flags=2)[0]
    extra.loc[dt, "source"] = "doctamper_" + set_name
    extra = extra[extra["source"] != "doctamper_TrainingSet"]
    if args.doctamper_per_set:
        is_dt = extra["source"].str.startswith("doctamper_")
        parts = [extra[~is_dt]] + [g.sample(min(len(g), args.doctamper_per_set), random_state=args.seed)
                                   for _, g in extra[is_dt].groupby("source")]
        extra = pd.concat([f for f in parts if not f.empty]) if any(not f.empty for f in parts) else extra
    if not extra.empty:
        frames.append(extra.drop(columns=["fixed_split"]))

    protocol = pd.concat(frames, ignore_index=True)
    dup = protocol["image_path"].duplicated()
    if dup.any():
        print(f"UYARI: {int(dup.sum())} yinelenen görüntü çıkarıldı.")
        protocol = protocol[~dup]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    protocol.to_csv(args.out, index=False)
    print(protocol.groupby(["split", "source", "label"]).size().to_string())
    print(f"Yazıldı: {args.out} ({len(protocol)} görüntü)")


if __name__ == "__main__":
    main()
