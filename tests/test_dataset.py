"""``src/dataset.py``: grup sızıntısı, holdout, ``image_key``, 8'e hizalı kırpma, boş maske yolu."""

import cv2
import numpy as np
import pandas as pd
import pytest

from src.dataset import (AlignedCrop, ForgeryDataset, apply_holdout, assign_splits, build_transforms, image_key,
                         read_mask, resolve_source_dir, verify_no_leakage)


def manifest(n_groups: int = 40) -> pd.DataFrame:
    rows = []
    for g in range(n_groups):
        issuer = ("garanti", "ziraat", "trendyol")[g % 3]
        for v in range(4):
            rows.append({"image_path": f"x/{issuer}_{g}_{v}.png", "mask_path": "", "label": int(v > 0),
                         "group_id": f"real:{issuer}/doc{g}", "source": "real_forged", "fixed_split": "",
                         "doc_type": issuer})
    return pd.DataFrame(rows)


def test_assign_splits_keeps_groups_together():
    splits = assign_splits(manifest(), seed=1)
    assert splits.groupby("group_id")["split"].nunique().max() == 1
    assert set(splits["split"]) == {"train", "val", "test"}


def test_verify_no_leakage_raises_on_shared_group():
    df = pd.DataFrame({"group_id": ["a", "a", "b"], "split": ["train", "test", "val"], "source": ["s"] * 3})
    with pytest.raises(AssertionError, match="sızıntı"):
        verify_no_leakage(df)


def test_apply_holdout_moves_whole_issuer_to_test():
    held = apply_holdout(manifest(), issuers=["Garanti"])
    splits = assign_splits(held, seed=0)
    garanti = splits[splits["group_id"].str.startswith("real:garanti/")]
    assert len(garanti) and (garanti["split"] == "test").all()
    others = splits.loc[~splits.index.isin(garanti.index), "split"]
    assert set(others) >= {"train", "val"}  # diğer kurumlar normal bölünmeye devam eder


def test_apply_holdout_does_not_touch_ood():
    df = manifest(3)
    df.loc[0, "fixed_split"] = "ood"
    held = apply_holdout(df, issuers=["garanti"])
    assert held.loc[0, "fixed_split"] == "ood"


@pytest.mark.parametrize("path, key", [
    ("data/8_real_forged/images/garanti_x__p0__00_orig.png", "8_real_forged__garanti_x__p0__00_orig"),
    ("data\\finetune_aiforgedoc\\TestingSet\\authentic\\X1.jpg", "TestingSet__authentic_X1"),
    ("lmdb://data/DocTamperV1-FCD::image-000000012", "DocTamperV1-FCD__image-000000012"),
])
def test_image_key(path, key):
    assert image_key(path) == key


def test_aligned_crop_origin_is_multiple_of_8():
    h = w = 200
    ys, xs = np.mgrid[:h, :w]
    image = np.stack([ys, xs, np.zeros_like(ys)], axis=-1).astype(np.uint8)  # piksel = koordinat
    mask = np.zeros((h, w), np.uint8)
    mask[150:160, 90:100] = 1
    crop = AlignedCrop(64, nonempty_p=0.5, rest=lambda **kw: kw)
    for _ in range(50):
        out = crop(image, mask)
        assert out["image"].shape[:2] == (64, 64)
        y0, x0 = int(out["image"][0, 0, 0]), int(out["image"][0, 0, 1])
        assert y0 % 8 == 0 and x0 % 8 == 0


def test_aligned_crop_pads_small_images_right_bottom():
    crop = AlignedCrop(64, nonempty_p=0.0, rest=lambda **kw: kw)
    out = crop(np.full((40, 30, 3), 7, np.uint8), np.ones((40, 30), np.uint8))
    assert out["image"].shape[:2] == (64, 64)
    assert out["mask"][:40, :30].all() and not out["mask"][40:, :].any()


def test_aligned_crop_rejects_unaligned_size():
    with pytest.raises(ValueError):
        AlignedCrop(60, 0.5, rest=lambda **kw: kw)


@pytest.mark.parametrize("empty", [None, "", float("nan")])
def test_empty_mask_path_gives_zero_mask(empty):
    mask = read_mask(empty, (5, 7))
    assert mask.shape == (5, 7) and mask.sum() == 0


def test_dataset_with_empty_mask_path(tmp_path):
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), np.full((48, 64, 3), 200, np.uint8))
    ds = ForgeryDataset(pd.DataFrame({"image_path": [str(path)], "mask_path": [""], "label": [0]}),
                        build_transforms(False), max_side=32)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (3, 24, 32)  # uzun kenar 32'ye küçültüldü
    assert sample["mask"].sum() == 0 and float(sample["label"]) == 0.0


def test_real_forged_v2_used_only_when_complete(tmp_path):
    (tmp_path / "8_real_forged").mkdir()
    (tmp_path / "8_real_forged" / "metadata.csv").write_text("image\n", encoding="utf-8")
    (tmp_path / "8_real_forged_v2").mkdir()
    assert resolve_source_dir(tmp_path, "real_forged").name == "8_real_forged"  # yarım v2 atlanır
    (tmp_path / "8_real_forged_v2" / "metadata.csv").write_text("image\n", encoding="utf-8")
    assert resolve_source_dir(tmp_path, "real_forged").name == "8_real_forged_v2"


def test_rerender_augment_keeps_mask_aligned_and_is_noop_when_off():
    from src.dataset import rerender_augment

    image = np.full((100, 200, 3), 120, np.uint8)
    mask = np.zeros((100, 200), np.uint8)
    mask[40:60, 80:120] = 1
    same_img, same_mask = rerender_augment(image, mask, p=0.0)
    assert same_img is image and same_mask is mask
    for _ in range(20):
        out_img, out_mask = rerender_augment(image, mask, p=1.0)
        assert out_img.shape[:2] == out_mask.shape and set(np.unique(out_mask)) <= {0, 1}
        assert out_mask.any()
        # Maske hâlâ yalnızca gri (120) bölgeyi kaplar; beyaz kenar boşluğuna taşmaz.
        assert (out_img[out_mask == 1] < 200).all()
