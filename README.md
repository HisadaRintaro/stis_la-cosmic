# stis_la_cosmic

HST/STIS 画像データに対する宇宙線除去パイプライン。  
[LA-Cosmic](http://www.astro.yale.edu/dokkum/lacosmic/)（Laplacian Cosmic Ray Identification）アルゴリズムの Python 実装 [`lacosmic`](https://lacosmic.readthedocs.io/en/stable/) を用いて、STIS FITS ファイルから宇宙線ヒットを検出・除去します。

## LA-Cosmic について

LA-Cosmic は、天文画像から宇宙線を除去するためのアルゴリズムです。ラプラシアンエッジ検出に基づいており、以下の論文で提案されました。

> **van Dokkum, P. G. (2001)**  
> *"Cosmic-Ray Rejection by Laplacian Edge Detection"*  
> PASP, 113, 1420  
> [ADS: 2001PASP..113.1420V](https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V/abstract)

本パッケージでは、上記アルゴリズムの Python 実装である [`lacosmic`](https://pypi.org/project/lacosmic/)（v1.4.0）パッケージを使用しています。`lacosmic` を使用した研究成果を公表する場合は、[Zenodo レコード](https://doi.org/10.5281/zenodo.6468623) への引用も含めてください。

## インストール

```bash
# Poetry を使用する場合
poetry install

# pip を使用する場合
pip install -e .
```

### 依存ライブラリ

- Python ≥ 3.13
- [astropy](https://www.astropy.org/) ≥ 7.2.0
- [lacosmic](https://lacosmic.readthedocs.io/) ≥ 1.4.0
- [matplotlib](https://matplotlib.org/) ≥ 3.10.8
- [tqdm](https://tqdm.github.io/) ≥ 4.67.3

## 使い方

### 基本的なワークフロー

```python
from stis_la_cosmic import InstrumentModel, ReaderCollection, ImageCollection
from pathlib import Path

# 1. FITS ファイルの探索
inst = InstrumentModel("HST/", "_crj", ".fits", depth=1)

# 2. FITS ファイルの読み込み
readers = ReaderCollection.from_paths(inst.path_list)

# 3. ImageCollection の生成（DQ フラグと LA-Cosmic パラメータを指定）
collection = ImageCollection.from_readers(
    readers,
    dq_flags=16,          # マスク対象の DQ ビットフラグ（16 = hot pixel）
    contrast=5.0,         # ラプラシアン/ノイズ比コントラスト閾値
    cr_threshold=5,       # 宇宙線検出のシグマクリッピング閾値
    neighbor_threshold=5, # 近傍ピクセルの検出閾値
    error=5,              # 誤差配列のスケール係数
)

# 4. 宇宙線の除去
cleaned = collection.remove_cosmic_ray()

# 5. 結果の FITS ファイル出力
output_dir = Path("output/")
cleaned.write_fits(output_dir=output_dir, overwrite=True)
```

### 処理の流れ

本パッケージでは、`lacosmic.remove_cosmics` を呼び出す前に以下の前処理を行います：

1. **DQ フラグによる Bad Pixel マスク生成** — FITS DQ 拡張のビットフラグに基づいて不良ピクセルを特定
2. **中央値補間** — マスク対象ピクセル（DQ bad pixel + 負の値を持つピクセル）を周囲 8 近傍の中央値で補間
3. **LA-Cosmic 実行** — 前処理済み画像に対して `lacosmic.remove_cosmics` を適用し、宇宙線を検出・除去

```python
# lacosmic.remove_cosmics の呼び出し（内部処理）
from lacosmic import remove_cosmics

clean_data, cr_mask = remove_cosmics(
    image_data,          # 前処理済みの画像データ
    contrast,            # コントラスト閾値
    cr_threshold,        # 宇宙線検出閾値
    neighbor_threshold,  # 近傍閾値
    error=error_array,   # 誤差配列
    mask=dq_mask,        # DQ bad pixel マスク
)
# clean_data: 宇宙線除去後の画像
# cr_mask: 検出された宇宙線の boolean マスク
```

### 可視化

```python
import matplotlib.pyplot as plt

# 画像の一覧表示
fig, ax = plt.subplots(2, 3, figsize=(10, 8))
cleaned.imshow(area=True, ax=ax)
fig.savefig("cleaned_images.png")
plt.close(fig)

# 宇宙線マスクの表示
cleaned.imshow_mask(mask_type="cr")

# DQ マスクの表示
cleaned.imshow_mask(mask_type="dq")
```

### FITS 出力

出力される FITS ファイルには以下の拡張が含まれます：

| HDU | EXTNAME  | 内容                          |
|-----|----------|-------------------------------|
| 0   | PRIMARY  | Primary Header（LACORR=True） |
| 1   | SCI      | 宇宙線除去後の科学画像        |
| 2   | ERR      | 誤差画像（元データ）          |
| 3   | DQ       | Data Quality フラグ（元データ）|
| 4   | LACOSMIC | 宇宙線マスク（boolean）      |

## 引用

本パッケージを利用した研究成果を公表する際は、以下を引用してください：

### LA-Cosmic アルゴリズム

```bibtex
@ARTICLE{2001PASP..113.1420V,
    author = {{van Dokkum}, Pieter G.},
    title = "{Cosmic-Ray Rejection by Laplacian Edge Detection}",
    journal = {PASP},
    year = 2001,
    volume = {113},
    pages = {1420-1427},
    doi = {10.1086/323894},
    adsurl = {https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V},
}
```

### lacosmic Python パッケージ

```bibtex
@software{lacosmic,
    author = {Larry Bradley},
    title = {lacosmic},
    url = {https://github.com/lacosmicx/lacosmic},
    doi = {10.5281/zenodo.6468623},
}
```

## ライセンス

TBD
