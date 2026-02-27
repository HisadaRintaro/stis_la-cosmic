"""処理前後のスペクトル比較図を保存するスクリプト.

HST ディレクトリの処理前画像 (_crj.fits) と
remove_cosmic_ray_result の各パラメータセットの処理後画像 (_lac.fits) を
ImageCollection.plot_spectrum_comparison で比較し、図を保存する。
"""

from stis_la_cosmic import InstrumentModel, ReaderCollection, ImageCollection
from pathlib import Path
from datetime import datetime

# =============================================================================
# 設定
# =============================================================================
HST_DIR = "HST/"
RESULT_DIR = Path("remove_cosmic_ray_result/")
EXCLUDE_FILES = ("o56503010_crj.fits",)
DQ_FLAGS = 16
# 比較するスリット方向のインデックス（y 軸）。
# 対象天体が写っている行を指定する。画像の中央付近が目安。
SLIT_INDEX = 572

# =============================================================================
# セットアップ
# =============================================================================
date = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path("Pictures/") / date / "spectrum_comparison"
save_dir.mkdir(parents=True, exist_ok=True)

# 処理前コレクション (_crj.fits) を読み込む
inst_before = InstrumentModel.load(
    HST_DIR,
    suffix="_crj",
    extension=".fits",
    depth=1,
    exclude_files=EXCLUDE_FILES,
)
reader_before = ReaderCollection.from_paths(inst_before.path_list)
before_collection = ImageCollection.from_readers(reader_before, dq_flags=DQ_FLAGS)

print(f"before_collection: {len(before_collection)} images")
print(f"Save directory: {save_dir}")

# =============================================================================
# 各パラメータセットについてスペクトル比較プロットを保存
# =============================================================================
param_dirs = sorted(d for d in RESULT_DIR.iterdir() if d.is_dir())
print(f"Parameter sets: {len(param_dirs)}")

for i, param_dir in enumerate(param_dirs, start=1):
    inst_after = InstrumentModel.load(
        str(param_dir),
        suffix="_lac",
        extension=".fits",
        depth=0,
    )
    reader_after = ReaderCollection.from_paths(inst_after.path_list)
    after_collection = ImageCollection.from_readers(reader_after, dq_flags=DQ_FLAGS)

    save_path = save_dir / f"{param_dir.name}_slit{SLIT_INDEX}.png"
    title = param_dir.name.replace("data_", "").replace("_", ", ")

    before_collection.plot_spectrum_comparison(
        other=after_collection,
        slit_index=SLIT_INDEX,
        save_path=save_path,
        title=title,
    )
    print(f"[{i}/{len(param_dirs)}] saved: {save_path.name}")

print(f"\nDone. All figures saved to {save_dir}")
