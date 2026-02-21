"""画像モデル.

STIS FITS ファイルから読み込んだ画像データを管理し、
宇宙線除去 (LA-Cosmic) および可視化機能を提供するモジュール。
"""

import PIL.BmpImagePlugin
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self
import numpy as np
from .fits_reader import ReaderCollection, STISFitsReader
from astropy.io import fits
import matplotlib.pyplot as plt
from lacosmic import remove_cosmics

@dataclass(frozen=True)
class ImageModel:
    """STIS 画像データの単一フレームモデル.

    1つの FITS HDU から取得した科学画像データとヘッダーを保持し、
    宇宙線除去および画像表示の機能を提供する。

    Attributes
    ----------
    image : np.ndarray
        2次元の科学画像データ配列
    header : fits.Header
        対応する FITS ヘッダー（HDU 1）
    source_path : Path | None
        元の FITS ファイルが格納されているディレクトリパス
    primary_header : fits.Header | None
        元の FITS ファイルの Primary Header（HDU 0）
    mask : np.ndarray | None
        DQ フラグから生成された bad pixel マスク（True = bad pixel）
    """

    image : np.ndarray
    header : fits.Header
    source_path : Path | None = None
    primary_header : fits.Header | None = None
    mask : np.ndarray | None = None
    dq_flags : int = 16

    def __repr__(self) -> str:
        mask_info = f"mask_count={self.mask.sum()}" if self.mask is not None else "mask=None"
        return f"ImageModel(image={self.image.shape}, header={type(self.header).__name__}), source_path={self.source_path}, primary_header={self.primary_header is not None}, {mask_info}"

    @classmethod
    def from_reader(
        cls,
        reader: STISFitsReader,
        index: int = 1,
        dq_index: int = 3,
        dq_flags: int = 16,
    ) -> Self:
        """STISFitsReader から ImageModel を生成する.

        Parameters
        ----------
        reader : STISFitsReader
            読み込み済みの FITS Reader
        index : int, optional
            画像データの HDU インデックス（デフォルト: 1、科学データ）
        dq_index : int, optional
            DQ 配列の HDU インデックス（デフォルト: 3）
        dq_flags : int, optional
            マスク対象の DQ ビットフラグ（デフォルト: 16 = hot pixel）。
            複数フラグはビット OR で指定（例: 16 | 256）

        Returns
        -------
        ImageModel
            生成されたモデル
        """
        try:
            dq = reader.image_data(dq_index)
        except KeyError:
            dq = None
        mask = (dq & dq_flags).astype(bool) if dq is not None else None
        return cls(
            image=reader.image_data(index),
            header=reader.header(index),
            source_path=reader.filename.parent,
            primary_header=reader.header(0),
            mask=mask,
            dq_flags=dq_flags,
        )

    @staticmethod
    def median_interpolate(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """マスク対象ピクセルを隣接8ピクセルの中央値で補間する.

        各 bad pixel に対して、周囲8近傍のうちマスクされていない
        ピクセルの中央値で値を置換する。有効な近傍がない場合は
        元の値を保持する。

        Parameters
        ----------
        image : np.ndarray
            2次元の画像データ配列
        mask : np.ndarray
            boolean マスク配列（True = bad pixel）

        Returns
        -------
        np.ndarray
            補間済みの画像データ配列（コピー）
        """
        result = image.copy()
        bad_y, bad_x = np.where(mask)
        ny, nx = image.shape
        for y, x in zip(bad_y, bad_x):
            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny_, nx_ = y + dy, x + dx
                    if 0 <= ny_ < ny and 0 <= nx_ < nx and not mask[ny_, nx_]:
                        neighbors.append(image[ny_, nx_])
            if neighbors:
                result[y, x] = np.median(neighbors)
        return result

    def remove_cosmic_ray(
        self,
        contrast: float = 5.0,
        cr_threshold: float = 5,
        neighbor_threshold: float = 5,
        error: float | None = None,
        mask_negative: bool = True,
        **kwargs
    ) -> tuple[Self, Self]:
        """LA-Cosmic アルゴリズムにより宇宙線を除去する.

        lacosmic.remove_cosmics を用いて画像データから宇宙線ヒットを
        検出・除去し、クリーン画像を持つ新しい ImageModel を返す。

        Parameters
        ----------
        contrast : float, optional
            ラプラシアン/ノイズ比のコントラスト閾値（デフォルト: 5.0）
        cr_threshold : float, optional
            宇宙線検出のシグマクリッピング閾値（デフォルト: 5）
        neighbor_threshold : float, optional
            近傍ピクセルの検出閾値（デフォルト: 5）
        error : float | None, optional
            誤差配列のスケール係数（デフォルト: 5）
        mask_negative : bool, optional
            True の場合、負の値を持つピクセルもマスク対象にする（デフォルト: True）
        **kwargs
            lacosmic.remove_cosmics に渡す追加キーワード引数

        Returns
        -------
        ImageModel
            宇宙線除去済み画像を持つ新しいインスタンス
        """
        # 1. DQ マスクの取得
        dq_mask = self.mask if self.mask is not None else np.zeros(self.shape, dtype=bool)

        # 2. Negative pixel マスクの生成
        neg_mask = (self.image < 0) if mask_negative else np.zeros(self.shape, dtype=bool)

        # 3. 合成マスク
        combined_mask = dq_mask | neg_mask

        # 4. マスク対象ピクセルを中央値補間
        if combined_mask.any():
            interpolated = self.median_interpolate(self.image, combined_mask)
        else:
            interpolated = self.image

        # 5. LA Cosmic 実行
        clean_image, mask = remove_cosmics(
            interpolated,
            contrast,
            cr_threshold,
            neighbor_threshold,
            error= error*np.ones(self.shape) if error is not None else None,
            mask=combined_mask,
            **kwargs
        )
        clean_model = replace(self,image=clean_image)
        mask_model = replace(self,image=mask)
        return clean_model, mask_model 
    
    @staticmethod
    def _resolve_output_path(
        source_path: Path | None,
        output_dir: Path | None,
        filename: str,
        output_suffix: str,
        overwrite: bool,
    ) -> Path:
        """出力パスを決定し、上書き防止チェックを行う.

        Parameters
        ----------
        source_path : Path | None
            元ファイルのディレクトリパス
        output_dir : Path | None
            出力先ディレクトリ。None の場合は source_path を使用する
        filename : str
            ファイルのルートネーム（ROOTNAME）
        output_suffix : str
            出力ファイルの接尾辞
        overwrite : bool
            既存ファイルの上書きを許可するか

        Returns
        -------
        Path
            出力先のフルパス

        Raises
        ------
        ValueError
            source_path が未設定かつ output_dir も指定されていない場合
        FileExistsError
            出力ファイルが既に存在し overwrite=False の場合
        """
        dest_dir = output_dir or source_path
        if dest_dir is None:
            raise ValueError(
                "出力先を決定できません。source_path が未設定のため、"
                "output_dir を指定してください。"
            )
        output_path = dest_dir / f"{filename}{output_suffix}.fits"
        if not overwrite and output_path.exists():
            raise FileExistsError(
                f"出力ファイルが既に存在します: {output_path}"
            )
        return output_path

    @staticmethod
    def _build_primary_header(
        primary_header: fits.Header | None,
    ) -> fits.Header:
        """Primary Header を準備し LACORR キーワードを CAL SWITCHES 末尾に挿入する.

        CALIBRATION SWITCHES セクションが存在する場合はその末尾、
        存在しない場合はヘッダー末尾に LACORR カードを挿入する。

        Parameters
        ----------
        primary_header : fits.Header | None
            元の Primary Header。None の場合は空ヘッダーを新規作成する

        Returns
        -------
        fits.Header
            LACORR キーワードを追加済みの Primary Header
        """
        header = primary_header.copy() if primary_header is not None else fits.Header()

        # CAL SWITCHES セクション末尾の空白カードインデックスを探す
        # パターン: keyword=='' かつ直前カードが非空白 = セクション末尾区切り
        section_end = None
        in_cal_section = False
        cards = list(header.cards)
        for i, card in enumerate(cards):
            if card.keyword == '' and 'CALIBRATION SWITCHES' in card.comment:
                in_cal_section = True
                continue
            if in_cal_section and card.keyword == '' and cards[i - 1].keyword != '':
                section_end = i
                break

        lacorr_card = fits.Card('LACORR', True, 'LA-Cosmic correction applied')
        if section_end is not None:
            header.insert(section_end, lacorr_card)
        else:
            header.append(lacorr_card)

        header.add_history('LA-Cosmic cosmic ray rejection applied (stis_la_cosmic)')
        return header

    def write_fits(
        self,
        output_suffix: str = "_lac",
        output_dir: Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """宇宙線除去済み画像を FITS ファイルとして出力する.

        PrimaryHDU の CALIBRATION SWITCHES セクション末尾に LACORR キーワードを追加し、
        ImageHDU に処理済みデータを格納して書き出す。

        Parameters
        ----------
        output_suffix : str, optional
            出力ファイルの接尾辞（デフォルト: "_lac"）
        output_dir : Path | None, optional
            出力先ディレクトリ。None の場合は source_path を使用する
        overwrite : bool, optional
            既存ファイルの上書きを許可するか（デフォルト: False）

        Returns
        -------
        Path
            出力した FITS ファイルのパス

        Raises
        ------
        ValueError
            source_path が設定されておらず output_dir も指定されていない場合
        FileExistsError
            出力ファイルが既に存在し overwrite=False の場合
        """
        output_path = self._resolve_output_path(
            self.source_path, output_dir, self.filename, output_suffix, overwrite
        )
        primary_header = self._build_primary_header(self.primary_header)
        primary_hdu = fits.PrimaryHDU(header=primary_header)
        image_hdu = fits.ImageHDU(data=self.image, header=self.header)
        fits.HDUList([primary_hdu, image_hdu]).writeto(output_path, overwrite=overwrite)
        return output_path

    def imshow(self,ax = None, **kwargs) -> plt.Axes: # pyright: ignore
        """画像データを matplotlib で表示する.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            描画先の Axes オブジェクト。None の場合は新規作成する
        **kwargs
            matplotlib.axes.Axes.imshow に渡す追加キーワード引数

        Returns
        -------
        matplotlib.axes.Axes
            描画に使用した Axes オブジェクト
        """
        if ax is None:
            _,ax = plt.subplots()
        ax.imshow(self.image,**kwargs)
        return ax

    def imshow_mask(self, ax=None, **kwargs) -> plt.Axes:  # pyright: ignore
        """マスク画像を matplotlib で表示する.

        DQ フラグから生成された bad pixel マスクを可視化する。
        マスクが設定されていない場合は全ゼロの画像を表示する。

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            描画先の Axes オブジェクト。None の場合は新規作成する
        **kwargs
            matplotlib.axes.Axes.imshow に渡す追加キーワード引数

        Returns
        -------
        matplotlib.axes.Axes
            描画に使用した Axes オブジェクト
        """
        if ax is None:
            _, ax = plt.subplots()
        mask_data = self.mask if self.mask is not None else np.zeros(self.shape, dtype=bool)
        ax.imshow(mask_data, cmap="gray", **kwargs)
        ax.set_title(f"{self.filename} (DQ Flag = {self.dq_flags})")
        return ax
    
    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape

    @property
    def filename(self) -> str:
        try:
            return self.header["ROOTNAME"] # pyright: ignore
        except KeyError:
            return "UNKNOWN"

    
@dataclass(frozen=True)
class ImageCollection:
    """複数の ImageModel をまとめて管理するコレクション.

    複数フレームの画像に対して一括での宇宙線除去と
    グリッド表示を提供する。LA-Cosmic パラメータを
    コレクション全体で共有する。

    Attributes
    ----------
    images : list[ImageModel]
        管理対象の ImageModel リスト
    contrast : float
        ラプラシアン/ノイズ比のコントラスト閾値（デフォルト: 5.0）
    cr_threshold : float
        宇宙線検出のシグマクリッピング閾値（デフォルト: 5）
    neighbor_threshold : float
        近傍ピクセルの検出閾値（デフォルト: 5）
    error : float
        誤差配列のスケール係数（デフォルト: 5）
    """

    images : list[ImageModel]
    contrast : float = 5.0
    cr_threshold : float = 5
    neighbor_threshold : float = 5
    error : float = 5


    def __repr__(self) -> str:
        return f"ImageCollection({len(self.images)} images, contrast={self.contrast}, cr_threshold={self.cr_threshold}, neighbor_threshold={self.neighbor_threshold}, error={self.error})"

    @classmethod
    def from_readers(
        cls,
        readers: ReaderCollection,
        index: int = 1,
        dq_index: int = 3,
        dq_flags: int = 16,
        contrast: float = 5.0,
        cr_threshold: float = 5,
        neighbor_threshold: float = 5,
        error: float = 5,
        ) -> Self:
        """ReaderCollection から ImageCollection を生成する.

        各 Reader の指定 HDU インデックスからデータとヘッダーを取得して
        ImageModel を構築し、コレクションとしてまとめる。

        Parameters
        ----------
        readers : ReaderCollection
            読み込み済みの FITS Reader コレクション
        index : int, optional
            画像データの HDU インデックス（デフォルト: 1、科学データ）
        dq_index : int, optional
            DQ 配列の HDU インデックス（デフォルト: 3）
        dq_flags : int, optional
            マスク対象の DQ ビットフラグ（デフォルト: 16 = hot pixel）。
            複数フラグはビット OR で指定（例: 16 | 256）
        contrast : float, optional
            ラプラシアン/ノイズ比のコントラスト閾値（デフォルト: 5.0）
        cr_threshold : float, optional
            宇宙線検出のシグマクリッピング閾値（デフォルト: 5）
        neighbor_threshold : float, optional
            近傍ピクセルの検出閾値（デフォルト: 5）
        error : float, optional
            誤差配列のスケール係数（デフォルト: 5）

        Returns
        -------
        ImageCollection
            生成されたコレクション
        """
        images = []
        for reader in readers:
            image = ImageModel.from_reader(
                reader,
                index=index,
                dq_index=dq_index,
                dq_flags=dq_flags,
            )
            images.append(image)
        return cls(
            images=images,
            contrast=contrast,
            cr_threshold=cr_threshold,
            neighbor_threshold=neighbor_threshold,
            error=error,
        )

    def remove_cosmic_ray(self, **kwargs) -> tuple[Self, Self]:
        """全画像から LA-Cosmic で宇宙線を一括除去する.

        コレクションが保持する LA-Cosmic パラメータを使用して、
        各 ImageModel に対して宇宙線除去を実行する。

        Parameters
        ----------
        **kwargs
            lacosmic.remove_cosmics に渡す追加キーワード引数

        Returns
        -------
        ImageCollection
            宇宙線除去済み画像を持つ新しいコレクション
        """
        images = []
        masks = []
        for image in self.images:
            clean_model, mask_model = (image.remove_cosmic_ray(
                contrast=self.contrast,
                cr_threshold=self.cr_threshold,
                neighbor_threshold=self.neighbor_threshold,
                error=self.error,
                **kwargs))
            images.append(clean_model)
            masks.append(mask_model)
        clean_collection_model = replace(self,images=images)
        mask_collection_model = replace(self,images=masks)
        return clean_collection_model, mask_collection_model

    def write_fits(
        self,
        output_suffix: str = "_lac",
        output_dir: Path | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        """全画像を FITS ファイルとして一括出力する.

        各 ImageModel の write_fits を呼び出し、
        元ファイルと同じディレクトリに新しい suffix で出力する。

        Parameters
        ----------
        output_suffix : str, optional
            出力ファイルの接尾辞（デフォルト: "_lac"）
        output_dir : Path | None, optional
            出力先ディレクトリ。None の場合は各画像の source_path を使用する
        overwrite : bool, optional
            既存ファイルの上書きを許可するか（デフォルト: False）

        Returns
        -------
        list[Path]
            出力した FITS ファイルパスのリスト

        Raises
        ------
        FileExistsError
            出力ファイルが既に存在し overwrite=False の場合
        """
        output_paths = []
        for image in self.images:
            path = image.write_fits(
                output_suffix=output_suffix,
                output_dir=output_dir,
                overwrite=overwrite,
            )
            output_paths.append(path)
        return output_paths

    def imshow(
        self,
        ax = None,
        vmax=1600,
        vmin=0,
        area:bool = False,
        x_center:int = 330,
        y_center:int = 550,
        half_width:int = 100,
        **kwargs
        ) -> plt.Axes: # pyright: ignore
        """全画像をサブプロットのグリッドで一覧表示する.

        2行3列のグリッドに各フレームを並べて表示する。
        area=True の場合、指定した中心座標と半幅で表示範囲を制限する。

        Parameters
        ----------
        ax : np.ndarray of matplotlib.axes.Axes, optional
            描画先の Axes 配列。None の場合は 2×3 グリッドを新規作成する
        vmax : float, optional
            カラーマップの最大値（デフォルト: 1600）
        vmin : float, optional
            カラーマップの最小値（デフォルト: 0）
        area : bool, optional
            True の場合、表示範囲を中心座標周辺に制限する（デフォルト: False）
        x_center : int, optional
            表示範囲の中心 x 座標（デフォルト: 330）
        y_center : int, optional
            表示範囲の中心 y 座標（デフォルト: 550）
        half_width : int, optional
            中心からの表示半幅（デフォルト: 100）
        **kwargs
            matplotlib.axes.Axes.imshow に渡す追加キーワード引数

        Returns
        -------
        np.ndarray of matplotlib.axes.Axes
            描画に使用した Axes 配列
        """
        if ax is None:
            _,ax = plt.subplots(2,3,figsize=(10,8))
        for i,image in enumerate(self.images):
            ax[i//3,i%3].imshow(image.image,vmax=vmax,vmin=vmin,**kwargs)
            ax[i//3,i%3].set_title(image.filename)
            if area:
                ax[i//3,i%3].set_xlim(x_center-half_width,x_center+half_width)
                ax[i//3,i%3].set_ylim(y_center-half_width,y_center+half_width)
        return ax

    def imshow_mask(
        self,
        ax=None,
        area: bool = False,
        x_center: int = 330,
        y_center: int = 550,
        half_width: int = 100,
        **kwargs
        ) -> plt.Axes:  # pyright: ignore
        """全画像のマスクをサブプロットのグリッドで一覧表示する.

        2行3列のグリッドに各フレームの DQ マスクを並べて表示する。
        area=True の場合、指定した中心座標と半幅で表示範囲を制限する。

        Parameters
        ----------
        ax : np.ndarray of matplotlib.axes.Axes, optional
            描画先の Axes 配列。None の場合は 2×3 グリッドを新規作成する
        area : bool, optional
            True の場合、表示範囲を中心座標周辺に制限する（デフォルト: False）
        x_center : int, optional
            表示範囲の中心 x 座標（デフォルト: 330）
        y_center : int, optional
            表示範囲の中心 y 座標（デフォルト: 550）
        half_width : int, optional
            中心からの表示半幅（デフォルト: 100）
        **kwargs
            matplotlib.axes.Axes.imshow に渡す追加キーワード引数

        Returns
        -------
        np.ndarray of matplotlib.axes.Axes
            描画に使用した Axes 配列
        """
        if ax is None:
            _, ax = plt.subplots(2, 3, figsize=(10, 8))
        for i, image in enumerate(self.images):
            mask_data = image.mask if image.mask is not None else np.zeros(image.shape, dtype=bool)
            ax[i // 3, i % 3].imshow(mask_data, cmap="gray", **kwargs)
            ax[i // 3, i % 3].set_title(f"{image.filename} (mask)")
            if area:
                ax[i // 3, i % 3].set_xlim(x_center - half_width, x_center + half_width)
                ax[i // 3, i % 3].set_ylim(y_center - half_width, y_center + half_width)
        return ax

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index) -> ImageModel:
        return self.images[index]
    
    def __iter__(self):
        return iter(self.images)
