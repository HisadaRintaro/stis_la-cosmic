"""画像モデル.

STIS FITS ファイルから読み込んだ画像データを管理し、
宇宙線除去 (LA-Cosmic) および可視化機能を提供するモジュール。
"""

from typing import Literal, cast
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self
import numpy as np
from .fits_reader import ReaderCollection, STISFitsReader
from astropy.io import fits
import matplotlib.pyplot as plt
from lacosmic import remove_cosmics
from scipy.ndimage import median_filter

@dataclass(frozen=True)
class ImageUnit:
    """data と header のペア（astropy.io.fits.ImageHDU の型安全なラッパー）.

    Attributes
    ----------
    data : np.ndarray
        画像データ配列
    header : fits.Header
        対応するヘッダー
    """

    data: np.ndarray
    header: fits.Header

    def __repr__(self) -> str:
        return f"ImageUnit(data={self.data.shape}, header={"fits.Header" if self.header else None})"

    def to_hdu(self) -> fits.ImageHDU:
        """fits.ImageHDU に変換する.

        bool 型の data は FITS で扱えないため uint8 に変換する。

        Returns
        -------
        fits.ImageHDU
            data と header を格納した ImageHDU
        """
        data = self.data.astype(np.uint8) if self.data.dtype == bool else self.data
        return fits.ImageHDU(data=data, header=self.header)

    @property
    def wavelength(self) -> np.ndarray | None:
        """ヘッダーの WCS キーワードから波長配列を生成する.

        CRVAL1（参照ピクセルの波長）と CDELT1 または CD1_1（波長/pixel）
        から各ピクセルの波長を計算する。必要なキーワードが存在しない場合は
        None を返す。

        Returns
        -------
        np.ndarray | None
            波長配列 [Å]。WCS 情報が不足している場合は None
        """
        crval1 = cast(float | None, self.header.get("CRVAL1"))
        cdelt1 = cast(float | None, self.header.get("CDELT1", self.header.get("CD1_1")))
        if crval1 is None or cdelt1 is None:
            return None
        crpix1 = cast(float, self.header.get("CRPIX1", 1.0))
        n_pixels = self.data.shape[1]
        return crval1 + cdelt1 * (np.arange(n_pixels) - (crpix1 - 1))


@dataclass(frozen=True)
class ImageModel:
    """STIS 画像データの単一フレームモデル.

    1つの FITS ファイルに対応する科学画像・誤差・DQ・マスクを
    ImageUnit として保持し、宇宙線除去および画像表示の機能を提供する。

    Attributes
    ----------
    primary_header : fits.Header
        Primary Header（HDU 0）
    sci : ImageUnit
        科学画像（SCI, HDU 1）の data / header ペア
    err : ImageUnit | None
        誤差（ERR, HDU 2）の data / header ペア。None の場合は存在しない
    dq : ImageUnit | None
        DQ（HDU 3）の data / header ペア。None の場合は存在しない
    cr_mask : ImageUnit | None
        宇宙線マスク。LA-Cosmic 後 HDU に追加される
    source_path : Path | None
        元の FITS ファイルが格納されているディレクトリパス
    dq_flags : int
        マスク対象の DQ ビットフラグ（デフォルト: 16 = hot pixel）
    dq_mask : np.ndarray | None
        DQ ビットフラグに基づいて生成された bad pixel マスク（True = bad pixel）。補完後とLA-Cosmic 後 HDU に追加される
    """

    primary_header: fits.Header
    sci: ImageUnit
    err: ImageUnit | None = None
    dq: ImageUnit | None = None
    cr_mask: ImageUnit | None = None
    source_path: Path | None = None
    dq_flags: int = 16
    dq_mask: np.ndarray | None = None

    def __repr__(self) -> str:
        dq_mask_info = (
            f"dq_mask_count={self.dq_mask.sum()}"
            if self.dq_mask is not None
            else "dq_mask=None"
        )
        cr_mask_info = (
            f"cr_mask_count={self.cr_mask.data.sum()}"
            if self.cr_mask is not None
            else "cr_mask=None"
        )
        return (
            f"ImageModel(\n"
            f"  sci={self.sci.data.shape},\n"
            f"  err={self.err is not None},\n"
            f"  dq={self.dq is not None},\n"
            f"  {cr_mask_info},\n"
            f"  source_path={self.source_path}\n"
            f"  dq_flags={self.dq_flags}\n"
            f"  {dq_mask_info},\n"
            f")"
        )

    @classmethod
    def from_reader(
        cls,
        reader: STISFitsReader,
        dq_flags: int = 16,
    ) -> Self:
        """STISFitsReader から ImageModel を生成する.

        Parameters
        ----------
        reader : STISFitsReader
            読み込み済みの FITS Reader
        dq_flags : int, optional
            マスク対象の DQ ビットフラグ（デフォルト: 16 = hot pixel）。
            複数フラグはビット OR で指定（例: 16 | 256）

        Returns
        -------
        ImageModel
            生成されたモデル
        """
        try:
            err = ImageUnit(
                data=reader.image_data(2),
                header=reader.header(2),
            )
        except KeyError:
            err = None
        try:
            dq_data = reader.image_data(3)
            dq = ImageUnit(
                data=dq_data,
                header=reader.header(3),
            )
            dq_mask = (dq_data & dq_flags).astype(bool)
        except KeyError:
            dq = None
            dq_mask = None
        return cls(
            primary_header=reader.header(0),
            sci=ImageUnit(
                data=reader.image_data(1),
                header=reader.header(1),
            ),
            err=err,
            dq=dq,
            dq_mask=dq_mask,
            source_path=reader.filename.parent,
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
        median = median_filter(image, size=3)
        result[mask] = median[mask]
        return result

    def interpolate_bad_pixels(
        self,
        mask_negative: bool = True,
    ) -> Self:
        """DQ マスクおよび負値ピクセルを中央値補間した ImageModel を返す.

        DQ フラグに基づく bad pixel マスクと、オプションで負の値を持つ
        ピクセルのマスクを合成し、該当ピクセルを周囲8近傍の中央値で補間する。

        Parameters
        ----------
        mask_negative : bool, optional
            True の場合、負の値を持つピクセルもマスク対象にする（デフォルト: True）

        Returns
        -------
        ImageModel
            補間済み ImageModel
        """
        dq_mask = (
            self.dq_mask
            if self.dq_mask is not None
            else np.zeros(self.shape, dtype=bool)
        )
        neg_mask = (
            (self.sci.data < 0)
            if mask_negative
            else np.zeros(self.shape, dtype=bool)
        )
        combined_mask = dq_mask | neg_mask

        if combined_mask.any():
            interpolated = self.median_interpolate(self.sci.data, combined_mask)
        else:
            interpolated = self.sci.data

        return replace(
            self,
            sci=replace(self.sci, data=interpolated),
            dq_mask=combined_mask,
        )

    def remove_cosmic_ray(
        self,
        contrast: float = 5.0,
        cr_threshold: float = 5,
        neighbor_threshold: float = 5,
        error: float | None = 5,
        mask_negative: bool = True,
        **kwargs,
    ) -> Self:
        """LA-Cosmic アルゴリズムにより宇宙線を除去する.

        interpolate_bad_pixels で前処理を行った後、
        lacosmic.remove_cosmics で宇宙線ヒットを検出・除去する。
        宇宙線マスクは戻り値の .cr_mask 属性に格納される。

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
            宇宙線除去済み画像を持つ新しいインスタンス。
            .cr_mask に宇宙線マスクが格納される
        """
        preprocessed= self.interpolate_bad_pixels(
            mask_negative=mask_negative
        )

        clean_data, cr_mask = remove_cosmics(
            preprocessed.sci.data,
            contrast,
            cr_threshold,
            neighbor_threshold,
            error=error * np.ones(self.shape) if error is not None else None,
            mask=preprocessed.dq_mask,
            **kwargs,
        )

        cr_mask_header = fits.Header()
        cr_mask_header["EXTNAME"] = "LACOSMIC"
        cr_mask_unit = ImageUnit(data=cr_mask, header=cr_mask_header)

        return replace(
            self,
            sci=replace(self.sci, data=clean_data),
            cr_mask=cr_mask_unit,
        )

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
        lacorr_applied: bool = True,
    ) -> fits.Header:
        """Primary Header を準備し、必要に応じて LACORR キーワードを挿入する.

        lacorr_applied=True の場合は CALIBRATION SWITCHES セクション末尾に
        LACORR カードと history を追加する。False の場合は追加しない。

        Parameters
        ----------
        primary_header : fits.Header | None
            元の Primary Header。None の場合は空ヘッダーを新規作成する
        lacorr_applied : bool, optional
            True の場合のみ LACORR=True キーワードと history を追加する
            （デフォルト: True）

        Returns
        -------
        fits.Header
            処理済みの Primary Header
        """
        header = primary_header.copy() if primary_header is not None else fits.Header()

        if not lacorr_applied:
            return header

        # CAL SWITCHES セクション末尾の空白カードインデックスを探す
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
        """画像を FITS ファイルとして出力する.

        LA-Cosmic が適用済み（mask の EXTNAME が 'LACOSMIC'）の場合は
        PrimaryHDU に LACORR=True キーワードと history を追加する。
        未適用のまま '_lac' suffix で書き出そうとした場合は UserWarning を発行する。

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
        lacorr_applied = (
            self.cr_mask is not None
            and self.cr_mask.header.get("EXTNAME") == "LACOSMIC"
        )
        if not lacorr_applied and output_suffix == "_lac":
            warnings.warn(
                f"{self.filename}: LA-Cosmic が未適用ですが '"
                f"{output_suffix}' suffix で書き出しています。"
                " remove_cosmic_ray() を実行済みか確認してください。",
                UserWarning,
                stacklevel=2,
            )
        output_path = self._resolve_output_path(
            self.source_path, output_dir, self.filename, output_suffix, overwrite
        )
        primary_header = self._build_primary_header(
            self.primary_header, lacorr_applied=lacorr_applied
        )
        hdu_list: list[fits.PrimaryHDU | fits.ImageHDU] = [
            fits.PrimaryHDU(header=primary_header),
            self.sci.to_hdu(),
        ]
        if self.err is not None:
            hdu_list.append(self.err.to_hdu())
        if self.dq is not None:
            hdu_list.append(self.dq.to_hdu())
        if self.cr_mask is not None:
            hdu_list.append(self.cr_mask.to_hdu())
        fits.HDUList(hdu_list).writeto(output_path, overwrite=overwrite)
        return output_path

    def plot_spectrum(
        self,
        slit_index: int,
        ax=None,
        **kwargs,
    ) -> plt.Axes:  # pyright: ignore
        """指定スリット位置での波長方向スペクトルをプロットする.

        sci の data から slit_index 行を取り出し、横軸を波長（またはピクセル）、
        縦軸をカウントとしてプロットする。

        Parameters
        ----------
        slit_index : int
            スリット方向（y 軸）のインデックス
        ax : matplotlib.axes.Axes, optional
            描画先の Axes オブジェクト。None の場合は新規作成する
        **kwargs
            matplotlib.axes.Axes.plot に渡す追加キーワード引数

        Returns
        -------
        matplotlib.axes.Axes
            描画に使用した Axes オブジェクト
        """
        if ax is None:
            _, ax = plt.subplots()

        spectrum = self.sci.data[slit_index, :]
        wavelength = self.sci.wavelength

        if wavelength is not None:
            ax.plot(wavelength, spectrum, **kwargs)
            ax.set_xlabel("Wavelength [Å]")
        else:
            ax.plot(spectrum, **kwargs)
            ax.set_xlabel("Pixel")

        ax.set_ylabel("Counts")
        ax.set_title(f"{self.filename} (slit index = {slit_index})")
        return ax

    def imshow(self, ax=None, **kwargs) -> plt.Axes:  # pyright: ignore
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
            _, ax = plt.subplots()
        ax.imshow(self.sci.data, **kwargs)
        return ax

    def imshow_mask(self, ax=None, **kwargs) -> plt.Axes:  # pyright: ignore
        """DQ マスク画像を matplotlib で表示する.

        DQ フラグから生成された bad pixel マスク（dq_mask）を可視化する。
        dq_mask が設定されていない場合は全ゼロの画像を表示する。

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
        mask_data = (
            self.dq_mask.data
            if self.dq_mask is not None
            else np.zeros(self.shape, dtype=bool)
        )
        ax.imshow(mask_data, cmap="gray", **kwargs)
        ax.set_title(f"{self.filename} (DQ Flag = {self.dq_flags})")
        return ax

    def imshow_cr_mask(self, ax=None, **kwargs) -> plt.Axes:  # pyright: ignore
        """LA-Cosmic マスク画像を matplotlib で表示する.

        remove_cosmic_ray() 後の宇宙線マスク（cr_mask）を可視化する。
        cr_mask が設定されていない場合は全ゼロの画像を表示する。

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
        mask_data = (
            self.cr_mask.data
            if self.cr_mask is not None
            else np.zeros(self.shape, dtype=bool)
        )
        ax.imshow(mask_data, cmap="gray", **kwargs)
        ax.set_title(f"{self.filename} (LA-Cosmic mask)")
        return ax

    @property
    def shape(self) -> tuple[int, int]:
        return self.sci.data.shape  # type: ignore[return-value]

    @property
    def filename(self) -> str:
        try:
            return self.sci.header["ROOTNAME"]  # pyright: ignore
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

    images: list[ImageModel]
    contrast: float = 5.0
    cr_threshold: float = 5
    neighbor_threshold: float = 5
    error: float = 5

    def __repr__(self) -> str:
        return (
            f"ImageCollection({len(self.images)} images, \n"
            + f"contrast={self.contrast}, \n"
            + f"cr_threshold={self.cr_threshold}, \n"
            + f"neighbor_threshold={self.neighbor_threshold}, \n"
            + f"error={self.error})\n"
        )

    @classmethod
    def from_readers(
        cls,
        readers: ReaderCollection,
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
        images = [
            ImageModel.from_reader(reader, dq_flags=dq_flags)
            for reader in readers
        ]
        return cls(
            images=images,
            contrast=contrast,
            cr_threshold=cr_threshold,
            neighbor_threshold=neighbor_threshold,
            error=error,
        )
    def interpolate_bad_pixels(self, **kwargs) -> Self:
        """全画像から data quality ビットフラグに基づいてhot pixelとnegative pixelを補間する。 

        各 ImageModelに対してlacosmic.interpolate_bad_pixelsを呼び出し、補間後の画像を生成する。
        bad_pixelsのマスクは各 ImageModel の .mask 属性に格納される。

        Parameters
        ----------
        **kwargs
            lacosmic.interpolate_bad_pixels に渡す追加キーワード引数

        Returns
        -------
        ImageCollection
            補間済み画像を持つ新しいコレクション。
            各 ImageModel の .mask に補間マスク（EXTNAME='BADPIXEL'）が格納される
        """
        images = [
            image.interpolate_bad_pixels(
                **kwargs,
            )
            for image in self.images
        ]
        return replace(self, images=images)

    def remove_cosmic_ray(self, **kwargs) -> Self:
        """全画像から LA-Cosmic で宇宙線を一括除去する.

        コレクションが保持する LA-Cosmic パラメータを使用して、
        各 ImageModel に対して宇宙線の除去を行い、
        宇宙線マスクは各 ImageModel の .mask 属性に格納される。

        Parameters
        ----------
        **kwargs
            lacosmic.remove_cosmics に渡す追加キーワード引数

        Returns
        -------
        ImageCollection
            宇宙線除去済み画像を持つ新しいコレクション。
            各 ImageModel の .mask に宇宙線マスク（EXTNAME='LACOSMIC'）が格納される
        """
        images = [
            image.remove_cosmic_ray(
                contrast=self.contrast,
                cr_threshold=self.cr_threshold,
                neighbor_threshold=self.neighbor_threshold,
                error=self.error,
                **kwargs,
            )
            for image in self.images
        ]
        return replace(self, images=images)

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
        return [
            image.write_fits(
                output_suffix=output_suffix,
                output_dir=output_dir,
                overwrite=overwrite,
            )
            for image in self.images
        ]

    @staticmethod
    def save_fig(ax: np.ndarray, save_path: Path | str, title: str | None = None) -> None:
        """Axes配列からFigureを取得し、タイトルを設定して画像を保存・クローズする."""
        fig = ax.flat[0].figure
        if title:
            fig.suptitle(title)
        fig.savefig(save_path)
        print(f"saved {save_path}")
        plt.close(fig)

    def imshow(
        self,
        ax=None,
        vmax=1600,
        vmin=0,
        area: bool = False,
        x_center: int = 330,
        y_center: int = 550,
        half_width: int = 100,
        save_path: Path | str | None = None,
        title: str | None = None,
        **kwargs,
    ) -> plt.Axes:  # pyright: ignore
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
        save_path : Path | str, optional
            保存先のファイルパス（指定された場合、描画後に保存して Figure を閉じる）
        title : str, optional
            Figure全体のタイトル
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
            ax[i // 3, i % 3].imshow(image.sci.data, vmax=vmax, vmin=vmin, **kwargs)
            ax[i // 3, i % 3].set_title(image.filename)
            if area:
                ax[i // 3, i % 3].set_xlim(x_center - half_width, x_center + half_width)
                ax[i // 3, i % 3].set_ylim(y_center - half_width, y_center + half_width)
                
        if save_path:
            self.save_fig(ax, save_path, title)
            
        return ax

    def imshow_mask(
        self,
        ax=None,
        mask_type: Literal["dq", "cr"] = "dq",
        area: bool = False,
        x_center: int = 330,
        y_center: int = 550,
        half_width: int = 100,
        save_path: Path | str | None = None,
        title: str | None = None,
        **kwargs,
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
        save_path : Path | str, optional
            保存先のファイルパス（指定された場合、描画後に保存して Figure を閉じる）
        title : str, optional
            Figure全体のタイトル
        **kwargs
            matplotlib.axes.Axes.imshow に渡す追加キーワード引数

        Returns
        -------
        np.ndarray of matplotlib.axes.Axes
            描画に使用した Axes 配列
        """
        if mask_type not in ("dq", "cr"):
            raise ValueError("mask_type must be 'dq' or 'cr'")
        if ax is None:
            _, ax = plt.subplots(2, 3, figsize=(10, 8))
        for i, image in enumerate(self.images):
            if mask_type == "dq":
                mask_data = image.dq_mask
                title_suffix = "DQ mask"
            elif mask_type == "cr":
                mask_data = (
                    image.cr_mask.data
                    if image.cr_mask is not None
                    else np.zeros(image.shape, dtype=bool)
                )
                title_suffix = "CR mask"
            ax[i // 3, i % 3].imshow(mask_data, cmap="gray", **kwargs)
            ax[i // 3, i % 3].set_title(f"{image.filename} ({title_suffix})")
            if area:
                ax[i // 3, i % 3].set_xlim(x_center - half_width, x_center + half_width)
                ax[i // 3, i % 3].set_ylim(y_center - half_width, y_center + half_width)
                
        if save_path:
            self.save_fig(ax, save_path, title)
            
        return ax

    def plot_spectrum_comparison(
        self,
        other: "ImageCollection",
        slit_index: int,
        labels: tuple[str, str] = ("before CR removal", "after CR removal"),
        image_source: Literal["self", "other"] = "other",
        area: bool = False,
        x_center: int = 330,
        y_center: int = 550,
        half_width: int = 100,
        imshow_kwargs: dict | None = None,
        save_path: Path | str | None = None,
        title: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """2つのコレクション間でスペクトルを比較プロットする.

        各フレームについて、self と other のスペクトルを同一 Axes 上に
        重ねてプロットし、左側2行3列のグリッドで一覧表示する。
        右側パネルにはイメージデータとスライス位置を水平線で表示する。

        Parameters
        ----------
        other : ImageCollection
            比較対象のコレクション（例: 宇宙線除去後）
        slit_index : int
            スリット方向（y 軸）のインデックス
        labels : tuple[str, str], optional
            凡例ラベル（self, other の順）。
            デフォルト: ("before CR removal", "after CR removal")
        image_source : {"self", "other"}, optional
            右側パネルに表示する画像の選択。
            "self" の場合は処理前、"other" の場合は処理後の画像を表示する
            （デフォルト: "other"）
        imshow_kwargs : dict, optional
            右側パネルの ImageModel.imshow に渡すキーワード引数
            （例: {"vmin": 0, "vmax": 1600}）
        save_path : Path | str | None, optional
            保存先のファイルパス（指定された場合、描画後に保存して Figure を閉じる）
        title : str | None, optional
            Figure 全体のタイトル
        **kwargs
            matplotlib.axes.Axes.plot に渡す追加キーワード引数（スペクトル用）

        Returns
        -------
        np.ndarray of matplotlib.axes.Axes
            描画に使用した Axes 配列（スペクトル用 2×3 + 画像パネル）

        Raises
        ------
        ValueError
            2つのコレクションの画像数が一致しない場合
        """
        if len(self) != len(other):
            raise ValueError(
                f"画像数が一致しません: self={len(self)}, other={len(other)}"
            )

        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1])

        # 左側: 2×3 のスペクトル比較プロット
        spec_axes = np.empty((2, 3), dtype=object)
        for row in range(2):
            for col in range(3):
                spec_axes[row, col] = fig.add_subplot(gs[row, col])

        for ax, img_self, img_other in zip(
            spec_axes.flat, self.images, other.images
        ):
            img_self.plot_spectrum(slit_index, ax=ax, label=labels[0], **kwargs)
            img_other.plot_spectrum(slit_index, ax=ax, label=labels[1], **kwargs)
            ax.legend(fontsize="small")

        # 未使用の Axes を非表示にする
        for ax in spec_axes.flat[len(self.images) :]:
            ax.set_visible(False)

        # 右側: イメージデータ + スライス位置表示
        ax_image = fig.add_subplot(gs[:, 3])
        source = other if image_source == "other" else self
        display_image = source.images[0]
        display_image.imshow(ax=ax_image, **(imshow_kwargs or {}))
        ax_image.axhline(
            y=slit_index, color="red", linestyle="--", linewidth=1.5,
            label=f"slit index = {slit_index}",
        )
        ax_image.legend(fontsize="small", loc="upper right")
        ax_image.set_title(f"{display_image.filename}")
        ax_image.set_xlabel("Pixel (dispersion)")
        ax_image.set_ylabel("Pixel (spatial)")
        if area:
            ax_image.set_xlim(x_center - half_width, x_center + half_width)
            ax_image.set_ylim(y_center - half_width, y_center + half_width)

        param_text = (
            f"contrast={self.contrast}  "
            f"cr_threshold={self.cr_threshold}  "
            f"neighbor_threshold={self.neighbor_threshold}  "
            f"error={self.error}"
        )
        fig.text(0.5, 0.01, param_text, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.93))

        if save_path:
            # save_fig は np.ndarray を期待するので spec_axes を渡す
            self.save_fig(spec_axes, save_path, title)

        return spec_axes

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> ImageModel:
        return self.images[index]

    def __iter__(self):
        return iter(self.images)
