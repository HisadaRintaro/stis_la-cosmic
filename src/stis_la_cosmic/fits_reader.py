"""STIS FITS ファイルリーダー.

STIS の FITS ファイルを1回のオープンで全データ（ヘッダー + 画像データ）を
読み取り、各モデルへ供給する Reader クラスを提供する。

STIS の FITS ファイル構成:

- HDU 0: PrimaryHDU（観測メタデータ）
- HDU 1: ImageHDU（科学データ）
- HDU 2: ImageHDU（統計的誤差）
- HDU 3: ImageHDU（品質フラグ）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, Iterator

import numpy as np
from astropy.io import fits  # type: ignore


@dataclass(frozen=True)
class STISFitsReader:
    """STIS FITS ファイルの読み取りを一元管理するクラス.

    ファイルを1回だけ開いて全 HDU のヘッダーとデータをキャッシュし、
    各モデルに必要な情報を提供する。

    Attributes
    ----------
    filename : Path
        FITS ファイルのパス
    headers : dict[int, fits.Header]
        HDU 番号をキーとするヘッダー辞書
    data : dict[int, np.ndarray]
        HDU 番号をキーとするデータ配列辞書
    """

    filename: Path
    headers: dict[int, fits.Header] = field(repr=False)
    data: dict[int, np.ndarray] = field(repr=False)

    @classmethod
    def open(cls, filename: Path) -> Self:
        """FITS ファイルを開いて全データを読み込む.

        ファイルを1回だけ開き、全 HDU のヘッダーとデータを
        メモリにキャッシュする。

        Parameters
        ----------
        filename : Path
            FITS ファイルのパス

        Returns
        -------
        STISFitsReader
            読み込み済みの Reader インスタンス
        """
        headers: dict[int, fits.Header] = {}
        data_dict: dict[int, np.ndarray] = {}

        with fits.open(filename) as hdul:  # type: ignore
            for i, hdu in enumerate(hdul):  # type: ignore
                headers[i] = hdu.header  # type: ignore
                if hasattr(hdu, "data") and isinstance(hdu.data, np.ndarray):  # type: ignore
                    data_dict[i] = hdu.data  # type: ignore

        return cls(filename=filename, headers=headers, data=data_dict)

    def header(self, hdu_number: int) -> fits.Header:
        """指定した HDU 番号のヘッダーを返す.

        Parameters
        ----------
        hdu_number : int
            HDU 番号

        Returns
        -------
        fits.Header
            指定 HDU のヘッダー

        Raises
        ------
        KeyError
            指定した HDU 番号が存在しない場合
        """
        if hdu_number not in self.headers:
            raise KeyError(f"HDU {hdu_number} のヘッダーが見つかりません")
        return self.headers[hdu_number]

    def image_data(self, hdu_number: int) -> np.ndarray:
        """指定した HDU 番号のデータ配列を返す.

        Parameters
        ----------
        hdu_number : int
            HDU 番号

        Returns
        -------
        np.ndarray
            指定 HDU のデータ配列

        Raises
        ------
        KeyError
            指定した HDU 番号のデータが存在しない場合
        """
        if hdu_number not in self.data:
            raise KeyError(f"HDU {hdu_number} のデータが見つかりません")
        return self.data[hdu_number]

    def spectrum_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """STIS のスペクトルデータ（科学データ、誤差、品質フラグ）を返す.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (HDU 1: 科学データ, HDU 2: 統計的誤差, HDU 3: 品質フラグ)
        """
        return self.image_data(1), self.image_data(2), self.image_data(3)

    def info(self) -> str:
        """読み込んだ HDU の概要情報を返す.

        Returns
        -------
        str
            HDU 番号、ヘッダーの有無、データ形状の一覧
        """
        lines = [f"STISFitsReader: {self.filename}"]
        for i in sorted(self.headers.keys()):
            shape = self.data[i].shape if i in self.data else "No data"
            lines.append(f"  HDU {i}: shape={shape}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ReaderCollection:
    """複数の STISFitsReader をまとめて管理するコレクション.

    ファイルリストを一括読み込みし、ヘッダー情報の確認や
    データの探索を効率的に行う。

    Attributes
    ----------
    readers : list[STISFitsReader]
        読み込み済み Reader のリスト
    """

    readers: list[STISFitsReader]

    @classmethod
    def from_paths(cls, paths: list[Path]) -> Self:
        """パスのリストから Reader を一括生成する.

        Parameters
        ----------
        paths : list[Path]
            FITS ファイルパスのリスト

        Returns
        -------
        ReaderCollection
            読み込み済みコレクション
        """
        return cls(readers=[STISFitsReader.open(p) for p in paths])

    def __len__(self) -> int:
        return len(self.readers)

    def __getitem__(self, index: int) -> STISFitsReader:
        return self.readers[index]

    def __iter__(self) -> Iterator[STISFitsReader]:
        return iter(self.readers)

    def info(self) -> str:
        """全ファイルのヘッダー概要を返す.

        Returns
        -------
        str
            各ファイルの HDU 概要を連結した文字列
        """
        return "\n\n".join(reader.info() for reader in self.readers)

