"""ファイル探索モデル.

観測データの FITS ファイルをディレクトリ構造から検索するためのモデル。
ディレクトリ、ファイル接尾辞、拡張子、およびディレクトリ深度を指定して
glob パターンによるファイル探索を行う。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self
from .fits_reader import STISFitsReader


@dataclass(frozen=True)
class InstrumentModel:
    """観測装置のファイル構成モデル.

    指定されたディレクトリ内から、接尾辞と拡張子のパターンに一致する
    FITS ファイルを検索する。

    Attributes
    ----------
    file_directory : str
        データファイルのルートディレクトリパス
    suffix : str
        ファイル名の接尾辞（例: "_flt"）
    extension : str
        ファイルの拡張子（例: ".fits"）
    depth : int
        ディレクトリ探索の深度（デフォルト: 1）
    exclude_files : tuple[str, ...]
        除外するファイル名のタプル。
        ファイルのステム名（拡張子なし）またはフルネーム（拡張子あり）で
        マッチしたファイルをリストから除外する。（デフォルト: ()）
    """

    file_directory: str
    suffix: str
    extension: str
    depth: int = 1
    exclude_files: tuple[str, ...] = ()

    @classmethod
    def load(
        cls,
        file_directory: str,
        suffix: str = "",
        extension: str = "",
        depth: int = 1,
        exclude_files: tuple[str, ...] = (),
    ) -> Self:
        """InstrumentModel を生成する.

        Parameters
        ----------
        file_directory : str
            データファイルのルートディレクトリパス
        suffix : str, optional
            ファイル名の接尾辞（デフォルト: ""）
        extension : str, optional
            ファイルの拡張子（デフォルト: ""）
        depth : int, optional
            ディレクトリ探索の深度（デフォルト: 1）
        exclude_files : tuple[str, ...], optional
            除外するファイル名のタプル（デフォルト: ()）

        Returns
        -------
        InstrumentModel
            生成されたモデル
        """
        return cls(
            file_directory=file_directory,
            suffix=suffix,
            extension=extension,
            depth=depth,
            exclude_files=exclude_files,
        )

    @property
    def path_list(self) -> list[Path]:
        """現在の設定に基づいてファイルパスの一覧を取得する.

        Returns
        -------
        list[Path]
            パターンに一致するファイルパスのソート済みリスト
        """
        path = Path(self.file_directory)
        pattern = "*/" * self.depth + f"*{self.suffix}{self.extension}"
        path_list = list(path.glob(pattern))
        if self.exclude_files:
            exclude_set = set(self.exclude_files)
            path_list = [
                p for p in path_list
                if p.stem not in exclude_set and p.name not in exclude_set
            ]
        path_list.sort()
        return path_list

    @property
    def reader_list(self) -> list[STISFitsReader]:
        """現在の設定に基づいて STISFitsReader のリストを生成する.

        path_list の各パスに対して FITS ファイルを開き、
        Reader インスタンスを返す。

        Returns
        -------
        list[STISFitsReader]
            パターンに一致するファイルから生成された Reader のリスト
        """
        return [STISFitsReader.open(p) for p in self.path_list]
