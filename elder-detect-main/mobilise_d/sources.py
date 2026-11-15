"""Mobilise-D CVS V1.0.0 原始发布包的定义、下载与完整性校验。

只处理实施计划表格列出的四个文件。原始包缓存在
``anti-fall/_datasets/Mobilise-D-CVS/21060991/``，后续运行按 MD5 复用；
缺少或哈希不符时才重新下载。任何派生数据都从缓存重建。
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from io import BytesIO
import json
from pathlib import Path
import shutil
from typing import Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile


RECORD_URL = "https://zenodo.org/records/21060991"
RECORD_API = "https://zenodo.org/api/records/21060991"
DOI = "10.5281/zenodo.21060991"
VERSION = "V1.0.0"
LICENSE = "CC BY-NC-ND 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-nd/4.0/"
PUBLICATION_DATE = "2026-06-30"
ATTRIBUTION = (
    "Mobilise-D Clinical Validation Study (CVS) dataset, " + VERSION + ", "
    "Becker et al., Zenodo, doi:" + DOI + ", " + LICENSE
)


@dataclass(frozen=True)
class ReleaseFile:
    """发布包中的一个文件。"""

    name: str
    md5: str
    bytes: int
    purpose: str

    @property
    def url(self) -> str:
        return RECORD_URL + "/files/" + quote(self.name) + "?download=1"


CLINICAL = ReleaseFile(
    name="Clinical Data.zip",
    md5="3b3d8305fa7ef6923341a85bfd4a3945",
    bytes=5_205_397,
    purpose="受试者、访视、日记与逐次跌倒日期",
)
DAILY = ReleaseFile(
    name="Aggregated DMO Data.zip",
    md5="b6497e3cdcad2166f3ba38d5d3d64e35",
    bytes=4_738_103,
    purpose="每日步态聚合与佩戴质量",
)
DICTIONARY = ReleaseFile(
    name="Data Dictionary.zip",
    md5="20c43429f655575c6868ea2ef8343176",
    bytes=154_697,
    purpose="字段单位、有效值与结局定义",
)
BOUTS = ReleaseFile(
    name="Walking Bout DMO Data.zip",
    md5="4e9b7952cf2015ee3cc6615c0e626323",
    bytes=955_711_374,
    purpose="逐步行段数值，用于估计同日测量标准误",
)

RELEASE_FILES: tuple[ReleaseFile, ...] = (CLINICAL, DAILY, DICTIONARY, BOUTS)
COHORT_FILES: tuple[ReleaseFile, ...] = (CLINICAL, DAILY, DICTIONARY)


def default_cache_dir() -> Path:
    """原始包缓存目录：``anti-fall/_datasets/Mobilise-D-CVS/21060991``。"""
    project_root = Path(__file__).resolve().parents[1]
    return project_root.parent / "_datasets" / "Mobilise-D-CVS" / "21060991"


def release_manifest(files: Iterable[ReleaseFile] = RELEASE_FILES) -> dict:
    """发布包来源、版本、许可与哈希清单。"""
    return {
        "dataset": "Mobilise-D Clinical Validation Study (CVS) dataset",
        "version": VERSION,
        "publication_date": PUBLICATION_DATE,
        "doi": DOI,
        "record_url": RECORD_URL,
        "record_api": RECORD_API,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "attribution": ATTRIBUTION,
        "redistribution": "原始文件与逐人预测仅保留在本地研究目录；对外只报告汇总结果。",
        "files": [
            {
                "name": item.name,
                "md5": item.md5,
                "bytes": item.bytes,
                "url": item.url,
                "purpose": item.purpose,
            }
            for item in files
        ],
    }


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def is_cached(path: Path, item: ReleaseFile) -> bool:
    return path.is_file() and path.stat().st_size == item.bytes and file_md5(path) == item.md5


def download(item: ReleaseFile, dest_dir: Path, *, timeout: float = 120.0) -> Path:
    """下载并校验单个发布文件；缓存命中时直接返回。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / item.name
    if is_cached(target, item):
        return target
    partial = dest_dir / (item.name + ".part")
    if partial.exists():
        partial.unlink()
    request = Request(item.url, headers={"User-Agent": "jiebang-research/1.0"})
    digest = md5()
    received = 0
    with urlopen(request, timeout=timeout) as response, partial.open("wb") as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            digest.update(block)
            received += len(block)
            handle.write(block)
    if received != item.bytes:
        partial.unlink(missing_ok=True)
        raise ValueError(f"{item.name} 大小不符：{received} != {item.bytes}")
    if digest.hexdigest() != item.md5:
        partial.unlink(missing_ok=True)
        raise ValueError(f"{item.name} MD5 不符：{digest.hexdigest()} != {item.md5}")
    shutil.move(partial, target)
    return target


def ensure_files(
    dest_dir: Path,
    files: Iterable[ReleaseFile] = RELEASE_FILES,
    *,
    offline: bool = False,
) -> dict[str, Path]:
    """确保发布包在本地可用；离线模式只校验已有缓存。"""
    resolved: dict[str, Path] = {}
    for item in files:
        target = dest_dir / item.name
        if offline:
            if not is_cached(target, item):
                raise FileNotFoundError(f"缓存缺失或哈希不符：{target}")
            resolved[item.name] = target
        else:
            resolved[item.name] = download(item, dest_dir)
    return resolved


def zip_inventory(path: Path) -> list[dict]:
    """压缩包成员清单：名称与解压后字节数。"""
    with ZipFile(path) as archive:
        return [
            {"name": info.filename, "bytes": info.file_size}
            for info in archive.infolist()
            if not info.is_dir() and "__MACOSX" not in info.filename
        ]


def member_by_suffix(archive: ZipFile, suffix: str) -> str:
    """按后缀唯一匹配一个成员，避免依赖压缩包内目录顺序。"""
    matches = [
        name for name in archive.namelist()
        if name.endswith(suffix) and "__MACOSX" not in name and not name.endswith("/")
    ]
    if len(matches) != 1:
        raise ValueError(f"后缀 {suffix} 匹配到 {len(matches)} 个成员：{matches[:5]}")
    return matches[0]


def nested_frames(archive_path: Path, inner_suffix: str) -> Iterator[tuple[str, bytes]]:
    """遍历两层压缩包：外层成员是 zip，内层按后缀取表。"""
    with ZipFile(archive_path) as outer:
        for name in outer.namelist():
            if "__MACOSX" in name or not name.lower().endswith(".zip"):
                continue
            with ZipFile(BytesIO(outer.read(name))) as inner:
                for member in inner.namelist():
                    if member.endswith(inner_suffix) and not member.endswith("/"):
                        yield name + "::" + member, inner.read(member)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False), encoding="utf-8")


def describe_error(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTPError {exc.code}: {exc.reason}"
    if isinstance(exc, URLError):
        return f"URLError: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"
