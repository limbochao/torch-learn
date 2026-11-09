#!/usr/bin/env python3
"""查询并下载 Triton-Ascend 每日 wheel，复用同目录 torch_npu 下载脚本的 OBS 和下载实现。

* ``--source``：``daily``（默认，OBS 时间戳目录）或 ``nightly``（华为云公开 nightly 索引）。
  两者是不同的发布来源，不会自动切换。
* ``--build``：``latest``（默认）、日期 ``YYYYMMDD`` 或精确时间戳 ``YYYYMMDDhhmmss``。
  自动选择范围内包含目标 Python 版本及架构的最新构建。
* ``--python-version``：支持 ``3.11``、``311``、``python311``，默认当前 Python 主次版本。
* ``--arch``：``x86_64`` 或 ``aarch64``，默认当前架构，也支持 ``amd64``、``x64``、``arm64``。
* ``--output-dir``：下载目录，默认当前工作目录。
* ``--list-builds N``：列出所选来源最近 N 个构建，默认关闭。
* ``--resolve-only``：只显示选中的对象和下载地址，默认关闭。
* ``--wheel-url``：已有完整 wheel URL 时直接使用，并从 URL 提取构建时间、Python 版本和架构，
  无需列目录；仍会校验 ``--build``、``--python-version`` 和 ``--arch``。
* ``--timeout``：单次请求超时秒数，默认 30，这是隐藏的调试配置。

OBS 自动识别依赖匿名列目录权限，403 时明确报错；公开 nightly 索引通常只包含日期而非完整时间戳，
不能用其推算 OBS 时间戳。HTTPS 与 torch_npu 脚本一致，不校验证书。仅下载，不安装。
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from download_torch_npu_daily import BuildError, ObjectInfo, ObsClient, normalize_arch, normalize_python_version


OBS_ENDPOINT = "https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com"
OBS_PREFIX = "triton-ascend/"
NIGHTLY_INDEX = "https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/triton-ascend/"
BUILD_RE = re.compile(r"\d{8}(?:\d{6})?")


class DailyClient(ObsClient):
    def __init__(self, timeout: float = 30.0) -> None:
        super().__init__(timeout)
        self.endpoint = OBS_ENDPOINT

    def _open(self, url: str, method: str = "GET"):
        request = urllib.request.Request(
            url, method=method, headers={"User-Agent": "triton-ascend-build-downloader/1.0"}
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context)
        except urllib.error.HTTPError as error:
            raise BuildError(f"request failed with HTTP {error.code}: {url}") from error
        except urllib.error.URLError as error:
            raise BuildError(f"request failed: {error.reason}") from error

    def object_url(self, key: str) -> str:
        if key.startswith(("https://", "http://")):
            parts = urllib.parse.urlsplit(key)
            path = urllib.parse.quote(urllib.parse.unquote(parts.path), safe="/")
            return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))
        return super().object_url(key)

    def head_object(self, obj: ObjectInfo) -> ObjectInfo:
        with self._open(self.object_url(obj.key), method="HEAD") as response:
            size = response.headers.get("Content-Length")
            if size is None or int(size) <= 0:
                raise BuildError(f"wheel has no positive Content-Length: {obj.key}")
            return ObjectInfo(obj.key, int(size), response.headers.get("Last-Modified", ""))


class WheelLinks(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.objects: list[ObjectInfo] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            url = urllib.parse.urljoin(self.base_url, href)
            if urllib.parse.urlsplit(url).scheme in ("https", "http") and wheel_info(url) is not None:
                self.objects.append(ObjectInfo(url, 0, ""))


def wheel_info(key: str) -> tuple[str, str, str] | None:
    path = urllib.parse.unquote(urllib.parse.urlsplit(key).path)
    filename = path.rsplit("/", 1)[-1]
    if not filename.startswith("triton_ascend-") or not filename.endswith(".whl"):
        return None
    parts = filename[:-4].rsplit("-", 3)
    if len(parts) != 4:
        return None
    version, python_tag, abi_tag, platforms = parts
    if python_tag != abi_tag or not re.fullmatch(r"cp3\d{1,2}", python_tag):
        return None
    parent = path.rsplit("/", 2)
    if len(parent) == 3 and re.fullmatch(r"\d{14}", parent[1]):
        build = parent[1]
    else:
        match = re.search(r"[.+]dev(\d{8}(?:\d{6})?)$", version)
        if match is None:
            return None
        build = match.group(1)
    return build, python_tag[2:], platforms


def list_obs_builds(client: DailyClient, date: str = "") -> list[str]:
    try:
        _, prefixes = client.list_objects(f"{OBS_PREFIX}{date}", delimiter="/")
    except BuildError as error:
        if isinstance(error.__cause__, urllib.error.HTTPError) and error.__cause__.code == 403:
            raise BuildError(
                "OBS denies anonymous directory listing (HTTP 403); cannot discover build timestamps. "
                "Use --wheel-url with a known wheel URL, or explicitly select the separate --source nightly."
            ) from error
        raise
    builds = [prefix.removeprefix(OBS_PREFIX).rstrip("/") for prefix in prefixes]
    return sorted((build for build in builds if re.fullmatch(r"\d{14}", build)), reverse=True)


def list_nightly_objects(client: DailyClient) -> list[ObjectInfo]:
    with client._open(NIGHTLY_INDEX) as response:
        parser = WheelLinks(response.geturl())
        parser.feed(response.read().decode("utf-8"))
    return list(dict.fromkeys(parser.objects))


def matching_build(candidate: str, build: str) -> bool:
    return build == "latest" or candidate == build or (len(build) == 8 and candidate.startswith(build))


def choose_wheel(
    objects: Iterable[ObjectInfo], build: str, python_tag: str, arch: str
) -> tuple[ObjectInfo, str] | None:
    matches: list[tuple[str, ObjectInfo]] = []
    for obj in objects:
        info = wheel_info(obj.key)
        if info is None:
            continue
        candidate, wheel_python, platforms = info
        if matching_build(candidate, build) and wheel_python == python_tag and all(
            tag.startswith("manylinux_") and tag.endswith(f"_{arch}") for tag in platforms.split(".")
        ):
            matches.append((candidate, obj))
    if not matches:
        return None
    newest = max(candidate for candidate, _ in matches)
    packages = [obj for candidate, obj in matches if candidate == newest]
    if len(packages) != 1:
        names = ", ".join(obj.key for obj in packages)
        raise BuildError(f"multiple matching wheels found for build {newest}: {names}")
    return packages[0], newest


def resolve_obs(client: DailyClient, build: str, python_tag: str, arch: str) -> tuple[ObjectInfo, str] | None:
    if len(build) == 14:
        builds = [build]
    else:
        builds = list_obs_builds(client, "" if build == "latest" else build)
    for candidate in builds:
        objects, _ = client.list_objects(f"{OBS_PREFIX}{candidate}/")
        objects = [obj for obj in objects if obj.size > 0]
        selection = choose_wheel(objects, candidate, python_tag, arch)
        if selection is not None:
            return selection
    return None


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover and download Triton-Ascend daily wheels.")
    parser.add_argument("--source", choices=("daily", "nightly"), default="daily")
    parser.add_argument(
        "--build",
        default="latest",
        help="daily date (YYYYMMDD, auto-select timestamp) or build ID (YYYYMMDDhhmmss), default: latest",
    )
    parser.add_argument(
        "--python-version",
        type=normalize_python_version,
        default=normalize_python_version(f"{sys.version_info.major}.{sys.version_info.minor}"),
        metavar="VERSION",
    )
    parser.add_argument("--arch", type=normalize_arch, default=normalize_arch(platform.machine()))
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--list-builds", type=int, metavar="N", help="list the newest N daily build IDs and exit")
    parser.add_argument("--resolve-only", action="store_true", help="print the selected URL without downloading")
    parser.add_argument("--wheel-url", help="use a known wheel URL without listing directories")
    parser.add_argument("--timeout", type=float, default=30.0, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.list_builds is None and args.build != "latest" and not BUILD_RE.fullmatch(args.build):
        parser.error("--build must be 'latest', YYYYMMDD, or YYYYMMDDhhmmss")
    if args.list_builds is not None and args.list_builds < 1:
        parser.error("--list-builds must be positive")
    if args.wheel_url and (args.list_builds is not None or args.source != "daily"):
        parser.error("--wheel-url cannot be combined with --list-builds or --source nightly")
    client = DailyClient(args.timeout)
    try:
        if args.wheel_url:
            parts = urllib.parse.urlsplit(args.wheel_url)
            if parts.scheme not in ("https", "http") or not parts.netloc:
                parser.error("--wheel-url must be an HTTP or HTTPS URL")
            selection = choose_wheel(
                [ObjectInfo(args.wheel_url, 0, "")], args.build, args.python_version, args.arch
            )
        elif args.source == "nightly":
            objects = list_nightly_objects(client)
            if args.list_builds is not None:
                builds = sorted({info[0] for obj in objects if (info := wheel_info(obj.key)) is not None}, reverse=True)
                print("\n".join(builds[:args.list_builds]))
                return 0
            selection = choose_wheel(objects, args.build, args.python_version, args.arch)
        else:
            if args.list_builds is not None:
                print("\n".join(list_obs_builds(client)[:args.list_builds]))
                return 0
            selection = resolve_obs(client, args.build, args.python_version, args.arch)
        if selection is None:
            version = f"{args.python_version[0]}.{args.python_version[1:]}"
            raise BuildError(f"no Python {version} {args.arch} wheel found for build {args.build!r}")
        package, build = selection
        if package.size == 0:
            package = client.head_object(package)
        print(f"Source: {'wheel URL' if args.wheel_url else args.source}")
        print(f"Build: {build}")
        print(f"Object: {package.key}")
        print(f"Size: {package.size}")
        print(f"Last modified: {package.last_modified}")
        print(f"URL: {client.object_url(package.key)}")
        if not args.resolve_only:
            filename = urllib.parse.unquote(urllib.parse.urlsplit(package.key).path).rsplit("/", 1)[-1]
            destination = args.output_dir / filename
            client.download(package, destination)
            print(f"Saved to: {destination.resolve()}")
        return 0
    except (BuildError, ET.ParseError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
