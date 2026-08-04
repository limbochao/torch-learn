#!/usr/bin/env python3
"""从华为云 OBS 查询并下载 torch_npu 分支对应的 PyTorch 构建包。

命令行配置：

* ``branch``：torch_npu 分支名，例如 ``master``、``v2.7.1``。无默认值；除使用
  ``--list-branches`` 外均为必填。
* ``--source``：构建来源，可选 ``daily``、``cache``，默认 ``daily``。``daily`` 保留带构建号的
  历史日构建；``cache`` 是按分支滚动更新的最新缓存。
* ``--build``：日构建号，可选 ``latest`` 或 ``YYYYMMDD.N`` 格式，默认 ``latest``。仅适用于
  ``--source daily``；``latest`` 会选择包含目标 Python 包的最新构建。
* ``--python-version``：目标 Python 版本，支持 ``3.11``、``311``、``python311`` 等写法，默认使用
  当前运行脚本的 Python 主次版本。
* ``--arch``：目标架构，通常为 ``x86_64`` 或 ``aarch64``，默认使用当前机器架构。``amd64``、``x64``
  会转换为 ``x86_64``，``arm64`` 会转换为 ``aarch64``；仅适用于 ``--source cache``。
* ``--output-dir``：下载目录，默认当前工作目录。
* ``--list-branches``：列出指定 source 下的可用分支后退出，默认关闭；使用时无需提供 ``branch``。
* ``--list-builds N``：列出分支最近 N 个日构建号后退出，N 必须为正整数，默认关闭；仅适用于
  ``--source daily``。
* ``--resolve-only``：只打印选中的对象信息和下载 URL，不下载文件，默认关闭。
* ``--timeout``：单次 OBS 请求超时秒数，接受浮点数，默认 ``30.0``。这是隐藏的调试配置。

HTTPS 请求默认不校验证书，行为等价于 ``curl -k``，用于兼容代理注入自签名证书的环境。
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import ssl
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ENDPOINT = "https://pytorch-package.obs.cn-north-4.myhuaweicloud.com"
BUILD_RE = re.compile(r"^(\d{8})\.(\d+)$")


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    last_modified: str


def normalize_python_version(value: str) -> str:
    match = re.fullmatch(r"(?:python)?(3)\.?([0-9]{1,2})", value.lower())
    if not match:
        raise argparse.ArgumentTypeError("expected a Python version such as 3.11, 311, or python311")
    return f"{match.group(1)}{match.group(2)}"


def normalize_arch(value: str) -> str:
    aliases = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}
    return aliases.get(value.lower(), value.lower())


def build_sort_key(build: str) -> tuple[int, int]:
    match = BUILD_RE.fullmatch(build)
    if not match:
        return (-1, -1)
    return int(match.group(1)), int(match.group(2))


class ObsClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self.endpoint = DEFAULT_ENDPOINT
        self.timeout = timeout
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    def _open(self, url: str):
        request = urllib.request.Request(url, headers={"User-Agent": "torch-npu-build-downloader/1.0"})
        try:
            return urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context)
        except urllib.error.HTTPError as error:
            raise BuildError(f"OBS request failed with HTTP {error.code}: {url}") from error
        except urllib.error.URLError as error:
            raise BuildError(f"OBS request failed: {error.reason}") from error

    def list_objects(self, prefix: str, delimiter: str | None = None) -> tuple[list[ObjectInfo], list[str]]:
        objects: list[ObjectInfo] = []
        prefixes: list[str] = []
        continuation_token: str | None = None

        while True:
            query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if delimiter is not None:
                query["delimiter"] = delimiter
            if continuation_token is not None:
                query["continuation-token"] = continuation_token
            url = f"{self.endpoint}/?{urllib.parse.urlencode(query)}"
            with self._open(url) as response:
                root = ET.fromstring(response.read())

            namespace = {"obs": root.tag.partition("}")[0].lstrip("{")}
            for item in root.findall("obs:Contents", namespace):
                key = item.findtext("obs:Key", namespaces=namespace)
                if key:
                    objects.append(
                        ObjectInfo(
                            key=key,
                            size=int(item.findtext("obs:Size", "0", namespace)),
                            last_modified=item.findtext("obs:LastModified", "", namespace),
                        )
                    )
            prefixes.extend(
                item.text
                for item in root.findall("obs:CommonPrefixes/obs:Prefix", namespace)
                if item.text
            )

            if root.findtext("obs:IsTruncated", "false", namespace).lower() != "true":
                break
            continuation_token = root.findtext("obs:NextContinuationToken", namespaces=namespace)
            if not continuation_token:
                raise BuildError("OBS returned a truncated listing without a continuation token")

        return objects, prefixes

    def object_url(self, key: str) -> str:
        return f"{self.endpoint}/{urllib.parse.quote(key, safe='/')}"

    def download(self, obj: ObjectInfo, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=destination.parent)
        downloaded = 0
        try:
            with os.fdopen(fd, "wb") as output, self._open(self.object_url(obj.key)) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    print(f"\rDownloading {destination.name}: {downloaded}/{obj.size} bytes", end="", file=sys.stderr)
            print(file=sys.stderr)
            if downloaded != obj.size:
                raise BuildError(f"size mismatch: expected {obj.size} bytes, downloaded {downloaded}")
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise


def extract_name(prefix: str, full_prefix: str) -> str:
    return full_prefix.removeprefix(prefix).rstrip("/")


def list_branches(client: ObsClient, source: str) -> list[str]:
    root = "pta/Daily/" if source == "daily" else "cache/"
    _, prefixes = client.list_objects(root, delimiter="/")
    return sorted(extract_name(root, prefix) for prefix in prefixes)


def list_builds(client: ObsClient, branch: str) -> list[str]:
    prefix = f"pta/Daily/{branch}/"
    _, prefixes = client.list_objects(prefix, delimiter="/")
    builds = [extract_name(prefix, item) for item in prefixes]
    return sorted((item for item in builds if BUILD_RE.fullmatch(item)), key=build_sort_key, reverse=True)


def choose_object(objects: Iterable[ObjectInfo], python_tag: str, arch: str | None = None) -> ObjectInfo | None:
    suffix = f"_py{python_tag}.tar.gz"
    cache_suffix = f"_{python_tag[0]}.{python_tag[1:]}_{arch}.tar.gz" if arch else None
    matches = [
        obj
        for obj in objects
        if obj.size > 0 and (obj.key.endswith(suffix) or obj.key.endswith(cache_suffix or "\0"))
    ]
    if len(matches) > 1:
        names = ", ".join(obj.key.rsplit("/", 1)[-1] for obj in matches)
        raise BuildError(f"multiple matching packages found: {names}")
    return matches[0] if matches else None


def resolve_daily(client: ObsClient, branch: str, build: str, python_tag: str) -> tuple[ObjectInfo, str]:
    if build != "latest" and not BUILD_RE.fullmatch(build):
        raise BuildError("--build must be 'latest' or have the form YYYYMMDD.N")
    builds = [build] if build != "latest" else list_builds(client, branch)
    if not builds:
        raise BuildError(f"no daily builds found for branch {branch!r}")

    for candidate in builds:
        prefix = f"pta/Daily/{branch}/{candidate}/"
        objects, _ = client.list_objects(prefix)
        package = choose_object(objects, python_tag)
        if package is not None:
            return package, candidate
    raise BuildError(f"no Python {python_tag[0]}.{python_tag[1:]} package found for branch {branch!r}")


def resolve_cache(client: ObsClient, branch: str, python_tag: str, arch: str) -> tuple[ObjectInfo, None]:
    objects, _ = client.list_objects(f"cache/{branch}/")
    package = choose_object(objects, python_tag, arch)
    if package is None:
        version = f"{python_tag[0]}.{python_tag[1:]}"
        raise BuildError(f"no cached Python {version} {arch} package found for branch {branch!r}")
    return package, None


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover and download public torch_npu branch builds from Huawei OBS."
    )
    parser.add_argument("branch", nargs="?", help="torch_npu branch, for example master or v2.7.1")
    parser.add_argument("--source", choices=("daily", "cache"), default="daily")
    parser.add_argument("--build", default="latest", help="daily build ID (YYYYMMDD.N), default: latest")
    parser.add_argument(
        "--python-version",
        type=normalize_python_version,
        default=normalize_python_version(f"{sys.version_info.major}.{sys.version_info.minor}"),
        metavar="VERSION",
    )
    parser.add_argument("--arch", type=normalize_arch, default=normalize_arch(platform.machine()))
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--list-branches", action="store_true", help="list branches and exit")
    parser.add_argument("--list-builds", type=int, metavar="N", help="list the newest N daily build IDs and exit")
    parser.add_argument("--resolve-only", action="store_true", help="print the selected URL without downloading")
    parser.add_argument("--timeout", type=float, default=30.0, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    client = ObsClient(args.timeout)
    try:
        if args.list_branches:
            print("\n".join(list_branches(client, args.source)))
            return 0
        if not args.branch:
            parser.error("branch is required unless --list-branches is used")
        if args.list_builds is not None:
            if args.source != "daily":
                parser.error("--list-builds is only valid with --source daily")
            if args.list_builds < 1:
                parser.error("--list-builds must be positive")
            print("\n".join(list_builds(client, args.branch)[: args.list_builds]))
            return 0

        if args.source == "daily":
            package, build = resolve_daily(client, args.branch, args.build, args.python_version)
        else:
            if args.build != "latest":
                parser.error("--build is only valid with --source daily")
            package, build = resolve_cache(client, args.branch, args.python_version, args.arch)

        url = client.object_url(package.key)
        print(f"Branch: {args.branch}")
        if build:
            print(f"Build: {build}")
        print(f"Object: {package.key}")
        print(f"Size: {package.size}")
        print(f"Last modified: {package.last_modified}")
        print(f"URL: {url}")
        if not args.resolve_only:
            destination = args.output_dir / package.key.rsplit("/", 1)[-1]
            client.download(package, destination)
            print(f"Saved to: {destination.resolve()}")
        return 0
    except (BuildError, ET.ParseError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
