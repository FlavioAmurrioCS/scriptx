#!/Users/flavio/opt/scriptx/installed_tools/sx/venv/bin/python
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "packaging",
#   "tomli >= 1.1.0 ; python_version < '3.11'",
# ]
# ///
# flake8: noqa: E501
from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import logging
import operator
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from functools import cache
from textwrap import dedent
from typing import TYPE_CHECKING
from typing import Literal
from typing import NamedTuple

if TYPE_CHECKING:
    from http.client import HTTPResponse
    from typing import Any

    from typing_extensions import NotRequired
    from typing_extensions import Protocol
    from typing_extensions import TypedDict
    from typing_extensions import Unpack

    _Params = dict[str, Any] | tuple[tuple[str, Any], ...] | list[tuple[str, Any]] | None

    HTTP_METHOD = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE"]

    class _CompleteRequestArgs(TypedDict):
        data: NotRequired[Mapping[str, Any] | None]
        verify: NotRequired[bool | str]
        headers: NotRequired[Mapping[str, Any] | None]
        json: NotRequired[Any | None]
        params: NotRequired[_Params]
        timeout: NotRequired[float | None]

    class InstalledTool(TypedDict):
        source: str
        venv: str
        path: str

    class RegistryItem(TypedDict):
        location: str
        description: NotRequired[str]
        name: NotRequired[str]

    class Registry(TypedDict):
        url: NotRequired[str]
        tools: dict[str, RegistryItem]

    class Cmd(Protocol):
        @classmethod
        def arg_parser(
            cls, parser: argparse.ArgumentParser | None = None
        ) -> argparse.ArgumentParser: ...
        def run(self) -> int: ...

    ScriptMetadata = TypedDict(
        "ScriptMetadata", {"requires-python": str, "dependencies": list[str]}
    )

    LinkMode = Literal["symlink", "copy", "hardlink"]

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def time_it(what: str) -> Generator[None, None, None]:
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        t1 = time.perf_counter_ns()
        logger.debug("%s took %.2f milliseconds.", what, (t1 - t0) / 1e6)


def http_request(  # noqa: C901, PLR0912
    url: str, *, method: HTTP_METHOD = "GET", **kwargs: Unpack[_CompleteRequestArgs]
) -> HTTPResponse:
    import urllib.parse
    from collections.abc import Mapping

    final_url = url
    params = kwargs.get("params")
    if params:
        parts = urllib.parse.urlsplit(url)
        base_pairs = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)

        if isinstance(params, Mapping):
            extra_pairs: list[tuple[str, str]] = []
            for k, v in params.items():
                if isinstance(v, (list, tuple)):
                    extra_pairs.extend((k, "" if item is None else str(item)) for item in v)
                else:
                    extra_pairs.append((k, "" if v is None else str(v)))
        else:
            extra_pairs = [(k, "" if v is None else str(v)) for k, v in params]

        new_query = urllib.parse.urlencode(
            base_pairs + extra_pairs, doseq=True, encoding="utf-8", errors="strict"
        )
        final_url = urllib.parse.urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path,
                new_query,
                parts.fragment,
            )
        )

    http_method = method.upper()
    import urllib.request

    headers = {k.title(): v for k, v in (kwargs.get("headers") or {}).items()}

    if kwargs.get("data") and kwargs.get("json"):
        msg = "Cannot set both 'data' and 'json'"
        raise ValueError(msg)

    data = kwargs.get("data")

    json_content = kwargs.get("json")
    if json_content is not None:
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        data = json.dumps(json_content).encode("utf-8")  # type: ignore[assignment]

    verify = kwargs.get("verify")
    verify = verify or False  # NOTE: Default to False if not provided
    import ssl

    context: ssl.SSLContext | None

    if verify is None:
        context = None
    elif isinstance(verify, (str, os.PathLike)):
        verify_str = str(verify)
        if os.path.isdir(verify_str):
            context = ssl.create_default_context(capath=verify_str)
        else:
            context = ssl.create_default_context(cafile=verify_str)
    else:
        context = ssl.create_default_context()
        if not verify:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(  # noqa: S310
        url=final_url,
        data=data,  # type: ignore[arg-type]
        headers=headers,
        # origin_req_host=None,
        # unverifiable=self.unverifiable,
        method=http_method,
    )
    logger.debug("Making HTTP %s request to %s", http_method, final_url)
    response: HTTPResponse = urllib.request.urlopen(  # noqa: S310
        url=req,
        timeout=kwargs.get("timeout"),
        # cafile=None, # Deprecated
        # capath=None, # Deprecated
        # cadefault=False, # Deprecated
        context=context,
    )

    return response


CONTENT_DISPOTION_PATTERN = re.compile(r'filename\*?=(?:UTF-8\'\')?\"?([^"]+)\"?')


@contextlib.contextmanager
def fetch_temp_file(url: str) -> Generator[str]:
    """Download a file from a URL to a temporary location."""
    from urllib.parse import urlparse

    response = http_request(url)
    filename: str = next(
        iter(CONTENT_DISPOTION_PATTERN.findall(response.headers["content-disposition"] or "")), None
    ) or os.path.basename(urlparse(url).path)
    with tempfile.TemporaryDirectory(suffix=filename) as tmp_dir:
        full_path = os.path.join(tmp_dir, filename)
        with open(full_path, "wb") as f:
            f.write(response.read())
        yield full_path


def subprocess_run(
    cmd: tuple[str, ...],
    *,
    check: bool = False,
    capture_output: bool = False,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    pretty_cmd = " ".join(shlex.quote(part) for part in cmd)
    logger.debug("Running command: %s", pretty_cmd)
    result = subprocess.run(  # noqa: S603
        cmd, check=False, capture_output=capture_output, text=True, env=env, cwd=cwd
    )
    if check and result.returncode != 0:
        logger.error("Command '%s' failed with return code %d.", pretty_cmd, result.returncode)
        if capture_output:
            logger.error("Stdout: %s", result.stdout)
            logger.error("Stderr: %s", result.stderr)
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


@cache
@time_it("Loading available python versions")
def pythons() -> Mapping[tuple[int, int, int], str]:
    """Return a mapping of available python versions to their executable paths."""
    ret: dict[tuple[int, int, int], str] = {}
    ret[(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)] = sys.executable
    # with contextlib.suppress(subprocess.CalledProcessError):
    #     cmd = ("uv", "python", "list", "--only-installed", "--output-format=json")
    #     result = subprocess_run(cmd, check=True, capture_output=True)
    #     if TYPE_CHECKING:

    #         class UVPythonElement(TypedDict):
    #             key: str
    #             version: str
    #             version_parts: dict[Literal["major", "minor", "patch"], int]
    #             path: str

    #     data: list[UVPythonElement] = json.loads(result.stdout)
    #     for item in data:
    #         version_parts = item["version_parts"]
    #         ret[(version_parts["major"], version_parts["minor"], version_parts["patch"])] = item[
    #             "path"
    #         ]
    for major, minor in [(3, i) for i in range(7, 15)]:
        exe_name = f"python{major}.{minor}"
        exe_path = shutil.which(exe_name)
        if exe_path:
            ret[(major, minor, 0)] = exe_path
    return dict(sorted(ret.items()))


def latest_python() -> str:
    """Return the path to the latest available python version."""
    for version, exe in sorted(pythons().items(), reverse=True):
        # NOTE: Due to not all libraries being compatible with Python 3.14 yet
        if version < (3, 14):
            return exe
    return sys.executable


def extract_script_metadata(content: str) -> ScriptMetadata:
    regex_pattern = r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
    captured_content = ""
    for match in re.finditer(regex_pattern, content):
        if match.group("type") == "script":
            captured_content = "".join(
                line[2:] if line.startswith("# ") else line[1:]
                for line in match.group("content").splitlines(keepends=True)
            )
            break
    if not captured_content.strip():
        return {"requires-python": ">=3.12", "dependencies": []}
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
        data: ScriptMetadata = tomllib.loads(captured_content)  # type: ignore[assignment,unused-ignore]
        logger.debug("Parsed script metadata using tomli/tomllib: %s", data)
        return data  # noqa: TRY300
    except ImportError:
        logger.debug("tomli or tomllib not available, falling back to regex parsing.")

    return extract_script_metadata_with_regex(captured_content)


def extract_script_metadata_with_regex(captured_content: str) -> ScriptMetadata:
    if "depedencies" not in captured_content and "requires-python" not in captured_content:
        return {"requires-python": ">=3.12", "dependencies": []}
    requires = re.search(
        r"^requires-python\s*=\s*(['\"]+)(?P<value>.+)(\1)$",
        captured_content,
        re.MULTILINE,
    )
    python_version = requires["value"] if requires else ">=3.12"

    dependendency_block = re.search(
        r"^dependencies\s*=\s*\[(?P<value>.+?)\]",
        captured_content,
        re.DOTALL | re.MULTILINE,
    )
    dependencies: list[str] = []
    if dependendency_block:
        deps_content = dependendency_block["value"]
        dependencies = [
            line.strip().rstrip(",")[1:-1]
            for line in deps_content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    return {"requires-python": python_version, "dependencies": dependencies}


def version_spec_to_predicate(version_spec: str) -> Callable[[tuple[int, int, int]], bool]:
    try:
        from packaging.specifiers import SpecifierSet

        logger.debug("Using packaging module for version spec parsing.")

        spec_set = SpecifierSet(version_spec)
        return lambda ver: ".".join(map(str, ver)) in spec_set
    except ImportError:
        logger.debug("packaging module not available, falling back to custom version spec parsing.")
    # NOTE: Consider using https://pypi.org/project/packaging/
    # ~=: Compatible release clause
    # ==: Version matching clause
    # !=: Version exclusion clause
    # <=, >=: Inclusive ordered comparison clause
    # <, >: Exclusive ordered comparison clause
    # ===: Arbitrary equality clause.
    # Write a regex to capture the operator and version
    ops_func: dict[str, Callable[..., bool]] = {
        "~=": operator.eq,
        "==": operator.eq,
        "!=": operator.ne,
        "<=": operator.le,
        ">=": operator.ge,
        "<": operator.lt,
        ">": operator.gt,
        "===": operator.eq,  # Treat === as ==
    }
    version_spec_regex = re.compile(r"^(?P<op>>=|<=|>|<|==|~=|!=|===)?\s*(?P<version>\d+(\.\d+)*)$")
    predicates: list[Callable[[tuple[int, int, int]], bool]] = []
    for _part in version_spec.split(","):
        part = _part.strip()
        if not part:
            continue
        cap = version_spec_regex.match(part)
        if not cap:
            msg = f"Invalid version spec: {part}"
            raise ValueError(msg)
        op: str = cap.group("op")
        version: str = cap.group("version")

        def pred(ver: tuple[int, int, int], op: str = op, version: str = version) -> bool:
            return ops_func[op](ver, tuple(map(int, version.split("."))))

        predicates.append(pred)
    if not predicates:
        msg = f"Invalid version spec: {version_spec}"
        raise ValueError(msg)

    def predicate(ver: tuple[int, int, int]) -> bool:
        return all(p(ver) for p in predicates)

    return predicate


def matching_python(version_spec: str) -> list[tuple[int, int, int]]:
    pred = version_spec_to_predicate(version_spec)
    return [k for k, _v in pythons().items() if pred(k)]


def _create_virtualenv_uv(path: str, dependencies: list[str], python_executable: str) -> str:
    uv_bin = shutil.which("uv")
    if not uv_bin:
        msg = "uv is not installed."
        raise RuntimeError(msg)
    uv_cmd_prefix = (
        uv_bin,
        "--quiet",
        # "--no-config",
        # "--native-tls",
        # "--working-directory=/tmp",
    )
    cmd: tuple[str, ...] = (*uv_cmd_prefix, "venv", f"--python={python_executable}", path)
    subprocess_run(cmd, check=True)
    if dependencies:
        cmd = (
            *uv_cmd_prefix,
            "pip",
            "install",
            f"--python={python_executable}",
            f"--prefix={path}",
            *dependencies,
        )
        subprocess_run(cmd, check=True, capture_output=True)
    return path


def _create_virtualenv_virtualenv(
    path: str, dependencies: list[str], python_executable: str
) -> str:
    virtualenv_bin = shutil.which("virtualenv")
    if not virtualenv_bin:
        msg = "virtualenv is not installed."
        raise RuntimeError(msg)
    cmd: tuple[str, ...] = (virtualenv_bin, path, "--no-download", f"--python={python_executable}")
    subprocess_run(cmd, check=True, capture_output=True)
    if dependencies:
        cmd = (os.path.join(path, "bin", "pip"), "install", *dependencies)
        subprocess_run(cmd, check=True, capture_output=True)
    return path


def _create_virtualenv_venv(path: str, dependencies: list[str], python_executable: str) -> str:
    cmd: tuple[str, ...] = (python_executable, "-m", "venv", path)
    subprocess_run(cmd, check=True, capture_output=True)
    if dependencies:
        cmd = (os.path.join(path, "bin", "pip"), "install", *dependencies)
        subprocess_run(cmd, check=True, capture_output=True)
    return path


def quick_atomic_delete(path: str) -> None:
    parent_dir = os.path.dirname(path)
    if not os.path.isdir(parent_dir):
        return
    tmp_dir = tempfile.TemporaryDirectory(dir=parent_dir)
    atexit.register(tmp_dir.__exit__, None, None, None)
    tmp_dir.__enter__()
    logger.debug("Moving %s to temporary location %s for deletion.", path, tmp_dir.name)
    if not os.path.exists(path):
        return
    os.replace(path, os.path.join(tmp_dir.name, "to_delete"))


def create_virtualenv(path: str, dependencies: list[str], python_executable: str) -> str:
    uv_bin = shutil.which("uv")
    if uv_bin:
        return _create_virtualenv_uv(
            path=path, dependencies=dependencies, python_executable=python_executable
        )
    virtualenv_bin = shutil.which("virtualenv")
    if virtualenv_bin:
        return _create_virtualenv_virtualenv(
            path=path, dependencies=dependencies, python_executable=python_executable
        )
    return _create_virtualenv_venv(
        path=path, dependencies=dependencies, python_executable=python_executable
    )


@dataclass
class RegistryStore(MutableMapping[str, "Registry"]):
    path: str
    _cache: dict[str, Registry] = field(default_factory=dict, repr=False, hash=False, init=False)

    def all_tools(self) -> Iterator[tuple[str, str, RegistryItem]]:
        """
        Yields tuples of (tool_registry_name, tool_name, item) for all tools in all registries.
        """
        yield from (
            (tool_registry_name, tool_name, item)
            for tool_registry_name, tool_registry in self.items()
            for tool_name, item in tool_registry["tools"].items()
        )

    def _items(self) -> tuple[str, ...]:
        if not os.path.exists(self.path):
            return ()
        return tuple(x.rstrip(".json") for x in os.listdir(self.path) if x.endswith(".json"))

    def __iter__(self) -> Iterator[str]:
        return iter(self._items())

    def __len__(self) -> int:
        return len(self._items())

    def resolve_location(self, name: str) -> str:
        return os.path.join(self.path, f"{name}.json")

    def __getitem__(self, key: str) -> Registry:
        if key not in self._cache:
            location = self.resolve_location(key)
            if not os.path.exists(location):
                raise KeyError(key)
            with open(location) as f:
                self._cache[key] = json.load(f)
        return self._cache[key]

    def __setitem__(self, key: str, value: Registry) -> None:
        self._cache[key] = value
        os.makedirs(self.path, exist_ok=True)
        with open(self.resolve_location(key), "w") as f:
            json.dump(value, f, indent=2)

    def __delitem__(self, key: str) -> None:
        self._cache.pop(key, None)
        location = self.resolve_location(key)
        if not os.path.exists(location):
            raise KeyError(key)
        os.remove(location)


def link_file(src: str, dest: str, link: LinkMode = "copy") -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if link == "symlink":
        os.symlink(src, dest)
    elif link == "hardlink":
        os.link(src, dest)
    else:  # copy
        shutil.copy2(src, dest)


@dataclass
class Inventory(Mapping[str, str]):
    path: str
    bin_path: str
    registry_store: RegistryStore

    def _install_file(
        self, src: str, link: LinkMode, name: str | None = None
    ) -> tuple[str, str] | tuple[None, None]:
        name = name or os.path.splitext(os.path.basename(src))[0].replace("_", "-")
        script_location = self._resolve_script_path(name)
        if name in self:
            print(f"Tool '{name}' is already installed.", file=sys.stderr)
            return None, None
        os.makedirs(os.path.dirname(script_location), exist_ok=True)
        if link == "symlink":
            os.symlink(src, script_location)
        elif link == "hardlink":
            os.link(src, script_location)
        else:  # copy
            shutil.copy2(src, script_location)
        return name, script_location

    def get_script(self, src: str, name: str | None, link: LinkMode) -> tuple[str, str, LinkMode]:
        if os.path.isfile(src):
            return src, name or os.path.splitext(os.path.basename(src))[0], link
        link = "copy"
        if not src.startswith(("http://", "https://")):
            # Handle registry installation
            logger.debug("Installing tool from registry: %s", src)
            t = next((t for t in self.registry_store.all_tools() if t[1] == src), None)
            if t is None:
                print(f"Tool '{src}' not found in any registry.", file=sys.stderr)
                raise SystemExit(1)
            _registry_name, tool_name, item = t
            name = name or tool_name
            src = item["location"]
        if not src.startswith(("http://", "https://")):
            msg = "Only URL installation is supported currently."
            raise ValueError(msg)
        with tempfile.NamedTemporaryFile(delete=False) as tmf:
            pass
        with fetch_temp_file(src) as location:
            link_file(location, tmf.name, link="copy")
            name = name or os.path.splitext(os.path.basename(location))[0]
            return tmf.name, name, link

    def update_metadata(self, name: str, metadata: InstalledTool) -> None:
        metadata_file = os.path.join(self.path, name, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def install(
        self, src: str, name: str | None = None, link: LinkMode = "copy"
    ) -> InstalledTool | None:
        original_src = src
        # Create a working directory to ensure all steps complete before installing
        # If not a file, download the file, hardcode the linkmode to "copy"
        # Create virtualenv from the script
        if name and name in self:
            print(f"Tool '{name}' is already installed.", file=sys.stderr)
            return None
        with tempfile.TemporaryDirectory() as tmpdir:
            location, name, link = self.get_script(src=src, name=name, link=link)
            name = name or os.path.splitext(os.path.basename(src))[0].replace("_", "-")
            final_install_location = os.path.join(self.path, name)
            if name in self:
                print(f"Tool '{name}' is already installed.", file=sys.stderr)
                return None
            install_script_path = os.path.join(tmpdir, "script.py")
            link_file(location, install_script_path, link=link)

            with open(install_script_path) as f:
                content = f.read()

            script_metadata = extract_script_metadata(content)
            python_executable = pythons()[matching_python(script_metadata["requires-python"])[0]]
            logger.debug("Using python executable %s for tool '%s'.", python_executable, name)
            # # Ensure it is a valid python script
            # with time_it("Validating script syntax"):
            #     cmd = (python_executable, "-m", "ast", install_script_path)
            #     _result = subprocess_run(cmd, check=True, capture_output=False)
            dependencies = script_metadata.get("dependencies", [])

            _venv_path = create_virtualenv(
                path=os.path.join(tmpdir, "venv"),
                dependencies=dependencies,
                python_executable=python_executable,
            )
            content_lines = content.splitlines(keepends=True)
            if content_lines and content_lines[0].startswith("#!"):
                content_lines.pop(0)
            content_lines.insert(0, f"#!{final_install_location}/venv/bin/python\n")
            with open(install_script_path, "w") as f:
                f.writelines(content_lines)
            os.chmod(install_script_path, 0o700)
            os.replace(tmpdir, final_install_location)

        metadata: InstalledTool = {
            "source": original_src,
            "venv": os.path.join(final_install_location, "venv"),
            "path": os.path.join(final_install_location, "script.py"),
        }
        self.update_metadata(name, metadata)

        bin_location = os.path.join(self.bin_path, name)
        os.makedirs(self.bin_path, exist_ok=True)
        os.symlink(metadata["path"], bin_location)
        return metadata

    def uninstall(self, name: str) -> None:
        script_path = self.get(name)
        if not script_path:
            print(f"Tool '{name}' is not installed.", file=sys.stderr)
            return
        metadata = self.get_metadata(name)

        # Remove symlink in bin path
        bin_path = os.path.join(self.bin_path, name)
        if os.path.exists(bin_path):
            os.unlink(bin_path)

        # Remove script directory, virtualenv is colocated
        tool_dir = os.path.dirname(script_path)
        quick_atomic_delete(tool_dir)

        if metadata:
            venv_path = metadata["venv"]
            # Shouldnt be needed as venv is inside tool_dir, but just in case
            quick_atomic_delete(venv_path)

    def list_scripts(self) -> list[str]:
        if not os.path.exists(self.path):
            return []
        return [name for name in os.listdir(self.path) if name in self]

    def get_metadata(self, name: str) -> InstalledTool | None:
        metadata_file = os.path.join(self.path, name, "metadata.json")
        if not os.path.exists(metadata_file):
            return None
        with open(metadata_file) as f:
            return json.load(f)

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_scripts())

    def __len__(self) -> int:
        return len(self.list_scripts())

    def _resolve_script_path(self, key: str) -> str:
        return os.path.join(self.path, key, "script.py")

    def __getitem__(self, key: str) -> str:
        if key not in self:
            raise KeyError(key)
        return self._resolve_script_path(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and os.path.exists(self._resolve_script_path(key))


_SCRIPTX_HOME_DEFAULT = "~/opt/scriptx"
_SCRIPTX_BIN_DEFAULT = os.path.join(_SCRIPTX_HOME_DEFAULT, "bin")

SCRIPTX_HOME = os.path.expanduser(os.getenv("SCRIPTX_HOME") or _SCRIPTX_HOME_DEFAULT)
SCRIPTX_BIN = os.path.expanduser(os.getenv("SCRIPTX_BIN") or _SCRIPTX_BIN_DEFAULT)
_REGISTRY_STORE_PATH = os.path.join(SCRIPTX_HOME, "registries")
REGISTRY_STORE = RegistryStore(path=_REGISTRY_STORE_PATH)
_INVENTORY_PATH = os.path.join(SCRIPTX_HOME, "installed_tools")
INVENTORY = Inventory(
    path=_INVENTORY_PATH,
    bin_path=SCRIPTX_BIN,
    registry_store=REGISTRY_STORE,
)


################################################################################
# region: Commands
################################################################################
VERSION = "0.1.0"


class InstallCmd(NamedTuple):
    """Install a tool from a URL or file path."""

    url_or_path: str
    name: str | None
    link: Literal["symlink", "copy", "hardlink"]

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.epilog = dedent("""\
        Example:
          %(prog)s https://example.com/tools/mytool.py
          %(prog)s ./file.py
          %(prog)s gh:owner/repo/path/to/tool.py
          %(prog)s https://example.com/tools/mytool.py --name toolname
          %(prog)s https://example.com/tools/mytool.py --python 3.11
        """)
        parser.add_argument(
            "url_or_path", type=str, help="URL or file path of the tool to install."
        )
        parser.add_argument(
            "--name",
            type=str,
            default=None,
            help="Optional name for the tool. If not provided, it will be derived from the URL or file path.",
        )
        parser.add_argument(
            "--link",
            type=str,
            default="copy",
            choices=["symlink", "copy", "hardlink"],
            help="Method to link the tool in the inventory when is a local file (default: copy).",
        )
        return parser

    def run(self) -> int:
        location = INVENTORY.install(self.url_or_path, name=self.name, link=self.link)
        if location is None:
            return 1
        print(f"Tool installed at: {location['path']}")
        return 0


class ReInstallCmd(NamedTuple):
    """Reinstall a previously installed tool."""

    tool: str

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.epilog = dedent("""\
        Example:
          %(prog)s mytool
          %(prog)s mytool --python 3.11
          %(prog)s mytool --python 3.11 --name newtoolname
        """)
        parser.add_argument("tool", type=str, help="Name of the tool to reinstall.")
        return parser

    def run(self) -> int:
        metadata = INVENTORY.get_metadata(self.tool)
        if metadata is None:
            print(f"Could not retrieve metadata for tool '{self.tool}'.")
            return 1
        with open(metadata["path"]) as f:
            content = f.read()

        script_metadata = extract_script_metadata(content)
        python_executable = pythons()[matching_python(script_metadata["requires-python"])[0]]
        dependencies = script_metadata.get("dependencies", [])
        quick_atomic_delete(metadata["venv"])

        _venv_path = create_virtualenv(
            path=metadata["venv"],
            dependencies=dependencies,
            python_executable=python_executable,
        )
        print(f"Tool '{self.tool}' has been reinstalled.")
        return 0


class UpgradeCmd(NamedTuple):
    """Upgrade an installed tool to the latest version."""

    tool: str

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.epilog = dedent("""\
        Example:
          %(prog)s mytool
        """)
        parser.add_argument("tool", type=str, help="Name of the tool to upgrade.")
        return parser

    def run(self) -> int:
        metadata = INVENTORY.get_metadata(self.tool)
        if metadata is None:
            print(f"Could not retrieve metadata for tool '{self.tool}'.")
            return 1
        if os.path.islink(metadata["path"]) or os.stat(metadata["path"]).st_nlink > 1:
            return ReInstallCmd(tool=self.tool).run()

        source = metadata["source"]

        UninstallCmd(tool=self.tool).run()
        ret = INVENTORY.install(source, name=self.tool, link="copy")
        if ret is None:
            print(f"Failed to upgrade tool '{self.tool}'.")
            return 1
        print(f"Tool '{self.tool}' has been upgraded.")
        return 0


class UninstallCmd(NamedTuple):
    """Uninstall a previously installed tool."""

    tool: str

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.epilog = dedent("""\
        Example:
          %(prog)s mytool
        """)
        parser.add_argument("tool", type=str, help="Name of the tool to uninstall.")
        return parser

    def run(self) -> int:
        if self.tool not in INVENTORY:
            print(f"Tool '{self.tool}' is not installed.")
            return 1
        INVENTORY.uninstall(self.tool)
        print(f"Tool '{self.tool}' has been uninstalled.")
        return 0


# class CleanupCmd(NamedTuple):
#     """<Brief description of the command>."""

#     @classmethod
#     def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
#         parser = parser or argparse.ArgumentParser()
#         parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
#         parser.formatter_class = argparse.RawTextHelpFormatter
#         parser.epilog = dedent("""\
#         Example:
#           %(prog)s <PLACEHOLDER_EXAMPLE>
#         """)
#         return parser

#     def run(self) -> int:
#         print(f"Executing {self}...")
#         return 0


class ListCmd(NamedTuple):
    """List all installed tools."""

    all: bool = False

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.epilog = dedent("""\
        Example:
          %(prog)s <PLACEHOLDER_EXAMPLE>
        """)
        parser.add_argument(
            "--all",
            action="store_true",
            help="List all tools, including those not currently installed.",
        )
        return parser

    def run(self) -> int:
        for line in INVENTORY.list_scripts():
            print(f" - {line}")
        return 0


# class SearchCmd(NamedTuple):
#     """<Brief description of the command>."""

#     @classmethod
#     def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
#         parser = parser or argparse.ArgumentParser()
#         parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
#         parser.formatter_class = argparse.RawTextHelpFormatter
#         parser.epilog = dedent("""\
#         Example:
#           %(prog)s <PLACEHOLDER_EXAMPLE>
#         """)
#         return parser

#     def run(self) -> int:
#         print(f"Executing {self}...")
#         return 0


# class ShowCmd(NamedTuple):
#     """<Brief description of the command>."""

#     @classmethod
#     def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
#         parser = parser or argparse.ArgumentParser()
#         parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
#         parser.formatter_class = argparse.RawTextHelpFormatter
#         parser.epilog = dedent("""\
#         Example:
#           %(prog)s <PLACEHOLDER_EXAMPLE>
#         """)
#         return parser

#     def run(self) -> int:
#         print(f"Executing {self}...")
#         return 0


class RunCmd(NamedTuple):
    """Run a specified tool with optional arguments."""

    tool_name: str
    args: list[str]

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.epilog = dedent("""\
        Example:
          %(prog)s mytool --arg1 value1 --arg2 value2
        """)
        parser.add_argument("tool_name", type=str, help="Name of the tool to run.")
        parser.add_argument(
            "args",
            type=str,
            nargs=argparse.REMAINDER,
            help="Arguments to pass to the tool.",
        )
        return parser

    def run(self) -> int:
        tool = INVENTORY.get(self.tool_name)
        if tool is None:
            print(f"Tool '{self.tool_name}' is not installed.")
            return 1
        cmd = (tool, *self.args)
        os.execvp(cmd[0], cmd)  # noqa: S606
        return 0


class EditCmd(NamedTuple):
    """Open script in editor."""

    tool_name: str
    editor: str | None

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.epilog = dedent("""\
        Example:
          %(prog)s mytool --arg1 value1 --arg2 value2
        """)
        parser.add_argument("tool_name", type=str, help="Name of the tool to run.")
        parser.add_argument(
            "--editor",
            type=str,
            help="Editor command to open the tool.",
        )
        return parser

    def run(self) -> int:
        metadata = INVENTORY.get_metadata(self.tool_name)
        if metadata is None:
            print(f"Tool '{self.tool_name}' is not installed.")
            return 1

        editor = (
            self.editor
            or shutil.which(os.getenv("EDITOR") or "UNKNOWN")
            or shutil.which("nvim")
            or shutil.which("code")
            or shutil.which("vim")
            or shutil.which("vi")
            or shutil.which("nano")
            or "vi"
        )
        cmd = (editor, metadata["path"])
        os.execvp(cmd[0], cmd)  # noqa: S606
        return 0


class RegistryAddCmd(NamedTuple):
    """Add a new registry."""

    src: str
    name: str | None
    link: LinkMode

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        # parser.epilog = dedent("""\
        # Example:
        #   %(prog)s https://example.com/registry.json --name myregistry
        #   %(prog)s gh:owner/repo
        #   %(prog)s gh:owner/repo/path/to/registry.json --name myregistry
        # """)
        parser.add_argument("src", type=str, help="URL or file path of the registry to add.")
        parser.add_argument(
            "name",
            type=str,
            default=None,
            nargs="?",
            help="Optional name for the registry. If not provided, it will be derived from the URL or file path.",
        )
        parser.add_argument(
            "--link",
            type=str,
            default="symlink",
            choices=["symlink", "copy", "hardlink"],
            help="Method to link the registry when is a local file (default: %(default)s).",
        )
        return parser

    def run(self) -> int:
        link = self.link
        with tempfile.TemporaryDirectory() as tmpdir:
            if not os.path.isfile(self.src):
                if not self.src.startswith(("http://", "https://")):
                    msg = "Only URL installation is supported currently."
                    raise ValueError(msg)
                with fetch_temp_file(self.src) as location:
                    name = self.name or os.path.splitext(os.path.basename(self.src))[0]
                    if name in REGISTRY_STORE:
                        print(f"Registry '{name}' already exists.", file=sys.stderr)
                        return 1
                    with open(location) as f:
                        try:
                            data: Registry = json.load(f)
                        except json.JSONDecodeError as e:
                            print(f"Failed to load registry from {self.src}: {e}", file=sys.stderr)
                            return 1
                    if "url" not in data:
                        data["url"] = self.src
                    with open(os.path.join(tmpdir, f"{name}.json"), "w") as f:
                        json.dump(data, f, indent=2)
                    # link_file(location, os.path.join(tmpdir, f"{name}.json"), link="copy")
            else:
                src = os.path.abspath(self.src)
                name = self.name or os.path.splitext(os.path.basename(src))[0]
                if name in REGISTRY_STORE:
                    print(f"Registry '{name}' already exists.", file=sys.stderr)
                    return 1
                link_file(src, os.path.join(tmpdir, f"{name}.json"), link)

            os.makedirs(REGISTRY_STORE.path, exist_ok=True)
            os.replace(
                os.path.join(tmpdir, f"{name}.json"),
                os.path.join(REGISTRY_STORE.path, f"{name}.json"),
            )
        return 0


class RegistryRemoveCmd(NamedTuple):
    """Remove a specified registry."""

    name: str

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        # parser.epilog = dedent("""\
        # Example:
        #   %(prog)s <PLACEHOLDER_EXAMPLE>
        # """)
        parser.add_argument("name", type=str, help="Name of the registry to remove.")
        return parser

    def run(self) -> int:
        del REGISTRY_STORE[self.name]
        return 0


class RegistryListCmd(NamedTuple):
    """List all registries."""

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        # parser.epilog = dedent("""\
        # Example:
        #   %(prog)s <PLACEHOLDER_EXAMPLE>
        # """)
        return parser

    def run(self) -> int:
        if not REGISTRY_STORE:
            print("No registries available.")
            return 0
        print("Available registries:")
        for k, v in REGISTRY_STORE.items():
            print(f" - {k}: {v.get('url', '<no url>')}")
        return 0


class RegistryUpdateCmd(NamedTuple):
    """Update specified registries or all if none specified."""

    repos: list[str]

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        # parser.epilog = dedent("""\
        # Example:
        #   %(prog)s registry1 registry2
        # """)
        parser.add_argument(
            "repos",
            type=str,
            nargs="*",
            help="Specific registries to update. If none provided, all registries will be updated.",
        )
        return parser

    def run(self) -> int:
        names = self.repos or list(REGISTRY_STORE.keys())
        for name in names:
            location = REGISTRY_STORE.resolve_location(name)
            if os.path.islink(location) or os.stat(location).st_nlink > 1:
                continue
            registry = REGISTRY_STORE[name]
            if "url" not in registry:
                print(f"Registry '{name}' does not have a URL to update from.", file=sys.stderr)
                continue
            url = registry["url"]
            RegistryRemoveCmd(name=name).run()
            RegistryAddCmd(src=url, name=name, link="copy").run()
        return 0


class SampleCmd(NamedTuple):
    """<Brief description of the command>."""

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.epilog = dedent("""\
        Example:
          %(prog)s <PLACEHOLDER_EXAMPLE>
        """)
        return parser

    def run(self) -> int:
        print(f"Executing {self}...")
        return 0


################################################################################
# endregion: Commands
################################################################################

SUB_COMMANDS: dict[str, type[Cmd]] = {
    "install": InstallCmd,
    "reinstall": ReInstallCmd,
    "upgrade": UpgradeCmd,
    "uninstall": UninstallCmd,
    # "cleanup": CleanupCmd,
    "list": ListCmd,
    # "search": SearchCmd,
    # "show": ShowCmd,
    "run": RunCmd,
    "edit": EditCmd,
    "registry-add": RegistryAddCmd,
    "registry-remove": RegistryRemoveCmd,
    "registry-list": RegistryListCmd,
    "registry-update": RegistryUpdateCmd,
}


def main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    # argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="scriptx")
    parser.description = "ScriptX v2 - A tool management system."
    parser.formatter_class = argparse.RawTextHelpFormatter
    # For more information, visit https://github.com/FlavioAmurrioCS/scriptx
    # Example usage:
    #   scriptx install https://example.com/tools/mytool.py
    #   scriptx reinstall mytool
    #   scriptx uninstall mytool
    #   scriptx list
    #   scriptx run mytool --arg1 value1 --arg2 value2
    #   scriptx upgrade mytool
    #   scriptx registry add https://example.com/registry.json --name myregistry
    #   scriptx registry remove myregistry
    #   scriptx registry list
    #   scriptx registry update
    parser.epilog = dedent(f"""\
    Environment variables:
      SCRIPTX_HOME  - Directory where ScriptX stores its data (default: {_SCRIPTX_HOME_DEFAULT})
      SCRIPTX_BIN   - Directory where ScriptX stores executable tools (default: {_SCRIPTX_BIN_DEFAULT})

    Add the following to your shell profile to include ScriptX tools in your PATH:
      export PATH="{_SCRIPTX_BIN_DEFAULT.replace("~", "${HOME}")}:${{PATH}}"
    """)

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity level."
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command")
    for cmd_name, cmd in SUB_COMMANDS.items():
        cmd_parser = subparsers.add_parser(cmd_name, help=cmd.__doc__ or None)
        cmd.arg_parser(cmd_parser)
    args = parser.parse_args(argv)
    command: str | None = args.command
    verbose: int = args.verbose
    if command is None:
        parser.print_help()
        return 1
    logging.basicConfig(
        level=logging.WARNING - (verbose * 10),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cls = SUB_COMMANDS[command]
    exclude_args = ("command", "verbose")
    try:
        cmd_instance = cls(
            **{k: v for k, v in vars(args).items() if k not in exclude_args}
        )  # pyrefly: ignore[bad-instantiation]
    except TypeError as e:
        print(
            f"BUG!!!!!: {cls.__name__} received arguments from the parser that do not match its expected attributes: {e}",
            file=sys.stderr,
        )
        return 1
    return cmd_instance.run()


if __name__ == "__main__":
    raise SystemExit(main())
