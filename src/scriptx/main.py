#!/Users/flavio/opt/scriptx/installed_tools/sx/venv/bin/python
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "tomli >= 1.1.0 ; python_version < '3.11'",
# ]
# ///
# flake8: noqa: E501
from __future__ import annotations

import argparse
import atexit
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
    from typing_extensions import NotRequired
    from typing_extensions import Protocol
    from typing_extensions import TypedDict

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


def func(s: str) -> Registry: ...


@cache
def pythons() -> Mapping[tuple[int, int], str]:
    """Return a mapping of available python versions to their executable paths."""
    ret: dict[tuple[int, int], str] = {}
    ret[(sys.version_info.major, sys.version_info.minor)] = sys.executable
    for major, minor in [(3, i) for i in range(7, 15)]:
        exe_name = f"python{major}.{minor}"
        exe_path = shutil.which(exe_name)
        if exe_path:
            ret[(major, minor)] = exe_path
    return ret


def latest_python() -> str:
    """Return the path to the latest available python version."""
    versions = pythons()
    if not versions:
        return sys.executable
    latest_version = max(versions.keys())
    return versions[latest_version]


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
    if not content.strip():
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


def matching_python(version_spec: str) -> list[str]:
    ops, _minor = version_spec.split("3.", maxsplit=1)
    version = (3, int(_minor.split(".", maxsplit=1)[0]))
    ops_func = {
        ">=": operator.ge,
        "<=": operator.le,
        ">": operator.gt,
        "<": operator.lt,
        "==": operator.eq,
        "~=": operator.eq,
    }[ops.strip()]
    ret = []
    for ver, u in pythons().items():
        if ops_func(ver, version):
            ret.append(u)
    return ret


def _create_virtualenv_uv(path: str, metadata: ScriptMetadata) -> str:
    uv_bin = shutil.which("uv")
    if not uv_bin:
        msg = "uv is not installed."
        raise RuntimeError(msg)
    requires_python = metadata["requires-python"]
    dependencies = metadata["dependencies"]
    env = {
        "UV_NATIVE_TLS": "true",
        "UV_NO_CONFIG": "true",
        "UV_WORKING_DIRECTORY": "/tmp",  # noqa: S108
        # "UV_PYTHON": requires_python,
    }
    uv_cmd_prefix = (
        uv_bin,
        "--quiet",
        # "--no-config",
        # "--native-tls",
        # "--working-directory=/tmp",
    )
    logger.debug("Creating virtualenv with uv using env: %s", env)
    cmd: tuple[str, ...] = (*uv_cmd_prefix, "venv", f"--python={requires_python}", path)
    subprocess_run(cmd, check=True, env=env)
    if dependencies:
        cmd = (
            *uv_cmd_prefix,
            "pip",
            "install",
            f"--prefix={path}",
            f"--python={requires_python}",
            *dependencies,
        )
        subprocess_run(cmd, check=True, env=env, capture_output=True)
    return path


def _create_virtualenv_virtualenv(path: str, metadata: ScriptMetadata) -> str:
    virtualenv_bin = shutil.which("virtualenv")
    if not virtualenv_bin:
        msg = "virtualenv is not installed."
        raise RuntimeError(msg)
    requires_python = metadata["requires-python"]
    dependencies = metadata["dependencies"]
    python_bin = next(iter(matching_python(requires_python)), latest_python())
    cmd: tuple[str, ...] = (virtualenv_bin, path, "--no-download", f"--python={python_bin}")
    subprocess_run(cmd, check=True, capture_output=True)
    if dependencies:
        cmd = (os.path.join(path, "bin", "pip"), "install", *dependencies)
        subprocess_run(cmd, check=True, capture_output=True)
    return path


def _create_virtualenv_venv(path: str, metadata: ScriptMetadata) -> str:
    requires_python = metadata["requires-python"]
    dependencies = metadata["dependencies"]
    python_bin = next(iter(matching_python(requires_python)), latest_python())
    cmd: tuple[str, ...] = (python_bin, "-m", "venv", path)
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


def create_virtualenv(script: str, path: str) -> str:
    with open(script) as f:
        content = f.read()
    metadata = extract_script_metadata(content)
    quick_atomic_delete(path)
    uv_bin = shutil.which("uv")
    if uv_bin:
        return _create_virtualenv_uv(path, metadata)
    virtualenv_bin = shutil.which("virtualenv")
    if virtualenv_bin:
        return _create_virtualenv_virtualenv(path, metadata)
    return _create_virtualenv_venv(path, metadata)


@dataclass
class RegistryStore(MutableMapping[str, "Registry"]):
    path: str
    _cache: dict[str, Registry] = field(default_factory=dict, repr=False, hash=False, init=False)

    def registry_add(self, location: str) -> None:
        self[location] = func(location)

    def registry_update_all(self) -> None:
        for k, v in self.items():
            url = v.get("url")
            if url:
                self[k] = func(k)

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

    def _resolve_location(self, name: str) -> str:
        return os.path.join(self.path, f"{name}.json")

    def __getitem__(self, key: str) -> Registry:
        if key not in self._cache:
            location = self._resolve_location(key)
            if not os.path.exists(location):
                raise KeyError(key)
            with open(location) as f:
                self._cache[key] = json.load(f)
        return self._cache[key]

    def __setitem__(self, key: str, value: Registry) -> None:
        self._cache[key] = value
        os.makedirs(self.path, exist_ok=True)
        with open(self._resolve_location(key), "w") as f:
            json.dump(value, f, indent=2)

    def __delitem__(self, key: str) -> None:
        self._cache.pop(key, None)
        location = self._resolve_location(key)
        if not os.path.exists(location):
            raise KeyError(key)
        os.remove(location)


@dataclass
class Inventory(Mapping[str, str]):
    path: str
    bin_path: str

    def _install_file(
        self, src: str, link: LinkMode, name: str | None = None
    ) -> tuple[str, str] | tuple[None, None]:
        name = name or os.path.splitext(os.path.basename(src))[0]
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

    def _install_url(
        self, url: str, name: str | None = None
    ) -> tuple[str, str] | tuple[None, None]:
        from urllib.parse import urlparse
        from urllib.request import urlretrieve

        parse_url = urlparse(url)
        name = name or os.path.splitext(os.path.basename(parse_url.path))[0]
        if name in self:
            print(f"Tool '{name}' is already installed.", file=sys.stderr)
            return None, None
        location, _http_message = urlretrieve(url)  # noqa: S310
        return self._install_file(location, link="copy", name=name)

    def _script_setup_virtualenv(self, script_location: str, path: str) -> str:
        virtualenv = create_virtualenv(script_location, path=path)
        with open(script_location) as f:
            content = f.readlines()
        if content and content[0].startswith("#!"):
            content.pop(0)
        content.insert(0, f"#!{virtualenv}/bin/python\n")
        with open(script_location, "w") as f:
            f.writelines(content)

        os.chmod(script_location, 0o700)
        return virtualenv

    def update_metadata(self, name: str, metadata: InstalledTool) -> None:
        metadata_file = os.path.join(self.path, name, "metadata.json")
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def install(
        self, src: str, name: str | None = None, link: LinkMode = "copy"
    ) -> InstalledTool | None:
        if os.path.isfile(src):
            logger.debug("Installing tool from file: %s", src)
            src = os.path.abspath(src)
            name, script_location = self._install_file(src, link=link, name=name)
        elif src.startswith(("http://", "https://")):
            logger.debug("Installing tool from URL: %s", src)
            name, script_location = self._install_url(src, name=name)
        else:
            logger.debug("Installing tool from registry: %s", src)
            if src not in REGISTRY_STORE:
                print(f"Tool '{src}' not found in any registry.", file=sys.stderr)
                return None
            registry = REGISTRY_STORE[src]
            tool_item = registry["tools"].get(src)
            if not tool_item:
                print(f"Tool '{src}' not found in registry '{src}'.", file=sys.stderr)
                return None
            tool_location = tool_item["location"]
            return self.install(tool_location, name=name, link=link)

        if name is None or script_location is None:
            return None

        virtualenv = self._script_setup_virtualenv(
            script_location, path=os.path.join(self.path, name, "venv")
        )

        bin_location = os.path.join(self.bin_path, name)
        os.makedirs(self.bin_path, exist_ok=True)
        os.symlink(script_location, bin_location)
        metadata: InstalledTool = {
            "source": src,
            "venv": virtualenv,
            "path": script_location,
        }
        self.update_metadata(name, metadata)
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
REGISTRY_STORE = RegistryStore(path=os.path.join(SCRIPTX_HOME, "registries"))
INVENTORY = Inventory(path=os.path.join(SCRIPTX_HOME, "installed_tools"), bin_path=SCRIPTX_BIN)


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
        INVENTORY._script_setup_virtualenv(metadata["path"], metadata["venv"])  # noqa: SLF001
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
            or os.getenv("EDITOR")
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

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.epilog = dedent("""\
        Example:
          %(prog)s https://example.com/registry.json --name myregistry
          %(prog)s gh:owner/repo
          %(prog)s gh:owner/repo/path/to/registry.json --name myregistry
        """)
        parser.add_argument("src", type=str, help="URL or file path of the registry to add.")
        parser.add_argument(
            "--name",
            type=str,
            default=None,
            help="Optional name for the registry. If not provided, it will be derived from the URL or file path.",
        )
        return parser

    def run(self) -> int:
        print(f"Executing {self}...")
        return 0


class RegistryRemoveCmd(NamedTuple):
    """Remove a specified registry."""

    name: str

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.epilog = dedent("""\
        Example:
          %(prog)s <PLACEHOLDER_EXAMPLE>
        """)
        parser.add_argument("name", type=str, help="Name of the registry to remove.")
        return parser

    def run(self) -> int:
        print(f"Executing {self}...")
        return 0


class RegistryListCmd(NamedTuple):
    """List all registries."""

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


class RegistryUpdateCmd(NamedTuple):
    """Update specified registries or all if none specified."""

    @classmethod
    def arg_parser(cls, parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser()
        parser.description = cls.__doc__ or "<PLACEHOLDER_DESCRIPTION>"
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.epilog = dedent("""\
        Example:
          %(prog)s registry1 registry2
        """)
        parser.add_argument(
            "repos",
            type=str,
            nargs="*",
            help="Specific registries to update. If none provided, all registries will be updated.",
        )
        return parser

    def run(self) -> int:
        print(f"Executing {self}...")
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
