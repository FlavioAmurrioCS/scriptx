from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from typing_extensions import Protocol
    from typing_extensions import TypedDict

    class ScriptXInstallerProtocol(Protocol):
        def install(self, src: str, name: str | None = None) -> None: ...

    class InstallUnit(TypedDict):
        src: str
        install_location: str
        venv: str

    SBOM = dict[str, InstallUnit]

SCRIPTX_HOME = os.path.expanduser("~/opt/scriptx")


@dataclass
class Registry:
    file: str = os.path.join(SCRIPTX_HOME, "registry.json")

    @cached_property
    def tools(self) -> SBOM:
        data: SBOM = {}
        import json
        from contextlib import suppress

        os.makedirs(os.path.dirname(self.file), exist_ok=True)

        with suppress(Exception), open(self.file) as f:
            data.update(json.load(f))
        import atexit

        atexit.register(self.save)
        return data

    def save(self) -> None:
        with open(self.file, "w") as f:
            import json

            json.dump(self.tools, f)

    def register(self, key: str, unit: InstallUnit) -> None:
        self.tools[key] = unit

    def get(self, key: str) -> tuple[str, InstallUnit | None]:
        if key in self.tools:
            return key, self.tools[key]
        for k, v in self.tools.items():
            if v["src"] == key:
                return k, self.tools[k]
        return key, None

    def unregister(self, key: str) -> InstallUnit | None:
        k, u = self.get(key)
        if u is not None:
            del self.tools[k]
        return u


class UvInstaller:
    bin_dir: str = os.path.join(SCRIPTX_HOME, "bin")

    def install(self, src: str, name: str | None = None) -> tuple[str, InstallUnit]:
        import stat
        import subprocess
        import tempfile
        import urllib.request

        original_src = src
        name = name or os.path.basename(src)
        with tempfile.TemporaryDirectory():
            if os.path.exists(src):
                original_src = os.path.abspath(src)
            elif src.startswith(("http://", "https://")):
                src, _m = urllib.request.urlretrieve(src)  # noqa: S310
            else:
                msg = f"{src} does not exist and is not a valid URL"
                print(msg)
                # raise FileNotFoundError(msg) from None
                raise SystemExit(1)

            from uv._find_uv import find_uv_bin

            cmd = (
                find_uv_bin(),
                "sync",
                "--native-tls",
                f"--script={src}",
            )

            result = subprocess.run(  # noqa: S603
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env={"VIRTUAL_ENV": ""},
            )
            if result.returncode != 0:
                print(result.stdout)
                print(result.stderr)
                msg = f"uv failed to install {src}"
                raise RuntimeError(msg)

            virtualenv = ""
            for line in result.stderr.splitlines():
                if " environment at: " in line:
                    virtualenv = line.split(" environment at: ", maxsplit=1)[-1]
            if not virtualenv:
                msg = f"Could not determine virtualenv location for {src}"
                raise RuntimeError(msg)

            with open(src) as f:
                file_content = f.readlines()
            if file_content[0].startswith("#!/"):
                file_content.pop(0)
            file_content = [f"#!{virtualenv}/bin/python\n", *file_content]
            filename = f"{self.bin_dir}/{name}"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.writelines(file_content)
            os.chmod(filename, os.stat(filename).st_mode | stat.S_IEXEC)
            print(f"{name} has been installed at: {filename}")
            # Check PATH includes bin_dir
            path_dirs = os.environ.get("PATH", "").split(os.pathsep)
            if self.bin_dir not in path_dirs:
                print(
                    f"Warning: {self.bin_dir} is not in your PATH. "
                    f"Please add it to run '{name}' from the command line."
                )
            return name, {
                "src": original_src,
                "install_location": filename,
                "venv": virtualenv,
            }

    def uninstall(self, unit: InstallUnit) -> None:
        from contextlib import suppress

        with suppress(Exception):
            os.remove(unit["install_location"])
        with suppress(Exception):
            import shutil

            shutil.rmtree(unit["venv"])


@dataclass
class ScriptX:
    registry: Registry = field(default_factory=Registry)
    installer: UvInstaller = field(default_factory=UvInstaller)

    def callback(self) -> None: ...

    def install(self, src: str, name: str | None = None) -> None:
        """Install a script from a local path or URL."""
        name, unit = self.installer.install(src, name)
        self.registry.register(name, unit)

    def uninstall(self, src: str) -> None:
        """Uninstall a script by name or source URL/path."""
        unit = self.registry.unregister(src)
        if unit is None:
            print(f"{src} is not installed")
            return
        self.installer.uninstall(unit)

    def list(self) -> None:
        """List all installed scripts."""
        from scriptx._utils import print_json

        print_json(data=self.registry.tools)

    # def update(self, src: str) -> None:
    #     print("Update executed")

    def run(self, ctx: typer.Context, script_name: str) -> None:
        """Run an installed script by name, passing any additional arguments."""
        _name, unit = self.registry.get(script_name)
        if unit is None:
            print(f"Script {script_name} is not installed.")
            raise typer.Exit(code=1)
        args = ctx.args
        os.execvp(unit["install_location"], [unit["install_location"], *args])  # noqa: S606

    def __call__(self, *args: object, **kwds: object) -> None:
        app = typer.Typer(add_completion=False)
        app.callback()(self.callback)
        app.command()(self.install)
        app.command()(self.uninstall)
        app.command()(self.list)
        # app.command()(self.update)
        app.command(
            add_help_option=False,
            no_args_is_help=True,
            context_settings={
                "allow_extra_args": True,
                "ignore_unknown_options": True,
            },
        )(self.run)
        return app(*args, **kwds)


main = ScriptX()

if __name__ == "__main__":
    main.install("poop.py")
