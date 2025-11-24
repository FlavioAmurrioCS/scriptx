from __future__ import annotations

import subprocess
import sys
from contextlib import suppress
from textwrap import dedent
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    import pathlib

    from scriptx.main import ScriptMetadata


def test__create_virtualenv_venv(tmp_path: pathlib.Path) -> None:
    metadata: ScriptMetadata = {
        "dependencies": ["certifi"],
        "requires-python": f">={sys.version_info.major}.{sys.version_info.minor}",
    }
    with mock.patch("scriptx.main.subprocess_run") as mock_run:
        mock_run.return_value = None
        from scriptx.main import _create_virtualenv_venv

        _create_virtualenv_venv(tmp_path.as_posix(), metadata)


def test__create_virtualenv_virtualenv(tmpdir: str) -> None:
    tmpdir = str(tmpdir)
    metadata: ScriptMetadata = {
        "dependencies": ["certifi"],
        "requires-python": f">={sys.version_info.major}.{sys.version_info.minor}",
    }
    with mock.patch("scriptx.main.subprocess_run") as mock_run:
        mock_run.return_value = None
        from scriptx.main import _create_virtualenv_virtualenv

        _create_virtualenv_virtualenv(tmpdir, metadata)


def test__create_virtualenv_uv(tmpdir: str) -> None:
    tmpdir = str(tmpdir)
    metadata: ScriptMetadata = {
        "dependencies": ["certifi"],
        "requires-python": f">={sys.version_info.major}.{sys.version_info.minor}",
    }
    with mock.patch("scriptx.main.subprocess_run") as mock_run:
        mock_run.return_value = None
        from scriptx.main import _create_virtualenv_uv

        _create_virtualenv_uv(tmpdir, metadata)


def test_main_help(capsys: pytest.CaptureFixture[str]) -> None:
    from scriptx.main import main

    print(type(capsys))
    with suppress(SystemExit):
        main(("--help",))
    stdout, _stderr = capsys.readouterr()
    assert "show this help message and exit" in stdout


def test_main_noargs(capsys: pytest.CaptureFixture[str]) -> None:
    from scriptx.main import main

    print(type(capsys))
    with suppress(SystemExit):
        main(())
    stdout, _stderr = capsys.readouterr()
    assert "show this help message and exit" in stdout


def test_subprocess_run_success() -> None:
    from scriptx.main import subprocess_run

    result = subprocess_run(("echo", "Hello, World!"), capture_output=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "Hello, World!"


def test_subprocess_run_failure() -> None:
    from scriptx.main import subprocess_run

    with pytest.raises(subprocess.CalledProcessError):
        subprocess_run(("cat", "DOESNOTEXIST"), capture_output=True, check=True)


def test_latest_python() -> None:
    mock_run: mock.MagicMock
    with mock.patch("scriptx.main.pythons") as mock_run:
        mock_run.return_value = {(3, 10): "/usr/bin/python3.10", (3, 9): "/usr/bin/python3.9"}
        from scriptx.main import latest_python

        path = latest_python()
        assert path == "/usr/bin/python3.10"
        mock_run.assert_called_once()
        mock_run.return_value = {}
        path = latest_python()
        assert path == sys.executable


SCRIPT_SIMPLE = """\
print("Hello, ScriptX!")
"""

SCRIPT_PEP723 = f"""\
#!/Users/flavio/opt/scriptx/installed_tools/sx/venv/bin/python
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "tomli >= 1.1.0 ; python_version < '3.11'",
# ]
# ///
{SCRIPT_SIMPLE}
"""
SCRIPT_PEP723_EMPTY = f"""\
#!/Users/flavio/opt/scriptx/installed_tools/sx/venv/bin/python
# /// script
# ///
{SCRIPT_SIMPLE}
"""


def test_extract_script_metadata() -> None:
    from scriptx.main import extract_script_metadata

    metadata = extract_script_metadata(SCRIPT_SIMPLE)
    assert metadata["dependencies"] == []
    assert metadata["requires-python"] == ">=3.12"
    metadata = extract_script_metadata(SCRIPT_PEP723)
    assert metadata["dependencies"] == ["tomli >= 1.1.0 ; python_version < '3.11'"]
    assert metadata["requires-python"] == ">=3.8"
    metadata = extract_script_metadata(SCRIPT_PEP723_EMPTY)
    assert metadata["dependencies"] == []
    assert metadata["requires-python"] == ">=3.12"


def test_extract_script_metadata_with_regex() -> None:
    from scriptx.main import extract_script_metadata_with_regex

    metadata = extract_script_metadata_with_regex("")
    assert metadata["dependencies"] == []
    assert metadata["requires-python"] == ">=3.12"
    content = dedent("""\
        requires-python = ">=3.8"
        dependencies = [
          "tomli >= 1.1.0 ; python_version < '3.11'",
        ]""")
    metadata = extract_script_metadata_with_regex(content)
    assert metadata["dependencies"] == ["tomli >= 1.1.0 ; python_version < '3.11'"]
    assert metadata["requires-python"] == ">=3.8"


def test_matching_python() -> None:
    mock_run: mock.MagicMock
    with mock.patch("scriptx.main.pythons") as mock_run:
        mock_run.return_value = {(3, 10): "/usr/bin/python3.10", (3, 9): "/usr/bin/python3.9"}
        from scriptx.main import matching_python

        path = matching_python(">=3.10")
        assert path == ["/usr/bin/python3.10"]


def test_quick_atomic_delete(tmp_path: pathlib.Path) -> None:
    from scriptx.main import quick_atomic_delete

    quick_atomic_delete(tmp_path.as_posix())
    assert not tmp_path.exists()


def test_registry_store(tmp_path: pathlib.Path) -> None:
    from scriptx.main import RegistryStore

    registry_store = RegistryStore(path=(tmp_path / "registries").as_posix())

    assert registry_store.path == (tmp_path / "registries").as_posix()


def test_inventory(tmp_path: pathlib.Path) -> None:
    registry_store_path = (tmp_path / "registries").as_posix()
    from scriptx.main import RegistryStore

    registry_store = RegistryStore(path=registry_store_path)
    inventory_path = (tmp_path / "installed_tools").as_posix()
    bin_path = (tmp_path / "bin").as_posix()
    from scriptx.main import Inventory

    inventory = Inventory(path=inventory_path, bin_path=bin_path, registry_store=registry_store)

    assert inventory.path == inventory_path


@pytest.mark.skip(reason="Just a template test function")
def test_typing(  # noqa: PLR0913
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
    capfdbinary: pytest.CaptureFixture[bytes],
    capsys: pytest.CaptureFixture[str],
    capsysbinary: pytest.CaptureFixture[bytes],
    capteesys: pytest.CaptureFixture[str],
    cache: pytest.Cache,
    monkeypatch: pytest.MonkeyPatch,
    pytestconfig: pytest.Config,
    recwarn: pytest.WarningsRecorder,
    tmp_path_factory: pytest.TempPathFactory,
    tmp_path: pathlib.Path,
    tmpdir_factory: pytest.TempdirFactory,
    tmpdir: pathlib.Path,  # LocalPath
) -> None: ...
