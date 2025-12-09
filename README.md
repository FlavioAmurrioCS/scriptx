# scriptx

[![PyPI - Version](https://img.shields.io/pypi/v/scriptx.svg)](https://pypi.org/project/scriptx)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scriptx.svg)](https://pypi.org/project/scriptx)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/FlavioAmurrioCS/scriptx/main.svg)](https://results.pre-commit.ci/latest/github/FlavioAmurrioCS/scriptx/main)

A lightweight manager for PEP 723 Python scripts — install, run, update, and manage standalone scripts with isolated environments.

-----

## Table of Contents

- [scriptx](#scriptx)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Install a tool](#install-a-tool)
    - [Display installed scripts](#display-installed-scripts)
    - [Execute the script via the tool](#execute-the-script-via-the-tool)
    - [Execute script directly](#execute-script-directly)
  - [License](#license)

## Installation

```console
pipx install scriptx
uv tool install scriptx
```

## Usage

```bash
[flavio@mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ scriptx --help
usage: scriptx [-h] [-v] [-V] {install,reinstall,upgrade,uninstall,list,run,edit,registry-add,registry-remove,registry-list,registry-update} ...

ScriptX v2 - A tool management system.

positional arguments:
  {install,reinstall,upgrade,uninstall,list,run,edit,registry-add,registry-remove,registry-list,registry-update}
    install             Install a tool from a URL or file path.
    reinstall           Reinstall a previously installed tool.
    upgrade             Upgrade an installed tool to the latest version.
    uninstall           Uninstall a previously installed tool.
    list                List all installed tools.
    run                 Run a specified tool with optional arguments.
    edit                Open script in editor.
    registry-add        Add a new registry.
    registry-remove     Remove a specified registry.
    registry-list       List all registries.
    registry-update     Update specified registries or all if none specified.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Increase verbosity level.
  -V, --version         show program's version number and exit

Environment variables:
  SCRIPTX_HOME  - Directory where ScriptX stores its data (default: ~/opt/scriptx)
  SCRIPTX_BIN   - Directory where ScriptX stores executable tools (default: ~/opt/scriptx/bin)

Add the following to your shell profile to include ScriptX tools in your PATH:
  export PATH="${HOME}/opt/scriptx/bin:${PATH}"
```

### Install a tool

Tool will:
- Download the publicly avaiable url(in this case a github page url) or a file on disk.
- Create a virtualenv for the script based on the inline script metadata.
- It will copy the script into the ~/opt/scriptx/bin directory.
- Update the `#!` `shebang` of the script to allow for faster execution.
- NOTE: You can pass in a alternative name for the script, by default it will use the basename of the SRC.

```bash
[flavio@mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ scriptx install --help
usage: scriptx install [-h] [--name NAME] [--link {symlink,copy,hardlink}] url_or_path

Install a tool from a URL or file path.

positional arguments:
  url_or_path           URL or file path of the tool to install.

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Optional name for the tool. If not provided, it will be derived from the URL or file path.
  --link {symlink,copy,hardlink}
                        Method to link the tool in the inventory when is a local file (default: copy).

Example:
  scriptx install ./file.py
  scriptx install https://example.com/tools/mytool.py
  scriptx install https://example.com/tools/mytool.py --name toolname

[flavio@mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ scriptx install https://flavioamurriocs.github.io/uv-to-pipfile/src/uv_to_pipfile/uv_to_pipfile.py
Tool 'uv_to_pipfile' installed successfully in /Users/flavio/opt/scriptx/bin/uv_to_pipfile.
Warning: /Users/flavio/opt/scriptx/bin is not in your PATH. ie (export PATH="${HOME}/opt/scriptx/bin:${PATH}")

[flavio@mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ head -n2 /Users/flavio/opt/scriptx/bin/uv_to_pipfile
#!/Users/flavio/opt/scriptx/installed_tools/uv_to_pipfile/venv/bin/python
# /// script
```

### Display installed scripts
```json
[flavio@mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ scriptx list
{
  "uv_to_pipfile": {
    "source": "https://flavioamurriocs.github.io/uv-to-pipfile/src/uv_to_pipfile/uv_to_pipfile.py",
    "venv": "/Users/flavio/opt/scriptx/installed_tools/uv_to_pipfile/venv",
    "path": "/Users/flavio/opt/scriptx/installed_tools/uv_to_pipfile/script.py"
  }
}
```

### Execute the script via the tool
```bash
[flavio@mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ scriptx run uv_to_pipfile --help
usage: script.py [-h] [--uv-lock UV_LOCK] [--pipfile-lock PIPFILE_LOCK]

Convert uv.lock to Pipfile.lock

optional arguments:
  -h, --help            show this help message and exit
  --uv-lock UV_LOCK     Path to the uv.lock file (default: ./uv.lock)
  --pipfile-lock PIPFILE_LOCK
                        Path to the Pipfile.lock file (default: ./Pipfile.lock)
```

### Execute script directly
NOTE: You must update your `PATH`
```bash
[flavio@Mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ export PATH="${HOME}/opt/scriptx/bin:${PATH}"

[flavio@Mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ uv_to_pipfile --help
usage: uv_to_pipfile [-h] [--uv-lock UV_LOCK] [--pipfile-lock PIPFILE_LOCK]

Convert uv.lock to Pipfile.lock

options:
  -h, --help            show this help message and exit
  --uv-lock UV_LOCK     Path to the uv.lock file (default: ./uv.lock)
  --pipfile-lock PIPFILE_LOCK
                        Path to the Pipfile.lock file (default: ./Pipfile.lock)
```


## License

`scriptx` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
