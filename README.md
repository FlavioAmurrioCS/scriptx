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
[flavio@Mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ scriptx --help

 Usage: scriptx [OPTIONS] COMMAND [ARGS]...

╭─ Options ───────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                     │
╰─────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────╮
│ install     Install a script from a local path or URL.                          │
│ uninstall   Uninstall a script by name or source URL/path.                      │
│ list        List all installed scripts.                                         │
│ run         Run an installed script by name, passing any additional arguments.  │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

### Install a tool

Tool will:
- Download the publicly avaiable url(in this case a github page url) or a file on disk.
- Create a virtualenv for the script based on the inline script metadata.
- It will copy the script into the ~/opt/scriptx/bin directory.
- Update the `#!` `shebang` of the script to allow for faster execution.
- NOTE: You can pass in a alternative name for the script, by default it will use the basename of the SRC.

```bash
[flavio@Mac ~/dev/github.com/FlavioAmurrioCS/scriptx][init ✗]
$ scriptx install --help

 Usage: scriptx install [OPTIONS] SRC

 Install a script from a local path or URL.

╭─ Arguments ──────────────────────────────────────╮
│ *    src      TEXT  [required]                   │
╰──────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────╮
│ --name        TEXT                               │
│ --help              Show this message and exit.  │
╰──────────────────────────────────────────────────╯

[flavio@Mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ scriptx install https://flavioamurriocs.github.io/uv-to-pipfile/src/uv_to_pipfile/uv_to_pipfile.py
uv_to_pipfile.py has been installed at: /Users/flavio/opt/scriptx/bin/uv_to_pipfile.py
Warning: /Users/flavio/opt/scriptx/bin is not in your PATH. Please add it to run 'uv_to_pipfile.py' from the command line.

[flavio@Mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ head -n2 /Users/flavio/opt/scriptx/bin/uv_to_pipfile.py
#!/Users/flavio/.cache/uv/environments-v2/tmplgn78zuu-f65b2dd2c079bd68/bin/python
# /// script
```

### Display installed scripts
```bash
[flavio@Mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ scriptx list
{
  "uv_to_pipfile.py": {
    "src": "https://flavioamurriocs.github.io/uv-to-pipfile/src/uv_to_pipfile/uv_to_pipfile.py",
    "install_location": "/Users/flavio/opt/scriptx/bin/uv_to_pipfile.py",
    "venv": "/Users/flavio/.cache/uv/environments-v2/tmplgn78zuu-f65b2dd2c079bd68"
  }
}

```

### Execute the script via the tool
```bash
[flavio@Mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ scriptx run uv_to_pipfile.py --help
usage: uv_to_pipfile.py [-h] [--uv-lock UV_LOCK] [--pipfile-lock PIPFILE_LOCK]

Convert uv.lock to Pipfile.lock

options:
  -h, --help            show this help message and exit
  --uv-lock UV_LOCK     Path to the uv.lock file (default: ./uv.lock)
  --pipfile-lock PIPFILE_LOCK
                        Path to the Pipfile.lock file (default: ./Pipfile.lock)
```

### Execute script directly
NOTE: You must update your `PATH`
```bash
[flavio@Mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ export PATH="/Users/flavio/opt/scriptx/bin:${PATH}"

[flavio@Mac ~/dev/github.com/FlavioAmurrioCS/scriptx]
$ uv_to_pipfile.py --help
usage: uv_to_pipfile.py [-h] [--uv-lock UV_LOCK] [--pipfile-lock PIPFILE_LOCK]

Convert uv.lock to Pipfile.lock

options:
  -h, --help            show this help message and exit
  --uv-lock UV_LOCK     Path to the uv.lock file (default: ./uv.lock)
  --pipfile-lock PIPFILE_LOCK
                        Path to the Pipfile.lock file (default: ./Pipfile.lock)
```


## License

`scriptx` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
