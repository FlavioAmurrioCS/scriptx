from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def print_json(  # noqa: PLR0913
    json: str | None = None,
    *,
    data: Any = None,  # noqa: ANN401
    indent: None | int | str = 2,
    highlight: bool = True,
    skip_keys: bool = False,
    ensure_ascii: bool = False,
    check_circular: bool = True,
    allow_nan: bool = True,
    default: Callable[[Any], Any] | None = None,
    sort_keys: bool = False,
) -> None:
    try:
        from rich import print_json as pjson

        pjson(
            json=json,
            data=data,
            indent=indent,
            highlight=highlight,
            skip_keys=skip_keys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            default=default,
            sort_keys=sort_keys,
        )
    except Exception:  # noqa: BLE001
        import json as pson

        print(
            pson.dumps(
                data,
                allow_nan=allow_nan,
                check_circular=check_circular,
                default=default,
                ensure_ascii=ensure_ascii,
                indent=indent,
                skipkeys=skip_keys,
                sort_keys=sort_keys,
            )
        )
