"""Local pre-flight wheel validation matching what PyPI runs on upload.

The PyPI/Warehouse upload endpoint runs ``validate_record`` and
``validate_entrypoints`` against every wheel. If either fails:

* ``validate_record`` ⇒ ``send_wheel_record_mismatch_email`` (today a
  warning; PyPI announced upload **rejection** after 2026-02-01).
* ``validate_entrypoints`` ⇒ upload is rejected.

This script runs the *exact* upstream functions (verbatim vendor in
``_vendored_warehouse_wheel.py``) so what passes here is what PyPI
will accept. It is invoked as a step of the release workflow after
the SBOM is injected, catching regressions before the wheel reaches
PyPI.

Usage::

    python scripts/validate_wheel.py path/to/one.whl ...
    python scripts/validate_wheel.py dist/

Exit status is 0 if every wheel passes, 1 otherwise; per-file
failures are reported on stderr.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Iterable

from _vendored_warehouse_wheel import (
    InvalidWheelEntryPointsError,
    InvalidWheelRecordError,
    MissingWheelRecordError,
    validate_entrypoints,
    validate_record,
)


WHEEL_GLOB = "*.whl"


class InvalidWheelFilenameError(ValueError):
    """Raised when a wheel's basename is not ``name-version-...`` shaped."""


def validate_wheel_file(path: Path) -> None:
    """Run PyPI's upload-time checks against the wheel at ``path``.

    :raises InvalidWheelFilenameError: If the basename lacks the two
        ``-`` separators the upstream parser unpacks.
    :raises MissingWheelRecordError: If the wheel has no RECORD or it
        is unreadable / malformed CSV / non-UTF-8.
    :raises InvalidWheelRecordError: If the RECORD entries do not
        match the wheel's non-directory ZIP entries.
    :raises InvalidWheelEntryPointsError: If ``entry_points.txt`` is
        present and malformed or has invalid entry-point names.
    """
    if path.name.count("-") < 2:
        raise InvalidWheelFilenameError(path.name)
    s = str(path)
    validate_record(s)
    validate_entrypoints(s)


def _expand_inputs(inputs: Iterable[str | os.PathLike[str]]) -> list[Path]:
    """Expand directories to ``*.whl`` files; pass files through.

    Directories are not recursed — ``python -m build`` deposits the
    wheel and sdist into a single flat ``dist/`` directory.
    """
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob(WHEEL_GLOB)))
        else:
            files.append(p)
    return files


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on any failure."""
    parser = argparse.ArgumentParser(
        prog="validate_wheel.py",
        description=(
            "Run PyPI's upload-time wheel checks (validate_record and "
            "validate_entrypoints) locally. Accepts wheel files and/or "
            "directories containing *.whl."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Wheel file paths or directories to validate.",
    )
    args = parser.parse_args(argv)

    files = _expand_inputs(args.inputs)
    if not files:
        print("validate_wheel.py: no wheel files found", file=sys.stderr)
        return 1

    failed = 0
    for path in files:
        try:
            validate_wheel_file(path)
        except MissingWheelRecordError:
            print(f"FAIL {path}: RECORD is missing or unreadable", file=sys.stderr)
            failed += 1
        except InvalidWheelRecordError as exc:
            print(f"FAIL {path}: RECORD mismatch: {exc}", file=sys.stderr)
            failed += 1
        except InvalidWheelEntryPointsError as exc:
            print(f"FAIL {path}: entry_points.txt invalid: {exc}", file=sys.stderr)
            failed += 1
        except InvalidWheelFilenameError as exc:
            print(f"FAIL {path}: malformed wheel filename: {exc}", file=sys.stderr)
            failed += 1
        else:
            print(f"OK   {path}")

    if failed:
        print(
            f"validate_wheel.py: {failed} / {len(files)} wheel(s) failed validation",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
