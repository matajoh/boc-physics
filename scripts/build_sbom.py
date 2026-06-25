"""Generate and inject a PEP 770 SBOM into a bocphysics wheel.

This script has two responsibilities:

1. **Generate** a CycloneDX 1.6 JSON document describing the
   bocphysics distribution being built. bocphysics is pure Python and
   bundles no third-party code in its wheel, so the SBOM describes
   exactly one piece of software: bocphysics itself. The runtime
   dependencies (bocpy, pyglet, webcolors) are *not* bundled in the
   archive; they are declared in the wheel's Core Metadata
   ``Requires-Dist`` and resolved at install time, so they do not
   belong in this PEP 770 archive SBOM.

2. **Inject** the generated SBOM into a built wheel under
   ``<dist>-<version>.dist-info/sboms/bocphysics.cdx.json`` per
   PEP 770 and regenerate ``RECORD``. The injection step uses only
   the Python standard library so it can run inside the release
   workflow without adding a build-time dependency.

Run ``python scripts/build_sbom.py --help`` for the available
subcommands.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as _dt
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
import uuid
import zipfile

SBOM_GENERATOR_VERSION = "0.1.0"
SBOM_FILENAME = "bocphysics.cdx.json"
PEP770_SBOM_SUBDIR = "sboms"

_SBOM_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://github.com/matajoh/boc-physics/sboms"
)


def _sbom_timestamp() -> str:
    """Return the SBOM ``metadata.timestamp`` as an ISO 8601 UTC string.

    Honours the freedesktop reproducible-build convention: if
    ``SOURCE_DATE_EPOCH`` is set in the environment to a parseable
    integer number of seconds since the Unix epoch, that value is used
    verbatim; otherwise we fall back to the current UTC time.

    Setting ``SOURCE_DATE_EPOCH`` to the commit timestamp yields
    byte-identical SBOMs across rebuilds, keeping wheel hashes stable.
    """
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    if raw is not None:
        try:
            epoch = int(raw)
        except ValueError:
            raise ValueError(
                f"SOURCE_DATE_EPOCH must be an integer, got {raw!r}"
            ) from None
        try:
            dt = _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc)
        except (OverflowError, OSError, ValueError):
            raise ValueError(
                f"SOURCE_DATE_EPOCH is out of range, got {epoch!r}"
            ) from None
    else:
        dt = _dt.datetime.now(_dt.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _sbom_serial_number(
    name: str,
    version: str,
    git_commit: str | None,
    wheel_filename: str | None,
) -> str:
    """Derive a deterministic ``urn:uuid:<UUIDv5>`` serial number.

    The serial number is a UUIDv5 (SHA-1 based) computed under
    :data:`_SBOM_NAMESPACE` from the canonical input tuple
    ``name@version+git_commit+wheel_filename``. Missing optional
    fields are encoded as empty strings so the input shape is fixed
    and the resulting UUID is byte-stable.

    Same inputs => same UUID, on every machine and every rebuild.
    """
    payload = (
        f"{name}@{version}+{git_commit or ''}+{wheel_filename or ''}"
    )
    return f"urn:uuid:{uuid.uuid5(_SBOM_NAMESPACE, payload)}"


def _wheel_purl(name: str, version: str) -> str:
    """Build a Package URL for the bocphysics wheel itself.

    PyPI purls do not encode the wheel tag; consumers wanting the
    exact wheel filename should use the ``cdx:python:wheel_filename``
    property attached to the root component.
    """
    return f"pkg:pypi/{name}@{version}"


def build_sbom_document(
    name: str,
    version: str,
    description: str,
    license_id: str,
    homepage_url: str,
    vcs_url: str,
    git_commit: str | None,
    wheel_filename: str | None,
) -> dict[str, Any]:
    """Construct the CycloneDX 1.6 JSON document for a bocphysics wheel.

    :param name: PEP 503-normalized distribution name (``"bocphysics"``).
    :param version: Distribution version (e.g. ``"0.6.0"``).
    :param description: One-line project description.
    :param license_id: SPDX license identifier (e.g. ``"MIT"``).
    :param homepage_url: Project homepage URL.
    :param vcs_url: VCS / repository URL.
    :param git_commit: Optional git commit SHA the wheel was built
        from. Stored as a property on the root component when set.
    :param wheel_filename: Optional wheel filename (basename) the
        SBOM is being embedded in. Stored as a property on the root
        component when set.
    :return: A CycloneDX 1.6 document as a ``dict`` ready to be
        serialised with ``json.dumps(..., indent=2, sort_keys=True)``.
    """
    bom_ref = _wheel_purl(name, version)

    root_component: dict[str, Any] = {
        "bom-ref": bom_ref,
        "type": "library",
        "name": name,
        "version": version,
        "purl": bom_ref,
        "description": description,
        "licenses": [{"license": {"id": license_id}}],
        "supplier": {
            "name": "Matthew Johnson",
            "url": [homepage_url],
        },
        "externalReferences": [
            {"type": "website", "url": homepage_url},
            {"type": "vcs", "url": vcs_url},
        ],
    }

    properties: list[dict[str, str]] = []
    if git_commit:
        properties.append(
            {"name": "cdx:python:git_commit", "value": git_commit}
        )
    if wheel_filename:
        properties.append(
            {"name": "cdx:python:wheel_filename", "value": wheel_filename}
        )
    if properties:
        root_component["properties"] = properties

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": _sbom_serial_number(
            name, version, git_commit, wheel_filename
        ),
        "version": 1,
        "metadata": {
            "timestamp": _sbom_timestamp(),
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "build_sbom.py",
                        "version": SBOM_GENERATOR_VERSION,
                        "vendor": "Matthew Johnson",
                    }
                ]
            },
            "component": root_component,
        },
        "components": [],
        "dependencies": [{"ref": bom_ref, "dependsOn": []}],
    }


def _record_row(arcname: str, data: bytes) -> tuple[str, str, str]:
    """Build a ``RECORD`` row for one entry in the wheel zip.

    :param arcname: Archive name of the entry inside the wheel.
    :param data: Raw bytes of the entry.
    :return: A ``(path, hash_spec, size)`` tuple matching the wheel
        ``RECORD`` CSV format described in PEP 427.
    """
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return arcname, f"sha256={b64}", str(len(data))


def _find_dist_info_dir(wheel: zipfile.ZipFile) -> str:
    """Return the ``<dist>-<version>.dist-info/`` arcname prefix.

    Every wheel contains exactly one ``*.dist-info/`` directory.
    Raises ``ValueError`` if the wheel is malformed.
    """
    prefixes: set[str] = set()
    for name in wheel.namelist():
        head = name.split("/", 1)[0]
        if head.endswith(".dist-info"):
            prefixes.add(head)
    if len(prefixes) != 1:
        raise ValueError(
            f"wheel contains {len(prefixes)} .dist-info directories: {sorted(prefixes)!r}"
        )
    return prefixes.pop()


def inject_sbom_into_wheel(
    wheel_path: Path,
    sbom_bytes: bytes,
) -> None:
    """Insert ``sbom_bytes`` into ``wheel_path`` and rewrite ``RECORD``.

    The wheel is replaced atomically: a new ``.whl`` is built in a
    sibling temporary file and renamed over the original only after
    being fully flushed.

    :param wheel_path: Path to the existing wheel to mutate.
    :param sbom_bytes: Serialised CycloneDX JSON document.
    :raises FileNotFoundError: If ``wheel_path`` does not exist.
    :raises ValueError: If the wheel is missing or has multiple
        ``.dist-info`` directories.
    """
    if not wheel_path.is_file():
        raise FileNotFoundError(wheel_path)

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=wheel_path.stem + ".",
        suffix=".whl.tmp",
        dir=wheel_path.parent,
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        with zipfile.ZipFile(wheel_path, "r") as src:
            dist_info = _find_dist_info_dir(src)
            sbom_arcname = f"{dist_info}/{PEP770_SBOM_SUBDIR}/{SBOM_FILENAME}"
            record_arcname = f"{dist_info}/RECORD"

            entries: list[tuple[zipfile.ZipInfo, bytes]] = []
            sbom_already_present = False
            for info in src.infolist():
                if info.filename == record_arcname:
                    continue
                if info.filename == sbom_arcname:
                    sbom_already_present = True
                    continue
                if info.is_dir():
                    continue
                with src.open(info) as f:
                    data = f.read()
                new_info = zipfile.ZipInfo(
                    filename=info.filename, date_time=info.date_time
                )
                new_info.compress_type = info.compress_type
                new_info.external_attr = info.external_attr
                new_info.create_system = info.create_system
                new_info.internal_attr = info.internal_attr
                new_info.extra = info.extra
                new_info.comment = info.comment
                entries.append((new_info, data))

            sbom_info = zipfile.ZipInfo(filename=sbom_arcname)
            sbom_info.compress_type = zipfile.ZIP_DEFLATED
            entries.append((sbom_info, sbom_bytes))

            record_buf = io.StringIO()
            writer = csv.writer(
                record_buf, delimiter=",", quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
            )
            for entry_info, data in entries:
                writer.writerow(_record_row(entry_info.filename, data))
            writer.writerow((record_arcname, "", ""))
            record_info = zipfile.ZipInfo(filename=record_arcname)
            record_info.compress_type = zipfile.ZIP_DEFLATED

            with zipfile.ZipFile(
                tmp_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as dst:
                for entry_info, data in entries:
                    dst.writestr(entry_info, data)
                dst.writestr(record_info, record_buf.getvalue())

        if sbom_already_present:
            print(
                f"build_sbom.py: wheel {wheel_path.name!r} already contained "
                f"{sbom_arcname!r}; replacing with freshly generated SBOM",
                file=sys.stderr,
            )

        shutil.move(str(tmp_path), str(wheel_path))
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _read_pyproject_metadata(repo_root: Path) -> dict[str, str]:
    """Extract bocphysics's distribution metadata from ``pyproject.toml``.

    Only the fields the SBOM needs are returned. Uses the stdlib
    ``tomllib`` (Python 3.11+); on 3.10 ``tomli`` is used as a
    fallback.
    """
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[import-not-found]

    with (repo_root / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)

    project = data["project"]
    urls = project.get("urls", {})
    return {
        "name": project["name"],
        "version": project["version"],
        "description": project["description"],
        "license": project["license"],
        "homepage": urls.get("Homepage", ""),
        "vcs": urls.get("Repository", ""),
    }


def _cmd_generate(args: argparse.Namespace) -> int:
    """Implement the ``generate`` subcommand."""
    if not args.git_commit and not args.wheel_filename:
        print(
            "error: build_sbom.py generate requires at least one of: "
            "--git-commit (or $GITHUB_SHA), --wheel-filename so the "
            "deterministic serialNumber distinguishes this build from "
            "other wheels of the same name+version.",
            file=sys.stderr,
        )
        return 2
    meta = _read_pyproject_metadata(Path(args.project_root))
    doc = build_sbom_document(
        name=meta["name"],
        version=meta["version"],
        description=meta["description"],
        license_id=meta["license"],
        homepage_url=meta["homepage"],
        vcs_url=meta["vcs"],
        git_commit=args.git_commit,
        wheel_filename=args.wheel_filename,
    )
    serialised = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        sys.stdout.write(serialised)
    else:
        Path(args.output).write_text(serialised, encoding="utf-8")
    return 0


def _cmd_inject(args: argparse.Namespace) -> int:
    """Implement the ``inject`` subcommand."""
    target = Path(args.target)
    if target.is_dir():
        wheels = sorted(target.glob("*.whl"))
        if len(wheels) != 1:
            print(
                f"error: expected exactly one .whl in {target}, found {len(wheels)}",
                file=sys.stderr,
            )
            return 1
        wheel_path = wheels[0]
    else:
        wheel_path = target

    if args.copy_to is not None:
        copy_target_dir = Path(args.copy_to)
        copy_target_dir.mkdir(parents=True, exist_ok=True)
        copied = copy_target_dir / wheel_path.name
        shutil.copyfile(wheel_path, copied)
        wheel_path = copied

    meta = _read_pyproject_metadata(Path(args.project_root))
    doc = build_sbom_document(
        name=meta["name"],
        version=meta["version"],
        description=meta["description"],
        license_id=meta["license"],
        homepage_url=meta["homepage"],
        vcs_url=meta["vcs"],
        git_commit=args.git_commit,
        wheel_filename=wheel_path.name,
    )
    sbom_bytes = (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    inject_sbom_into_wheel(wheel_path, sbom_bytes)
    print(f"injected SBOM into {wheel_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Top-level CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="build_sbom.py", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Path to the bocphysics repository root (default: parent of this script).",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    gen = subparsers.add_parser(
        "generate", help="Emit a CycloneDX 1.6 JSON SBOM for the wheel."
    )
    gen.add_argument(
        "--output",
        "-o",
        default="-",
        help="Output path, or '-' for stdout (default: stdout).",
    )
    gen.add_argument(
        "--git-commit",
        default=os.environ.get("GITHUB_SHA"),
        help="Commit SHA to embed (default: $GITHUB_SHA).",
    )
    gen.add_argument(
        "--wheel-filename",
        default=None,
        help="Filename of the wheel the SBOM will be embedded in.",
    )
    gen.set_defaults(func=_cmd_generate)

    inj = subparsers.add_parser(
        "inject",
        help="Generate an SBOM and inject it into an existing wheel.",
    )
    inj.add_argument(
        "target",
        help="Wheel file or directory containing exactly one wheel.",
    )
    inj.add_argument(
        "--copy-to",
        default=None,
        help=(
            "If given, copy the wheel into this directory before injecting; "
            "the original is left unchanged."
        ),
    )
    inj.add_argument(
        "--git-commit",
        default=os.environ.get("GITHUB_SHA"),
        help="Commit SHA to embed (default: $GITHUB_SHA).",
    )
    inj.set_defaults(func=_cmd_inject)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
