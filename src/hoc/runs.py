"""Run directories, manifests and overwrite protection.

Every analysis writes into its own run directory:

    runs/<name>/                      main configuration
    runs/<name>/variants/<variant>/   a config with ``[run] variant = "..."``

The name comes from ``--run``, then ``$HOC_RUN``, then ``"main"``. A
``--subject-limit N`` run always gets the suffix ``__nN``, so a smoke run can
never write into a full run's directory.

The name ``results`` points at the legacy pre-cleanup tree in ``results/``
(see ``baseline/PROVENANCE.md``). It can be read but never written.

Each run directory holds ``manifest.json`` with the config hash, subject set,
input checksums and, per stage, the git commit, package versions, arguments,
timing and output checksums. A stage refuses to run when the directory was made
with a different config or subject set, or when the stage has already finished,
unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import metadata as _metadata
from pathlib import Path

from .paths import REPO_ROOT

RUNS_ROOT_ENV = "HOC_RUNS_ROOT"
RUN_ENV = "HOC_RUN"
DEFAULT_RUN = "main"
LEGACY_RUN = "results"
LEGACY_VARIANT_DIRS = {"fraction_5pct": "results_fraction_5pct"}
MANIFEST_NAME = "manifest.json"

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "ripser",
    "persim",
    "bctpy",
    "networkx",
    "neuromaps",
    "netneurotools",
    "nilearn",
    "nibabel",
    "statsmodels",
    "gudhi",
    "joblib",
    "matplotlib",
)
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class RunConflictError(RuntimeError):
    """Raised when a stage would overwrite or mix outputs in a run directory."""


# --------------------------------------------------------------------------- paths


def runs_root() -> Path:
    """Directory that holds all run directories (``$HOC_RUNS_ROOT`` or ``runs/``)."""
    root = os.environ.get(RUNS_ROOT_ENV)
    return Path(root).resolve() if root else REPO_ROOT / "runs"


def run_name(run: str | None = None, subject_limit: int | None = None) -> str:
    """Resolve the run name from the argument, ``$HOC_RUN`` or the default."""
    name = run or os.environ.get(RUN_ENV) or DEFAULT_RUN
    if subject_limit is not None:
        suffix = f"__n{int(subject_limit)}"
        if name == LEGACY_RUN:
            raise RunConflictError("The legacy results/ tree cannot be used for a --subject-limit run.")
        if not name.endswith(suffix):
            name = name + suffix
    return name


def resolve_run_dir(
    run: str | None = None,
    *,
    variant: str | None = None,
    subject_limit: int | None = None,
) -> Path:
    """Return the absolute run directory for a run name and optional variant."""
    name = run_name(run, subject_limit)
    if name == LEGACY_RUN:
        root = REPO_ROOT / "results"
        if variant:
            return root / LEGACY_VARIANT_DIRS.get(variant, f"results_{variant}")
        return root
    root = runs_root() / name
    return root / "variants" / variant if variant else root


def is_legacy(path: Path) -> bool:
    """True when ``path`` lies inside the read-only legacy results tree."""
    legacy = (REPO_ROOT / "results").resolve()
    try:
        Path(path).resolve().relative_to(legacy)
    except ValueError:
        return False
    return True


def config_variant(config: dict) -> str | None:
    """Variant name declared by a config (``[run] variant``), if any."""
    value = config.get("run", {}).get("variant")
    return str(value) if value else None


# --------------------------------------------------------------------------- helpers


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_state() -> dict:
    """Commit, branch and whether tracked files differ from the commit."""
    commit = _git("rev-parse", "HEAD")
    dirty_files = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty_files) if dirty_files is not None else None,
        "dirty_files": dirty_files.splitlines() if dirty_files else [],
    }


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = _metadata.version(name)
        except _metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def blas_info() -> dict:
    """Which BLAS/LAPACK numpy is linked against (best effort)."""
    try:
        import numpy as np

        config = np.show_config(mode="dicts")
        deps = config.get("Build Dependencies", {})
        return {key: {k: deps[key].get(k) for k in ("name", "version")} for key in ("blas", "lapack") if key in deps}
    except Exception:  # noqa: BLE001 - informational only
        return {}


def environment_record() -> dict:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "blas": blas_info(),
        "threads": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
    }


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def collect_inputs(config: dict) -> list[Path]:
    """Input files named by a config's ``[paths]`` block."""
    paths_cfg = config.get("paths", {})
    files: list[Path] = []
    raw = paths_cfg.get("raw_connectomes")
    if raw:
        files.append(REPO_ROOT / raw if not Path(raw).is_absolute() else Path(raw))
    atlas_dir = paths_cfg.get("atlas_dir")
    if atlas_dir:
        atlas = REPO_ROOT / atlas_dir if not Path(atlas_dir).is_absolute() else Path(atlas_dir)
        if atlas.is_dir():
            files.extend(sorted(p for p in atlas.iterdir() if p.is_file() and p.name != ".DS_Store"))
    return [p for p in files if p.exists()]


# --------------------------------------------------------------------------- manifest


def read_manifest(run_dir: str | Path) -> dict | None:
    path = Path(run_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(run_dir / MANIFEST_NAME)


@dataclass
class RunContext:
    """One stage of an analysis writing into a run directory.

    Typical use::

        ctx = RunContext.from_args("01_primary", args, config_path=args.config)
        ctx.begin(subject_ids)
        ...  # write outputs under ctx.root
        ctx.finish()
    """

    stage: str
    config_path: Path | None = None
    run: str | None = None
    subject_limit: int | None = None
    overwrite: bool = False
    argv: list[str] = field(default_factory=lambda: list(sys.argv))
    config: dict = field(default_factory=dict)
    variant: str | None = None
    root: Path = field(init=False)
    _started: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.config_path is not None:
            self.config_path = Path(self.config_path)
            if not self.config_path.is_absolute():
                self.config_path = REPO_ROOT / self.config_path
            if not self.config:
                from .config import load_config

                self.config = load_config(self.config_path)
        if self.variant is None:
            self.variant = config_variant(self.config)
        self.root = resolve_run_dir(self.run, variant=self.variant, subject_limit=self.subject_limit)
        if is_legacy(self.root):
            raise RunConflictError(
                f"{_relative(self.root)} is the legacy pre-cleanup tree and is read-only. "
                "Choose another --run name."
            )

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_args(cls, stage: str, args: argparse.Namespace, *, config_path: str | Path | None = None) -> "RunContext":
        return cls(
            stage=stage,
            config_path=config_path if config_path is not None else getattr(args, "config", None),
            run=getattr(args, "run", None),
            subject_limit=getattr(args, "subject_limit", None),
            overwrite=bool(getattr(args, "overwrite", False)),
        )

    # ------------------------------------------------------------------ properties

    @property
    def name(self) -> str:
        return run_name(self.run, self.subject_limit)

    @property
    def config_sha256(self) -> str | None:
        return sha256_file(self.config_path) if self.config_path else None

    def path(self, *parts: str) -> Path:
        """Path inside the run directory, with parent folders created."""
        target = self.root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    # ------------------------------------------------------------------ lifecycle

    def _check_compatible(self, manifest: dict, subject_ids: list | None) -> None:
        problems = []
        cfg_hash = self.config_sha256
        if cfg_hash and manifest.get("config", {}).get("sha256") not in (None, cfg_hash):
            problems.append(
                f"config differs (run made with {manifest['config'].get('path')}, "
                f"sha256 {manifest['config']['sha256'][:12]}; now {cfg_hash[:12]})"
            )
        if manifest.get("subject_limit") != self.subject_limit:
            problems.append(f"subject limit differs ({manifest.get('subject_limit')} vs {self.subject_limit})")
        if subject_ids is not None and manifest.get("subjects") is not None:
            if [str(s) for s in manifest["subjects"]] != [str(s) for s in subject_ids]:
                problems.append("subject set differs")
        stage_record = manifest.get("stages", {}).get(self.stage, {})
        if stage_record.get("status") == "complete":
            problems.append(f"stage '{self.stage}' already finished at {stage_record.get('finished')}")
        if problems and not self.overwrite:
            raise RunConflictError(
                f"Refusing to write into {_relative(self.root)}: " + "; ".join(problems)
                + ". Use a new --run name, or pass --overwrite to replace it."
            )

    def require_stage(self, stage: str) -> dict:
        """Fail unless ``stage`` finished in this run with the same config."""
        manifest = read_manifest(self.root) or {}
        record = manifest.get("stages", {}).get(stage)
        if not record or record.get("status") != "complete":
            raise RunConflictError(f"Stage '{stage}' has not finished in {_relative(self.root)}; run it first.")
        return record

    def begin(self, subject_ids: list | None = None, *, hash_inputs: bool = True) -> "RunContext":
        manifest = read_manifest(self.root)
        if manifest is not None:
            self._check_compatible(manifest, subject_ids)
        else:
            manifest = {
                "run": self.name,
                "variant": self.variant,
                "created": _now(),
                "config": {
                    "path": _relative(self.config_path) if self.config_path else None,
                    "sha256": self.config_sha256,
                },
                "subject_limit": self.subject_limit,
                "subjects": None,
                "inputs": {},
                "stages": {},
            }
        if self.overwrite and self.config_path is not None:
            manifest["config"] = {"path": _relative(self.config_path), "sha256": self.config_sha256}
        if subject_ids is not None:
            manifest["subjects"] = [str(s) if not isinstance(s, (int, float)) else s for s in subject_ids]
        if hash_inputs and self.config:
            for path in collect_inputs(self.config):
                key = _relative(path)
                stat = path.stat()
                cached = manifest["inputs"].get(key)
                if cached and cached.get("size") == stat.st_size and cached.get("mtime") == stat.st_mtime:
                    continue
                manifest["inputs"][key] = {"sha256": sha256_file(path), "size": stat.st_size, "mtime": stat.st_mtime}
        if self.config_path is not None:
            snapshot = self.root / "config" / self.config_path.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(self.config_path.read_bytes())
        self._started = _now()
        manifest["stages"][self.stage] = {
            "status": "running",
            "started": self._started,
            "argv": self.argv,
            "overwrite": self.overwrite,
            "git": git_state(),
            "environment": environment_record(),
        }
        _write_manifest(self.root, manifest)
        return self

    def finish(self, outputs: list[str | Path] | None = None, **extra: object) -> None:
        """Mark the stage complete and record checksums of its outputs.

        ``outputs`` defaults to every file in the run directory modified since
        the stage started, excluding variant folders and ``logs/`` folders.
        Stages that run concurrently should pass ``outputs`` explicitly.
        """
        manifest = read_manifest(self.root) or {"stages": {}}
        record = manifest["stages"].setdefault(self.stage, {})
        if outputs is None:
            started = _dt.datetime.fromisoformat(self._started).timestamp() if self._started else 0.0
            outputs = [
                p
                for p in self.root.rglob("*")
                if p.is_file()
                and p.name not in (MANIFEST_NAME, MANIFEST_NAME + ".tmp")
                and "variants" not in p.relative_to(self.root).parts
                and "logs" not in p.relative_to(self.root).parts
                and p.stat().st_mtime >= started - 1
            ]
        record["outputs"] = {
            str(Path(p).resolve().relative_to(self.root.resolve())): sha256_file(p) for p in sorted(map(Path, outputs))
        }
        record.update({"status": "complete", "finished": _now(), **extra})
        _write_manifest(self.root, manifest)


# --------------------------------------------------------------------------- CLI glue


def add_run_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--run`` and ``--overwrite`` to a script's parser."""
    parser.add_argument(
        "--run",
        default=None,
        help=f"Run name under runs/ (default: ${RUN_ENV} or '{DEFAULT_RUN}'; 'results' = legacy tree, read-only).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing a finished stage or a run made with a different config/subject set.",
    )
    return parser


def stage_from_argv(
    stage: str,
    *,
    config_path: str | Path | None = "config/default.toml",
    argv: list[str] | None = None,
    variant: str | None = None,
) -> RunContext:
    """Build a RunContext for scripts that take no other arguments.

    Reads ``--run``, ``--overwrite`` and ``--subject-limit`` from the command line
    (unknown arguments are left alone).
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_run_arguments(parser)
    parser.add_argument("--subject-limit", type=int, default=None)
    known, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return RunContext(
        stage=stage,
        config_path=config_path,
        run=known.run,
        subject_limit=known.subject_limit,
        overwrite=known.overwrite,
        variant=variant,
    )


def reader_run_dir(argv: list[str] | None = None, *, variant: str | None = None) -> Path:
    """Run directory for read-only consumers (figure scripts, notebooks).

    Uses ``--run`` from ``argv`` if present, else ``$HOC_RUN``, else the default,
    and fails clearly when the directory does not exist.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", default=None)
    known, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    path = resolve_run_dir(known.run, variant=variant)
    if not path.exists():
        raise FileNotFoundError(
            f"Run directory {_relative(path)} does not exist. Pass --run <name> or set ${RUN_ENV} "
            f"(use '{LEGACY_RUN}' for the legacy results/ tree)."
        )
    return path


def writable_root(run_dir: str | Path) -> Path:
    """Where read-only consumers of ``run_dir`` may write derived files.

    For a normal run this is the run directory itself. For the legacy
    ``results/`` tree it is ``runs/results-derived/``, so the baseline files are
    never touched.
    """
    run_dir = Path(run_dir)
    if not is_legacy(run_dir):
        return run_dir
    legacy = (REPO_ROOT / "results").resolve()
    rel = run_dir.resolve().relative_to(legacy)
    return runs_root() / "results-derived" / rel


def figures_dir(run_dir: str | Path, *parts: str) -> Path:
    """Figure output folder for a run (``<run>/figures/...``), created on demand."""
    path = writable_root(run_dir) / "figures"
    path = path.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def partial_stage(stage: str, **subsets: tuple[object, object]) -> str:
    """Stage name for a run over a subset of the work.

    Each keyword is ``name=(selected, full)``. If any selection differs from the
    full set, the stage gets a ``:partial[...]`` suffix, so a partial run never
    counts as the finished stage that downstream scripts require.
    """
    parts = []
    for key, (selected, full) in subsets.items():
        sel = sorted(selected) if isinstance(selected, (list, tuple, set)) else selected
        ful = sorted(full) if isinstance(full, (list, tuple, set)) else full
        if sel != ful:
            shown = ",".join(map(str, sel)) if isinstance(sel, list) else str(sel)
            parts.append(f"{key}={shown}")
    return f"{stage}:partial[{';'.join(parts)}]" if parts else stage
