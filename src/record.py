# src/record.py
"""Record of the Week 3 analysis: guarded output paths, file hashes and results/week3/manifest.json.

    python -m src.record SNAPSHOT.sha256

compares a `sha256sum` listing of results/ taken before the analysis with the files as they are now and
stores the outcome in the manifest. New outputs only go to results/week3/ and figures/.
"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "week3"
FIGURES = ROOT / "figures"
MANIFEST = OUT / "manifest.json"
PACKAGES = ["numpy", "pandas", "scipy", "matplotlib"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def guarded(path: Path) -> Path:
    """Path of a new output. Anything outside results/week3/ and figures/ is refused."""
    path = Path(path).resolve()
    assert OUT in path.parents or FIGURES in path.parents, f"refusing to write {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def source_state() -> dict:
    changed = [line[3:] for line in git("status", "--porcelain", "--", "src", "figures").splitlines() if line.endswith(".py")]
    return {"repo_commit": git("rev-parse", "HEAD"), "uncommitted_or_new_py_files": changed,
            "src_sha256": {p.name: sha256(p) for p in sorted((ROOT / "src").glob("*.py"))}}


def hashes(paths: list[Path]) -> dict[str, str]:
    return {relative(p): sha256(p) for p in sorted(set(map(Path, paths)))}


def read_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"steps": {}}


def write_manifest(manifest: dict) -> None:
    guarded(MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")


def record(step: str, command: str, inputs: list[Path], outputs: list[Path], settings: dict | None = None) -> None:
    """One entry per step; a rerun replaces the entry of the same step."""
    manifest = read_manifest()
    manifest["steps"][step] = {
        "date": date.today().isoformat(), "command": command, "python": platform.python_version(),
        "packages": {name: version(name) for name in PACKAGES}, "source": source_state(),
        "settings": settings or {}, "inputs_sha256": hashes(inputs), "outputs_sha256": hashes(outputs)}
    write_manifest(manifest)


def check_snapshot(snapshot: Path) -> dict:
    """Every file of the snapshot must still exist with the same hash."""
    rows = [line.split(maxsplit=1) for line in Path(snapshot).read_text().splitlines() if line.strip()]
    changed = [name for digest, name in rows if not (ROOT / name).exists() or sha256(ROOT / name) != digest]
    return {"date": date.today().isoformat(), "snapshot": relative(snapshot), "snapshot_sha256": sha256(snapshot),
            "files_in_snapshot": len(rows), "changed_or_missing": changed, "unchanged": not changed}


def main() -> None:
    outcome = check_snapshot(Path(sys.argv[1]))
    manifest = read_manifest()
    manifest["results_before_week3_unchanged"] = outcome
    write_manifest(manifest)
    print(f"{outcome['files_in_snapshot']} files, changed or missing: {outcome['changed_or_missing'] or 'none'}")


if __name__ == "__main__":
    main()
