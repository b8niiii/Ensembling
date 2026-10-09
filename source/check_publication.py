"""Check the candidate Git publication without staging, committing or pushing."""
from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"hf_[A-Za-z0-9]{25,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def candidate_files():
    raw = git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    return sorted({name.decode() for name in raw.split(b"\0")
                   if name and (ROOT / name.decode()).is_file()})


def check():
    files = candidate_files()
    public_paths = {(ROOT / name).resolve() for name in files}
    issues, notebooks = [], []
    # Compare exact local credential values without printing or exporting them.
    credentials = []
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            name, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and ("TOKEN" in name or "KEY" in name or "PASSWORD" in name):
                value = value.strip().strip("\"'")
                if len(value) >= 12:
                    credentials.append(value.encode())

    def contains_secret(raw):
        return any(pattern.search(raw) for pattern in SECRET_PATTERNS) or any(
            credential in raw for credential in credentials)

    for name in files:
        path = ROOT / name
        if (name.split("/")[0] in {"data", "models", "model_cache", ".venv", "outputs"}
                or (path.name.startswith(".env") and path.name != ".env.example")):
            issues.append({"path": name, "issue": "private/local artifact included"})
        if path.stat().st_size >= 10 * 1024**2:
            issues.append({"path": name, "issue": "public file exceeds the project 10 MiB limit"})
        raw = path.read_bytes()
        if contains_secret(raw):
            issues.append({"path": name, "issue": "possible embedded credential; value withheld"})
        if path.suffix == ".py":
            ast.parse(raw.decode(), filename=name)
        markdown = raw.decode() if path.suffix == ".md" else ""
        if path.suffix == ".ipynb":
            notebook = json.loads(raw)
            markdown = "\n".join("".join(c["source"]) for c in notebook["cells"]
                                 if c["cell_type"] == "markdown")
            assignments = {}
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse("".join(cell["source"]), filename=name)
                for node in tree.body:
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                assignments[target.id] = node.value.value
                if any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
                    issues.append({"path": name, "issue": "saved notebook error output"})
            for flag in ("RUN_API", "RUN_LOCAL_INFERENCE", "RUN_RELEX_BASE", "RUN_RELEX_LARGE",
                         "DOWNLOAD_DATA", "PREFETCH_ASSETS", "FETCH_ENABLED_MODEL_ASSETS",
                         "ANALYZE_LOCAL_RESULTS", "ANALYZE_LOCAL_SCORES"):
                if assignments.get(flag) is True:
                    issues.append({"path": name, "issue": f"execution enabled by default: {flag}"})
            if path.name == "Full_validation_relex.ipynb" and assignments.get("RUN_GLIDRE"):
                issues.append({"path": name, "issue": "GLiDRE inference enabled by default"})
            notebooks.append(name)
        for target in re.findall(r"\]\(([^)]+)\)", markdown):
            if "://" in target or target.startswith("#"):
                continue
            target = target.strip("<>").split("#", 1)[0]
            resolved = (path.parent / target).resolve()
            is_public = resolved in public_paths or (
                resolved.is_dir() and any(p.is_relative_to(resolved) for p in public_paths)
            )
            if target and not is_public:
                issues.append({"path": name, "issue": "link target absent from public file set",
                               "target": target})
    # Inspect existing Git history as well as the proposed working-tree files.
    historical_blobs = set()
    for line in git("rev-list", "--objects", "--all").splitlines():
        object_id = line.split(b" ", 1)[0].decode()
        if git("cat-file", "-t", object_id).strip() == b"blob":
            historical_blobs.add(object_id)
    for object_id in historical_blobs:
        raw = git("cat-file", "blob", object_id)
        if contains_secret(raw):
            issues.append({"git_object": object_id, "issue": "possible historical credential; value withheld"})
        if len(raw) >= 100 * 1024**2:
            issues.append({"git_object": object_id, "issue": "historical blob exceeds 100 MiB"})
    return {"success": not issues, "candidate_file_count": len(files),
            "candidate_bytes": sum((ROOT / name).stat().st_size for name in files),
            "largest_candidate": max(files, key=lambda n: (ROOT / n).stat().st_size),
            "notebooks_checked": notebooks, "historical_blobs_checked": len(historical_blobs),
            "issues": issues, "files": files,
            "scope": "local candidates and reachable Git history; no staging, commit, push or provider request"}


if __name__ == "__main__":
    report = check()
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
    raise SystemExit(0 if report["success"] else 1)
