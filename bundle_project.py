#!/usr/bin/env python3
"""
bundle_project.py  -  drop in the project root and run:  python bundle_project.py

Writes ONE text file (default: project_bundle.txt) in this format:

    ===== file name : models/quantum_ml.py
    <file contents>

    ===== file name : main.py
    <file contents>

Edit the CONFIG block below, or override from the command line, e.g.:
    python bundle_project.py --exclude-dir notebooks --exclude-glob "test_*.py"
    python bundle_project.py --include-ext .tex --out bundle_with_paper.txt
    python bundle_project.py --dry-run          # list what would be included, write nothing
"""
import argparse
import fnmatch
import os
import re
import sys
from pathlib import Path

# ============================ CONFIG ============================
OUTPUT_FILE = "project_bundle.txt"

# Directories skipped by NAME anywhere in the tree (case-insensitive).
EXCLUDE_DIRS = {
    "venv", ".venv", "env", "__pycache__", ".git", ".idea", ".vscode",
    "node_modules", "runs", "data", "dist", "build", ".pytest_cache",
    ".mypy_cache", ".ipynb_checkpoints",
}

# Files skipped by glob (matched against file name AND relative path).
EXCLUDE_FILE_GLOBS = [
    "*_old.py",              # quantum_ml_old.py, yolo_model_old.py, config_old.py
    "*_v0.*",
    "gemini-code-*.py",      # scratch files
    "benchmark_compare_1.py",
    "*.pyc", "*.lock", ".gitignore",
    "project_bundle*.txt",
]

# Only these extensions are bundled (add ".tex" to include the paper source).
INCLUDE_EXTS = {
    ".py", ".ps1", ".md", ".txt", ".yaml", ".yml", ".json", ".toml",
    ".cfg", ".ini", ".csv", ".sh", ".bat",
}

# Relative-path globs that are ALWAYS included, even inside excluded dirs.
ALWAYS_INCLUDE = [
    "data/dataset.yaml",
    "requirements.txt",
]

MAX_KB = 300          # skip files larger than this (large CSV/JSON dumps)
REDACT_SECRETS = True # mask API keys / tokens / passwords
INCLUDE_TREE = True   # first block = project tree
# ================================================================

SECRET_PATTERNS = [
    (re.compile(r"""(?i)((?:api[_-]?key|secret|token|password|passwd)\s*[=:]\s*)(["'])[^"'\n]{8,}\2"""),
     r"\1\2***REDACTED***\2"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "***REDACTED***"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{35}"), "***REDACTED***"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"), "***REDACTED***"),
]


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def matches_any(rel: str, name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel, p) for p in patterns)


def is_always(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in ALWAYS_INCLUDE)


def is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(4096)
    except OSError:
        return True


def collect(root: Path, exclude_dirs, exclude_globs, include_exts, max_kb, skip_names):
    included, skipped = [], []
    excl_lower = {d.lower() for d in exclude_dirs}
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        keep = []
        for d in sorted(dirnames):
            rel = rel_posix(base / d, root)
            if d.lower() in excl_lower:
                # still descend if an ALWAYS_INCLUDE file lives below it
                if any(p.replace("\\", "/").startswith(rel + "/") for p in ALWAYS_INCLUDE):
                    keep.append(d)
                else:
                    skipped.append((rel + "/", "excluded dir"))
            else:
                keep.append(d)
        dirnames[:] = keep

        for fn in sorted(filenames):
            p = base / fn
            rel = rel_posix(p, root)
            if fn in skip_names:
                continue
            in_excluded_dir = any(part.lower() in excl_lower for part in Path(rel).parts[:-1])
            if is_always(rel):
                pass
            elif in_excluded_dir:
                continue
            elif matches_any(rel, fn, exclude_globs):
                skipped.append((rel, "excluded glob"))
                continue
            elif p.suffix.lower() not in include_exts:
                skipped.append((rel, "extension not included"))
                continue
            if p.stat().st_size > max_kb * 1024:
                skipped.append((rel, f"larger than {max_kb} KB"))
                continue
            if is_binary(p):
                skipped.append((rel, "binary"))
                continue
            included.append(p)
    return included, skipped


def build_tree(paths, root: Path) -> str:
    return "\n".join(rel_posix(p, root) for p in paths)


def redact(text: str) -> str:
    for pat, repl in SECRET_PATTERNS:
        text = pat.sub(repl, text)
    return text


def main():
    ap = argparse.ArgumentParser(description="Bundle project files into one text file.")
    ap.add_argument("--root", default=".", help="project root (default: current dir)")
    ap.add_argument("--out", default=OUTPUT_FILE)
    ap.add_argument("--exclude-dir", action="append", default=[], help="extra dir name to skip")
    ap.add_argument("--exclude-glob", action="append", default=[], help="extra file glob to skip")
    ap.add_argument("--include-ext", action="append", default=[], help="extra extension, e.g. .tex")
    ap.add_argument("--max-kb", type=int, default=MAX_KB)
    ap.add_argument("--no-redact", action="store_true")
    ap.add_argument("--no-tree", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    exclude_dirs = set(EXCLUDE_DIRS) | set(args.exclude_dir)
    exclude_globs = list(EXCLUDE_FILE_GLOBS) + args.exclude_glob
    include_exts = {e.lower() if e.startswith(".") else "." + e.lower()
                    for e in (set(INCLUDE_EXTS) | set(args.include_ext))}
    out_path = (root / args.out).resolve()
    skip_names = {Path(__file__).name, out_path.name}

    files, skipped = collect(root, exclude_dirs, exclude_globs, include_exts,
                             args.max_kb, skip_names)

    if args.dry_run:
        print(f"Would include {len(files)} files:")
        for p in files:
            print("  +", rel_posix(p, root))
        print(f"\nSkipped {len(skipped)} entries:")
        for rel, why in skipped:
            print(f"  - {rel}  ({why})")
        return

    do_redact = REDACT_SECRETS and not args.no_redact
    total_chars, redacted_files = 0, []
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        if INCLUDE_TREE and not args.no_tree:
            out.write("===== file name : _PROJECT_TREE.txt\n")
            out.write(build_tree(files, root) + "\n\n")
        for p in files:
            rel = rel_posix(p, root)
            text = p.read_text(encoding="utf-8", errors="replace")
            if do_redact:
                new = redact(text)
                if new != text:
                    redacted_files.append(rel)
                text = new
            out.write(f"===== file name : {rel}\n")
            out.write(text.rstrip("\n") + "\n\n")
            total_chars += len(text)

    print(f"Wrote {out_path.name}: {len(files)} files, ~{total_chars/1000:.0f}K chars "
          f"(~{total_chars//4/1000:.0f}K tokens)")
    if redacted_files:
        print("Secrets masked in:", ", ".join(redacted_files))
    print(f"Skipped {len(skipped)} entries (run with --dry-run to see why).")
    print("Review the output for anything private before sharing.")


if __name__ == "__main__":
    sys.exit(main())
