import ast
import json
import time
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path.cwd()
CHECKPOINT_FILE = Path("audit_checkpoint.json")
REPORT_JSON = Path("audit_report_v2.json")
REPORT_MD = Path("AUDIT_SUMMARY_v2.md")

# ═════════════════════════════════════════════════════════════
# Stage 1: Directory Discovery + Auto-Exclusion
# ═════════════════════════════════════════════════════════════

EXCLUDE_DIR_NAMES = {
    'venv', '.venv', 'env', '__pycache__', '.git', '.tox',
    '.pytest_cache', 'build', 'dist', '.eggs', '.idea', '.vscode',
    '.mypy_cache', 'htmlcov', 'coverage', '.egg-info', 'node_modules',
    '.npm', '.yarn', 'Scripts', 'bin', 'include', 'share', 'lib',
    'Lib', 'libs'
}

EXCLUDE_PATH_SUBSTRINGS = [
    'site-packages', 'dist-packages', 'lib/python', 'Lib/python',
    'lib64/python', 'Lib64/python', 'python3.', 'python2.'
]

def should_exclude_dir(dir_path):
    parts = [p.lower() for p in dir_path.parts]
    for part in parts:
        if part in EXCLUDE_DIR_NAMES:
            return True, f"matched_exclude_dir:{part}"
        for sub in EXCLUDE_PATH_SUBSTRINGS:
            if sub.lower() in part:
                return True, f"matched_substring:{sub}"
    return False, ""

print("=" * 60)
print("Stage 1: Directory Discovery")
print("=" * 60)

all_dirs = []
excluded_dirs = []
for dp in REPO_ROOT.rglob("*"):
    if dp.is_dir():
        exc, reason = should_exclude_dir(dp)
        if exc:
            excluded_dirs.append((str(dp.relative_to(REPO_ROOT)), reason))
        else:
            all_dirs.append(dp)

print(f">>> Total directories scanned: {len(all_dirs) + len(excluded_dirs)}")
print(f">>> Directories excluded (external): {len(excluded_dirs)}")
print(f">>> Directories kept (project): {len(all_dirs)}")

# ═════════════════════════════════════════════════════════════
# Stage 2: Filtered File Scan
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Stage 2: Filtered .py File Scan")
print("=" * 60)

PY_FILES = []
for d in all_dirs:
    for f in d.glob("*.py"):
        # Double-check: file itself not in excluded path
        exc, reason = should_exclude_dir(f.parent)
        if not exc:
            PY_FILES.append(f)

PY_FILES = sorted(set(PY_FILES))
TOTAL = len(PY_FILES)

print(f">>> Python files found (after filtering): {TOTAL}")

if TOTAL == 0:
    print(">>> FATAL: No .py files found. Check directory structure.")
    sys.exit(1)

if TOTAL > 2000:
    print(f">>> WARNING: {TOTAL} files is unusually high for a trading project.")
    print(">>> This may indicate incomplete filtering. Proceeding anyway...")

# ═════════════════════════════════════════════════════════════
# Stage 3: Batched AST Analysis (Chunked for Terminal Safety)
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Stage 3: Batched AST Analysis (50 files per batch)")
print("=" * 60)

BATCH_SIZE = 50

file_data = {}
imports_graph = defaultdict(list)
reverse_graph = defaultdict(list)
broken_files = []
orphan_candidates = []
entry_points = []

known_entry_names = {"main", "beast_final", "run", "app", "server", "cli", "manage", "train", "backtest", "execute", "bot", "engine", "trader"}

def is_local_module(module_parts, repo_root):
    if not module_parts:
        return False
    rel_path = Path(*module_parts)
    if (repo_root / rel_path).with_suffix(".py").exists():
        return True
    if (repo_root / rel_path / "__init__.py").exists():
        return True
    if (repo_root / module_parts[0]).with_suffix(".py").exists():
        return True
    return False

def extract_imports(node):
    results = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            parts = alias.name.split(".") if alias.name else []
            results.append((parts, False, 0))
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        parts = module.split(".") if module else []
        results.append((parts, node.level > 0, node.level))
    return results

def analyze_file(py_path):
    rel_path = str(py_path.relative_to(REPO_ROOT)).replace("\\", "/")
    result = {
        "relative_path": rel_path,
        "filename": py_path.name,
        "lines": 0,
        "imports_local": [],
        "imports_external": [],
        "classes": [],
        "functions": [],
        "docstring": None,
        "syntax_valid": True,
        "error": None,
    }
    try:
        src = py_path.read_text(encoding="utf-8", errors="ignore")
        result["lines"] = len(src.splitlines())
        tree = ast.parse(src)
        result["docstring"] = ast.get_docstring(tree)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for parts, is_rel, level in extract_imports(node):
                    if is_rel:
                        result["imports_local"].append(".".join(parts) if parts else ".")
                        imports_graph[rel_path].append("[relative]")
                    else:
                        if is_local_module(parts, REPO_ROOT):
                            result["imports_local"].append(".".join(parts))
                            imports_graph[rel_path].append(".".join(parts))
                        else:
                            result["imports_external"].append(".".join(parts))
            elif isinstance(node, ast.ClassDef):
                result["classes"].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                result["functions"].append(node.name)
    except SyntaxError as e:
        result["syntax_valid"] = False
        result["error"] = f"SyntaxError line {e.lineno}: {e.msg}"
        broken_files.append(rel_path)
    except Exception as e:
        result["error"] = str(e)
        broken_files.append(rel_path)
    return rel_path, result

# Load checkpoint if exists
start_idx = 0
if CHECKPOINT_FILE.exists():
    try:
        cp = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        file_data = cp.get("file_data", {})
        broken_files = cp.get("broken_files", [])
        imports_graph = defaultdict(list, cp.get("imports_graph", {}))
        start_idx = cp.get("last_processed_index", 0)
        print(f">>> Resuming from checkpoint: {start_idx}/{TOTAL}")
    except Exception:
        print(">>> Checkpoint corrupt. Starting from scratch.")
        start_idx = 0

# Process in batches
total_batches = (TOTAL + BATCH_SIZE - 1) // BATCH_SIZE
for batch_num in range(start_idx // BATCH_SIZE, total_batches):
    batch_start = batch_num * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, TOTAL)
    batch = PY_FILES[batch_start:batch_end]

    t0 = time.time()
    for pf in batch:
        rp, data = analyze_file(pf)
        file_data[rp] = data

    batch_time = time.time() - t0
    valid_in_batch = sum(1 for f in batch if str(f.relative_to(REPO_ROOT)).replace("\\", "/") not in broken_files)
    broken_in_batch = len(batch) - valid_in_batch

    print(f">>> BATCH {batch_num + 1}/{total_batches} | Files: {len(batch)} | Valid: {valid_in_batch} | Broken: {broken_in_batch} | Time: {batch_time:.2f}s")

    # Save checkpoint
    checkpoint = {
        "last_processed_index": batch_end,
        "file_data": file_data,
        "broken_files": broken_files,
        "imports_graph": dict(imports_graph),
    }
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")

# ═════════════════════════════════════════════════════════════
# Stage 4: Dependency Graph + Orphan Detection + Report
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Stage 4: Dependency Analysis & Report Generation")
print("=" * 60)

all_rel_paths = set(file_data.keys())

# Build reverse graph
for rel_path, data in file_data.items():
    for imp in data["imports_local"]:
        if imp == "[relative]":
            continue
        parts = imp.split(".")
        candidates = [
            "/".join(parts) + ".py",
            "/".join(parts[:-1]) + "/" + parts[-1] + ".py" if len(parts) > 1 else parts[-1] + ".py",
            "/".join(parts) + "/__init__.py",
        ]
        for cand in candidates:
            if cand in all_rel_paths:
                reverse_graph[cand].append(rel_path)
                break

# Detect orphans and entry points
for rp in all_rel_paths:
    if not reverse_graph[rp]:
        orphan_candidates.append(rp)
        name = Path(rp).stem.lower()
        if any(e in name for e in known_entry_names):
            entry_points.append(rp)

# External dependency stats
ext_counter = defaultdict(int)
for data in file_data.values():
    for e in data["imports_external"]:
        top = e.split(".")[0]
        ext_counter[top] += 1

# ═════════════════════════════════════════════════════════════
# JSON Report
# ═════════════════════════════════════════════════════════════

report = {
    "repo_root": str(REPO_ROOT),
    "total_py_files": TOTAL,
    "syntax_valid_count": TOTAL - len(broken_files),
    "syntax_broken_count": len(broken_files),
    "broken_files": broken_files,
    "orphan_files": orphan_candidates,
    "entry_points": entry_points,
    "dependency_graph": dict(imports_graph),
    "reverse_dependency_graph": {k: v for k, v in reverse_graph.items()},
    "files": file_data,
    "excluded_directories": [{"path": p, "reason": r} for p, r in excluded_dirs[:100]],  # limit size
}

REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f">>> JSON report saved: {REPORT_JSON}")

# ═════════════════════════════════════════════════════════════
# Markdown Summary
# ═════════════════════════════════════════════════════════════

md = []
md.append("# AUDIT SUMMARY v2 — Atom 0.1 (Corrected)")
md.append(f"**Repository:** `{REPO_ROOT}`")
md.append(f"**Total .py files (filtered):** {TOTAL}")
md.append(f"**Syntax valid:** {TOTAL - len(broken_files)} / {TOTAL}")
md.append(f"**Syntax broken:** {len(broken_files)}")
md.append(f"**Orphan files:** {len(orphan_candidates)}")
md.append(f"**Entry points:** {len(entry_points)}")
md.append("")

if broken_files:
    md.append("## Broken Files (Syntax Errors)")
    for bf in broken_files:
        info = file_data.get(bf, {})
        err = info.get("error", "Unknown")
        md.append(f"- `{bf}` → {err}")
    md.append("")

md.append("## Orphaned Files (No Incoming Imports)")
md.append(f"Count: {len(orphan_candidates)}")
for of in sorted(orphan_candidates):
    flag = " [ENTRY POINT?]" if of in entry_points else ""
    md.append(f"- `{of}`{flag}")
md.append("")

md.append("## Entry Points Detected")
md.append(f"Count: {len(entry_points)}")
for ep in sorted(entry_points):
    md.append(f"- `{ep}`")
md.append("")

md.append("## Top External Dependencies")
for pkg, cnt in sorted(ext_counter.items(), key=lambda x: -x[1])[:20]:
    md.append(f"- `{pkg}`: {cnt} imports")
md.append("")

md.append("## File Inventory (Lines / Classes / Functions)")
for rp in sorted(file_data.keys()):
    d = file_data[rp]
    md.append(f"- `{rp}`: {d['lines']} lines, {len(d['classes'])} classes, {len(d['functions'])} funcs")

REPORT_MD.write_text("\n".join(md), encoding="utf-8")
print(f">>> Markdown report saved: {REPORT_MD}")

# Cleanup checkpoint
if CHECKPOINT_FILE.exists():
    CHECKPOINT_FILE.unlink()
    print(">>> Checkpoint cleaned.")

# ═════════════════════════════════════════════════════════════
# Final Console Summary
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f">>> TOTAL PYTHON FILES FOUND: {TOTAL}")
print(f">>> SYNTAX VALID: {TOTAL - len(broken_files)} / {TOTAL}")
print(f">>> BROKEN FILES: {len(broken_files)}")
print(f">>> ORPHAN FILES: {len(orphan_candidates)}")
print(f">>> ENTRY POINTS: {len(entry_points)}")
print(f">>> EXCLUDED DIRS: {len(excluded_dirs)}")
print(">>> OUTPUTS: audit_report_v2.json + AUDIT_SUMMARY_v2.md")
