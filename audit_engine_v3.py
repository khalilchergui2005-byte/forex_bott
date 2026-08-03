import ast
import json
import time
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path.cwd()
CHECKPOINT_FILE = Path("audit_checkpoint_v3.json")
REPORT_JSON = Path("audit_report_v3.json")
REPORT_MD = Path("AUDIT_SUMMARY_v3.md")

# ═════════════════════════════════════════════════════════════
# Stage 0: Structure Discovery (No Exclusion)
# ═════════════════════════════════════════════════════════════

SELF_FILES = {
    'audit_engine.py', 'audit_engine_v2.py', 'full_audit.py',
    'merge_and_holdout.py', 'restore_20y_data.py', 'test_fix.py',
    'audit_engine_v3.py'
}

def is_external_path(path):
    """استبعاد بناءً على المحتوى المؤكد فقط — لا أسماء عامة."""
    path_str = str(path).lower().replace("\\", "/")
    critical = [
        '/site-packages/', '/dist-packages/', '/venv/', '/.venv/', '/env/',
        '/__pycache__/', '/.git/', '/pip/', '/setuptools/', '/wheel/',
        '/pkg_resources/', '/tox/', '/pytest_cache/', '/node_modules/',
        '/.npm/', '/.yarn/', '/lib/python', '/lib64/python', '/scripts/python'
    ]
    for marker in critical:
        if marker in path_str:
            return True, marker
    if path.name.lower() in {'pip.exe', 'python.exe', 'pythonw.exe', 'activate', 'activate.bat'}:
        return True, "python_exe"
    return False, ""

print("=" * 70)
print("STAGE 0: STRUCTURE DISCOVERY")
print("=" * 70)

all_py_files = []
for f in REPO_ROOT.rglob("*.py"):
    if f.name in SELF_FILES:
        continue
    all_py_files.append(f)

# عرض حسب المجلد الأعلى
by_top = defaultdict(list)
for f in all_py_files:
    rel = f.relative_to(REPO_ROOT)
    parts = rel.parts
    if len(parts) >= 2:
        top = parts[0] + ("/" + parts[1] if len(parts) > 2 else "")
    else:
        top = "[root]"
    by_top[top].append(str(rel).replace("\\", "/"))

print(f">>> Total .py files discovered (excluding audit tools): {len(all_py_files)}")
print()

for d, files in sorted(by_top.items(), key=lambda x: -len(x[1]))[:20]:
    ext = sum(1 for f in files if is_external_path(REPO_ROOT / f)[0])
    real = len(files) - ext
    print(f"{d}: {len(files)} total | External: {ext} | Project: {real}")
    if real > 0 and len(files) <= 30:
        for f in files:
            if not is_external_path(REPO_ROOT / f)[0]:
                print(f"    -> {f}")

# ═════════════════════════════════════════════════════════════
# Stage 1: Smart Filtering
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STAGE 1: SMART FILTERING")
print("=" * 70)

project_files = []
external_files = []
for f in all_py_files:
    is_ext, reason = is_external_path(f)
    if is_ext:
        external_files.append((str(f.relative_to(REPO_ROOT)).replace("\\", "/"), reason))
    else:
        project_files.append(f)

print(f">>> External files excluded: {len(external_files)}")
print(f">>> Project files kept: {len(project_files)}")

# حماية إضافية: إذا كان العدد لا يزال ضخماً
if len(project_files) > 2000:
    print(">>> WARNING: Still >2000 files. Applying directory-size filter...")
    by_dir = defaultdict(int)
    for f in project_files:
        by_dir[f.parent] += 1
    big_dirs = {d for d, c in by_dir.items() if c > 500}
    project_files = [f for f in project_files if f.parent not in big_dirs]
    print(f">>> After size filter: {len(project_files)} files")

if len(project_files) == 0:
    print(">>> FATAL: No project files found after filtering.")
    sys.exit(1)

if len(project_files) > 2000:
    print(">>> FATAL: Too many files. Manual review required.")
    sys.exit(1)

TOTAL = len(project_files)
PY_FILES = sorted(project_files)

# ═════════════════════════════════════════════════════════════
# Stage 2: Batched AST Analysis
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STAGE 2: Batched AST Analysis (50 files per batch)")
print("=" * 70)

BATCH_SIZE = 50
file_data = {}
imports_graph = defaultdict(list)
reverse_graph = defaultdict(list)
broken_files = []
orphan_candidates = []
entry_points = []

known_entry_names = {"main", "beast_final", "run", "app", "server", "cli", "manage", "train", "backtest", "execute", "bot", "engine", "trader", "forex", "strategy"}

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

# Load checkpoint
start_idx = 0
if CHECKPOINT_FILE.exists():
    try:
        cp = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        file_data = cp.get("file_data", {})
        broken_files = cp.get("broken_files", [])
        imports_graph = defaultdict(list, {k: v for k, v in cp.get("imports_graph", {}).items()})
        start_idx = cp.get("last_processed_index", 0)
        print(f">>> Resuming from checkpoint: {start_idx}/{TOTAL}")
    except Exception as e:
        print(f">>> Checkpoint corrupt ({e}). Starting from scratch.")
        start_idx = 0

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
    batch_rps = [str(f.relative_to(REPO_ROOT)).replace("\\", "/") for f in batch]
    broken_in_batch = sum(1 for rp in batch_rps if rp in broken_files)
    valid_in_batch = len(batch) - broken_in_batch

    print(f">>> BATCH {batch_num + 1}/{total_batches} | Files: {len(batch)} | Valid: {valid_in_batch} | Broken: {broken_in_batch} | Time: {batch_time:.2f}s")

    checkpoint = {
        "last_processed_index": batch_end,
        "file_data": file_data,
        "broken_files": broken_files,
        "imports_graph": dict(imports_graph),
    }
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")

# ═════════════════════════════════════════════════════════════
# Stage 3: Dependency Graph + Report
# ═════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STAGE 3: Dependency Analysis & Report Generation")
print("=" * 70)

all_rel_paths = set(file_data.keys())

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

for rp in all_rel_paths:
    if not reverse_graph[rp]:
        orphan_candidates.append(rp)
        name = Path(rp).stem.lower()
        if any(e in name for e in known_entry_names):
            entry_points.append(rp)

ext_counter = defaultdict(int)
for data in file_data.values():
    for e in data["imports_external"]:
        top = e.split(".")[0]
        ext_counter[top] += 1

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
}

REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f">>> JSON report saved: {REPORT_JSON}")

md = []
md.append("# AUDIT SUMMARY v3 — Atom 0.1 (Content-Based)")
md.append(f"**Repository:** `{REPO_ROOT}`")
md.append(f"**Total .py files (project only):** {TOTAL}")
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

if CHECKPOINT_FILE.exists():
    CHECKPOINT_FILE.unlink()
    print(">>> Checkpoint cleaned.")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f">>> TOTAL PYTHON FILES FOUND: {TOTAL}")
print(f">>> SYNTAX VALID: {TOTAL - len(broken_files)} / {TOTAL}")
print(f">>> BROKEN FILES: {len(broken_files)}")
print(f">>> ORPHAN FILES: {len(orphan_candidates)}")
print(f">>> ENTRY POINTS: {len(entry_points)}")
print(">>> OUTPUTS: audit_report_v3.json + AUDIT_SUMMARY_v3.md")
