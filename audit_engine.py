import ast
import json
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path.cwd()
PY_FILES = sorted(REPO_ROOT.rglob("*.py"))
TOTAL = len(PY_FILES)

# الهياكل
file_data = {}
imports_graph = defaultdict(list)      # ملف -> [الملفات التي يستوردها]
reverse_graph = defaultdict(list)      # ملف -> [الملفات التي تستورده]
broken_files = []
orphan_candidates = []
entry_points = []

def is_local_module(module_parts, repo_root):
    """التحقق إذا كان مسار الوحدة يتوافق مع ملف .py أو حزمة موجودة في المستودع."""
    if not module_parts:
        return False
    rel_path = Path(*module_parts)
    # التحقق كملف: a/b/c.py
    if (repo_root / rel_path).with_suffix(".py").exists():
        return True
    # التحقق كحزمة: a/b/c/__init__.py
    if (repo_root / rel_path / "__init__.py").exists():
        return True
    # التحقق من المستوى الأعلى فقط: a.py
    if (repo_root / module_parts[0]).with_suffix(".py").exists():
        return True
    return False

def extract_imports(node):
    """إرجاع قائمة (اسم_الوحدة، نسبي؟، المستوى) من عقدة الاستيراد."""
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
    rel_path = py_path.relative_to(REPO_ROOT)
    result = {
        "relative_path": str(rel_path).replace("\\", "/"),
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
                        imports_graph[str(rel_path)].append("[relative]")
                    else:
                        if is_local_module(parts, REPO_ROOT):
                            result["imports_local"].append(".".join(parts))
                            imports_graph[str(rel_path)].append(".".join(parts))
                        else:
                            result["imports_external"].append(".".join(parts))
            elif isinstance(node, ast.ClassDef):
                result["classes"].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                result["functions"].append(node.name)
    except SyntaxError as e:
        result["syntax_valid"] = False
        result["error"] = f"SyntaxError line {e.lineno}: {e.msg}"
        broken_files.append(str(rel_path))
    except Exception as e:
        result["error"] = str(e)
        broken_files.append(str(rel_path))
    return result

# المرحلة 1: تحليل كل الملفات
print(f">>> AUDIT STARTED: Scanning {TOTAL} Python files...")
for pf in PY_FILES:
    data = analyze_file(pf)
    file_data[str(pf.relative_to(REPO_ROOT)).replace("\\", "/")] = data

# المرحلة 2: بناء المخطط المعكوس لكشف اليتامى
all_rel_paths = set(file_data.keys())
for rel_path, data in file_data.items():
    for imp in data["imports_local"]:
        if imp == "[relative]":
            continue
        # ربط اسم الاستيراد بمسارات الملفات المحتملة
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

# المرحلة 3: كشف اليتامى ونقاط الدخول
known_entry_names = {"main", "beast_final", "run", "app", "server", "cli", "manage", "train", "backtest", "execute"}
for rp in all_rel_paths:
    if not reverse_graph[rp]:
        orphan_candidates.append(rp)
        name = Path(rp).stem.lower()
        if any(e in name for e in known_entry_names):
            entry_points.append(rp)

# المرحلة 4: تقرير JSON
report = {
    "repo_root": str(REPO_ROOT),
    "total_py_files": TOTAL,
    "syntax_valid_count": TOTAL - len(broken_files),
    "syntax_broken_count": len(broken_files),
    "broken_files": broken_files,
    "orphan_files": orphan_candidates,
    "entry_points": entry_points,
    "files": file_data,
    "dependency_graph": dict(imports_graph),
    "reverse_dependency_graph": {k: v for k, v in reverse_graph.items()},
}

with open("audit_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# المرحلة 5: ملخص Markdown للمدير
md = []
md.append("# AUDIT SUMMARY — Atom 0.1")
md.append(f"**Repository:** `{REPO_ROOT}`")
md.append(f"**Total .py files:** {TOTAL}")
md.append(f"**Syntax valid:** {TOTAL - len(broken_files)} / {TOTAL}")
md.append(f"**Syntax broken:** {len(broken_files)}")
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
ext_counter = defaultdict(int)
for data in file_data.values():
    for e in data["imports_external"]:
        top = e.split(".")[0]
        ext_counter[top] += 1
for pkg, cnt in sorted(ext_counter.items(), key=lambda x: -x[1])[:20]:
    md.append(f"- `{pkg}`: {cnt} imports")
md.append("")

md.append("## File Inventory (Lines / Classes / Functions)")
for rp in sorted(file_data.keys()):
    d = file_data[rp]
    md.append(f"- `{rp}`: {d['lines']} lines, {len(d['classes'])} classes, {len(d['functions'])} funcs")

with open("AUDIT_SUMMARY.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f">>> TOTAL PYTHON FILES FOUND: {TOTAL}")
print(f">>> SYNTAX VALID: {TOTAL - len(broken_files)} / {TOTAL}")
print(f">>> BROKEN FILES: {len(broken_files)}")
print(f">>> ORPHAN FILES: {len(orphan_candidates)}")
print(f">>> ENTRY POINTS: {len(entry_points)}")
print(">>> OUTPUTS: audit_report.json + AUDIT_SUMMARY.md")
