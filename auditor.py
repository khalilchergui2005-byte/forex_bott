import os
import re

EXCLUDE_DIRS = {'venv', '__pycache__', '.pytest_cache', '.git', 'logs', 'data', 'models', 'reports'}
EXCLUDE_FILES = {'auditor.py', 'test_rm.py', 'test_risk_manager_pro.py'}

MAGIC_PATTERNS = [
    (r'=\s*\d+\.?\d*', 'assignment'),
    (r'[*/]\s*\d+\.?\d*', 'multiplication/division'),
]

def audit_file(filepath):
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        return issues
    for i, line in enumerate(lines, 1):
        for pattern, ptype in MAGIC_PATTERNS:
            matches = re.findall(pattern, line)
            for m in matches:
                val = re.search(r'\d+\.?\d*', m).group()
                if float(val) == 0.0 or float(val) == 1.0:
                    continue
                if ptype == 'assignment' and any(kw in line for kw in ['import', 'def ', 'class ', '#', '"""', "'''", 'path', 'file', 'log', 'print', 'axes', 'fig', 'dpi', 'fontsize', 'lw', 'alpha', 'color', 'rotation', 'axhline', 'fill_between', 'set_title', 'savefig', 'xaxis', 'plot', 'span', 'ewm', 'rolling', 'quantile', 'index', 'read_csv', 'to_csv', 'DateOffset', 'timedelta', 'resample', 'agg', 'map', 'isin', 'shift', 'diff', 'cummax', 'cumsum', 'dropna', 'sort_index', 'strftime', 'Timestamp', 'datetime', 'LineWidth', 'linewidth']):
                    continue
                issues.append((i, ptype, val.strip(), line.strip()[:80]))
    return issues

def main():
    root = '.'
    total_issues = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            if fname.endswith('.py') and fname not in EXCLUDE_FILES:
                fpath = os.path.join(dirpath, fname)
                issues = audit_file(fpath)
                if issues:
                    print(f'\n📄 {fpath} ({len(issues)} issues):')
                    for line_no, ptype, val, code in issues:
                        print(f'   ⚠️  سطر {line_no} [{ptype}]: {val}  | {code}')
                        total_issues += 1
    print(f'\n{"="*60}')
    print(f'🔍 تم فحص المشروع. إجمالي التحذيرات: {total_issues}')
    if total_issues == 0:
        print('✅ لا توجد قيم ثابتة مشبوهة.')

if __name__ == '__main__':
    main()
