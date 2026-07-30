import os
import ast

PROJECT_DIR = r"C:\Users\GTINF\Desktop\forex_bot"
EXCLUDE_DIRS = {"venv", "__pycache__", ".git", "logs", "reports", "models", "data"}

print("=" * 60)
print("   PROJECT LEGACY AUDIT (Full File Map)")
print("=" * 60)

# 1. Non-Python Files (Data, Models, Configs)
print("\n[1] DATA & MODEL FILES (Assets):")
for root, dirs, files in os.walk(PROJECT_DIR):
    # Skip excluded dirs
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for file in files:
        if not file.endswith('.py'):
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            if size > 1024: # > 1KB
                rel_path = os.path.relpath(file_path, PROJECT_DIR)
                print(f"  - {rel_path} ({size/1024:.1f} KB)")

# 2. Python Files Analysis (Extracting Docstrings)
print("\n[2] PYTHON FILES MAP (Functionality):")
for root, dirs, files in os.walk(PROJECT_DIR):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for file in sorted(files):
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, PROJECT_DIR)
            
            docstring = "No description"
            try:
                with open(file_path, encoding='utf-8') as f:
                    content = f.read(2048) # Read first 2KB
                    tree = ast.parse(content)
                    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
                        docstring = tree.body[0].value.value.strip().split('\n')[0][:80]
            except:
                docstring = "Parse error or empty"
                
            print(f"  [{rel_path}] -> {docstring}")

print("\n" + "=" * 60)
print("Audit Complete. Review the list above.")
