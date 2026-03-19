"""
Validate submission zip structure and constraints.

Checks:
- run.py at zip root
- Allowed file types only
- ≤10 Python files
- ≤3 weight files
- Total uncompressed size ≤420 MB
- No blocked imports in Python files
"""
import re
import zipfile
from pathlib import Path

ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".cfg", ".pt", ".pth", ".onnx", ".safetensors", ".npy"}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
MAX_FILES = 1000
MAX_PY_FILES = 10
MAX_WEIGHT_FILES = 3
MAX_SIZE_MB = 420

BLOCKED_IMPORTS = [
    r"\bimport\s+os\b",
    r"\bfrom\s+os\b",
    r"\bimport\s+sys\b",
    r"\bfrom\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bfrom\s+subprocess\b",
    r"\bimport\s+socket\b",
    r"\bfrom\s+socket\b",
    r"\bimport\s+pickle\b",
    r"\bfrom\s+pickle\b",
    r"\bimport\s+requests\b",
    r"\bfrom\s+requests\b",
    r"\bimport\s+multiprocessing\b",
    r"\bfrom\s+multiprocessing\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\s*\(",
    r"\bgetattr\s*\(",
]


def validate_zip(zip_path: str) -> list[str]:
    """Validate submission zip. Returns list of issues (empty = valid)."""
    issues = []
    zip_path = Path(zip_path)

    if not zip_path.exists():
        return [f"File not found: {zip_path}"]

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # Check run.py at root
        if "run.py" not in names:
            issues.append("run.py not found at zip root")

        # Check for nested run.py
        nested = [n for n in names if n.endswith("run.py") and n != "run.py"]
        if nested and "run.py" not in names:
            issues.append(f"run.py is nested: {nested[0]} — must be at root")

        # File count
        if len(names) > MAX_FILES:
            issues.append(f"Too many files: {len(names)} > {MAX_FILES}")

        # File type check
        py_count = 0
        weight_count = 0
        total_size = 0

        for info in zf.infolist():
            if info.is_dir():
                continue

            ext = Path(info.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                issues.append(f"Disallowed file type: {info.filename}")

            if ext == ".py":
                py_count += 1
            if ext in WEIGHT_EXTENSIONS:
                weight_count += 1

            total_size += info.file_size

        if py_count > MAX_PY_FILES:
            issues.append(f"Too many Python files: {py_count} > {MAX_PY_FILES}")

        if weight_count > MAX_WEIGHT_FILES:
            issues.append(f"Too many weight files: {weight_count} > {MAX_WEIGHT_FILES}")

        total_mb = total_size / (1024 * 1024)
        if total_mb > MAX_SIZE_MB:
            issues.append(f"Total size too large: {total_mb:.1f} MB > {MAX_SIZE_MB} MB")

        # Check for blocked imports in Python files
        for name in names:
            if name.endswith(".py"):
                try:
                    content = zf.read(name).decode("utf-8")
                    for pattern in BLOCKED_IMPORTS:
                        matches = re.findall(pattern, content)
                        if matches:
                            issues.append(f"Blocked import/call in {name}: {matches[0]}")
                except Exception:
                    issues.append(f"Could not read {name}")

    return issues


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", help="Path to submission zip file")
    args = parser.parse_args()

    issues = validate_zip(args.zip_path)
    if issues:
        print("VALIDATION FAILED:")
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("VALIDATION PASSED")
        with zipfile.ZipFile(args.zip_path) as zf:
            total = sum(i.file_size for i in zf.infolist() if not i.is_dir())
            print(f"  Files: {len(zf.namelist())}")
            print(f"  Size: {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
