"""
Builds the submission ZIP: CSD358_<Member1>_<Member2>.zip in the project folder.

Run from the project root:  python scripts/package_submission.py
Edit MEMBERS and GITHUB_LINK below before the final run.
"""

import os
import sys
import zipfile

MEMBERS = ["AryanRastogi", "RaghavGarg"]
GITHUB_LINK = "TODO(link)"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PDF = os.path.join("docs", "report", "main.pdf")
DATASET_ZIP = os.path.join("data", "raw", "beir", "trec-covid.zip")

LOCAL_EXCLUDE_FILE = os.path.join(PROJECT_ROOT, ".git", "info", "exclude")
EXCLUDED_FOLDERS = {"index_out", "node_modules", "archive", "reference", ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
EXCLUDED_PATHS = {
    os.path.join("web", "dist"),
    os.path.join("data", "raw"),
    os.path.join("data", "cache"),
}
EXCLUDED_SUFFIXES = (".pyc", ".aux", ".log", ".fdb_latexmk", ".fls", ".synctex.gz")
EXCLUDED_FILES = {"build.log", ".env", ".dev.vars", "secrets.toml"}

BUILD_TEXT = """EvidenceLens: build instructions

The ZIP leaves out index_out/ (the search index, about 500 MB) and the extracted dataset. Both are rebuilt
from data/raw/beir/trec-covid.zip, which is included.

1. Python 3.12 or newer and Node.js. In the project folder:
     python -m venv .venv
     .venv\\Scripts\\activate          (Windows; source .venv/bin/activate on macOS and Linux)
     pip install -r requirements.txt
2. Build the index, the thesaurus and the study-type labels (this reads data/raw/beir/trec-covid.zip):
     python -m pipeline.build_index
     python -m pipeline.build_thesaurus
     python -m pipeline.build_evidence
   word2vec training uses several workers and is not bit-reproducible, so the rebuilt thesaurus can differ
   slightly from the one behind the report. The time for these steps was not measured for this ZIP.
3. Build the web app and start the server:
     cd web
     npm install
     npm run build
     cd ..
     uvicorn api.main:app
   Open http://localhost:8000 (health check: http://localhost:8000/api/health).
4. Tests: python -m unittest discover tests
   Evaluation (about 90 seconds): python -m evaluation.run_eval, then see evaluation/summary.md.

The report is docs/report/main.pdf (LaTeX source docs/report/main.tex). The README explains the project.
"""


def local_excludes():
    """Paths this working copy keeps out of version control (the lines of .git/info/exclude); they stay out of the ZIP too."""
    if not os.path.exists(LOCAL_EXCLUDE_FILE):
        return set()
    with open(LOCAL_EXCLUDE_FILE, encoding="utf-8") as f:
        lines = [line.strip().rstrip("/") for line in f]
    return {line for line in lines if line and not line.startswith("#")}


def is_excluded(relative_path, local_entries):
    parts = relative_path.split(os.sep)
    posix_path = "/".join(parts)
    if any(posix_path == entry or posix_path.startswith(entry + "/") or entry in parts for entry in local_entries):
        return True
    if any(part in EXCLUDED_FOLDERS for part in parts[:-1]):
        return True
    if any(relative_path == path or relative_path.startswith(path + os.sep) for path in EXCLUDED_PATHS):
        return relative_path != DATASET_ZIP
    return parts[-1] in EXCLUDED_FILES or relative_path.endswith(EXCLUDED_SUFFIXES)


def project_files():
    local_entries = local_excludes()
    for folder, folder_names, file_names in os.walk(PROJECT_ROOT):
        folder_names[:] = [name for name in folder_names if name not in EXCLUDED_FOLDERS]
        for name in file_names:
            relative_path = os.path.relpath(os.path.join(folder, name), PROJECT_ROOT)
            if not is_excluded(relative_path, local_entries) and not name.startswith("CSD358_"):
                yield relative_path


def check_inputs():
    missing = [path for path in (REPORT_PDF, DATASET_ZIP) if not os.path.exists(os.path.join(PROJECT_ROOT, path))]
    if missing:
        sys.exit("Cannot build the ZIP, these files are missing: " + ", ".join(missing)
                 + ("\nBuild the report first: cd docs/report && latexmk -pdf main.tex" if REPORT_PDF in missing else ""))


def main():
    check_inputs()
    zip_name = "CSD358_" + "_".join(MEMBERS) + ".zip"
    zip_path = os.path.join(PROJECT_ROOT, zip_name)
    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(os.path.join(PROJECT_ROOT, REPORT_PDF), "main.pdf")
        archive.writestr("GITHUB_LINK.txt", GITHUB_LINK + "\n")
        archive.writestr("BUILD.md", BUILD_TEXT)
        file_count += 3
        for relative_path in sorted(project_files()):
            archive.write(os.path.join(PROJECT_ROOT, relative_path), "source/" + relative_path.replace(os.sep, "/"))
            file_count += 1
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Built {zip_name}: {file_count} files, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
