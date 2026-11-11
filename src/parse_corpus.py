"""Orchestration script for downloading, extracting, parsing, and validating SpamAssassin corpus."""

import os
from pathlib import Path
import sys
import tarfile
import urllib.request
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import parse_email_file


CORPUS_URLS = {
    "easy_ham": {
        "filename": "20030228_easy_ham.tar.bz2",
        "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
        "label": 0,
        "extracted_folder": "easy_ham",
    },
    "hard_ham": {
        "filename": "20030228_hard_ham.tar.bz2",
        "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
        "label": 0,
        "extracted_folder": "hard_ham",
    },
    "spam": {
        "filename": "20030228_spam.tar.bz2",
        "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
        "label": 1,
        "extracted_folder": "spam",
    },
}


def download_archives(archives_dir: Path) -> dict:
    """Download and validate SpamAssassin archives, reusing existing valid files."""
    archives_dir.mkdir(parents=True, exist_ok=True)
    archive_paths = {}

    print("=" * 60)
    print("TASK 2: DOWNLOADING AND VERIFYING SPAMASSASSIN ARCHIVES")
    print("=" * 60)

    for category, meta in CORPUS_URLS.items():
        tar_path = archives_dir / meta["filename"]
        archive_paths[category] = tar_path
        url = meta["url"]

        is_valid = False
        if tar_path.exists():
            try:
                if tarfile.is_tarfile(str(tar_path)):
                    is_valid = True
                    size_mb = tar_path.stat().st_size / (1024 * 1024)
                    print(f"[{category.upper()}] Existing archive valid ({size_mb:.2f} MB): {tar_path.name}")
            except Exception:
                is_valid = False

        if not is_valid:
            print(f"[{category.upper()}] Downloading from {url}...")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(tar_path, "wb") as out_file:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)

            if not tarfile.is_tarfile(str(tar_path)):
                raise ValueError(f"Downloaded archive for {category} is corrupt or not a valid tarfile: {tar_path}")

            size_mb = tar_path.stat().st_size / (1024 * 1024)
            print(f"[{category.upper()}] Download completed successfully. Size: {size_mb:.2f} MB")

    return archive_paths


def extract_archives(archive_paths: dict, extracted_base_dir: Path) -> dict:
    """Extract archives into isolated category directories under extracted_base_dir."""
    extracted_base_dir.mkdir(parents=True, exist_ok=True)
    category_dirs = {}

    print("\n" + "=" * 60)
    print("EXTRACTING ARCHIVES INTO ISOLATED DIRECTORIES")
    print("=" * 60)

    for category, tar_path in archive_paths.items():
        target_dir = extracted_base_dir / category
        category_dirs[category] = target_dir

        # Check if already extracted (target directory has files other than .gitkeep)
        existing_files = [f for f in target_dir.glob("*") if f.name != ".gitkeep"] if target_dir.exists() else []

        if existing_files:
            print(f"[{category.upper()}] Target directory already extracted with {len(existing_files)} files. Skipping re-extraction.")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{category.upper()}] Extracting {tar_path.name} to {target_dir}...")
            with tarfile.open(str(tar_path), "r:bz2") as tar:
                # Safe extraction
                for member in tar.getmembers():
                    # Handle internal folder structure (e.g. member named easy_ham/0001...)
                    member_path = Path(member.name)
                    # Extract directly into target_dir stripping leading category folder if present
                    if len(member_path.parts) > 1 and member_path.parts[0] in [category, "easy_ham", "hard_ham", "spam", "easy_ham_2", "spam_2"]:
                        sub_path = Path(*member_path.parts[1:])
                    else:
                        sub_path = member_path

                    if member.isdir():
                        (target_dir / sub_path).mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        dest_file = target_dir / sub_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as src_f, open(dest_file, "wb") as dst_f:
                            dst_f.write(src_f.read())

            extracted_count = len([f for f in target_dir.glob("*") if f.is_file()])
            print(f"[{category.upper()}] Extracted {extracted_count} items into {target_dir}")

    return category_dirs


def parse_and_compile_corpus(category_dirs: dict, output_csv_path: Path):
    """Parse all emails across categories and construct the dataset DataFrame."""
    print("\n" + "=" * 60)
    print("TASK 3: PARSING EMAILS VIA RFC-822 / MIME PARSER")
    print("=" * 60)

    records = []
    skipped_files = []
    parsing_errors = []

    for category, cat_dir in category_dirs.items():
        meta = CORPUS_URLS[category]
        label = meta["label"]
        print(f"\nProcessing category: {category} (Label: {label}) from {cat_dir}...")

        cat_files = sorted([f for f in cat_dir.rglob("*") if f.is_file()])
        cat_parsed = 0

        for fpath in cat_files:
            record, error_msg = parse_email_file(fpath)
            if error_msg:
                if "Skipped" in error_msg:
                    skipped_files.append((category, fpath.name, error_msg))
                else:
                    parsing_errors.append((category, fpath.name, error_msg))
                continue

            if record is not None:
                record["label"] = label
                record["category"] = category
                records.append(record)
                cat_parsed += 1

        print(f"  Successfully parsed {cat_parsed} emails from {category}.")

    # Build DataFrame
    df = pd.DataFrame(records, columns=["text", "label", "category", "subject", "body"])
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"\nDataset saved successfully to: {output_csv_path} (Total records: {len(df)})")

    # Log skipped and errors
    if skipped_files:
        print(f"\nSkipped Non-Email / Metadata Files ({len(skipped_files)}):")
        for cat, fname, reason in skipped_files:
            print(f"  - [{cat}] {fname}: {reason}")

    if parsing_errors:
        print(f"\nParsing Errors Logged ({len(parsing_errors)}):")
        for cat, fname, err in parsing_errors:
            print(f"  - [{cat}] {fname}: {err}")
    else:
        print("\nParsing Errors Logged: 0 (No file parsing errors encountered)")

    return df, skipped_files, parsing_errors


def validate_dataset(csv_path: Path, skipped_count: int, error_count: int):
    """Reload and perform integrity and validation checks."""
    print("\n" + "=" * 60)
    print("TASK 4: INTEGRITY AND VALIDATION CHECKS")
    print("=" * 60)

    df = pd.read_csv(csv_path, keep_default_na=False)

    print(f"1. Serialization Check: Successfully reloaded {csv_path} using Pandas.")
    print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")

    # Category Breakdown
    cat_counts = df["category"].value_counts()
    print("\n2. Category Breakdown:")
    for cat in ["easy_ham", "hard_ham", "spam"]:
        count = cat_counts.get(cat, 0)
        pct = (count / len(df)) * 100
        print(f"   - {cat:<10}: {count:>5} messages ({pct:>5.2f}%)")

    # Class Balance
    label_counts = df["label"].value_counts()
    not_spam_cnt = label_counts.get(0, 0)
    spam_cnt = label_counts.get(1, 0)
    print("\n3. Overall Class Balance:")
    print(f"   - Not Spam (0): {not_spam_cnt} ({not_spam_cnt / len(df) * 100:.2f}%)")
    print(f"   - Spam (1)    : {spam_cnt} ({spam_cnt / len(df) * 100:.2f}%)")

    # Data Quality Metrics
    empty_text_mask = df["text"].astype(str).str.strip() == ""
    empty_text_count = empty_text_mask.sum()

    # Exact duplicates: distinguish empty text from non-empty duplicated content
    non_empty_df = df[~empty_text_mask]
    exact_duplicates_count = non_empty_df.duplicated(subset=["text"], keep="first").sum()

    print("\n4. Data Quality Metrics:")
    print(f"   - Empty or null 'text' records: {empty_text_count}")
    print(f"   - Exact duplicate emails (non-empty): {exact_duplicates_count}")
    print(f"   - Skipped non-email files: {skipped_count}")
    print(f"   - Total parsing errors / unreadable files: {error_count}")

    # Label Mapping Verification
    label_0_cats = set(df[df["label"] == 0]["category"].unique())
    label_1_cats = set(df[df["label"] == 1]["category"].unique())
    assert label_0_cats == {"easy_ham", "hard_ham"}, f"Unexpected categories for label 0: {label_0_cats}"
    assert label_1_cats == {"spam"}, f"Unexpected categories for label 1: {label_1_cats}"
    print("\n5. Label Mapping Verification: STRICT PASS")
    print(f"   - Label 0 strictly maps to: {sorted(list(label_0_cats))}")
    print(f"   - Label 1 strictly maps to: {sorted(list(label_1_cats))}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS TABLE")
    print("=" * 60)
    summary_df = pd.DataFrame([
        {"Metric": "easy_ham Messages", "Value": cat_counts.get("easy_ham", 0)},
        {"Metric": "hard_ham Messages", "Value": cat_counts.get("hard_ham", 0)},
        {"Metric": "spam Messages", "Value": cat_counts.get("spam", 0)},
        {"Metric": "Total Corpus Messages", "Value": len(df)},
        {"Metric": "Not Spam (0) Ratio", "Value": f"{not_spam_cnt / len(df) * 100:.2f}% ({not_spam_cnt})"},
        {"Metric": "Spam (1) Ratio", "Value": f"{spam_cnt / len(df) * 100:.2f}% ({spam_cnt})"},
        {"Metric": "Empty 'text' Records", "Value": empty_text_count},
        {"Metric": "Exact Duplicate Emails", "Value": exact_duplicates_count},
        {"Metric": "Skipped Metadata Files", "Value": skipped_count},
        {"Metric": "File Parsing Errors", "Value": error_count},
    ])
    print(summary_df.to_string(index=False))
    print("=" * 60)


def main():
    archives_dir = PROJECT_ROOT / "dataset" / "archives"
    extracted_base_dir = archives_dir / "extracted"
    output_csv = PROJECT_ROOT / "dataset" / "spam_assassin.csv"

    archive_paths = download_archives(archives_dir)
    category_dirs = extract_archives(archive_paths, extracted_base_dir)
    df, skipped, errors = parse_and_compile_corpus(category_dirs, output_csv)
    validate_dataset(output_csv, len(skipped), len(errors))


if __name__ == "__main__":
    main()

