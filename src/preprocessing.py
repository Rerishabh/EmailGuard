"""Email text extraction, cleaning, and feature engineering preprocessing module."""

import email
from email import policy
from email.message import EmailMessage
import os
from pathlib import Path
import re
import tarfile
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def is_attachment(part) -> bool:
    """Check if an email MIME part represents a file attachment.
    
    Identifies attachments via:
    1. Content-Disposition set to 'attachment'
    2. Presence of a filename parameter in Content-Disposition or Content-Type
    """
    try:
        disposition = part.get_content_disposition()
        if disposition == "attachment":
            return True
    except Exception:
        pass

    try:
        filename = part.get_filename()
        if filename:
            return True
    except Exception:
        pass

    return False


def decode_payload_bytes(payload: bytes) -> str:
    """Safely decode payload bytes using UTF-8, CP1252, Latin-1, or replacement fallback."""
    if not payload:
        return ""
    for enc in ["utf-8", "cp1252", "latin-1"]:
        try:
            return payload.decode(enc)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def extract_plain_text_from_html(html_str: str) -> str:
    """Fallback utility to strip HTML tags and extract readable text when no text/plain part exists."""
    # Remove script and style elements
    clean = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", html_str)
    # Replace line break and paragraph tags with newlines
    clean = re.sub(r"(?i)<(br|p|div|tr)[^>]*>", "\n", clean)
    # Strip remaining HTML tags
    clean = re.sub(r"<[^>]+>", " ", clean)
    # Unescape common HTML entities
    clean = (
        clean.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    # Normalize multiple whitespace within lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in clean.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def extract_email_content(msg: EmailMessage) -> Tuple[str, str]:
    """Extract Subject and Body from a parsed EmailMessage.
    
    MIME Extraction Rules:
    - Extract only 'text/plain' MIME parts.
    - Exclude MIME parts with Content-Disposition: attachment.
    - Exclude parts with attachment filenames.
    - For multipart/alternative messages, extract the text/plain representation without duplicating content.
    - Do not include HTML markup/tags when a plain-text version is available.
    - If no text/plain parts exist, fall back to stripping text from text/html parts.
    - Safely drop non-text binary attachments, base64 blobs, image payloads, and executable content.
    - Decode content safely using UTF-8 with fallback decoding (latin-1, cp1252, or errors='replace').
    """
    # 1. Extract Subject safely
    subject = ""
    try:
        raw_subject = msg.get("Subject", "")
        if raw_subject is not None:
            subject = str(raw_subject).strip()
    except Exception:
        subject = ""

    # 2. Extract Body
    text_plain_parts: List[str] = []
    text_html_parts: List[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            # Skip container multipart structures
            if part.is_multipart():
                continue

            # Exclude attachments
            if is_attachment(part):
                continue

            content_type = part.get_content_type()

            # Process text/plain parts
            if content_type == "text/plain":
                part_text = ""
                try:
                    # Attempt standard email policy decoding
                    content = part.get_content()
                    if isinstance(content, str):
                        part_text = content
                    elif isinstance(content, bytes):
                        part_text = decode_payload_bytes(content)
                except Exception:
                    # Fallback to raw decoded payload
                    raw_payload = part.get_payload(decode=True)
                    if isinstance(raw_payload, bytes):
                        part_text = decode_payload_bytes(raw_payload)
                    elif isinstance(raw_payload, str):
                        part_text = raw_payload

                if part_text and part_text.strip():
                    text_plain_parts.append(part_text.strip())

            # Collect text/html parts as a contingency if NO text/plain exists
            elif content_type == "text/html":
                html_text = ""
                try:
                    content = part.get_content()
                    if isinstance(content, str):
                        html_text = content
                    elif isinstance(content, bytes):
                        html_text = decode_payload_bytes(content)
                except Exception:
                    raw_payload = part.get_payload(decode=True)
                    if isinstance(raw_payload, bytes):
                        html_text = decode_payload_bytes(raw_payload)
                    elif isinstance(raw_payload, str):
                        html_text = raw_payload

                if html_text and html_text.strip():
                    text_html_parts.append(html_text)

    else:
        # Non-multipart message
        content_type = msg.get_content_type()
        if not is_attachment(msg):
            if content_type == "text/plain":
                try:
                    content = msg.get_content()
                    if isinstance(content, str):
                        text_plain_parts.append(content.strip())
                    elif isinstance(content, bytes):
                        text_plain_parts.append(decode_payload_bytes(content).strip())
                except Exception:
                    raw_payload = msg.get_payload(decode=True)
                    if isinstance(raw_payload, bytes):
                        text_plain_parts.append(decode_payload_bytes(raw_payload).strip())
                    elif isinstance(raw_payload, str):
                        text_plain_parts.append(raw_payload.strip())
            elif content_type == "text/html":
                try:
                    content = msg.get_content()
                    raw = content if isinstance(content, str) else decode_payload_bytes(content)
                except Exception:
                    raw_payload = msg.get_payload(decode=True)
                    raw = decode_payload_bytes(raw_payload) if isinstance(raw_payload, bytes) else str(raw_payload or "")
                if raw and raw.strip():
                    text_html_parts.append(raw)

    # Prioritize text/plain parts. If present, HTML is completely excluded.
    if text_plain_parts:
        body = "\n\n".join(text_plain_parts).strip()
    elif text_html_parts:
        # Fallback only when no text/plain version was provided in the message
        stripped_parts = [extract_plain_text_from_html(h) for h in text_html_parts]
        body = "\n\n".join(p for p in stripped_parts if p).strip()
    else:
        body = ""

    return subject, body


def parse_email_file(file_path: Path) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Parse a single raw email file on disk using BytesParser with default policy.
    
    Returns:
        (record_dict, error_message)
    """
    # Skip non-email metadata files
    if file_path.name in ["cmds", ".DS_Store", "Thumbs.db"] or file_path.name.startswith("."):
        return None, f"Skipped metadata file: {file_path.name}"

    try:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        if not raw_bytes:
            return None, f"Skipped empty file: {file_path.name}"

        msg = email.parser.BytesParser(policy=policy.default).parsebytes(raw_bytes)
        subject, body = extract_email_content(msg)
        text = f"{subject}\n\n{body}".strip()

        return {
            "subject": subject,
            "body": body,
            "text": text,
        }, None

    except Exception as e:
        return None, f"Parsing error in {file_path.name}: {str(e)}"


def clean_dataset(input_csv_path: Path, output_csv_path: Path) -> pd.DataFrame:
    """Read raw email CSV, validate schema, deduplicate on full 'text', and persist cleaned CSV.
    
    Guarantees:
    - input_csv_path and output_csv_path are converted to Path objects.
    - Resolves and verifies paths to ensure they do not target the same file (collision guard).
    - Validates presence and ordering of exact five columns: ['text', 'label', 'category', 'subject', 'body'].
    - Deduplication evaluates the full 'text' column, keeping first occurrences (keep='first').
    - Preserves literal string representation (keep_default_na=False) without text mutations.
    - Persists cleaned DataFrame to output_csv_path and returns the resulting DataFrame.
    """
    input_csv_path = Path(input_csv_path)
    output_csv_path = Path(output_csv_path)

    if input_csv_path.resolve() == output_csv_path.resolve():
        raise ValueError(
            f"input_csv_path and output_csv_path must not resolve to the same file: {input_csv_path}"
        )

    df = pd.read_csv(input_csv_path, keep_default_na=False)
    expected_cols = ["text", "label", "category", "subject", "body"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Schema mismatch: expected {expected_cols}, got {list(df.columns)}")

    # Deduplicate strictly on full text
    df_cleaned = df.drop_duplicates(subset=["text"], keep="first").copy()

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_csv_path, index=False, encoding="utf-8")
    return df_cleaned


# =====================================================================
# Scikit-Learn Compatible Preprocessing Transformers (Stage 4)
# =====================================================================

def _validate_1d_input(X) -> List[str]:
    """Validate that input is a 1D sequence (list, Series, or 1D ndarray).
    
    Handles missing (None, NaN) values by converting them to empty strings.
    Raises ValueError on 2D arrays, DataFrames, or nested sequences.
    """
    if isinstance(X, pd.DataFrame):
        raise ValueError("Expected 1D array-like input, got DataFrame.")
    if hasattr(X, "ndim") and X.ndim > 1:
        raise ValueError(f"Expected 1D array-like input, got {X.ndim}D array.")
    if isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], (list, tuple, np.ndarray, pd.Series)):
        raise ValueError("Expected 1D array-like input, got 2D sequence.")
    
    out: List[str] = []
    for item in X:
        if item is None:
            out.append("")
        elif isinstance(item, float) and np.isnan(item):
            out.append("")
        else:
            out.append(str(item))
    return out


def _get_nltk_stopwords() -> set:
    """Retrieve NLTK English stopwords set offline with defensive validation.
    
    Raises LookupError if the corpus is not present on disk, ensuring no network calls.
    """
    try:
        words = stopwords.words("english")
        return set(words)
    except LookupError as e:
        raise LookupError(
            "NLTK English stopwords resource is missing and offline execution is required. "
            "Please ensure the 'stopwords' corpus is available in your NLTK data directory."
        ) from e


def _normalize_minimal(text: str) -> str:
    """Pipeline A text normalization preserving casing, punctuation, symbols, and URLs."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _normalize_standard_nlp(text: str, stop_words: set, stemmer: PorterStemmer) -> str:
    """Pipeline B classical NLP normalization: lowercase, strip punctuation/digits, filter stopwords, stem."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = text.split()
    filtered_tokens = [
        stemmer.stem(t) for t in tokens if not t.isdigit() and t not in stop_words and len(t) > 0
    ]
    return " ".join(filtered_tokens)


def _normalize_domain_aware(text: str) -> str:
    """Pipeline C lexical normalization: maps domain entities to deterministic placeholder tokens."""
    if not text:
        return ""
    # 1. Full currency expressions ($500, € 12.50, 100 dollars, 50 USD)
    text = re.sub(
        r"[$€£¥]\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*(?:dollars?|cents?|usd|gbp|eur)\b",
        " currencyplaceholder ",
        text,
        flags=re.IGNORECASE,
    )
    # 2. Standalone currency symbols
    text = re.sub(r"[$€£¥]", " currencyplaceholder ", text)
    # 3. URLs
    text = re.sub(r"https?://\S+|www\.\S+", " urlplaceholder ", text)
    # 4. Email addresses
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", " emailplaceholder ", text)
    # 5. Remaining standalone numeric sequences
    text = re.sub(r"\b\d+(?:[.,]\d+)?\b", " numberplaceholder ", text)
    # 6. Exclamation bursts (2 or more consecutive exclamation marks)
    text = re.sub(r"!{2,}", " exclamationburstplaceholder ", text)
    # Normalize erratic whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_heuristics_single(text: str) -> List[float]:
    """Extract exactly 6 audited structural heuristic features from an individual email string."""
    if not text:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    alpha_count = sum(1 for c in text if c.isalpha())
    uppercase_ratio = sum(1 for c in text if c.isupper()) / float(alpha_count) if alpha_count > 0 else 0.0
    all_caps_word_count = float(len(re.findall(r"\b[A-Z]{3,}\b", text)))
    exclamation_count = float(text.count("!"))
    question_count = float(text.count("?"))
    currency_symbol_count = float(len(re.findall(r"[$€£¥]", text)))
    url_count = float(len(re.findall(r"https?://\S+|www\.\S+", text)))

    return [
        uppercase_ratio,
        all_caps_word_count,
        exclamation_count,
        question_count,
        currency_symbol_count,
        url_count,
    ]


class MinimalTextTransformer(BaseEstimator, TransformerMixin):
    """Pipeline A text normalizer preserving casing, URLs, punctuation, and symbols.
    
    Normalizes excessive whitespace (\r\n -> \n, collapses tabs and multiple spaces to a
    single space, and strips boundary whitespace) while retaining raw textual characteristics.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        _validate_1d_input(X)
        return self

    def transform(self, X):
        X_clean = _validate_1d_input(X)
        transformed = [_normalize_minimal(text) for text in X_clean]
        return np.array(transformed, dtype=object)


class StandardNLPTextTransformer(BaseEstimator, TransformerMixin):
    """Pipeline B text preprocessor applying classical NLP transformations.
    
    Transforms raw email strings by:
    1. Converting text to lowercase.
    2. Replacing punctuation and special characters with spaces via regex.
    3. Splitting into tokens and removing standalone numeric tokens.
    4. Filtering out English stopwords using NLTK.
    5. Applying Porter stemming to remaining tokens.
    6. Returning space-separated stemmed strings.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        _validate_1d_input(X)
        return self

    def transform(self, X):
        X_clean = _validate_1d_input(X)
        stop_words = _get_nltk_stopwords()
        stemmer = PorterStemmer()
        transformed = [_normalize_standard_nlp(text, stop_words, stemmer) for text in X_clean]
        return np.array(transformed, dtype=object)


class DomainAwareTextTransformer(BaseEstimator, TransformerMixin):
    """Pipeline C lexical preprocessor mapping spam-indicative entities to placeholder tokens.
    
    Normalizes entities in strict deterministic order:
    1. Full currency expressions -> currencyplaceholder
    2. Standalone currency symbols -> currencyplaceholder
    3. URLs -> urlplaceholder
    4. Email addresses -> emailplaceholder
    5. Remaining standalone numeric sequences -> numberplaceholder
    6. Exclamation bursts (2+ !) -> exclamationburstplaceholder
    
    Preserves casing and surrounding words while normalizing erratic whitespace.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        _validate_1d_input(X)
        return self

    def transform(self, X):
        X_clean = _validate_1d_input(X)
        transformed = [_normalize_domain_aware(text) for text in X_clean]
        return np.array(transformed, dtype=object)


class EmailHeuristicFeatures(BaseEstimator, TransformerMixin):
    """Pipeline C dense transformer extracting 6 structural and statistical heuristic features.
    
    Extracts from raw email text:
    1. uppercase_ratio: Ratio of uppercase letters to total alphabetic characters.
    2. all_caps_word_count: Count of words with >=3 uppercase letters.
    3. exclamation_count: Total occurrences of '!'.
    4. question_count: Total occurrences of '?'.
    5. currency_symbol_count: Total occurrences of currency symbols ($ € £ ¥).
    6. url_count: Total count of explicit URLs.
    
    Outputs a 2D NumPy array of shape (N, 6) with float64 values.
    """

    FEATURE_NAMES = [
        "uppercase_ratio",
        "all_caps_word_count",
        "exclamation_count",
        "question_count",
        "currency_symbol_count",
        "url_count",
    ]

    def __init__(self):
        pass

    def fit(self, X, y=None):
        _validate_1d_input(X)
        return self

    def transform(self, X):
        X_clean = _validate_1d_input(X)
        features = np.zeros((len(X_clean), 6), dtype=np.float64)
        for i, text in enumerate(X_clean):
            features[i] = _extract_heuristics_single(text)
        return features

    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURE_NAMES, dtype=object)

