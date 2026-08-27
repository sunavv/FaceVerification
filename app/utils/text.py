import re
import unicodedata
from typing import List, Optional, Tuple


def normalize_text(text: str) -> str:
    """
    Normalize text:
    - Unicode NFD decomposition and strip diacritical marks (accents)
    - Lowercase
    - Replace non-alphanumeric (except spaces) with space
    - Collapse multiple spaces into single space
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""
    # Strip combining diacritics (accents)
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    # Convert to lowercase
    text = text.lower()
    # Replace punctuation and special characters with spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compare_names(name1: Optional[str], name2: Optional[str], threshold: float = 0.85) -> Tuple[bool, float]:
    """
    Compare two names with multi-level matching:
    1. Exact normalized match
    2. Token set / subset match (e.g. 'JOHN DOE' vs 'DOE JOHN' or 'JOHN M DOE')
    3. Character sequence similarity / Levenshtein-like ratio
    
    Returns (is_match, similarity_score).
    """
    if not name1 or not name2:
        return False, 0.0

    n1 = normalize_text(name1)
    n2 = normalize_text(name2)

    if not n1 or not n2:
        return False, 0.0

    # 1. Exact normalized match
    if n1 == n2:
        return True, 1.0

    # 2. Token-level matching
    tokens1 = set(n1.split())
    tokens2 = set(n2.split())

    if tokens1 == tokens2 and len(tokens1) > 0:
        return True, 1.0

    # Subsets (e.g., first and last name match, middle name omitted)
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    jaccard = len(intersection) / max(len(union), 1)

    if len(intersection) >= min(len(tokens1), len(tokens2)) and len(intersection) >= 2:
        # High confidence subset match
        return True, max(jaccard, 0.9)

    # 3. Normalized string similarity (Levenshtein-based ratio)
    char_sim = compute_string_similarity(n1, n2)
    is_match = char_sim >= threshold or jaccard >= threshold

    return is_match, float(max(char_sim, jaccard))


def compute_string_similarity(s1: str, s2: str) -> float:
    """Compute character-level similarity between two strings."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    dist = dp[len1][len2]
    max_len = max(len1, len2)
    return 1.0 - (dist / max_len)


def extract_candidate_name_from_lines(lines: List[str]) -> Optional[str]:
    """
    Heuristic name extractor from OCR detected lines.
    Looks for labels such as 'Name:', 'Full Name:', 'Given Name:', 'Holder:', or
    identifies capitalized name patterns while skipping common ID keywords.
    """
    if not lines:
        return None

    # Blacklist keywords that shouldn't be names
    blacklist = {
        "identity", "card", "republic", "driving", "license", "passport",
        "national", "date", "birth", "sex", "gender", "address", "signature",
        "citizenship", "country", "expiry", "issue", "number", "id", "authority",
        "department", "ministry", "government", "state", "federal", "male", "female"
    }

    # Pass 1: Look for explicit field labels
    label_patterns = [
        r"(?:name|full\s*name|given\s*name|surname|holder|holder\'s\s*name)\s*[:\-\. ]+\s*([a-zA-Z\s\.\-]{2,40})",
        r"(?:name|full\s*name)\s*[\n\r]*([a-zA-Z\s\.\-]{2,40})",
    ]

    for line in lines:
        clean_line = line.strip()
        for pat in label_patterns:
            match = re.search(pat, clean_line, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                # Clean up candidate
                candidate = re.sub(r"[^a-zA-Z\s\.\-]", "", candidate).strip()
                if len(candidate) >= 3 and not any(kw in candidate.lower().split() for kw in blacklist):
                    return candidate.upper()

    # Pass 2: Look for line right after a label like 'NAME' or 'FULL NAME'
    for i, line in enumerate(lines):
        norm = normalize_text(line)
        if norm in {"name", "full name", "given name", "holder name"} and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            cand = re.sub(r"[^a-zA-Z\s\.\-]", "", next_line).strip()
            if len(cand) >= 3 and not any(kw in cand.lower().split() for kw in blacklist):
                return cand.upper()

    # Pass 3: Search for prominent capitalized alphabetic lines (e.g. 'JOHN DOE')
    for line in lines:
        candidate = line.strip()
        # Must be mostly uppercase letters and spaces
        alpha_only = re.sub(r"[^a-zA-Z\s]", "", candidate).strip()
        words = alpha_only.split()
        
        if 2 <= len(words) <= 4:
            if all(len(w) >= 2 for w in words):
                lower_words = set(w.lower() for w in words)
                if not lower_words.intersection(blacklist):
                    # Check uppercase ratio
                    if sum(1 for c in alpha_only if c.isupper()) / max(len(alpha_only), 1) > 0.6:
                        return alpha_only.upper()

    # Fallback: Pick first non-blacklisted alphabetic phrase with 2+ words
    for line in lines:
        alpha_only = re.sub(r"[^a-zA-Z\s]", "", line).strip()
        words = alpha_only.split()
        if 2 <= len(words) <= 4 and all(len(w) >= 2 for w in words):
            if not set(w.lower() for w in words).intersection(blacklist):
                return alpha_only.upper()

    return None
