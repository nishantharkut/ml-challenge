"""
Safe Text Normalization, Script Preservation, Suffix Standardization, and Address Parsing.
Protects Indic vowel marks (Mn/Mc) under NFKC and normalizes multi-language business data.
"""
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass


_MISSING_TEXT = frozenset({'', 'null', '<null>', 'none', 'nan'})


@dataclass(frozen=True, slots=True)
class TextViews:
    """Immutable deterministic views derived from one raw text value."""

    normalized: str
    folded: str
    compact: str
    tokens: tuple[str, ...]
    numeric_tokens: frozenset[str]

    @property
    def core(self) -> str:
        """Compatibility alias for the whitespace-free compact view."""
        return self.compact

# Comprehensive legal designations (English, Indian, French, Global)
LEGAL_SUFFIX_MAP = {
    'pvt': 'private', 'pvtltd': 'private limited', 'ltd': 'limited',
    'llp': 'llp', 'inc': 'incorporated', 'corp': 'corporation',
    'co': 'company', 'llc': 'llc', 'plc': 'plc', 'sa': 'sa',
    'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sasu', 'gmbh': 'gmbh',
    'ag': 'ag', 'nv': 'nv', 'bv': 'bv', 'pty': 'proprietary',
    'pte': 'private', 'intl': 'international', 'svc': 'services',
    'svcs': 'services', 'mfg': 'manufacturing', 'assoc': 'associates',
    'bros': 'brothers', 'dept': 'department', 'enterprises': 'enterprises',
    'enterprise': 'enterprises', 'private': 'private', 'limited': 'limited',
    'incorporated': 'incorporated', 'corporation': 'corporation',
    'company': 'company'
}

def detect_script(text):
    """Detect dominant script of text."""
    if not text:
        return 'Latin'
    scripts = defaultdict(int)
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, '')
            if 'DEVANAGARI' in name: scripts['Devanagari'] += 1
            elif 'TAMIL' in name: scripts['Tamil'] += 1
            elif 'TELUGU' in name: scripts['Telugu'] += 1
            elif 'KANNADA' in name: scripts['Kannada'] += 1
            elif 'BENGALI' in name: scripts['Bengali'] += 1
            elif 'GUJARATI' in name: scripts['Gujarati'] += 1
            elif 'MALAYALAM' in name: scripts['Malayalam'] += 1
            elif 'GURMUKHI' in name: scripts['Gurmukhi'] += 1
            elif 'ORIYA' in name: scripts['Odia'] += 1
            else: scripts['Latin'] += 1
    if not scripts:
        return 'Latin'
    return max(scripts, key=scripts.get)

def _is_missing_text(text: object) -> bool:
    if text is None:
        return True
    return str(text).strip().casefold() in _MISSING_TEXT


def _normalize_text(text: object) -> str:
    if _is_missing_text(text):
        return ''

    # NFKC handles compatibility characters; casefold is deterministic and
    # stronger than lower() for multilingual comparisons (for example, ß -> ss).
    text = unicodedata.normalize('NFKC', str(text)).casefold()

    cleaned_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        # L*: letters, N*: numbers, Mn/Mc: Indic and other meaningful marks.
        if cat.startswith('L') or cat.startswith('N') or cat in ('Mn', 'Mc'):
            cleaned_chars.append(ch)
        elif cat.startswith('Z') or ch in (' ', '\t', '\n', ',', '-', '/', '.', '&'):
            cleaned_chars.append(' ')

    return re.sub(r'\s+', ' ', ''.join(cleaned_chars)).strip()


def _fold_latin_accents(text: str) -> str:
    """Remove marks attached to Latin letters without damaging other scripts."""
    folded = []
    latin_base = False
    for ch in unicodedata.normalize('NFD', text):
        category = unicodedata.category(ch)
        if category.startswith('M'):
            if not latin_base:
                folded.append(ch)
            continue

        folded.append(ch)
        latin_base = category.startswith('L') and 'LATIN' in unicodedata.name(ch, '')

    return unicodedata.normalize('NFC', ''.join(folded))


def text_views(text: object) -> TextViews:
    """Return immutable normalized, folded, compact, token, and number views."""
    normalized = _normalize_text(text)
    if not normalized:
        return TextViews('', '', '', (), frozenset())

    tokens = tuple(normalized.split())
    return TextViews(
        normalized=normalized,
        folded=_fold_latin_accents(normalized),
        compact=''.join(tokens),
        tokens=tokens,
        numeric_tokens=frozenset(re.findall(r'\d+', normalized)),
    )


def numeric_tokens(text: object) -> frozenset[str]:
    """Return the immutable set of numeric tokens in ``text``."""
    return text_views(text).numeric_tokens


def normalize_text_safe(text: object) -> str:
    """
    Script-safe normalization using Unicode NFKC.
    Preserves Indic vowel modifiers (Unicode categories Mn, Mc).
    Strips noise, punctuation, and standardizes spacing.
    """
    return text_views(text).normalized

def extract_core_and_suffix(name_tokens):
    """Separate business name into core name and standardized legal suffix."""
    if not name_tokens:
        return "", ""
    core = list(name_tokens)
    suffix_parts = []
    # Check trailing tokens
    for _ in range(min(3, len(core))):
        last_tok = core[-1]
        std_suffix = LEGAL_SUFFIX_MAP.get(last_tok, None)
        if std_suffix:
            suffix_parts.insert(0, std_suffix)
            core.pop()
        else:
            break
            
    core_str = " ".join(core) if core else " ".join(name_tokens)
    suffix_str = " ".join(suffix_parts)
    return core_str, suffix_str

def parse_structured_address(addr_raw, addr_norm):
    """
    Extract numbers, postal codes, and indicators from address.
    """
    if not addr_norm:
        return {
            'is_empty': True,
            'numbers': set(),
            'postal_code': None,
            'has_null_literal': ('null' in str(addr_raw).lower() if addr_raw else False)
        }
        
    # Extract numbers (premise, street numbers, door numbers)
    raw_numbers = re.findall(r'\b\d+[-/]?\w*\b', addr_norm)
    numbers = set(re.findall(r'\b\d+\b', addr_norm))
    
    # Postal code heuristics: 5 digits (US/France) or 6 digits (India)
    postal_match = re.search(r'\b\d{5,6}\b', addr_norm)
    postal_code = postal_match.group(0) if postal_match else None
    
    return {
        'is_empty': False,
        'numbers': numbers,
        'postal_code': postal_code,
        'has_null_literal': ('null' in str(addr_raw).lower() if addr_raw else False)
    }

_TRANSLIT_ENGINE = None

def get_translit_engine():
    global _TRANSLIT_ENGINE
    if _TRANSLIT_ENGINE is None:
        try:
            from .transliteration import TransliterationEngine
            _TRANSLIT_ENGINE = TransliterationEngine()
        except Exception:
            _TRANSLIT_ENGINE = None
    return _TRANSLIT_ENGINE

def preprocess_record(record):
    """
    Multi-view preprocessing for a single record row dict.
    Returns structured record dictionary.
    """
    eid = record.get('entity_id', '') or ''
    bname = record.get('business_name', '') or ''
    baddr = record.get('business_address', '') or ''
    country = (record.get('country', '') or '').strip().upper()
    
    name_views = text_views(bname)
    addr_views = text_views(baddr)
    name_norm = name_views.normalized
    addr_norm = addr_views.normalized
    
    name_tokens = name_norm.split()
    addr_tokens = addr_norm.split()
    
    core_name, suffix = extract_core_and_suffix(name_tokens)
    script = detect_script(bname)
    addr_struct = parse_structured_address(baddr, addr_norm)
    is_indic = (script != 'Latin')
    
    name_translit = ""
    if is_indic:
        engine = get_translit_engine()
        if engine:
            name_translit = engine.transliterate_text(name_norm, is_indic=True)
    
    return {
        'entity_id': eid,
        'country': country,
        'name_raw': bname,
        'name_norm': name_norm,
        'name_folded': name_views.folded,
        'name_compact': name_views.compact,
        'name_numeric_tokens': name_views.numeric_tokens,
        'name_core': core_name,
        'name_suffix': suffix,
        'name_tokens': name_tokens,
        'name_translit': name_translit,
        'addr_raw': baddr,
        'addr_norm': addr_norm,
        'addr_folded': addr_views.folded,
        'addr_compact': addr_views.compact,
        'addr_numeric_tokens': addr_views.numeric_tokens,
        'addr_tokens': addr_tokens,
        'addr_empty': addr_struct['is_empty'],
        'addr_numbers': addr_struct['numbers'],
        'addr_postal': addr_struct['postal_code'],
        'script': script,
        'is_indic': is_indic
    }
