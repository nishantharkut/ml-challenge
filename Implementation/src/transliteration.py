"""
Transliteration and Cross-Script Alignment Module.
Provides auto-mined GT dictionary translation and fallback phonetic romanization.
"""
import os
import json
from collections import defaultdict
from .config import CHECKPOINT_DIR

try:
    import anyascii
    HAS_ANYASCII = True
except ImportError:
    HAS_ANYASCII = False

class TransliterationEngine:
    def __init__(self, dictionary_path=None):
        self.dict_path = dictionary_path or os.path.join(CHECKPOINT_DIR, "mined_dictionary.json")
        self.translit_dict = {}
        self.load_dictionary()
        
    def load_dictionary(self):
        """Load mined dictionary if exists."""
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                self.translit_dict = json.load(f)
                
    def save_dictionary(self, dictionary):
        """Save mined dictionary atomically."""
        self.translit_dict = dictionary
        with open(self.dict_path + ".tmp", 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=2)
        if os.path.exists(self.dict_path):
            os.remove(self.dict_path)
        os.rename(self.dict_path + ".tmp", self.dict_path)
        
    def mine_from_pairs(self, cross_script_pairs, min_count=5, min_confidence=0.8):
        """
        Mine high-confidence word-level alignments from cross-script pairs.
        cross_script_pairs: list of tuples (latin_name_tokens, indic_name_tokens)
        """
        cooccur = defaultdict(lambda: defaultdict(int))
        indic_totals = defaultdict(int)
        
        for latin_tokens, indic_tokens in cross_script_pairs:
            l_set = set(latin_tokens)
            i_set = set(indic_tokens)
            for i_tok in i_set:
                indic_totals[i_tok] += 1
                for l_tok in l_set:
                    cooccur[i_tok][l_tok] += 1
                    
        mined = {}
        for i_tok, counts in cooccur.items():
            tot = indic_totals[i_tok]
            if tot < min_count:
                continue
            best_l, best_count = max(counts.items(), key=lambda x: x[1])
            conf = best_count / tot
            if conf >= min_confidence:
                mined[i_tok] = best_l
                
        self.save_dictionary(mined)
        return mined

    def transliterate_tokens(self, tokens):
        """
        Translate tokens using dictionary first, anyascii as fallback.
        Returns list of Latin tokens.
        """
        out = []
        for t in tokens:
            if t in self.translit_dict:
                out.append(self.translit_dict[t])
            elif HAS_ANYASCII:
                out.append(anyascii.anyascii(t).lower())
            else:
                out.append(t)
        return out
        
    def transliterate_text(self, text, is_indic=False):
        """Transliterate text string to Latin characters."""
        if not is_indic or not text:
            return text
        tokens = text.split()
        trans_tokens = self.transliterate_tokens(tokens)
        return " ".join(trans_tokens)
