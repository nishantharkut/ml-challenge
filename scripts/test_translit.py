import sys
sys.stdout.reconfigure(encoding='utf-8')
import anyascii
import unidecode

# Test cases from our actual dataset
test_cases = [
    ('रेड वेंचर्स प्राइवेट लिमिटेड', 'Red Ventures Private Limited', 'Hindi business name'),
    ('एसएस फूड प्राइवेट लिमिटेड', 'Ss Food Private Limited', 'Hindi business name'),
    ('ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி', 'Raj Investments LLP', 'Tamil business name'),
    ('బాలాజీ ఇన్వెస్ట్\u200cమెంట్ ప్రైవేట్ లిమిటెడ్', 'Balaji Investment Pvt Ltd', 'Telugu business name'),
    ('होटल एंटरप्राइजेज लिमिटेड', 'Hotel Enterprises Limited', 'Hindi business name'),
    ('स्वस्तिक ॐ सॉल्यूशंस एलएलपी', 'Swastik Om Solutions LLP', 'Hindi with Om symbol'),
    ('எல்எல்பி', 'LLP', 'Tamil acronym'),
    ('उत्तर प्रदेश', 'Uttar Pradesh', 'Hindi state'),
    ('महाराष्ट्र', 'Maharashtra', 'Hindi state'),
    ('हरियाणा', 'Haryana', 'Hindi state'),
    ('தமிழ்நாடு', 'Tamil Nadu', 'Tamil state'),
    ('Société Générale', 'Societe Generale', 'French'),
    ('Chordia &-Pártners Ltd', 'Chordia and-Partners Ltd', 'Mixed diacritics'),
]

print("TRANSLITERATION QUALITY TEST")
print("=" * 120)
for original, expected, desc in test_cases:
    aa = anyascii.anyascii(original)
    ud = unidecode.unidecode(original)
    
    # Check if transliterated version would match the expected via lowercase
    aa_lower = aa.lower().strip()
    expected_lower = expected.lower().strip()
    
    # Compute simple token overlap
    aa_tokens = set(aa_lower.split())
    exp_tokens = set(expected_lower.split())
    overlap = len(aa_tokens & exp_tokens)
    total = max(len(aa_tokens), len(exp_tokens), 1)
    
    print(f"\n[{desc}]")
    print(f"  Original:  {original}")
    print(f"  Expected:  {expected}")
    print(f"  anyascii:  {aa}")
    print(f"  unidecode: {ud}")
    print(f"  Token overlap (anyascii vs expected): {overlap}/{total} = {overlap/total:.0%}")
