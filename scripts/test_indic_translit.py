import sys
sys.stdout.reconfigure(encoding='utf-8')
from indic_transliteration import sanscript

tests = [
    ('प्राइवेट लिमिटेड', sanscript.DEVANAGARI, 'Private Limited'),
    ('रेड वेंचर्स', sanscript.DEVANAGARI, 'Red Ventures'),
    ('एसएस फूड', sanscript.DEVANAGARI, 'Ss Food'),
    ('महाराष्ट्र', sanscript.DEVANAGARI, 'Maharashtra'),
    ('हरियाणा', sanscript.DEVANAGARI, 'Haryana'),
    ('उत्तर प्रदेश', sanscript.DEVANAGARI, 'Uttar Pradesh'),
    ('होटल एंटरप्राइजेज', sanscript.DEVANAGARI, 'Hotel Enterprises'),
    ('एलएलपी', sanscript.DEVANAGARI, 'LLP'),
    ('स्वस्तिक', sanscript.DEVANAGARI, 'Swastik'),
    ('सॉल्यूशंस', sanscript.DEVANAGARI, 'Solutions'),
]

print("INDIC-TRANSLITERATION QUALITY TEST (Devanagari)")
print("=" * 110)
for text, script, expected in tests:
    itrans = sanscript.transliterate(text, script, sanscript.ITRANS)
    iast = sanscript.transliterate(text, script, sanscript.IAST)
    print(f"  {text:30s} ITRANS: {itrans:30s} IAST: {iast:30s} Expected: {expected}")

# Test Tamil
print("\nTamil tests:")
tamil_tests = [
    ('ராஜ்', sanscript.TAMIL, 'Raj'),
    ('இன்வெஸ்ட்மெண்ட்ஸ்', sanscript.TAMIL, 'Investments'),
    ('எல்எல்பி', sanscript.TAMIL, 'LLP'),
    ('தமிழ்நாடு', sanscript.TAMIL, 'Tamil Nadu'),
]
for text, script, expected in tamil_tests:
    itrans = sanscript.transliterate(text, script, sanscript.ITRANS)
    iast = sanscript.transliterate(text, script, sanscript.IAST)
    print(f"  {text:30s} ITRANS: {itrans:30s} IAST: {iast:30s} Expected: {expected}")

# Test Telugu
print("\nTelugu tests:")
telugu_tests = [
    ('బాలాజీ', sanscript.TELUGU, 'Balaji'),
    ('ప్రైవేట్', sanscript.TELUGU, 'Private'),
    ('లిమిటెడ్', sanscript.TELUGU, 'Limited'),
    ('తెలంగాణ', sanscript.TELUGU, 'Telangana'),
]
for text, script, expected in telugu_tests:
    itrans = sanscript.transliterate(text, script, sanscript.ITRANS)
    iast = sanscript.transliterate(text, script, sanscript.IAST)
    print(f"  {text:30s} ITRANS: {itrans:30s} IAST: {iast:30s} Expected: {expected}")
