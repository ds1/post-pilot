# test_spacy.py
import numpy
print(f"NumPy version: {numpy.__version__}")

import spacy
print(f"Spacy version: {spacy.__version__}")

nlp = spacy.load("en_core_web_sm")
print("✓ Spacy loaded successfully!")