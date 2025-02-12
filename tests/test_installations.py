# In a Python console or script
import openai
import spacy

# Check OpenAI version
print(f"OpenAI version: {openai.__version__}")  # Should show 0.28.0

# Check spaCy model
nlp = spacy.load("en_core_web_sm")
print(f"SpaCy model loaded: {nlp}")  # Should show that the model loaded successfully