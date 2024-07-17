import spacy
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import openai
from api_init import load_settings, init_openai
import logging
from transformers import logging as transformers_logging

# Set logging level for transformers
transformers_logging.set_verbosity_error()

# Set up logging for your application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load settings
settings = load_settings()

# Initialize OpenAI
init_openai()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=100)

# Load a specific model for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
revision = "af0f99b"  # Specify the revision
model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_content(content):
    """
    Analyze the given content and return various linguistic features.
    """
    doc = nlp(content)
    
    analysis = {
        'word_count': len(doc),
        'sentence_count': len(list(doc.sents)),
        'entities': [(ent.text, ent.label_) for ent in doc.ents],
        'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
        'readability': textstat.flesch_reading_ease(content),
        'sentiment': get_sentiment(content),
        'top_keywords': get_top_keywords([content])
    }
    
    return analysis

def get_sentiment(text, max_length=512):
    """
    Perform sentiment analysis on the given text, handling long texts by chunking.
    """
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    results = sentiment_analyzer(chunks)
    
    # Aggregate results
    positive_score = sum(result['score'] for result in results if result['label'] == 'POSITIVE')
    negative_score = sum(result['score'] for result in results if result['label'] == 'NEGATIVE')
    
    if positive_score > negative_score:
        return {'label': 'POSITIVE', 'score': positive_score / len(chunks)}
    else:
        return {'label': 'NEGATIVE', 'score': negative_score / len(chunks)}

def get_top_keywords(texts, n=5):
    """
    Extract top n keywords from a list of texts using TF-IDF.
    """
    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()
    
    top_keywords = []
    for i in range(len(texts)):
        feature_index = tfidf_matrix[i,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        top_keywords.append([feature_names[i] for i, score in sorted_scores[:n]])
    
    return top_keywords[0] if len(texts) == 1 else top_keywords

def generate_content(prompt, max_tokens=100):
    """
    Generate content using OpenAI's GPT model.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

def generate_ab_variant(content, insight):
    """
    Generate an A/B test variant based on the original content and an insight.
    """
    prompt = f"""
    Original content: "{content}"
    
    Insight for improvement: {insight}
    
    Create a variation of this content that incorporates the insight while maintaining the original message. 
    The variation should be noticeably different but not drastically so. 
    Ensure the content is suitable for social media.
    """
    
    return generate_content(prompt, max_tokens=200)

def summarize_text(text, max_length=100):
    """
    Summarize the given text to a specified maximum length.
    """
    prompt = f"Summarize the following text in no more than {max_length} characters:\n\n{text}"
    return generate_content(prompt, max_tokens=max_length // 4)  # Assuming average token is 4 characters

def generate_hashtags(text, num_hashtags=3):
    """
    Generate relevant hashtags for the given text.
    """
    prompt = f"Generate {num_hashtags} relevant hashtags for the following text:\n\n{text}"
    hashtags = generate_content(prompt, max_tokens=50)
    return [tag.strip() for tag in hashtags.split() if tag.startswith('#')]

def generate_content_suggestions(content_calendar):
    """
    Generate content suggestions based on the content calendar's past performance.
    """
    top_posts = content_calendar.get_top_performing_posts(n=5)
    suggestions = []

    for post in top_posts:
        analysis = analyze_content(post['content'])
        
        suggestion = {
            'platform': post['platform'],
            'original_content': post['content'],
            'engagement_score': post['engagement_score'],
            'insights': []
        }

        # Add insights based on the analysis
        if analysis['sentiment']['label'] == 'POSITIVE':
            suggestion['insights'].append("Use positive sentiment in your content")
        
        if analysis['readability'] > 70:
            suggestion['insights'].append("Maintain high readability scores")
        
        if len(analysis['entities']) > 3:
            suggestion['insights'].append("Include multiple relevant entities in your content")
        
        top_keywords = ', '.join(analysis['top_keywords'][:3])
        suggestion['insights'].append(f"Consider using these keywords: {top_keywords}")

        suggestions.append(suggestion)

    return suggestions

def apply_insights_to_future_content(content_calendar, suggestions):
    """
    Apply insights from suggestions to future content in the calendar.
    """
    future_posts = content_calendar.get_future_posts()
    
    for post in future_posts:
        relevant_suggestions = [s for s in suggestions if s['platform'] == post['platform']]
        if relevant_suggestions:
            # Choose a random suggestion to apply
            suggestion = random.choice(relevant_suggestions)
            
            # Apply insights
            new_content = post['content']
            for insight in suggestion['insights']:
                if "sentiment" in insight.lower():
                    new_content = improve_sentiment(new_content)
                elif "readability" in insight.lower():
                    new_content = improve_readability(new_content)
                elif "keywords" in insight.lower():
                    new_content = incorporate_keywords(new_content, suggestion['top_keywords'])
            
            # Update the post with new content
            content_calendar.update_post(post['index'], content=new_content)

def improve_sentiment(content):
    # Placeholder function to improve sentiment
    return content + " 😊"

def improve_readability(content):
    # Placeholder function to improve readability
    sentences = content.split('.')
    return '. '.join([s.strip() for s in sentences if s.strip()])

def incorporate_keywords(content, keywords):
    # Placeholder function to incorporate keywords
    return content + f" #{' #'.join(keywords)}"