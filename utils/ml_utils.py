import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import logging
import random
from api_init import load_settings
from utils.nlp_utils import generate_ab_variant
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load a specific model for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Set up logging
logging.basicConfig(filename='logs/ml_utils.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load settings
settings = load_settings()
SAMPLE_SIZE_AB_TEST = settings.get('SAMPLE_SIZE_AB_TEST', 0.2)
SIGNIFICANCE_LEVEL = settings.get('SIGNIFICANCE_LEVEL', 0.05)

def extract_features(df):
    """
    Extract features from the content calendar dataframe.
    """
    # Text features
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    text_features = tfidf.fit_transform(df['content']).toarray()
    
    # Time features
    df['hour'] = pd.to_datetime(df['time_slot']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['due_date']).dt.dayofweek
    
    # Platform one-hot encoding
    platform_dummies = pd.get_dummies(df['platform'], prefix='platform')
    
    # Combine all features
    features = np.hstack((
        text_features,
        df[['hour', 'day_of_week']].values,
        platform_dummies.values
    ))
    
    feature_names = (
        [f'tfidf_{i}' for i in range(100)] +
        ['hour', 'day_of_week'] +
        list(platform_dummies.columns)
    )
    
    return features, feature_names

def train_model(X, y):
    """
    Train a Random Forest model to predict engagement scores.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f"Model trained. MSE: {mse:.4f}, R2: {r2:.4f}")
    
    return model, scaler

def predict_engagement(model, scaler, features):
    """
    Predict engagement score for new content.
    """
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)

def get_feature_importance(model, feature_names):
    """
    Get feature importance from the trained model.
    """
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    return feature_importance.sort_values('importance', ascending=False)

def create_ab_tests(calendar, suggestions):
    """
    Create A/B test variants based on suggestions.
    """
    future_posts = calendar.get_future_posts()
    posts_to_test = np.random.choice(future_posts, size=int(len(future_posts) * SAMPLE_SIZE_AB_TEST), replace=False)
    
    for post in posts_to_test:
        relevant_suggestions = [s for s in suggestions if s['platform'] == post['platform']]
        if relevant_suggestions:
            insight = np.random.choice(relevant_suggestions)['insight']
            variant_content = generate_ab_variant(post['content'], insight)
            calendar.add_ab_test(post['index'], variant_content)
            logging.info(f"Created A/B test variant for post {post['index']}")

def analyze_ab_test_results(calendar):
    """
    Analyze the results of A/B tests.
    """
    ab_tests = calendar.get_ab_test_results()
    results = []
    
    for test in ab_tests:
        original = test['original']
        variant = test['variant']
        
        t_statistic, p_value = stats.ttest_ind([original['engagement_score']], [variant['engagement_score']])
        
        if p_value < SIGNIFICANCE_LEVEL:
            winner = "Variant" if variant['engagement_score'] > original['engagement_score'] else "Original"
            results.append({
                'original_id': original['post_id'],
                'variant_id': variant['post_id'],
                'winner': winner,
                'p_value': p_value,
                'original_score': original['engagement_score'],
                'variant_score': variant['engagement_score']
            })
            logging.info(f"A/B test result: {winner} won. Original ID: {original['post_id']}, Variant ID: {variant['post_id']}")
        else:
            logging.info(f"A/B test inconclusive. Original ID: {original['post_id']}, Variant ID: {variant['post_id']}")
    
    return results

def incorporate_ab_test_results(calendar, ab_results):
    """
    Incorporate insights from A/B test results into the content strategy.
    """
    for result in ab_results:
        if result['winner'] == 'Variant':
            winning_post = calendar.get_post_by_id(result['variant_id'])
            insight = f"This content performed well in A/B testing: {winning_post['content']}"
            calendar.update_post(winning_post['index'], applied_insights=insight)
            logging.info(f"Incorporated winning variant insight for post {winning_post['index']}")

def optimize_content_strategy(calendar):
    """
    Use machine learning to optimize the content strategy.
    """
    df = pd.DataFrame(calendar.get_past_posts())
    
    if len(df) < 10:  # Arbitrary threshold, adjust as needed
        logging.warning("Not enough data to train a model yet.")
        return
    
    X, feature_names = extract_features(df)
    y = df['engagement_score']
    
    model, scaler = train_model(X, y)
    
    feature_importance = get_feature_importance(model, feature_names)
    logging.info("Top 10 important features:")
    logging.info(feature_importance.head(10).to_string(index=False))
    
    # Use the model to predict engagement for future posts
    future_posts = calendar.get_future_posts()
    future_df = pd.DataFrame(future_posts)
    future_X, _ = extract_features(future_df)
    future_predictions = predict_engagement(model, scaler, future_X)
    
    for i, post in enumerate(future_posts):
        calendar.update_post(post['index'], predicted_engagement=future_predictions[i])
        logging.info(f"Updated predicted engagement for post {post['index']}")
    
    return feature_importance