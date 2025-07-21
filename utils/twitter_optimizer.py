# post-pilot/utils/twitter_optimizer.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
# from transformers import pipeline  # REMOVED
import spacy
from datetime import datetime
import logging
import openai

class TwitterContentOptimizer:
    def __init__(self, content_calendar):
        self.content_calendar = content_calendar
        self.nlp = spacy.load("en_core_web_sm")
        # REMOVED transformer sentiment analyzer
        # self.sentiment_analyzer = pipeline("sentiment-analysis", 
        #     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        #     revision="714eb0f")
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def extract_time_features(self, df):
        """Extract temporal engagement patterns"""
        df['hour'] = pd.to_datetime(df['time_slot'], format='%H:%M').dt.hour
        df['day_of_week'] = pd.to_datetime(df['due_date']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time block features
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 17)).astype(int)
        df['is_evening'] = ((df['hour'] >= 17) & (df['hour'] < 22)).astype(int)
        
        return df
    
    def simple_sentiment_analysis(self, text):
        """Simple rule-based sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best', 'happy', 'excited']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'poor', 'worst', 'sad', 'angry', 'disappointed']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 1  # Positive
        elif negative_count > positive_count:
            return 0  # Negative
        else:
            return 0.5  # Neutral
    
    def extract_content_features(self, texts):
        """Extract rich content features"""
        features = []
        
        for text in texts:
            doc = self.nlp(text)
            
            # Basic text features
            word_count = len(doc)
            char_count = len(text)
            avg_word_length = char_count / word_count if word_count > 0 else 0
            
            # NLP features
            entity_count = len(doc.ents)
            hashtag_count = text.count('#')
            mention_count = text.count('@')
            url_count = sum(1 for token in doc if token.like_url)
            
            # Simple sentiment instead of transformer-based
            sentiment_score = self.simple_sentiment_analysis(text)
            
            features.append({
                'word_count': word_count,
                'char_count': char_count,
                'avg_word_length': avg_word_length,
                'entity_count': entity_count,
                'hashtag_count': hashtag_count,
                'mention_count': mention_count,
                'url_count': url_count,
                'sentiment_score': sentiment_score
            })
            
        return pd.DataFrame(features)
    
    def train_model(self):
        """Train the engagement prediction model"""
        try:
            # Get historical data
            past_posts = self.content_calendar.get_past_posts()
            if not past_posts:
                logging.warning("No past posts available for training")
                return False
                
            df = pd.DataFrame(past_posts)
            
            # Filter for Twitter posts only
            df = df[df['platform'] == 'Twitter']
            
            if len(df) < 10:
                logging.warning("Insufficient Twitter training data")
                return False
                
            # Extract features
            df = self.extract_time_features(df)
            content_features = self.extract_content_features(df['content'].tolist())
            
            # Combine all features
            feature_cols = ['hour', 'day_of_week', 'is_weekend', 
                           'is_morning', 'is_afternoon', 'is_evening']
            
            # Check if all required columns exist
            for col in feature_cols:
                if col not in df.columns:
                    logging.error(f"Missing column: {col}")
                    return False
            
            X = pd.concat([
                df[feature_cols].reset_index(drop=True),
                content_features.reset_index(drop=True)
            ], axis=1)
            
            # Fill NaN values in engagement_score with 0
            y = df['engagement_score'].fillna(0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model with time series split
            tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 2))
            best_score = float('-inf')
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                self.model.fit(X_train, y_train)
                score = self.model.score(X_val, y_val)
                
                if score > best_score:
                    best_score = score
            
            logging.info(f"Model trained with best R2 score: {best_score}")
            return True
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return False
    
    def generate_optimized_content(self, topic, target_engagement=None):
        """Generate content optimized for engagement"""
        # Simple template-based generation without OpenAI
        templates = [
            f"🚀 Exciting news about {topic}! What are your thoughts? #DigitalStrategy #Innovation",
            f"💡 Did you know? {topic} is changing the game! Share your experience below 👇 #TechTrends",
            f"🎯 {topic}: The future is here! How are you preparing? Let's discuss! #FutureOfWork",
            f"📈 Breaking down {topic} - what you need to know today! Thread 🧵 #Learning #Growth",
            f"🔥 Hot take: {topic} will transform our industry. Agree or disagree? #Disruption"
        ]
        
        # Select a random template
        import random
        content = random.choice(templates)
        
        # Ensure it's within Twitter limit
        if len(content) > 280:
            content = content[:277] + "..."
        
        # Predict engagement score for this content
        try:
            time_features = self.extract_time_features(pd.DataFrame({
                'time_slot': [datetime.now().strftime('%H:%M')],
                'due_date': [datetime.now().date()]
            }))
            
            content_features = self.extract_content_features([content])
            
            feature_cols = ['hour', 'day_of_week', 'is_weekend', 
                           'is_morning', 'is_afternoon', 'is_evening']
            
            features = pd.concat([
                time_features[feature_cols].reset_index(drop=True),
                content_features.reset_index(drop=True)
            ], axis=1)
            
            # Scale and predict
            features_scaled = self.scaler.transform(features)
            predicted_engagement = self.model.predict(features_scaled)[0]
            
            return content, predicted_engagement
        except:
            # If prediction fails, return content with default score
            return content, 50.0
    
    def schedule_optimized_content(self, topic, time_slots):
        """Generate and schedule optimized content"""
        content, predicted_engagement = self.generate_optimized_content(topic)
        
        if content:
            try:
                # Use the first available time slot
                best_slot = time_slots[0] if time_slots else datetime.now()
                
                # Schedule the post
                post_data = {
                    'due_date': pd.to_datetime(best_slot).date().strftime('%Y-%m-%d'),
                    'time_slot': pd.to_datetime(best_slot).strftime('%H:%M'),
                    'platform': 'Twitter',
                    'content_type': 'post',
                    'subject': topic,
                    'content': content,
                    'predicted_engagement': predicted_engagement,
                    'is_variant': False
                }
                
                self.content_calendar.add_post(post_data)
                logging.info(f"Scheduled optimized content for {best_slot}")
                return True
            except Exception as e:
                logging.error(f"Error scheduling content: {str(e)}")
                return False
                
        return False