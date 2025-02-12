# post-pilot/utils/twitter_optimizer.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import spacy
from datetime import datetime
import logging
import openai

class TwitterContentOptimizer:
    def __init__(self, content_calendar):
        self.content_calendar = content_calendar
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f")
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
        df['is_morning'] = (df['hour'] >= 6) & (df['hour'] < 12).astype(int)
        df['is_afternoon'] = (df['hour'] >= 12) & (df['hour'] < 17).astype(int)
        df['is_evening'] = (df['hour'] >= 17) & (df['hour'] < 22).astype(int)
        
        return df
    
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
            
            # Sentiment
            sentiment = self.sentiment_analyzer(text)[0]
            sentiment_score = 1 if sentiment['label'] == 'POSITIVE' else 0
            
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
            df = pd.DataFrame(self.content_calendar.get_past_posts())
            
            if len(df) < 10:
                logging.warning("Insufficient training data")
                return False
                
            # Extract features
            df = self.extract_time_features(df)
            content_features = self.extract_content_features(df['content'])
            
            # Combine all features
            X = pd.concat([
                df[['hour', 'day_of_week', 'is_weekend', 
                    'is_morning', 'is_afternoon', 'is_evening']],
                content_features
            ], axis=1)
            
            y = df['engagement_score']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model with time series split
            tscv = TimeSeriesSplit(n_splits=5)
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
        prompt = f"""Write a Twitter post about {topic} that will maximize engagement.
The post should:
- Use relevant hashtags
- Be clear and concise
- Include a call to action
- Be engaging and conversation-starting
- Stay within Twitter's 280 character limit
- Use emojis appropriately"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a social media expert who writes engaging Twitter posts that drive high engagement."},
                    {"role": "user", "content": prompt}
                ],
                n=3,
                temperature=0.7,
                max_tokens=100
            )
            
            # Get all generated candidates
            candidates = [choice.message['content'].strip() for choice in response.choices]
            
            # Predict engagement for each candidate
            best_content = None
            best_score = float('-inf')
            
            for content in candidates:
                # Skip if content exceeds Twitter's limit
                if len(content) > 280:
                    continue
                    
                # Extract features for prediction
                time_features = self.extract_time_features(pd.DataFrame({
                    'time_slot': [datetime.now().strftime('%H:%M')],
                    'due_date': [datetime.now().date()]
                }))
                
                content_features = self.extract_content_features([content])
                
                features = pd.concat([
                    time_features.iloc[[0]].reset_index(drop=True),
                    content_features
                ], axis=1)
                
                # Scale and predict
                features_scaled = self.scaler.transform(features)
                predicted_engagement = self.model.predict(features_scaled)[0]
                
                if predicted_engagement > best_score:
                    best_score = predicted_engagement
                    best_content = content
            
            return best_content, best_score
            
        except Exception as e:
            logging.error(f"Error generating content: {str(e)}")
            return None, None
    
    def schedule_optimized_content(self, topic, time_slots):
        """Generate and schedule optimized content"""
        content, predicted_engagement = self.generate_optimized_content(topic)
        
        if content:
            try:
                # Find best time slot based on historical patterns
                best_slot = max(time_slots, key=lambda slot: 
                    self.model.predict(self.scaler.transform([[
                        pd.to_datetime(slot).hour,
                        pd.to_datetime(slot).dayofweek,
                        1 if pd.to_datetime(slot).dayofweek in [5, 6] else 0,
                        1 if pd.to_datetime(slot).hour in range(6, 12) else 0,
                        1 if pd.to_datetime(slot).hour in range(12, 17) else 0,
                        1 if pd.to_datetime(slot).hour in range(17, 22) else 0
                    ]]))[0]
                )
                
                # Schedule the post
                post_data = {
                    'due_date': pd.to_datetime(best_slot).date(),
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