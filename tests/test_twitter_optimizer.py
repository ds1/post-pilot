# post-pilot/tests/test_twitter_optimizer.py

import sys
import os
import json
import unittest
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import openai
from unittest.mock import patch, MagicMock

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.content_calendar import ContentCalendar
from utils.twitter_optimizer import TwitterContentOptimizer

class TestTwitterOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('test_twitter_optimizer.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        cls.logger = logging.getLogger(__name__)
        
        # Load settings and set OpenAI key
        with open('settings.json', 'r') as f:
            settings = json.load(f)
        openai.api_key = settings.get('OPENAI_API_KEY')

        # Initialize components
        cls.calendar = ContentCalendar()
        cls.optimizer = TwitterContentOptimizer(cls.calendar)

        # Initialize test data
        cls.setup_test_data()

    @classmethod
    def setup_test_data(cls):
        """Create test data for the content calendar"""
        test_posts = []
        start_date = datetime.now() - timedelta(days=30)
        
        # Generate historical posts with engagement data
        for i in range(30):
            post_date = start_date + timedelta(days=i)
            test_posts.append({
                'due_date': post_date.date(),
                'platform': 'Twitter',
                'content_type': 'post',
                'subject': 'Test Post',
                'content': f'Test post {i} about #DigitalMarketing with some #Strategy content',
                'time_slot': '12:00',  # Fixed time slot for consistency
                'engagement_score': float(i % 5) * 10,  # Varied engagement scores
                'likes': float(i % 5) * 5,
                'retweets': float(i % 3) * 2,
                'replies': float(i % 2),
                'impressions': float(i % 5) * 100,
                'post_id': f'test_post_{i}',  # Add post_id
                'is_variant': False,  # Add is_variant flag
                'author_email': 'test@example.com'  # Add author email
            })
        
        # Create DataFrame and pre-process features
        df = pd.DataFrame(test_posts)
        
        # Pre-process temporal features
        df = cls.optimizer.extract_time_features(df)
        
        # Add content features
        content_features = cls.optimizer.extract_content_features(df['content'].tolist())
        
        # Combine all features
        df = pd.concat([
            df,
            content_features
        ], axis=1)
        
        # Save to CSV and update calendar
        df.to_csv('test_content_calendar.csv', index=False)
        cls.calendar.df = df.copy()

    def setUp(self):
        """Set up for each test"""
        self.logger.info("Setting up test case")

    def test_model_training(self):
        """Test if the model can be trained on historical data"""
        self.logger.info("Testing model training")
        training_success = self.optimizer.train_model()
        self.assertTrue(training_success, "Model training should succeed with test data")

    def test_content_generation(self):
        """Test content generation capabilities"""
        self.logger.info("Testing content generation")
        
        # First, train the model
        self.optimizer.train_model()
        
        # Create a more detailed mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message={
                    'content': 'Test #DigitalMarketing post with #engagement 🚀'
                }
            )
        ]
        
        # Create test features DataFrame
        feature_data = {
            'hour': [12],
            'day_of_week': [0],
            'is_weekend': [0],
            'is_morning': [0],
            'is_afternoon': [1],
            'is_evening': [0],
            'word_count': [10],
            'char_count': [50],
            'avg_word_length': [5],
            'entity_count': [2],
            'hashtag_count': [2],
            'mention_count': [0],
            'url_count': [0],
            'sentiment_score': [1]
        }
        
        test_features = pd.DataFrame(feature_data)
        
        with patch('openai.ChatCompletion.create', return_value=mock_response), \
             patch.object(self.optimizer, 'extract_time_features', return_value=test_features.iloc[:, :6]), \
             patch.object(self.optimizer, 'extract_content_features', return_value=test_features.iloc[:, 6:]), \
             patch.object(self.optimizer.scaler, 'transform', return_value=test_features.values):
            
            topic = "Digital Marketing"
            content, score = self.optimizer.generate_optimized_content(topic)
            
            self.assertIsNotNone(content, "Generated content should not be None")
            self.assertIsNotNone(score, "Engagement score prediction should not be None")
            self.assertTrue(len(content) > 0, "Generated content should not be empty")
            self.assertTrue(len(content) <= 280, "Content should respect Twitter's character limit")

    def test_content_scheduling(self):
        """Test the content scheduling functionality"""
        self.logger.info("Testing content scheduling")
        
        # First, train the model
        self.optimizer.train_model()
        
        # Create a more detailed mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message={
                    'content': 'Test #DigitalMarketing post with #engagement 🚀'
                }
            )
        ]
        
        # Create test features DataFrame with ALL expected features
        feature_data = {
            # Time features
            'hour': [12],
            'day_of_week': [0],
            'is_weekend': [0],
            'is_morning': [0],
            'is_afternoon': [1],
            'is_evening': [0],
            # Content features
            'word_count': [10],
            'char_count': [50],
            'avg_word_length': [5],
            'entity_count': [2],
            'hashtag_count': [2],
            'mention_count': [0],
            'url_count': [0],
            'sentiment_score': [1]
        }
        
        test_features = pd.DataFrame(feature_data)
        scaled_features = np.array(test_features)  # Create scaled features array
        
        # Generate test time slots
        start_date = datetime.now() + timedelta(days=1)
        time_slots = [
            (start_date + timedelta(days=x)).strftime('%Y-%m-%d %H:%M')
            for x in range(7)
            for h in [9, 12, 15, 17]
        ]
        
        with patch('openai.ChatCompletion.create', return_value=mock_response), \
             patch.object(self.optimizer, 'extract_time_features', return_value=test_features.iloc[:, :6]), \
             patch.object(self.optimizer, 'extract_content_features', return_value=test_features.iloc[:, 6:]), \
             patch.object(self.optimizer.scaler, 'transform', return_value=scaled_features):
            
            scheduling_result = self.optimizer.schedule_optimized_content(
                topic="Digital Marketing",
                time_slots=time_slots
            )
            
            self.assertTrue(scheduling_result, "Content scheduling should succeed")
            
            # Verify the scheduled post exists in the calendar
            recent_posts = self.calendar.get_future_posts()
            self.assertTrue(len(recent_posts) > 0, "Calendar should contain the scheduled post")
            
            # Verify the content of the scheduled post
            self.assertEqual(
                recent_posts[0]['content'],
                'Test #DigitalMarketing post with #engagement 🚀'
            )

    def test_feature_extraction(self):
        """Test feature extraction from content"""
        self.logger.info("Testing feature extraction")
        
        test_texts = [
            "Test post about #DigitalMarketing with a link https://example.com",
            "Another #test post with @mention and #hashtags"
        ]
        
        content_features = self.optimizer.extract_content_features(test_texts)
        
        self.assertIsInstance(content_features, pd.DataFrame, "Should return a DataFrame")
        self.assertTrue(len(content_features) == 2, "Should have features for both test texts")
        
        # Check if all expected features are present
        expected_features = [
            'word_count', 'char_count', 'avg_word_length', 'entity_count',
            'hashtag_count', 'mention_count', 'url_count', 'sentiment_score'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, content_features.columns,
                         f"Feature {feature} should be present")

    def test_time_features(self):
        """Test temporal feature extraction"""
        self.logger.info("Testing temporal feature extraction")
        
        test_df = pd.DataFrame({
            'time_slot': ['09:00', '15:00', '20:00'],
            'due_date': [datetime.now().date()] * 3
        })
        
        time_features = self.optimizer.extract_time_features(test_df)
        
        expected_columns = [
            'hour', 'day_of_week', 'is_weekend',
            'is_morning', 'is_afternoon', 'is_evening'
        ]
        
        for col in expected_columns:
            self.assertIn(col, time_features.columns,
                         f"Time feature {col} should be present")

    def tearDown(self):
        """Clean up after each test"""
        self.logger.info("Tearing down test case")

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        # Remove test files
        try:
            os.remove('test_content_calendar.csv')
            os.remove('test_twitter_optimizer.log')
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    unittest.main(verbosity=2)