# post-pilot/tests/load_test_data.py

import sys
import os
import pandas as pd
from datetime import datetime

# Get the absolute paths to the directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from utils.content_calendar import ContentCalendar

def load_test_data():
    # Initialize content calendar with path relative to project root
    calendar_path = os.path.join(PROJECT_ROOT, 'content_calendar.csv')
    calendar = ContentCalendar(calendar_path)
    
    # Read the analytics data from tests directory
    analytics_path = os.path.join(TESTS_DIR, 'account_analytics_content_20241114_20250212.csv')
    if os.path.exists(analytics_path):
        print(f"Found analytics file at: {analytics_path}")
        content_analytics = pd.read_csv(analytics_path)
    else:
        print(f"Warning: Analytics file not found at {analytics_path}")
        return calendar
    
    # Convert analytics data to content calendar format
    for _, row in content_analytics.iterrows():
        post_data = {
            'due_date': pd.to_datetime(row['Date']).strftime('%Y-%m-%d'),
            'platform': 'Twitter',
            'content_type': 'post',
            'subject': 'Twitter update',
            'content': row['Post text'] if pd.notna(row['Post text']) else '',
            'time_slot': '12:00',  # Default time
            'post_id': row['Post id'],
            'engagement_score': row['Engagements'],
            'likes': row['Likes'],
            'shares': row['Share'],
            'comments': row['Replies'],
            'impressions': row['Impressions'],
            'is_variant': False
        }
        calendar.add_post(post_data)
    
    # Save the calendar
    calendar.save()
    print(f"Loaded {len(content_analytics)} posts into content calendar")
    print(f"Calendar saved to: {calendar_path}")
    return calendar

if __name__ == "__main__":
    calendar = load_test_data()
    print("\nCalendar Statistics:")
    stats = calendar.get_engagement_stats()
    print(f"Average engagement: {stats['average']:.2f}")
    print(f"Max engagement: {stats['max']:.2f}")
    print(f"Min engagement: {stats['min']:.2f}")
    print("\nPlatform engagement:", stats['by_platform'])