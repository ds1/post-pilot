#import tweepy
import logging
import json
from datetime import datetime
import pandas as pd
from api_init import init_twitter, init_linkedin, init_facebook

def post_to_twitter(content):
    try:
        twitter_api = init_twitter()
        response = twitter_api.create_tweet(text=content)
        tweet_id = response.data['id']
        logging.info(f"Posted to Twitter: {content[:50]}...")
        return tweet_id
    except Exception as e:
        logging.error(f"Error posting to Twitter: {str(e)}")
        return None

def post_to_linkedin(content):
    try:
        linkedin_api = init_linkedin()
        response = linkedin_api.post(content)
        logging.info(f"Posted to LinkedIn: {content[:50]}...")
        return response['id']
    except Exception as e:
        logging.error(f"Error posting to LinkedIn: {e}")
        return None

def post_to_facebook(content):
    try:
        facebook_api = init_facebook()
        response = facebook_api.put_object("me", "feed", message=content)
        logging.info(f"Posted to Facebook: {content[:50]}...")
        return response['id']
    except Exception as e:
        logging.error(f"Error posting to Facebook: {e}")
        return None

def post_content(calendar, row):
    platform = row['platform']
    content = row['content']
    post_id = None
    
    now = datetime.now()
    post_datetime = datetime.combine(pd.to_datetime(row['due_date']).date(), pd.to_datetime(row['time_slot']).time())
    
    if post_datetime <= now:
        if platform == "Twitter":
            post_id = post_to_twitter(content)
        elif platform == "LinkedIn":
            post_id = post_to_linkedin(content)
        elif platform == "Facebook":
            post_id = post_to_facebook(content)
        
        if post_id:
            calendar.update_post(row.name, post_id=post_id, posted_at=now.strftime("%Y-%m-%d %H:%M"))
            logging.info(f"Posted content to {platform} at {now.strftime('%Y-%m-%d %H:%M')}")
        else:
            logging.error(f"Failed to post content to {platform} at {now.strftime('%Y-%m-%d %H:%M')}")
    else:
        logging.info(f"Scheduled post for {platform} at {post_datetime.strftime('%Y-%m-%d %H:%M')}")

def update_engagement_metrics(calendar):
    # Implement this function to update engagement metrics
    pass

# def test_twitter_post():
#     content = "This is a test post from my Social Media Optimizer app!"
#     try:
#         tweet = twitter_api.update_status(content)
#         print(f"Successfully posted to Twitter. Tweet ID: {tweet.id}")
#     except Exception as e:
#         print(f"Error posting to Twitter: {e}")

# Call this function from your main.py or gui.py to test