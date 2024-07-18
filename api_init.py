import json
import tweepy
from linkedin_api import Linkedin
from facebook import GraphAPI
import openai
import logging

def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("settings.json file not found. Please configure your settings in the app.")
        return {}

settings = load_settings()

#OAuth1.0a
# def init_twitter():
#     auth = tweepy.OAuthHandler(
#         settings.get('TWITTER_API_KEY', ''),
#         settings.get('TWITTER_API_SECRET', ''),
#         callback='http://127.0.0.1:8000/callback'
#     )
#     auth.set_access_token(
#         settings.get('TWITTER_ACCESS_TOKEN', ''),
#         settings.get('TWITTER_ACCESS_TOKEN_SECRET', '')
#     )
#     return tweepy.API(auth)

#OAuth 2
def init_twitter():
    try:
        client = tweepy.Client(
            consumer_key=settings.get('TWITTER_API_KEY', ''),
            consumer_secret=settings.get('TWITTER_API_SECRET', ''),
            access_token=settings.get('TWITTER_ACCESS_TOKEN', ''),
            access_token_secret=settings.get('TWITTER_ACCESS_TOKEN_SECRET', '')
        )
        return client
    except Exception as e:
        logging.error(f"Error initializing Twitter client: {str(e)}")
        return None

def init_linkedin():
    return Linkedin(
        settings.get('LINKEDIN_CLIENT_ID', ''),
        settings.get('LINKEDIN_CLIENT_SECRET', '')
    )

def init_facebook():
    return GraphAPI(access_token=settings.get('FACEBOOK_ACCESS_TOKEN', ''))

def init_openai():
    openai.api_key = settings.get('OPENAI_API_KEY', '')