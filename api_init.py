import json
import tweepy
from linkedin_api import Linkedin
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

def init_twitter():
    client = tweepy.Client(
        consumer_key=settings.get('TWITTER_API_KEY', ''),
        consumer_secret=settings.get('TWITTER_API_SECRET', ''),
        access_token=settings.get('TWITTER_ACCESS_TOKEN', ''),
        access_token_secret=settings.get('TWITTER_ACCESS_TOKEN_SECRET', '')
    )
    return client

def init_linkedin():
    return Linkedin(
        settings.get('LINKEDIN_CLIENT_ID', ''),
        settings.get('LINKEDIN_CLIENT_SECRET', '')
    )

def init_openai():
    openai.api_key = settings.get('OPENAI_API_KEY', '')