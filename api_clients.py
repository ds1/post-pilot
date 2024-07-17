import json
import logging
import tweepy
from linkedin_api import Linkedin
from facebook import GraphAPI
import openai

def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("settings.json file not found. Please configure your settings in the app.")
        return {}

settings = load_settings()

def init_twitter():
    auth = tweepy.OAuthHandler(
        settings.get('TWITTER_API_KEY', ''),
        settings.get('TWITTER_API_SECRET', '')
    )
    auth.set_access_token(
        settings.get('TWITTER_ACCESS_TOKEN', ''),
        settings.get('TWITTER_ACCESS_TOKEN_SECRET', '')
    )
    return tweepy.API(auth)

def get_linkedin_api():
    try:
        return Linkedin(
            settings.get('LINKEDIN_CLIENT_ID', ''),
            settings.get('LINKEDIN_CLIENT_SECRET', '')
        )
    except Exception as e:
        logging.error(f"Error initializing LinkedIn API: {str(e)}")
        return None

def init_facebook():
    return GraphAPI(access_token=settings.get('FACEBOOK_ACCESS_TOKEN', ''))

def init_openai():
    openai.api_key = settings.get('OPENAI_API_KEY', '')

twitter_api = init_twitter()
facebook_api = init_facebook()
init_openai()