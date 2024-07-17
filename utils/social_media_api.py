from api_handlers import post_content, update_engagement_metrics

# You can keep any other utility functions here if needed

# Initialize logging
logging.basicConfig(filename='logs/social_media_api.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Twitter API setup
def init_twitter():
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
    return tweepy.API(auth)

twitter_api = init_twitter()

# Mock LinkedIn API
class MockLinkedIn:
    def post(self, content):
        post_id = f"linkedin_{random.randint(1000, 9999)}"
        logging.info(f"Mock post to LinkedIn: {content[:50]}...")
        return {'id': post_id}

    def get_post(self, post_id):
        return {
            'likes': random.randint(0, 100),
            'comments': random.randint(0, 20),
            'shares': random.randint(0, 10)
        }

linkedin_api = MockLinkedIn()

# Facebook API setup
def init_facebook():
    return GraphAPI(access_token=FACEBOOK_ACCESS_TOKEN)

facebook_api = init_facebook()

def post_to_twitter(content):
    try:
        tweet = twitter_api.update_status(content)
        logging.info(f"Posted to Twitter: {content[:50]}...")
        return tweet.id
    except Exception as e:
        logging.error(f"Error posting to Twitter: {e}")
        return None

def post_to_linkedin(content):
    try:
        linkedin_api = get_linkedin_api()
        if linkedin_api:
            response = linkedin_api.post(content)
            logging.info(f"Posted to LinkedIn: {content[:50]}...")
            return response['id']
        else:
            logging.error("LinkedIn API not initialized")
            return None
    except Exception as e:
        logging.error(f"Error posting to LinkedIn: {e}")
        return None

def post_to_facebook(content):
    try:
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
    
    if platform == "Twitter":
        post_id = post_to_twitter(content)
    elif platform == "LinkedIn":
        post_id = post_to_linkedin(content)
    elif platform == "Facebook":
        post_id = post_to_facebook(content)
    
    if post_id:
        calendar.update_post(row.name, post_id=post_id, posted_at=datetime.now().isoformat())

def get_twitter_engagement(post_id):
    try:
        tweet = twitter_api.get_status(post_id)
        return {
            'likes': tweet.favorite_count,
            'retweets': tweet.retweet_count,
            'comments': 0  # Twitter API doesn't provide comment count easily
        }
    except Exception as e:
        logging.error(f"Error getting Twitter engagement: {e}")
        return None

def get_linkedin_engagement(post_id):
    return linkedin_api.get_post(post_id)

def get_facebook_engagement(post_id):
    try:
        post = facebook_api.get_object(post_id, fields='likes.summary(true),comments.summary(true),shares')
        return {
            'likes': post['likes']['summary']['total_count'],
            'comments': post['comments']['summary']['total_count'],
            'shares': post.get('shares', {}).get('count', 0)
        }
    except Exception as e:
        logging.error(f"Error getting Facebook engagement: {e}")
        return None

def get_engagement_metrics(platform, post_id):
    if platform == "Twitter":
        return get_twitter_engagement(post_id)
    elif platform == "LinkedIn":
        return get_linkedin_engagement(post_id)
    elif platform == "Facebook":
        return get_facebook_engagement(post_id)
    else:
        logging.error(f"Unknown platform: {platform}")
        return None

def calculate_engagement_score(metrics):
    if metrics is None:
        return 0
    return metrics.get('likes', 0) + (metrics.get('comments', 0) * 2) + (metrics.get('shares', 0) * 3)

def update_engagement_metrics(calendar):
    for index, row in calendar.df.iterrows():
        if pd.notnull(row['post_id']) and (pd.isnull(row['engagement_score']) or 
                                           (datetime.now() - pd.to_datetime(row['posted_at'])).days <= 7):
            metrics = get_engagement_metrics(row['platform'], row['post_id'])
            if metrics:
                engagement_score = calculate_engagement_score(metrics)
                calendar.update_post(index, 
                                     likes=metrics.get('likes', 0),
                                     comments=metrics.get('comments', 0),
                                     shares=metrics.get('shares', metrics.get('retweets', 0)),
                                     engagement_score=engagement_score,
                                     last_updated=datetime.now().isoformat())