import pandas as pd
import random
from datetime import datetime, timedelta

def generate_content(platform):
    topics = ["Domain Investing", "SEO", "Web Development", "Digital Marketing", "E-commerce"]
    actions = ["Tips for", "The Future of", "How to Optimize", "Understanding", "Top Trends in"]
    
    topic = random.choice(topics)
    action = random.choice(actions)
    
    if platform == "Twitter":
        return f"{action} {topic}. #DigitalStrategy #OnlineBusiness"
    elif platform == "LinkedIn":
        return f"{action} {topic}: A Comprehensive Guide for Professionals. Share your thoughts in the comments!"
    else:  # Facebook
        return f"🚀 {action} {topic} 🚀\n\nLearn how this can transform your online presence. Click the link in our bio to read more!"

def generate_calendar():
    start_date = datetime.now().date()
    platforms = ["Twitter", "LinkedIn", "Facebook"]
    data = []
    
    for i in range(28):  # 4 weeks
        current_date = start_date + timedelta(days=i)
        for platform in platforms:
            time_slot = f"{random.randint(9, 17):02d}:00"
            data.append({
                "due_date": current_date.strftime("%Y-%m-%d"),
                "platform": platform,
                "content_type": "post",
                "subject": f"{platform} update",
                "content": generate_content(platform),
                "author_email": "social_media_team@example.com",
                "time_slot": time_slot,
                "post_id": None,
                "engagement_score": None,
                "likes": None,
                "comments": None,
                "shares": None,
                "is_variant": False,
                "original_post_id": None
            })
    
    return pd.DataFrame(data)

# Generate the calendar
calendar_df = generate_calendar()

# Save to CSV
calendar_df.to_csv("social_media_optimizer/content_calendar.csv", index=False)

print("Generated content calendar for 4 weeks and saved to content_calendar.csv")