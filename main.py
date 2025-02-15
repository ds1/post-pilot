import json
import schedule
import time
import logging
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
from utils.content_calendar import ContentCalendar
from api_handlers import post_content, update_engagement_metrics
from utils.nlp_utils import generate_content_suggestions, apply_insights_to_future_content
from utils.ml_utils import optimize_content_strategy, create_ab_tests, analyze_ab_test_results, incorporate_ab_test_results
from api_init import load_settings, init_openai
from utils.twitter_optimizer import TwitterContentOptimizer

settings = load_settings()

init_openai()

logging.basicConfig(filename='logs/content_optimization.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

content_calendar = ContentCalendar()

def run_content_optimization():
    logging.info("Running content optimization process...")
    
    # Initialize the Twitter optimizer
    twitter_optimizer = TwitterContentOptimizer(content_calendar)
    
    # Train the model
    if twitter_optimizer.train_model():
        # Generate and schedule new content for upcoming week
        topics = ["Digital Marketing", "SEO", "Web Development", 
                 "Social Media Strategy", "Content Marketing"]
        
        # Generate time slots for next week
        from datetime import datetime, timedelta
        start_date = datetime.now() + timedelta(days=1)
        time_slots = [
            (start_date + timedelta(days=x)).strftime('%Y-%m-%d %H:%M')
            for x in range(7)
            for h in [9, 12, 15, 17]  # Multiple slots per day
        ]
        
        # Schedule optimized content
        for topic in topics:
            twitter_optimizer.schedule_optimized_content(topic, time_slots)
    
    # Run existing optimization processes
    suggestions = generate_content_suggestions(content_calendar)
    if suggestions:
        create_ab_tests(content_calendar, suggestions)
        apply_insights_to_future_content(content_calendar, suggestions)
    winning_post = analyze_ab_test_results(content_calendar)
    incorporate_ab_test_results(content_calendar, winning_post)
    feature_importance = optimize_content_strategy(content_calendar)

def schedule_posts():
    schedule.clear()  # Clear existing schedule
    for index, row in content_calendar.df.iterrows():
        if pd.isnull(row['post_id']):
            post_datetime = pd.to_datetime(f"{row['due_date']} {row['time_slot']}")
            schedule.every().day.at(post_datetime.strftime("%H:%M")).do(post_content, content_calendar, row).tag(f"{row['platform']}-{row['due_date']}-{row['time_slot']}")

def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("settings.json file not found. Please configure your settings in the app.")
        return {}

class SchedulerThread(QThread):
    update_signal = pyqtSignal(str)

    def run(self):
        schedule_posts()
        schedule.every(15).minutes.do(update_engagement_metrics, content_calendar)
        schedule.every().day.at("00:00").do(run_content_optimization)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
            self.update_signal.emit("Scheduler is running...")

def run_scheduler():
    scheduler_thread = SchedulerThread()
    scheduler_thread.start()
    return scheduler_thread

if __name__ == "__main__":
    logging.info("Starting social media scheduler and content optimization system...")
    scheduler_thread = run_scheduler()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping scheduler...")
        scheduler_thread.terminate()
        scheduler_thread.wait()
    logging.info("Scheduler stopped.")