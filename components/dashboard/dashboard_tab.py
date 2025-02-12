import os
import json
import math
import logging
from datetime import date, datetime
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QUrl, QTimer, QObject, pyqtSlot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Bridge(QObject):
    def __init__(self, content_calendar):
        super().__init__()
        self.content_calendar = content_calendar

    @pyqtSlot(result=str)
    def readAnalyticsFile(self):
        try:
            # Get raw analytics data first
            engagement_stats = self.content_calendar.get_engagement_stats()
            posts = self.content_calendar.get_past_posts()
            
            # Log raw data
            logger.info(f"Raw engagement stats: {engagement_stats}")
            logger.info(f"Number of posts: {len(posts)}")
            
            # Handle engagement stats
            cleaned_stats = {}
            for key, value in engagement_stats.items():
                if isinstance(value, dict):
                    cleaned_nested = {}
                    for k, v in value.items():
                        cleaned_nested[k] = float(v) if v is not None else 0.0
                    cleaned_stats[key] = cleaned_nested
                else:
                    cleaned_stats[key] = float(value) if value is not None else 0.0
            
            # Handle posts
            cleaned_posts = []
            total_engagement = 0
            total_impressions = 0
            
            for post in posts:
                cleaned_post = {}
                for key, value in post.items():
                    if key in ['engagement_score', 'likes', 'comments', 'shares', 'impressions']:
                        # Handle numeric fields
                        try:
                            if isinstance(value, str) and value.lower() == 'nan':
                                cleaned_val = 0.0
                            else:
                                cleaned_val = float(value) if value not in (None, '', 'inf', '-inf') else 0.0
                            cleaned_post[key] = cleaned_val
                            if key == 'engagement_score':
                                total_engagement += cleaned_val
                            elif key == 'impressions':
                                total_impressions += cleaned_val
                        except (ValueError, TypeError):
                            cleaned_post[key] = 0.0
                    else:
                        # Handle non-numeric fields
                        if isinstance(value, str) and value.lower() == 'nan':
                            cleaned_post[key] = ""  # Convert 'nan' strings to empty string
                        else:
                            cleaned_post[key] = str(value) if value is not None else ""
                
                cleaned_posts.append(cleaned_post)

            # Prepare analytics data
            analytics_data = {
                "engagement_stats": cleaned_stats,
                "posts": cleaned_posts,
                "summary": {
                    "total_engagement": total_engagement,
                    "total_impressions": total_impressions,
                    "engagement_rate": (total_engagement / total_impressions * 100) if total_impressions > 0 else 0.0
                }
            }
            
            # Validate and clean analytics data
            def clean_value(val):
                if isinstance(val, (int, float)):
                    return 0.0 if math.isnan(val) or math.isinf(val) else float(val)
                elif isinstance(val, str) and val.lower() == 'nan':
                    return ""
                elif isinstance(val, dict):
                    return {k: clean_value(v) for k, v in val.items()}
                elif isinstance(val, list):
                    return [clean_value(v) for v in val]
                return str(val) if val is not None else ""

            # Clean the entire analytics data structure
            analytics_data = {
                "engagement_stats": clean_value(cleaned_stats),
                "posts": [clean_value(post) for post in cleaned_posts],
                "summary": {
                    "total_engagement": float(total_engagement),
                    "total_impressions": float(total_impressions),
                    "engagement_rate": float((total_engagement / total_impressions * 100) if total_impressions > 0 else 0.0)
                }
            }
            
            # Convert to JSON and verify
            try:
                json_str = json.dumps(analytics_data)
                # Verify the JSON is valid
                json.loads(json_str)
                logger.info(f"Successfully generated valid JSON data")
                return json_str
            except Exception as json_error:
                logger.error(f"JSON error: {str(json_error)}")
                return json.dumps({
                    "engagement_stats": {},
                    "posts": [],
                    "summary": {
                        "total_engagement": 0,
                        "total_impressions": 0,
                        "engagement_rate": 0
                    }
                })
                
        except Exception as e:
            logger.error(f"Error in readAnalyticsFile: {str(e)}", exc_info=True)
            return json.dumps({"error": str(e)})

class DashboardTab(QWidget):
    def __init__(self, content_calendar):
        super().__init__()
        self.content_calendar = content_calendar
        self.setup_ui()
        self.setup_timer()

    def setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view)
        
        self.channel = QWebChannel()
        self.bridge = Bridge(self.content_calendar)
        self.channel.registerObject('bridge', self.bridge)
        self.web_view.page().setWebChannel(self.channel)
        
        dashboard_html = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'static',
            'dashboard.html'
        )
        
        if os.path.exists(dashboard_html):
            self.web_view.setUrl(QUrl.fromLocalFile(dashboard_html))
            logger.info(f"Loaded dashboard HTML from: {dashboard_html}")
        else:
            logger.error(f"Dashboard HTML not found at: {dashboard_html}")

    def setup_timer(self):
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_dashboard)
        self.refresh_timer.start(300000)  # Refresh every 300 seconds

    def refresh_dashboard(self):
        try:
            self.web_view.page().runJavaScript('window.location.reload()')
            logger.info("Dashboard refreshed successfully")
        except Exception as e:
            logger.error(f"Error refreshing dashboard: {str(e)}")