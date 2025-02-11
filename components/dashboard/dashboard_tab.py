import os
import json
import logging
from datetime import date, datetime
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebChannel
from PyQt5.QtCore import QUrl, QTimer, QObject, pyqtSlot

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

class Bridge(QObject):
    def __init__(self, content_calendar):
        super().__init__()
        self.content_calendar = content_calendar

    @pyqtSlot(result=str)
    def readAnalyticsFile(self):
        try:
            # Get analytics data from content calendar
            analytics_data = {
                'engagement_stats': self.content_calendar.get_engagement_stats(),
                'posts': self.content_calendar.get_past_posts()
            }
            # Convert to JSON with custom encoder
            return json.dumps(analytics_data, cls=DateTimeEncoder)
        except Exception as e:
            logging.error(f"Error reading analytics data: {str(e)}")
            return json.dumps({'error': str(e)})

class DashboardTab(QWidget):
    def __init__(self, content_calendar):
        super().__init__()
        self.content_calendar = content_calendar
        self.setup_ui()
        self.setup_timer()

    def setup_ui(self):
        """Initialize the UI components"""
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create web view
        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view)
        
        # Setup bridge
        self.channel = QWebChannel()
        self.bridge = Bridge(self.content_calendar)
        self.channel.registerObject('bridge', self.bridge)
        self.web_view.page().setWebChannel(self.channel)
        
        # Load dashboard HTML
        dashboard_html = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'static',
            'dashboard.html'
        )
        
        if os.path.exists(dashboard_html):
            self.web_view.setUrl(QUrl.fromLocalFile(dashboard_html))
            logging.info(f"Loaded dashboard HTML from: {dashboard_html}")
        else:
            logging.error(f"Dashboard HTML not found at: {dashboard_html}")

    def setup_timer(self):
        """Setup refresh timer"""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_dashboard)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds

    def refresh_dashboard(self):
        """Refresh the dashboard data"""
        try:
            self.web_view.page().runJavaScript('window.location.reload()')
            logging.info("Dashboard refreshed successfully")
        except Exception as e:
            logging.error(f"Error refreshing dashboard: {str(e)}")