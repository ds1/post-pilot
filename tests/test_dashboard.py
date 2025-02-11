# post-pilot/tests/test_dashboard.py

import sys
import os
import logging

# Get the absolute paths to the directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

try:
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from PyQt5.QtCore import QTimer
    from components.dashboard.dashboard_tab import DashboardTab
    from load_test_data import load_test_data
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)

def setup_logging():
    log_file = os.path.join(TESTS_DIR, 'test_dashboard.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("Starting dashboard test")

    # Create required directories if they don't exist
    dashboard_dir = os.path.join(PROJECT_ROOT, 'components', 'dashboard')
    static_dir = os.path.join(dashboard_dir, 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    try:
        # Initialize the application
        app = QApplication(sys.argv)
        
        # Load test data into content calendar
        logger.info("Loading test data...")
        content_calendar = load_test_data()
        
        # Create dashboard
        logger.info("Creating dashboard...")
        dashboard = DashboardTab(content_calendar)
        dashboard.resize(1200, 800)
        dashboard.show()
        
        # Create a timer to refresh the dashboard
        refresh_timer = QTimer()
        refresh_timer.timeout.connect(dashboard.refresh_dashboard)
        refresh_timer.start(30000)  # Refresh every 30 seconds
        
        logger.info("Dashboard initialized successfully")
        
        # Start the application
        return app.exec_()
    
    except Exception as e:
        logger.error(f"Error in dashboard test: {str(e)}", exc_info=True)
        if 'app' in locals():
            QMessageBox.critical(None, "Error", f"Dashboard test failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())