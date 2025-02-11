import os
import sys
import logging
import traceback
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, 
                             QLabel, QHBoxLayout, QComboBox, QMessageBox, QTabWidget, QTableWidget, 
                             QTableWidgetItem, QLineEdit, QFormLayout, QDateEdit, QTimeEdit, QDialog,
                             QGroupBox, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate, QTime
from PyQt5.QtGui import QIcon
import pandas as pd
from main import SchedulerThread, content_calendar, run_content_optimization
from utils.nlp_utils import generate_ab_variant
from utils.content_calendar import ContentCalendar
from components.dashboard.dashboard_tab import DashboardTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Post Pilot")
        self.setGeometry(100, 100, 1000, 800)

        self.content_calendar = ContentCalendar()
        self.scheduler_thread = None
        self.optimization_thread = None

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Main tab
        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)
        
        # Add controls section
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Scheduler")
        self.start_button.clicked.connect(self.start_scheduler)
        controls_layout.addWidget(self.start_button)
        
        self.optimize_button = QPushButton("Run Optimization")
        self.optimize_button.clicked.connect(self.run_optimization)
        controls_layout.addWidget(self.optimize_button)
        
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)
        
        # Add A/B testing section
        ab_group = QGroupBox("A/B Testing")
        ab_layout = QVBoxLayout()
        
        self.post_selector = QComboBox()
        ab_layout.addWidget(self.post_selector)
        
        generate_button = QPushButton("Generate Variant")
        generate_button.clicked.connect(self.generate_variant)
        ab_layout.addWidget(generate_button)
        
        ab_group.setLayout(ab_layout)
        main_layout.addWidget(ab_group)
        
        # Add log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        main_layout.addWidget(self.log_display)
        
        self.tab_widget.addTab(main_tab, "Main")

        # Content Calendar tab
        self.calendar_tab = ContentCalendarTab(self.content_calendar)
        self.tab_widget.addTab(self.calendar_tab, "Content Calendar")

        # Analytics tab
        self.analytics_tab = AnalyticsTab(self.content_calendar)
        self.tab_widget.addTab(self.analytics_tab, "Analytics")

        # Settings tab
        self.settings_tab = SettingsTab()
        self.tab_widget.addTab(self.settings_tab, "Settings")

        # Update post selector
        self.update_post_selector()

    def start_scheduler(self):
        if self.scheduler_thread is None:
            self.scheduler_thread = SchedulerThread()
            self.scheduler_thread.update_signal.connect(self.update_log)
            self.scheduler_thread.start()
            self.start_button.setEnabled(False)
            self.start_button.setText("Scheduler Running")

    def run_optimization(self):
        try:
            run_content_optimization()
            self.update_log("Content optimization completed.")
        except Exception as e:
            self.update_log(f"Error during optimization: {str(e)}")
            logging.error(f"Optimization error: {str(e)}\n{traceback.format_exc()}")

    def update_log(self, message):
        self.log_display.append(message)

    def update_post_selector(self):
        self.post_selector.clear()
        future_posts = self.content_calendar.get_future_posts()
        for post in future_posts:
            self.post_selector.addItem(f"{post['due_date']} - {post['platform']}: {post['content'][:30]}...", post['index'])

    def generate_variant(self):
        selected_index = self.post_selector.currentData()
        if selected_index is not None:
            post = self.content_calendar.df.loc[selected_index]
            variant = generate_ab_variant(post['content'], "Improve engagement")
            
            self.content_calendar.add_ab_test(selected_index, variant)
            self.update_post_selector()
            self.calendar_tab.update_table()
            
            QMessageBox.information(self, "Variant Generated", f"New variant created for post on {post['due_date']}:\n\n{variant}")
            self.update_log(f"Generated variant for post {selected_index}")
        else:
            QMessageBox.warning(self, "Error", "No post selected")

class ContentCalendarTab(QWidget):
    post_updated = pyqtSignal()

    def __init__(self, content_calendar):
        super().__init__()
        self.content_calendar = content_calendar
        self.column_mapping = {
            "Due Date": "due_date",
            "Platform": "platform",
            "Content Type": "content_type",
            "Subject": "subject",
            "Content": "content",
            "Time Slot": "time_slot",
            "Engagement Score": "engagement_score",
            "Is Variant": "is_variant"
        }
        layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().sectionClicked.connect(self.sort_table)
        self.update_table()
        layout.addWidget(self.table)
        
        button_layout = QHBoxLayout()
        add_button = QPushButton("Add Post")
        add_button.clicked.connect(self.add_post)
        button_layout.addWidget(add_button)
        
        edit_button = QPushButton("Edit Post")
        edit_button.clicked.connect(self.edit_post)
        button_layout.addWidget(edit_button)
        
        delete_button = QPushButton("Delete Post")
        delete_button.clicked.connect(self.delete_post)
        button_layout.addWidget(delete_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def sort_table(self, logical_index):
        if self.table.horizontalHeader().sortIndicatorSection() == logical_index:
            order = Qt.DescendingOrder if self.table.horizontalHeader().sortIndicatorOrder() == Qt.AscendingOrder else Qt.AscendingOrder
        else:
            order = Qt.AscendingOrder
        
        column_label = self.table.horizontalHeaderItem(logical_index).text()
        column_name = self.column_mapping.get(column_label)
        
        if column_name:
            self.content_calendar.df.sort_values(by=column_name, ascending=(order == Qt.AscendingOrder), inplace=True)
            self.update_table()
            self.table.horizontalHeader().setSortIndicator(logical_index, order)
        else:
            logging.warning(f"Column '{column_label}' not found in DataFrame")    

    def update_table(self):
        try:
            self.table.setColumnCount(8)
            self.table.setHorizontalHeaderLabels(["Due Date", "Platform", "Content Type", "Subject", "Content", "Time Slot", "Engagement Score", "Is Variant"])
            self.table.setRowCount(len(self.content_calendar.df))
            
            for i, (_, row) in enumerate(self.content_calendar.df.iterrows()):
                self.table.setItem(i, 0, QTableWidgetItem(str(row['due_date'])))  # Convert date to string
                self.table.setItem(i, 1, QTableWidgetItem(row['platform']))
                self.table.setItem(i, 2, QTableWidgetItem(row['content_type']))
                self.table.setItem(i, 3, QTableWidgetItem(row['subject']))
                self.table.setItem(i, 4, QTableWidgetItem(row['content']))
                self.table.setItem(i, 5, QTableWidgetItem(row['time_slot']))
                self.table.setItem(i, 6, QTableWidgetItem(str(row['engagement_score'])))
                self.table.setItem(i, 7, QTableWidgetItem('Yes' if row.get('is_variant', False) else 'No'))
            
            self.table.resizeColumnsToContents()
        except Exception as e:
            logging.error(f"Error updating table: {str(e)}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to update table: {str(e)}")

    def add_post(self):
        try:
            dialog = PostDialog(self.content_calendar)
            if dialog.exec_():
                self.update_table()
                self.post_updated.emit()
        except Exception as e:
            logging.error(f"Error adding post: {str(e)}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to add post: {str(e)}")
    
    def edit_post(self):
        try:
            selected_items = self.table.selectedItems()
            if selected_items:
                row = selected_items[0].row()
                post_index = self.content_calendar.df.index[row]
                dialog = PostDialog(self.content_calendar, post_index)
                if dialog.exec_():
                    self.update_table()
                    self.post_updated.emit()
            else:
                QMessageBox.warning(self, "Error", "No post selected")
        except Exception as e:
            logging.error(f"Error editing post: {str(e)}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to edit post: {str(e)}")
    
    def delete_post(self):
        try:
            selected_items = self.table.selectedItems()
            if selected_items:
                row = selected_items[0].row()
                post_index = self.content_calendar.df.index[row]
                self.content_calendar.df = self.content_calendar.df.drop(post_index)
                self.content_calendar.save()
                self.update_table()
                self.post_updated.emit()
            else:
                QMessageBox.warning(self, "Error", "No post selected")
        except Exception as e:
            logging.error(f"Error deleting post: {str(e)}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to delete post: {str(e)}")

class PostDialog(QDialog):
    def __init__(self, content_calendar, post_index=None):
        super().__init__()
        self.content_calendar = content_calendar
        self.post_index = post_index
        self.setWindowTitle("Add/Edit Post")
        self.setModal(True)
        
        try:
            layout = QFormLayout()
            
            self.due_date = QDateEdit()
            self.due_date.setCalendarPopup(True)
            self.due_date.setDate(QDate.currentDate())
            layout.addRow("Due Date:", self.due_date)
            
            self.platform = QComboBox()
            self.platform.addItems(["Twitter", "LinkedIn", "Facebook"])
            layout.addRow("Platform:", self.platform)
            
            self.content_type = QLineEdit()
            layout.addRow("Content Type:", self.content_type)
            
            self.subject = QLineEdit()
            layout.addRow("Subject:", self.subject)
            
            self.content = QTextEdit()
            layout.addRow("Content:", self.content)
            
            self.time_slot = QTimeEdit()
            self.time_slot.setDisplayFormat("HH:mm")  # Set display format to exclude seconds
            self.time_slot.setTime(QTime.currentTime())
            layout.addRow("Time Slot:", self.time_slot)
            
            button_box = QHBoxLayout()
            save_button = QPushButton("Save")
            save_button.clicked.connect(self.save_post)
            button_box.addWidget(save_button)
            
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(self.reject)
            button_box.addWidget(cancel_button)
            
            layout.addRow(button_box)
            
            self.setLayout(layout)
            
            if post_index is not None:
                self.load_post_data()
        except Exception as e:
            logging.error(f"Error initializing PostDialog: {str(e)}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to initialize dialog: {str(e)}")
    
    def save_post(self):
        try:
            post_data = {
                'due_date': self.due_date.date().toString("yyyy-MM-dd"),
                'platform': self.platform.currentText(),
                'content_type': self.content_type.text(),
                'subject': self.subject.text(),
                'content': self.content.toPlainText(),
                'time_slot': self.time_slot.time().toString("HH:mm"),  # Format time without seconds
                'engagement_score': 0,
                'is_variant': False
            }
            
            logging.info(f"Attempting to save post: {post_data}")
            
            self.content_calendar.add_post(post_data)
            logging.info("Post added successfully")
            
            self.accept()
            logging.info("PostDialog accepted")
        except Exception as e:
            logging.error(f"Error saving post: {str(e)}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to save post: {str(e)}")

    def load_post_data(self):
        try:
            post = self.content_calendar.df.loc[self.post_index]
            self.due_date.setDate(pd.to_datetime(post['due_date']).date())
            self.platform.setCurrentText(post['platform'])
            self.content_type.setText(post['content_type'])
            self.subject.setText(post['subject'])
            self.content.setPlainText(post['content'])
            self.time_slot.setTime(pd.to_datetime(post['time_slot']).time())
        except Exception as e:
            logging.error(f"Error loading post data: {str(e)}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to load post data: {str(e)}")        

class AnalyticsTab(QWidget):
    def __init__(self, content_calendar):
        super().__init__()
        self.content_calendar = content_calendar
        layout = QVBoxLayout()
        
        self.analytics_text = QTextEdit()
        self.analytics_text.setReadOnly(True)
        layout.addWidget(self.analytics_text)
        
        refresh_button = QPushButton("Refresh Analytics")
        refresh_button.clicked.connect(self.refresh_analytics)
        layout.addWidget(refresh_button)
        
        self.setLayout(layout)
        self.refresh_analytics()
    
    def refresh_analytics(self):
        analytics_text = "Content Calendar Analytics:\n\n"
        
        # Overall engagement statistics
        analytics_text += "Overall Engagement:\n"
        analytics_text += f"Average Engagement Score: {self.content_calendar.df['engagement_score'].mean():.2f}\n"
        analytics_text += f"Highest Engagement Score: {self.content_calendar.df['engagement_score'].max():.2f}\n"
        analytics_text += f"Lowest Engagement Score: {self.content_calendar.df['engagement_score'].min():.2f}\n\n"
        
        # Engagement by platform
        analytics_text += "Engagement by Platform:\n"
        platform_engagement = self.content_calendar.df.groupby('platform')['engagement_score'].mean()
        for platform, score in platform_engagement.items():
            analytics_text += f"{platform}: {score:.2f}\n"
        analytics_text += "\n"
        
        # Top performing posts
        analytics_text += "Top 5 Performing Posts:\n"
        top_posts = self.content_calendar.df.nlargest(5, 'engagement_score')
        for _, post in top_posts.iterrows():
            analytics_text += f"- {post['platform']} ({post['due_date']}): {post['content'][:50]}... (Score: {post['engagement_score']:.2f})\n"
        analytics_text += "\n"
        
        self.analytics_text.setPlainText(analytics_text)

class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        ab_group = QGroupBox("A/B Testing Settings")
        ab_layout = QFormLayout()
        self.sample_size = QLineEdit()
        self.significance_level = QLineEdit()
        ab_layout.addRow("A/B Test Sample Size:", self.sample_size)
        ab_layout.addRow("Significance Level:", self.significance_level)
        ab_group.setLayout(ab_layout)
        scroll_layout.addWidget(ab_group)
        
        twitter_group = QGroupBox("Twitter API Settings")
        twitter_layout = QFormLayout()
        self.twitter_api_key = QLineEdit()
        self.twitter_api_secret = QLineEdit()
        self.twitter_access_token = QLineEdit()
        self.twitter_access_token_secret = QLineEdit()
        twitter_layout.addRow("API Key:", self.twitter_api_key)
        twitter_layout.addRow("API Secret:", self.twitter_api_secret)
        twitter_layout.addRow("Access Token:", self.twitter_access_token)
        twitter_layout.addRow("Access Token Secret:", self.twitter_access_token_secret)
        twitter_group.setLayout(twitter_layout)
        scroll_layout.addWidget(twitter_group)
        
        linkedin_group = QGroupBox("LinkedIn API Settings")
        linkedin_layout = QFormLayout()
        self.linkedin_client_id = QLineEdit()
        self.linkedin_client_secret = QLineEdit()
        linkedin_layout.addRow("Client ID:", self.linkedin_client_id)
        linkedin_layout.addRow("Client Secret:", self.linkedin_client_secret)
        linkedin_group.setLayout(linkedin_layout)
        scroll_layout.addWidget(linkedin_group)
        
        facebook_group = QGroupBox("Facebook API Settings")
        facebook_layout = QFormLayout()
        self.facebook_access_token = QLineEdit()
        facebook_layout.addRow("Access Token:", self.facebook_access_token)
        facebook_group.setLayout(facebook_layout)
        scroll_layout.addWidget(facebook_group)
        
        openai_group = QGroupBox("OpenAI API Settings")
        openai_layout = QFormLayout()
        self.openai_api_key = QLineEdit()
        openai_layout.addRow("API Key:", self.openai_api_key)
        openai_group.setLayout(openai_layout)
        scroll_layout.addWidget(openai_group)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)
        
        self.setLayout(layout)
        
        self.load_settings()

    def load_settings(self):
        if os.path.exists('settings.json'):
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                self.sample_size.setText(str(settings.get('SAMPLE_SIZE_AB_TEST', '')))
                self.significance_level.setText(str(settings.get('SIGNIFICANCE_LEVEL', '')))
                self.twitter_api_key.setText(settings.get('TWITTER_API_KEY', ''))
                self.twitter_api_secret.setText(settings.get('TWITTER_API_SECRET', ''))
                self.twitter_access_token.setText(settings.get('TWITTER_ACCESS_TOKEN', ''))
                self.twitter_access_token_secret.setText(settings.get('TWITTER_ACCESS_TOKEN_SECRET', ''))
                self.linkedin_client_id.setText(settings.get('LINKEDIN_CLIENT_ID', ''))
                self.linkedin_client_secret.setText(settings.get('LINKEDIN_CLIENT_SECRET', ''))
                self.facebook_access_token.setText(settings.get('FACEBOOK_ACCESS_TOKEN', ''))
                self.openai_api_key.setText(settings.get('OPENAI_API_KEY', ''))

    def save_settings(self):
        settings = {
            'SAMPLE_SIZE_AB_TEST': float(self.sample_size.text()),
            'SIGNIFICANCE_LEVEL': float(self.significance_level.text()),
            'TWITTER_API_KEY': self.twitter_api_key.text(),
            'TWITTER_API_SECRET': self.twitter_api_secret.text(),
            'TWITTER_ACCESS_TOKEN': self.twitter_access_token.text(),
            'TWITTER_ACCESS_TOKEN_SECRET': self.twitter_access_token_secret.text(),
            'LINKEDIN_CLIENT_ID': self.linkedin_client_id.text(),
            'LINKEDIN_CLIENT_SECRET': self.linkedin_client_secret.text(),
            'FACEBOOK_ACCESS_TOKEN': self.facebook_access_token.text(),
            'OPENAI_API_KEY': self.openai_api_key.text()
        }
        
        with open('settings.json', 'w') as f:
            json.dump(settings, f)
        
        QMessageBox.information(self, "Settings Saved", "Your settings have been saved successfully.")

class OptimizationThread(QThread):
    update_signal = pyqtSignal(str)

    def run(self):
        run_content_optimization()
        self.update_signal.emit("Content optimization completed.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())