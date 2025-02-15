import pandas as pd
import threading
import logging
import traceback
import os
import json
import time
import shutil
from queue import Queue
from datetime import datetime

class ContentCalendar:
    def __init__(self, file_path='content_calendar.csv'):
        self.file_path = file_path
        self.lock = threading.Lock()
        self.save_queue = Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        self.df = pd.DataFrame(columns=['due_date', 'platform', 'content_type', 'subject', 'content', 
                                      'time_slot', 'engagement_score', 'is_variant'])
        self.load()

    def load(self):
        try:
            if os.path.exists(self.file_path):
                # Read the CSV file
                self.df = pd.read_csv(self.file_path)
                
                # Convert due_date using mixed format
                self.df['due_date'] = pd.to_datetime(
                    self.df['due_date'],
                    format='mixed',
                    dayfirst=False
                ).dt.date
                
                # Handle time_slot conversion
                if 'time_slot' in self.df.columns:
                    def convert_time(time_str):
                        if pd.isna(time_str):
                            return '00:00'
                        try:
                            # First try HH:MM format
                            return pd.to_datetime(time_str, format='%H:%M').strftime('%H:%M')
                        except:
                            try:
                                # Then try HH:MM:SS format
                                return pd.to_datetime(time_str, format='%H:%M:%S').strftime('%H:%M')
                            except:
                                try:
                                    # Finally try full datetime format and extract time
                                    return pd.to_datetime(time_str).strftime('%H:%M')
                                except:
                                    return '00:00'  # Default time if all parsing fails
                    
                    self.df['time_slot'] = self.df['time_slot'].apply(convert_time)
                
                # Ensure other numeric columns are properly typed
                numeric_columns = ['engagement_score', 'likes', 'comments', 'shares']
                for col in numeric_columns:
                    if col in self.df.columns:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
                # Ensure boolean columns are properly typed
                if 'is_variant' in self.df.columns:
                    self.df['is_variant'] = self.df['is_variant'].astype(bool)
                
                logging.info(f"Loaded content calendar from {self.file_path}")
            else:
                logging.warning(f"File {self.file_path} not found. Using empty DataFrame.")
                self.df = pd.DataFrame(columns=['due_date', 'platform', 'content_type', 'subject', 'content', 
                                              'time_slot', 'engagement_score', 'is_variant'])
        except Exception as e:
            logging.error(f"Error loading content calendar: {str(e)}\n{traceback.format_exc()}")
            raise

    def _save_worker(self):
        while True:
            df_to_save = self.save_queue.get()
            if df_to_save is None:
                break
            self._save_to_file(df_to_save)
            self.save_queue.task_done()

    def _save_to_file(self, df):
        try:
            logging.info("Attempting to save content calendar")
            temp_file = self.file_path + '.temp'
            backup_file = self.file_path + '.bak'
            
            # Convert date to string format before saving
            df = df.copy()
            df['due_date'] = df['due_date'].astype(str)
            
            # Save to temporary file
            df.to_csv(temp_file, index=False)
            
            # If original file exists, create backup
            if os.path.exists(self.file_path):
                shutil.copy2(self.file_path, backup_file)
            
            # Replace original with new file
            shutil.move(temp_file, self.file_path)
            
            logging.info("Content calendar saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving content calendar: {str(e)}\n{traceback.format_exc()}")
            # Try to restore from backup if save failed
            if os.path.exists(backup_file):
                shutil.copy2(backup_file, self.file_path)
                logging.info("Restored from backup file")

    def save(self):
        with self.lock:
            df_copy = self.df.copy()
        self.save_queue.put(df_copy)

    def add_post(self, post_data):
        try:
            logging.info(f"Adding new post: {post_data}")
            # Convert due_date string to date object if it isn't already
            if isinstance(post_data['due_date'], str):
                post_data['due_date'] = datetime.strptime(post_data['due_date'], "%Y-%m-%d").date()
            
            # Ensure time_slot is in HH:MM format
            if 'time_slot' in post_data and isinstance(post_data['time_slot'], str):
                try:
                    post_data['time_slot'] = datetime.strptime(post_data['time_slot'], "%H:%M").strftime("%H:%M")
                except ValueError:
                    try:
                        post_data['time_slot'] = datetime.strptime(post_data['time_slot'], "%H:%M:%S").strftime("%H:%M")
                    except ValueError:
                        post_data['time_slot'] = "00:00"
            
            with self.lock:
                new_row = pd.DataFrame([post_data])
                self.df = pd.concat([self.df, new_row], ignore_index=True)
                logging.info(f"Post added to DataFrame. New DataFrame shape: {self.df.shape}")
            self.save()
            logging.info("Post added and save initiated")
        except Exception as e:
            logging.error(f"Error adding post: {str(e)}\n{traceback.format_exc()}")
            raise

    def update_post(self, index, **kwargs):
        try:
            with self.lock:
                for key, value in kwargs.items():
                    self.df.at[index, key] = value
                self.save()
            logging.info(f"Post {index} updated successfully with {kwargs}")
        except Exception as e:
            logging.error(f"Error updating post: {str(e)}\n{traceback.format_exc()}")
            raise

    def delete_post(self, index):
        with self.lock:
            self.df = self.df.drop(index)
            self.save()

    def get_future_posts(self):
        future_posts = self.df[pd.isnull(self.df['post_id'])].to_dict('records')
        for i, post in enumerate(future_posts):
            post['index'] = self.df[pd.isnull(self.df['post_id'])].index[i]
        return future_posts

    def get_past_posts(self):
        return self.df[pd.notnull(self.df['post_id'])].to_dict('records')

    def add_ab_test(self, original_post_index, variant_content):
        with self.lock:
            original_post = self.df.loc[original_post_index].to_dict()
            variant_post = original_post.copy()
            variant_post['content'] = variant_content
            variant_post['is_variant'] = True
            variant_post['original_post_id'] = original_post['post_id']
            variant_post['post_id'] = None  # Reset post_id for the variant
            variant_post['engagement_score'] = None  # Reset engagement score
            
            self.df = self.df.append(variant_post, ignore_index=True)
            self.save()

    def get_ab_test_results(self):
        ab_tests = self.df[self.df['is_variant'] == True]
        results = []
        for _, variant in ab_tests.iterrows():
            original = self.df.loc[self.df['post_id'] == variant['original_post_id']].iloc[0]
            if pd.notnull(variant['engagement_score']) and pd.notnull(original['engagement_score']):
                results.append({
                    'original': original.to_dict(),
                    'variant': variant.to_dict()
                })
        return results

    def get_top_performing_posts(self, n=5):
        return self.df.nlargest(n, 'engagement_score').to_dict('records')

    def get_engagement_stats(self):
        return {
            'average': self.df['engagement_score'].mean(),
            'max': self.df['engagement_score'].max(),
            'min': self.df['engagement_score'].min(),
            'by_platform': self.df.groupby('platform')['engagement_score'].mean().to_dict()
        }

    def get_post_by_id(self, post_id):
        return self.df[self.df['post_id'] == post_id].to_dict('records')[0]

    def get_posts_by_date_range(self, start_date, end_date):
        mask = (self.df['due_date'] >= start_date) & (self.df['due_date'] <= end_date)
        return self.df.loc[mask].to_dict('records')

    def get_posts_by_platform(self, platform):
        return self.df[self.df['platform'] == platform].to_dict('records')

    def update_engagement_metrics(self, post_id, likes, comments, shares):
        with self.lock:
            index = self.df[self.df['post_id'] == post_id].index[0]
            self.df.at[index, 'likes'] = likes
            self.df.at[index, 'comments'] = comments
            self.df.at[index, 'shares'] = shares
            self.df.at[index, 'engagement_score'] = likes + (comments * 2) + (shares * 3)  # Example scoring
            self.save()

    def get_content_insights(self):
        return {
            'avg_content_length': self.df['content'].str.len().mean(),
            'avg_engagement_by_content_length': self.df.groupby(pd.cut(self.df['content'].str.len(), bins=5))['engagement_score'].mean().to_dict(),
            'top_subjects': self.df.groupby('subject')['engagement_score'].mean().nlargest(5).to_dict()
        }

    def __del__(self):
        self.save_queue.put(None)  # Signal the save thread to exit
        self.save_thread.join()    # Wait for the save thread to finish