import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = "ytc_database.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    video_url TEXT NOT NULL,
                    title TEXT,
                    processed_at TEXT NOT NULL,
                    chunk_count INTEGER,
                    status TEXT DEFAULT 'completed'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id)
                )
            """)
            conn.commit()
            logger.info("Database initialized")
    
    def add_video(self, video_id: str, video_url: str, chunk_count: int, title: str = ""):
        """Add processed video to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO videos (video_id, video_url, title, processed_at, chunk_count) VALUES (?, ?, ?, ?, ?)",
                (video_id, video_url, title, datetime.now().isoformat(), chunk_count)
            )
            conn.commit()
            logger.info(f"Added video {video_id} to database")
    
    def get_all_videos(self) -> List[Dict]:
        """Get all processed videos"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM videos ORDER BY processed_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_video(self, video_id: str) -> Optional[Dict]:
        """Get specific video"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def delete_video(self, video_id: str):
        """Delete video and its chat history"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chat_history WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
            conn.commit()
            logger.info(f"Deleted video {video_id}")
    
    def add_chat(self, video_id: str, question: str, answer: str):
        """Save chat interaction"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_history (video_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
                (video_id, question, answer, datetime.now().isoformat())
            )
            conn.commit()
    
    def get_chat_history(self, video_id: str) -> List[Dict]:
        """Get chat history for a video"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM chat_history WHERE video_id = ? ORDER BY created_at ASC",
                (video_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
