from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, chunk_size: int = 1700, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
            r'youtube\.com\/embed\/([^&\n?#]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def load_transcript(self, video_url: str) -> str:
        """Load transcript from YouTube video"""
        try:
            loader = YoutubeLoader.from_youtube_url(
                video_url, 
                add_video_info=False,
                language=["en"]
            )
            docs = loader.load()
            transcript = " ".join(doc.page_content for doc in docs)
            logger.info(f"Successfully loaded transcript (length: {len(transcript)})")
            return transcript
        except Exception as e:
            logger.error(f"Failed to load transcript: {e}")
            raise Exception(f"Could not load transcript: {str(e)}")
    
    def chunk_transcript(self, transcript: str) -> List:
        """Split transcript into chunks"""
        chunks = self.splitter.create_documents([transcript])
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def process_video(self, video_url: str) -> tuple[str, List]:
        """Full pipeline: load and chunk transcript"""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        transcript = self.load_transcript(video_url)
        chunks = self.chunk_transcript(transcript)
        return video_id, chunks
