from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import time
import os
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001", store_dir: str = "vector_stores"):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key
        )
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
    
    def create_vector_store(self, chunks: List, video_id: str, batch_size: int = 10) -> FAISS:
        """Create FAISS vector store from chunks with batching"""
        vectorstore = None
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embedding=self.embeddings)
                logger.info(f"Created vector store with batch 0-{min(i+batch_size, len(chunks))}")
            else:
                vectorstore.add_documents(batch)
                logger.info(f"Added batch {i}-{min(i+batch_size, len(chunks))}")
            
            # Rate limiting to avoid API quotas
            if i + batch_size < len(chunks):
                time.sleep(2)
        
        # Save to disk
        self.save_vector_store(vectorstore, video_id)
        return vectorstore
    
    def save_vector_store(self, vectorstore: FAISS, video_id: str):
        """Persist vector store to disk"""
        path = os.path.join(self.store_dir, video_id)
        vectorstore.save_local(path)
        logger.info(f"Saved vector store to {path}")
    
    def load_vector_store(self, video_id: str) -> Optional[FAISS]:
        """Load vector store from disk"""
        path = os.path.join(self.store_dir, video_id)
        if os.path.exists(path):
            vectorstore = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded vector store from {path}")
            return vectorstore
        return None
    
    def delete_vector_store(self, video_id: str):
        """Delete vector store from disk"""
        path = os.path.join(self.store_dir, video_id)
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
            logger.info(f"Deleted vector store {path}")
