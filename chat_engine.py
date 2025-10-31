from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatEngine:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.2):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature
        )
        
        self.prompt = PromptTemplate(
            template="""
                You are a helpful assistant that answers questions about YouTube videos.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, say you don't know.
                Be concise but informative.

                Context from video:
                {context}

                Question: {question}

                Answer:
            """,
            input_variables=['context', 'question']
        )
    
    @staticmethod
    def format_docs(retrieved_docs):
        """Format retrieved documents into context string"""
        return "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    def create_chain(self, vectorstore: FAISS, k: int = 4):
        """Create RAG chain"""
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(self.format_docs),
            'question': RunnablePassthrough()
        })
        
        chain = parallel_chain | self.prompt | self.llm | StrOutputParser()
        return chain
    
    def chat(self, vectorstore: FAISS, question: str, k: int = 4) -> str:
        """Get answer for a question"""
        try:
            chain = self.create_chain(vectorstore, k)
            answer = chain.invoke(question)
            logger.info(f"Answered question: {question[:50]}...")
            return answer
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            raise
