import os
from langchain_openai import OpenAIEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import OPENAI_API_KEY

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Loading OpenAI embedding model...")
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set. Please check your .env file or environment variables.")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
        logger.info("Successfully loaded OpenAI embedding model")
        return embeddings
    except Exception as e:
        error_msg = CustomException(f"Error loading OpenAI embedding model: {e}")
        logger.error(str(error_msg))
        raise error_msg