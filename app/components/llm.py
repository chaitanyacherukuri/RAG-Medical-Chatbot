from langchain_groq import ChatGroq
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import GROQ_API_KEY

logger = get_logger(__name__)

def load_llm(model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key: str = GROQ_API_KEY):
    try:
        logger.info("Loading LLM from groq...")
        llm = ChatGroq(
            model=model_name,
            temperature=0.3,
            groq_api_key=groq_api_key,
            max_tokens=256
        )
        logger.info("Successfully loaded LLM from groq")
        return llm
    except Exception as e:
        error_msg = CustomException(f"Error loading LLM from groq: {e}")
        logger.error(str(error_msg))
        return None
