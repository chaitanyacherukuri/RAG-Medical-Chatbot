import os
from langchain_community.vectorstores import FAISS
from app.components.emeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

def load_vector_store():
    try:
        logger.info("Loading vector store...")
        embedding_model = get_embedding_model()
        if not os.path.exists(DB_FAISS_PATH):
            raise CustomException("Vector store does not exist")
        
        vector_store = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        logger.info("Successfully loaded vector store")
        return vector_store
    except Exception as e:
        error_msg = CustomException(f"Error loading vector store: {e}")
        logger.error(str(error_msg))
        raise error_msg
    
def save_vector_store(chunks):
    try:
        if not chunks:
            raise CustomException("No chunks were found")
        
        logger.info("Creating a vector store...")
        embedding_model = get_embedding_model()
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        logger.info("Successfully created a vector store...")
        vector_store.save_local(DB_FAISS_PATH)
        logger.info("Successfully saved vector store")
        return vector_store
    except Exception as e:
        error_msg = CustomException(f"Error creating vector store: {e}")
        logger.error(str(error_msg))
        raise error_msg