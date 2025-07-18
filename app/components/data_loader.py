from app.components.pdf_loader import load_pdf_files, create_text_chunks
from app.components.vector_store import save_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DATA_PATH

logger = get_logger(__name__)

def process_and_save_pdfs():
    try:
        logger.info("Making the vector store...")

        documents = load_pdf_files()

        text_chunks = create_text_chunks(documents)

        save_vector_store(text_chunks)

        logger.info("Successfully created the vector store...")
    except Exception as e:
        error_msg = CustomException(f"Error creating vector store: {e}")
        logger.error(str(error_msg))
    
if __name__ == "__main__":
    process_and_save_pdfs()