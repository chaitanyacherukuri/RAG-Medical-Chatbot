from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

prompt_template = """ Answer the medical question in maximum 2-3 lines using only the information provided in the context\n
context: 
{context}

question: 
{query}
"""

def create_prompt():
    return PromptTemplate(
        template=prompt_template, input_variables=["context", "query"]
    )

def create_qa_chain():
    try:
        vector_store = load_vector_store()

        if vector_store is None:
            raise CustomException("Vector store is empty or not found")
        
        llm = load_llm()

        if llm is None:
            raise CustomException("LLM is not loaded")
        
        prompt = create_prompt()

        parser = StrOutputParser()

        retriever = vector_store.as_retriever(search_kwargs={'k': 1})

        parallel_chain = RunnableParallel({
            "query": RunnablePassthrough(),
            "context": retriever | RunnableLambda(lambda docs: "\n\n".join([doc.page_content for doc in docs]))
        })

        qa_chain = parallel_chain | prompt | llm | parser

        logger.info("Successfully created the QA chain")

        return qa_chain
    
    except Exception as e:
        error_message = CustomException("Failed to make a QA chain", e)
        logger.error(str(error_message))
        return None
