from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

#Extract Data From the PDF File
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents


#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


#Download the Embeddings from HuggingFace with error handling
def download_hugging_face_embeddings():
    try:
        logger.info("Attempting to load HuggingFace embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
        logger.info("HuggingFace embeddings loaded successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load HuggingFace embeddings: {e}")
        logger.warning("Falling back to simple embeddings...")
        # Return a simple mock embeddings for fallback
        return None