from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings


# Extract text from PDF
def load_pdf(file_path):
    loader = DirectoryLoader(
        file_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: list[Document]) -> list[Document]:
    """
    Given a list of document objects, return a new list of document objects containing
    only 'source' in metadata and the original document content.
    """
    minimal_docs: list[Document] = []
    for doc in docs:
        src = doc.metadata.get('source')
        minimal_docs.append(
            Document(
            page_content=doc.page_content,
            metadata={'source': src}
            )
        )
    return minimal_docs


# Split the data into smaller chunks for processing
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

