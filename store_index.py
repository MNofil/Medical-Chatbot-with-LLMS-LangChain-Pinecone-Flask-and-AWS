from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from src.helper import load_pdf, filter_to_minimal_docs, text_split, download_embeddings
from langchain_pinecone import PineconeVectorStore

extract_text=load_pdf("data")

# Filter and split the text
minimal_docs = filter_to_minimal_docs(extract_text)
texts_chunk = text_split(minimal_docs)

# Download embeddings
embedding = download_embeddings()

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("Pinecone Key Loaded:", PINECONE_API_KEY is not None)
print("OpenAI Key Loaded:", OPENAI_API_KEY is not None)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define index
index_name = "medical-chatbot"

# Create index if not exists
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,   # ✅ changed from index_name to name
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to index
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)
