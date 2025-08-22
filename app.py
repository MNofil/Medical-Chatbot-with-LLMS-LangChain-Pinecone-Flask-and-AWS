from flask import Flask, render_template, request, jsonify
from src.helper import download_embeddings,load_pdf, filter_to_minimal_docs, text_split
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embedding = download_embeddings()

index_name = "medical-chatbot"

extract_text=load_pdf("data")

# Filter and split the text
minimal_docs = filter_to_minimal_docs(extract_text)
texts_chunk = text_split(minimal_docs)

docsearch = PineconeVectorStore.from_documents(
    index_name=index_name,
    embedding=embedding,
    documents=texts_chunk,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(
    model="llama3-8b-8192",
    groq_api_key="xxxxxxxx"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['POST','GET'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)