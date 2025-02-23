
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from call_llm import call_llm_offline
app = FastAPI()

# Request model
class QueryRequest(BaseModel):
    query: str
    pdf1_text: str
    pdf2_text: str


# Function to create vector store for RAG retrieval
def create_rag_retriever(pdf_texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = [Document(page_content=text) for text in pdf_texts]
    split_texts = text_splitter.split_documents(documents)

    # Use SentenceTransformers for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_texts, embeddings)

    return vector_store.as_retriever()


@app.post("/query/")
def get_query_response(request: QueryRequest):
    pdf_texts = [request.pdf1_text, request.pdf2_text]
    retriever = create_rag_retriever(pdf_texts)

    # Retrieve relevant context from vector DB
    retrieved_docs = retriever.get_relevant_documents(request.query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Generate response using Mistral-7B
    prompt = f"Context: {context}\n\nQuestion: {request.query}\n\nAnswer:"
    # response = llm_pipeline(prompt, max_length=300, truncation=True)[0]["generated_text"]
    response = call_llm_offline(user_query=prompt)

    return {"response": response}
