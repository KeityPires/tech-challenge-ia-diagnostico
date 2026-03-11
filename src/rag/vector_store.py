from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def build_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [
        Document(page_content=doc["content"], metadata={"source": doc["source"]})
        for doc in documents
    ]
    return FAISS.from_documents(docs, embeddings)