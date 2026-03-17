from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_vector_store(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_documents = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(
        split_documents,
        embedding_model
    )

    return vector_store


def save_vector_store(vector_store, path: str):
    vector_store.save_local(path)


def load_vector_store(path: str):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True
    )