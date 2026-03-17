from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_vector_store(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = FAISS.from_documents(docs, embeddings)

    num_chunks = len(docs)

    return vector_store, num_chunks

def save_vector_store(vector_store, path: str):
    vector_store.save_local(path)
    print(f"Base vetorial salva em: {path}")


def load_vector_store(path: str):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    print(f"Base vetorial carregada de: {path}")
    return vector_store