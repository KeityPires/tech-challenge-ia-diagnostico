def retrieve_context(vector_store, query: str, k: int = 3):
    results = vector_store.similarity_search(query, k=k)
    return results