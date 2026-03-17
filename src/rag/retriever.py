def get_retriever(vector_store, k: int = 3):
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_context(retriever, query: str):
    return retriever.invoke(query)