def format_sources(docs):
    sources = []

    for i, doc in enumerate(docs):
        metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})

        sources.append({
            "label": f"Fonte {i+1}",
            "collection": metadata.get("collection", "desconhecida"),
            "source_file": metadata.get("source_file", "desconhecido"),
            "id": metadata.get("id", f"doc_{i+1}")
        })

    return sources