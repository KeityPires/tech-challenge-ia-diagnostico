from pathlib import Path

def load_documents(folder_path: str):
    documents = []
    for file_path in Path(folder_path).glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append({
                "source": file_path.name,
                "content": f.read()
            })
    return documents