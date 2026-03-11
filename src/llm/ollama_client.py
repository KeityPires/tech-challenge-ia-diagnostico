from langchain_community.chat_models import ChatOllama

def get_llm(model_name: str = "mistral", temperature: float = 0.2):
    return ChatOllama(
        model=model_name,
        temperature=temperature
    )