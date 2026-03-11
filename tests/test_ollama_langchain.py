from src.llm.ollama_client import get_llm

llm = get_llm()

response = llm.invoke("Responda em português: o que é câncer de mama?")

print(response.content)