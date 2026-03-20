# tech-challenge-ia-diagnostico
Projeto de IA para suporte à decisão médica — Pós Tech FIAP

# Tech Challenge - Fase 1: Diagnóstico de Câncer de Mama
Este projeto tem como objetivo o desenvolvimento de um algoritmo de Machine Learning capaz de classificar um paciente com um tumor como 'Maligno' ou 'Benigno', utilizando o dataset *Breast Cancer Wisconsin (Diagnostic)*.

Modelos utilizados:
- K-Nearest Neighbors (KNN)
- Decision Tree

# Tech Challenge - Fase 2: Diagnóstico de Câncer de Mama
Na Fase 2, os modelos foram otimizados por meio de Algoritmos Genéticos e integrados a uma Large Language Model (LLM) para geração de explicações clínicas em linguagem natural, com foco em interpretabilidade e suporte à decisão médica.

Principais evoluções:
- Otimização com DEAP
- Explicabilidade com SHAP
- Integração com OpenAI GPT

# Tech Challenge - Fase 3: Assistente Médico Virtual
Na Fase 3, o projeto evolui para um sistema completo de assistência médica baseado em LLM, utilizando arquitetura RAG (Retrieval-Augmented Generation).

O sistema é capaz de:
- Responder perguntas médicas
- Consultar base de conhecimento (MedQuAD)
- Explicar suas respostas (Explainability)
- Aplicar regras de segurança (Guardrails)
- Registrar logs das interações
- Indicar fontes utilizadas nas respostas

# Arquitetura do Sistema
Pipeline principal:
Pergunta do usuário
- Guardrails (segurança)
- Retriever (FAISS)
- Contexto (MedQuAD)
- LLM (Mistral via Ollama)
- Resposta médica
- Fontes (Explainability)
- Logging

Orquestração com LangGraph:
guardrails -> retrieve -> generate -> format -> log

# Dataset Utilizado

Fase 1:
- Breast Cancer Wisconsin (Diagnostic)

Fase 3:
- MedQuAD (Medical Question Answering Dataset)

Repositório:
https://github.com/abachaa/MedQuAD

Coleção utilizada:
- 1_CancerGov_QA

# Tecnologias Utilizadas
- Python 3.10  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- SHAP  
- DEAP  
- LangChain  
- LangGraph  
- FAISS  
- Ollama  
- Docker  
- Jupyter Notebook  

# Estrutura do Projeto
tech-challenge-ia-diagnostico/
│
├── src/
│ ├── preprocess.py
│ ├── model.py
│ ├── evaluate.py
│ ├── genetic_optimization_tree.py
│ ├── genetic_optimization_knn.py
│ ├── llm_interpretation.py
│ ├── utils.py
│ │
│ ├── rag/
│ │ ├── documents_loader.py
│ │ ├── vector_store.py
│ │ └── retriever.py
│ │
│ ├── llm/
│ │ ├── ollama_client.py
│ │ ├── prepare_dataset.py
│ │ └── fine_tuning.py
│ │
│ ├── assistant/
│ │ ├── medical_assistant.py
│ │ └── response_formatter.py
│ │
│ ├── security/
│ │ ├── guardrails.py
│ │ └── logging_system.py
│ │
│ ├── workflows/
│ │ └── langgraph_flow.py
│
├── notebooks/
│ ├── 02_exploracao_dados_cancer_mama.ipynb
│ └── 03_medquad_exploracao.ipynb
│
├── data/
│ └── medical_qa/
│
├── tests/
│
├── requirements.txt
├── Dockerfile
└── README.md

# Segurança (Guardrails)
O sistema implementa regras para evitar respostas inadequadas:

- Não fornece diagnóstico definitivo  
- Não prescreve tratamento  
- Não fornece dosagens  
- Bloqueia perguntas de alto risco  

Classificação:
- Low: informativo  
- Medium: interpretação sensível  
- High: bloqueado  

# Logging
As interações são registradas em:

data/assistant_logs.jsonl

Informações registradas:
- timestamp  
- pergunta  
- resposta (resumo)  
- nível de risco  
- status  
- número de documentos recuperados  
- fontes utilizadas  

# Explainability
Todas as respostas incluem:

- fontes consultadas  
- identificação do documento  
- coleção  
- arquivo original  

# Como Executar o Projeto com Docker
- docker build -t tech-challenge-ia .

# Executar o container
- docker run -p 8888:8888 tech-challenge-ia

# Abrir o Jupyter Notebook
- Acesse no navegador:
http://localhost:8888

# Modelos Treinados
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Mistral (LLM local via Ollama)

## Documentação da API
Este projeto integra a API da OpenAI (GPT) para interpretação dos resultados dos modelos e geração de explicações clínicas em linguagem natural.
A documentação completa de configuração, uso e prompt engineering encontra-se em:
- 'docs/api_openai.md'

## Observação sobre o Ollama

Para a execução da Fase 3, o Ollama deve estar instalado e em execução no ambiente local, com os modelos necessários previamente baixados.

Exemplo:

ollama pull mistral
ollama pull nomic-embed-text
ollama serve

Limitações:
- Não substitui avaliação médica profissional
- Respostas limitadas à base MedQuAD
- Dependência da qualidade da recuperação (RAG)
- Pipeline de fine-tuning pode ser expandido

👩‍💻 Autora

Keity Pires
📧 keityrcpires@gmail.com

Pós-Tech FIAP
🗓️ 2025


