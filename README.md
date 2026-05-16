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
Na Fase 3, o projeto evolui para um sistema de assistência médica baseado em LLM, utilizando arquitetura RAG (Retrieval-Augmented Generation) e Fine Tuning.

O sistema é capaz de:
- Responder perguntas médicas
- Consultar base de conhecimento (MedQuAD)
- Explicar suas respostas (Explainability)
- Aplicar regras de segurança (Guardrails)
- Registrar logs das interações
- Indicar fontes utilizadas nas respostas



Utilizando o dataset:
- Breast Cancer Wisconsin (Diagnostic)

Modelos utilizados:
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

Nesta fase foram desenvolvidos:
- análise exploratória de dados;
- pré-processamento;
- treinamento de modelos;
- avaliação de métricas;
- visualizações estatísticas.

# Tech Challenge - Fase 2: Diagnóstico de Câncer de Mama

Na Fase 2, os modelos foram otimizados por meio de Algoritmos Genéticos e integrados a uma Large Language Model (LLM) para geração de explicações clínicas em linguagem natural, com foco em interpretabilidade e suporte à decisão médica.

Principais evoluções:
- otimização com DEAP;
- explicabilidade com SHAP;
- integração com OpenAI GPT;
- interpretação automática dos resultados;
- suporte à análise clínica textual.

# Tech Challenge - Fase 3: Assistente Médico Virtual

Na Fase 3, o projeto evolui para um sistema de assistência médica baseado em LLM, utilizando arquitetura RAG (Retrieval-Augmented Generation) e Fine Tuning.

O sistema é capaz de:
- responder perguntas médicas;
- consultar base de conhecimento (MedQuAD);
- explicar suas respostas;
- aplicar regras de segurança (Guardrails);
- registrar logs das interações;
- indicar fontes utilizadas nas respostas.

# Tech Challenge - Fase 4: Pipeline Multimodal para Apoio Preventivo

Na Fase 4, o projeto evolui para uma arquitetura multimodal focada em apoio preventivo em saúde da mulher, utilizando:
- análise de vídeo;
- análise de áudio;
- transcrição automática;
- fusão multimodal;
- IA generativa;
- visão computacional;
- análise acústica.

O sistema realiza:
- detecção de emoções aparentes;
- análise vocal;
- identificação de sinais complementares de desconforto emocional;
- geração de interpretação multimodal;
- classificação de risco preventivo.

Importante:
- o sistema NÃO realiza diagnóstico médico;
- o sistema NÃO realiza diagnóstico psicológico;
- a análise possui finalidade exclusivamente preventiva e educacional.

# Arquitetura do Sistema

## Pipeline RAG

Pergunta do usuário
- Guardrails (segurança)
- Retriever (FAISS / ChromaDB)
- Contexto (MedQuAD)
- LLM (Mistral via Ollama)
- Resposta médica
- Fontes (Explainability)
- Logging

---

## Pipeline Multimodal

Upload de vídeo e áudio
- AWS S3
- YOLOv8
- AWS Rekognition
- Speech Recognition
- Análise acústica
- Fusão multimodal
- LLM interpretativa
- Guardrails
- Resposta final

# Orquestração com LangGraph

Fluxo principal:
guardrails -> retrieve -> generate -> format -> log

Fluxo multimodal:
upload -> video_analysis -> audio_analysis -> multimodal_fusion -> generate -> alert

# Dataset Utilizado

## Fase 1
- Breast Cancer Wisconsin (Diagnostic)

## Fase 3
- MedQuAD (Medical Question Answering Dataset)

Repositório:
https://github.com/abachaa/MedQuAD

Coleção utilizada:
- 1_CancerGov_QA

## Fase 4

### Vídeos
Os vídeos utilizados na validação multimodal foram obtidos principalmente a partir de:

- Pexels (https://www.pexels.com/pt-br/videos/)

Foram utilizados:
- vídeos curtos;
- vídeos simulando consultas;
- expressões emocionais aparentes;
- cenários de tristeza, medo, desconforto e neutralidade.

Objetivo:
- validar o pipeline multimodal;
- avaliar análise facial;
- testar fusão entre áudio e vídeo;
- reduzir falso positivo em cenários emocionais distintos.

### Áudios

Os áudios utilizados foram:
- sintéticos;
- gravados manualmente;
- transcritos em português;
- combinados aos vídeos para simulação multimodal.

As falas continham cenários como:
- ansiedade;
- medo;
- fadiga;
- sofrimento emocional;
- cenários neutros.

# Tecnologias Utilizadas

## Machine Learning
- Scikit-learn
- SHAP
- DEAP

## IA Generativa e LLMs
- LangChain
- LangGraph
- Ollama
- OpenAI GPT
- Mistral

## Processamento Multimodal
- OpenCV
- YOLOv8
- AWS Rekognition
- librosa
- SpeechRecognition
- moviepy

## Vetorização e Busca Semântica
- FAISS
- ChromaDB

## Cloud
- AWS S3
- AWS Rekognition

## Ambiente
- Python 3.10
- Docker
- Jupyter Notebook

# Estrutura do Projeto

```text
tech-challenge-ia-diagnostico/
│
├── src/
│
│ ├── assistant/
│ │ ├── medical_assistant.py
│ │ ├── prompts.py
│ │ └── response_formatter.py
│ │
│ ├── llm/
│ │ ├── fine_tuning.py
│ │ └── ollama_client.py
│ │
│ ├── multimodal/
│ │ ├── __init__.py
│ │ ├── alert_generator.py
│ │ ├── audio_processor.py
│ │ ├── media_utils.py
│ │ ├── multimodal_fusion.py
│ │ └── video_processor.py
│ │
│ ├── rag/
│ │ ├── documents_loader.py
│ │ ├── retriever.py
│ │ └── vector_store.py
│ │
│ ├── security/
│ │ ├── guardrails.py
│ │ └── logging_system.py
│ │
│ ├── workflows/
│ │ └── langgraph_flow.py
│ │
│ ├── config.py
│ ├── data_preprocessing.py
│ ├── evaluation.py
│ ├── genetic_optimization_knn.py
│ ├── genetic_optimization_tree.py
│ ├── llm_interpretation.py
│ ├── model_training.py
│ └── utils.py
│
├── tests/
│ ├── test_ga_and_llm.py
│ ├── test_multimodal_pipeline.py
│ ├── test_ollama_connection.py
│ ├── test_ollama_langchain.py
│ └── test_pipeline.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

# Funcionalidades Implementadas

## Áudio
- transcrição automática;
- análise de pitch;
- análise de energia vocal;
- análise de pausas;
- detecção de hesitação;
- classificação emocional textual;
- interpretação acústica cautelosa.

## Vídeo
- detecção de pessoas com YOLOv8;
- detecção facial com AWS Rekognition;
- análise de emoções aparentes;
- emotion percentages;
- emotion transitions;
- análise temporal de frames.

## Fusão Multimodal
- ponderação:
  - áudio: 60%;
  - vídeo: 40%.
- geração de score final;
- classificação de risco;
- interpretação multimodal;
- alertas preventivos.

# Segurança (Guardrails)

O sistema implementa regras para evitar respostas inadequadas:

- não fornece diagnóstico definitivo;
- não prescreve tratamento;
- não fornece dosagens;
- bloqueia perguntas de alto risco;
- reforça limitações clínicas do sistema.

Classificação:
- Low
- Medium
- High

# Logging

As interações são registradas em:

```text
data/assistant_logs.jsonl
```

Informações registradas:
- timestamp;
- pergunta;
- resposta resumida;
- nível de risco;
- status;
- número de documentos recuperados;
- fontes utilizadas.

# Testes Automatizados

O projeto possui testes automatizados utilizando:
- pytest.

Cobertura:
- pipeline multimodal;
- integração com Ollama;
- LangChain;
- algoritmos genéticos;
- funções auxiliares.

# Fine-Tuning (Kaggle)

O fine-tuning foi executado em ambiente Kaggle devido a limitações de hardware local.

Bibliotecas utilizadas:
- Transformers;
- PEFT;
- Accelerate;
- PyTorch;
- LoRA.

# Como Executar o Projeto

## Instalar dependências

```bash
pip install -r requirements.txt
```

## Executar testes

```bash
pytest -v
```

## Executar com Docker

### Build da imagem

```bash
docker build -t tech-challenge-ia .
```

### Executar container

```bash
docker run -p 8888:8888 tech-challenge-ia
```

## Abrir o Jupyter Notebook

Acesse:
```text
http://localhost:8888
```

# Modelos Utilizados

## Machine Learning
- KNN
- Decision Tree

## LLMs
- Mistral via Ollama
- OpenAI GPT

## Visão Computacional
- YOLOv8
- AWS Rekognition

# Documentação da API

Este projeto integra:
- OpenAI GPT;
- Ollama;
- LangChain;
- LangGraph.

# Observação sobre o Ollama

Para execução da Fase 3 e Fase 4, o Ollama deve estar instalado localmente.

Exemplo:

```bash
ollama pull mistral
ollama pull nomic-embed-text
ollama serve
```

# Limitações

- Não substitui avaliação médica profissional;
- Não realiza diagnóstico clínico;
- Emoções representam apenas estados aparentes;
- Sinais acústicos são complementares;
- Dependência da qualidade do áudio e vídeo;
- Dependência da recuperação documental no pipeline RAG.

👩‍💻 Autora

Keity Pires
📧 keityrcpires@gmail.com

Pós-Tech FIAP
🗓️ 2025


