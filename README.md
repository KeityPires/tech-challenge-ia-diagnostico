# tech-challenge-ia-diagnostico
Projeto de IA para suporte a diagnóstico médico — Pós Tech Fiap
# Tech Challenge - Fase 1: Diagnóstico de Câncer de Mama
Este projeto tem como objetivo o desenvolvimento de um algoritmo de Machine Learning capaz de classificar um paciente com um tumor 'Maligno' ou 'Benigno', através do treinamento com o dataset *Breast Cancer Wisconsin (Diagnostic)*. Utilizando os modelos de Árvore de Decisão e K-Nearest Neighbors para chegar ao resultado esperado.

# Tecnologias Utilizadas
- Python 3.10  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- SHAP (interpretação dos modelos)  
- Docker & Jupyter Notebook  

# Estrutura do Projeto
tech-challenge-ia-diagnostico/
├── src/
│ ├── preprocess.py # Funções de pré-processamento e visualização
│ ├── model.py # Treinamento de modelos KNN e Árvore de Decisão
│ ├── evaluate.py # Avaliação e métricas dos modelos
│ └── utils.py # Testes e funções auxiliares
│
├── notebooks/
│ └── 02_exploracao_dados_cancer_mama.ipynb # Notebook principal
│
├── data/
│ └── data.csv # Base de dados utilizada
│
├── requirements.txt # Dependências do projeto
├── Dockerfile # Configuração do container Docker
└── README.md # Este arquivo

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

👩‍💻 Autora

Keity Pires
📧 keityrcpires@gmail.com

Pós-Tech FIAP
🗓️ 2025


