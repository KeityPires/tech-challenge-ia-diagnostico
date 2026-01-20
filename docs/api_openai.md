Documentação da API – Integração com OpenAI GPT

Projeto: Tech Challenge – Diagnóstico de Câncer de Mama 

1. Visão Geral

Este projeto integra uma Large Language Model (LLM) por meio da API da OpenAI (GPT) para realizar a interpretação em linguagem natural dos resultados gerados pelos modelos de Machine Learning (KNN e Árvore de Decisão), previamente otimizados por Algoritmos Genéticos.

A LLM é utilizada para:

- Traduzir métricas estatísticas em explicações clínicas compreensíveis;
- Gerar laudos textuais de apoio à decisão médica;
- Produzir insights acionáveis a partir de probabilidades e importâncias de atributos;
- Preparar a base para futura integração com dados textuais (Fase 3 do projeto).

2. Configuração do Ambiente

2.1 Chave de API
Crie um arquivo .env na raiz do projeto contendo:

'OPENAI_API_KEY=SEU_TOKEN_AQUI'

O carregamento da variável de ambiente é realizado com a biblioteca python-dotenv.

2.2 Dependências
As bibliotecas necessárias estão listadas em requirements.txt, incluindo:

- openai
- python-dotenv

3. Módulo de Integração
Arquivo responsável:

src/llm_interpretation.py

Principais responsabilidades:

- Construção de prompts médicos estruturados;
- Envio das requisições à API da OpenAI;
- Pós-processamento e formatação das respostas;
- Geração de laudos e explicações em linguagem técnica.

4. Principais Funções

4. Estruturas e Funções Principais

4.1 Estrutura de Entrada de Caso

@dataclass
class CaseExplanationInput:
    model_name: str
    pred_label: int
    p_maligno: float
    top_features: Optional[Dict[str, float]]
    clinical_notes: str


Representa um caso clínico individual com:

- Modelo utilizado
- Classe predita (0=benigno, 1=maligno)
- Probabilidade de malignidade
- Variáveis mais relevantes
- Notas clínicas textuais (preparação para Fase 3)

4.2 Prompt para Explicação de Caso

build_prompt_case(x: CaseExplanationInput) -> str

Gera um prompt estruturado para:

- Classificação de risco (baixo / moderado / alto);
- Explicação clínica em linguagem natural;
- Sugestão de ação (triagem, exame complementar, acompanhamento);
- Declaração ética de não substituição do médico.

4.3 Prompt para Relatório de Métricas

build_prompt_report(metrics: Dict[str, Any]) -> str

- Transforma métricas quantitativas (Accuracy, Recall, F1, Matriz de Confusão) em:
- Insights clínicos acionáveis;
- Análise de falsos negativos;
- Trade-off FN vs FP;
- Recomendações de melhoria do modelo.

4.4 Chamada à LLM

call_gpt(prompt: str, model: str = "gpt-4.1-mini", temperature: float = 0.2) -> str

Responsável por:

- Enviar o prompt à API da OpenAI;
- Receber a resposta do modelo GPT;
- Retornar o texto gerado para pós-processamento.

4.5 Avaliação da Qualidade das Respostas

checklist_quality(text: str) -> Dict[str, int]

Realiza uma avaliação heurística baseada em:

- Coerência com métricas
- Clareza para médicos
- Foco em redução de falsos negativos
- Presença de ação acionável

Inclusão de aviso ético

5. Exemplo de Uso
from src.llm_interpretation import (
    CaseExplanationInput,
    build_prompt_case,
    build_prompt_report,
    call_gpt,
    checklist_quality
)

case = CaseExplanationInput(
    model_name="DecisionTree_GA",
    pred_label=1,
    p_maligno=0.87,
    top_features={"radius_mean": 0.82, "concavity_mean": 0.77},
    clinical_notes=""
)

prompt = build_prompt_case(case)
response = call_gpt(prompt)
quality = checklist_quality(response)

print(response)
print(quality)

6. Prompt Engineering

O prompt enviado à LLM segue os seguintes princípios:

- Contextualização clínica: o modelo assume o papel de assistente médico.

Estruturação:

- Dados do paciente (features relevantes);
- Métricas do modelo;
- Classe predita e probabilidades;
- Instruções de linguagem:
- Clareza e objetividade;

Restrições éticas:

- Indicação explícita de que a resposta não substitui diagnóstico profissional.

7. Avaliação das Respostas

A qualidade das respostas da LLM é avaliada de forma qualitativa considerando:

- Coerência clínica;
- Aderência às métricas fornecidas;
- Clareza da linguagem;
- Consistência com literatura médica básica;
- Ausência de afirmações categóricas sem ressalvas.

8. Considerações Éticas

A LLM é utilizada exclusivamente como ferramenta de apoio.
Nenhuma decisão clínica automatizada é tomada.
As respostas sempre incluem recomendações de validação por profissional de saúde.