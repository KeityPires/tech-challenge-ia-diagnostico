from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
from openai import OpenAI


@dataclass
class CaseExplanationInput:
    model_name: str
    pred_label: int            
    p_maligno: float           
    top_features: Optional[Dict[str, float]] = None
    clinical_notes: str = ""   


def build_prompt_case(x: CaseExplanationInput) -> str:
    """
    Prompt para explicar um diagnóstico individual (por paciente).
    """
    features_txt = "Não disponível (dados estruturados não fornecidos neste exemplo)."
    if x.top_features:
        features_txt = ", ".join([f"{k}={v:.3f}" for k, v in x.top_features.items()])

    return f"""
Você é um assistente clínico para apoio ao diagnóstico de câncer de mama.
Gere uma explicação em linguagem natural para um médico, baseada APENAS nos dados fornecidos.

Regras:
- Não invente informações (ex.: exames, sintomas, histórico) que não estejam nos dados.
- Se faltarem dados clínicos, diga explicitamente o que falta.
- Priorize sensibilidade (redução de falsos negativos).
- Tom profissional, objetivo, sem sensacionalismo.
- Inclua o aviso: "Não substitui avaliação médica".

Dados do caso:
- Modelo: {x.model_name}
- Predição do modelo: {x.pred_label} (0=benigno, 1=maligno)
- Probabilidade de malignidade: {x.p_maligno:.2f}
- Principais variáveis (se disponíveis): {features_txt}
- Observações textuais do paciente (Fase 3, pode estar vazio): {x.clinical_notes}

Tarefa:
1) Classifique o risco (baixo/moderado/alto) baseado na probabilidade.
2) Explique o significado clínico do resultado (3 bullets).
3) Sugira uma ação genérica (triagem / exame complementar / acompanhamento).
4) Liste limitações do modelo e quais informações adicionais ajudariam (ex.: histórico, exame de imagem, laudo).
""".strip()


def build_prompt_report(metrics: Dict[str, Any]) -> str:
    """
    Prompt para transformar métricas do modelo em insights acionáveis para médicos.
    """
    return f"""
Você é um assistente de IA para suporte clínico. Gere um resumo executivo para médicos
sobre o desempenho do modelo de diagnóstico de câncer de mama.

Regras:
- Converta números em implicações clínicas.
- Destaque risco de falsos negativos (classe maligna=1).
- Não invente dados.
- Saída curta (máximo 12 linhas), em bullets.
- Inclua o aviso: "Não substitui avaliação médica".

Métricas no conjunto de teste:
- Modelo: {metrics['model_name']}
- Accuracy: {metrics['acc']:.3f}
- Recall (maligno=1): {metrics['rec']:.3f}
- F1 (maligno=1): {metrics['f1']:.3f}
- Matriz de confusão (TN, FP, FN, TP): {metrics['tn']}, {metrics['fp']}, {metrics['fn']}, {metrics['tp']}

Tarefa:
- Explique o que significa esse resultado na prática clínica.
- Cite explicitamente o número de falsos negativos e o risco associado.
- Descreva o trade-off FN vs FP e quando isso é aceitável.
- Sugira próximos passos para melhoria (ex.: calibrar probabilidades, ajustar pesos de classe, integrar texto na Fase 3).
""".strip()


def checklist_quality(text: str) -> Dict[str, int]:
    """
    Checklist simples
    """
    t = text.lower()

    foco_fn = 2 if ("falso negativo" in t or "sensibilidade" in t or "recall" in t) else 1
    acao = 2 if ("triagem" in t or "exame" in t or "acompanhamento" in t) else 1
    aviso = 2 if ("não substitui" in t or "nao substitui" in t) else 1

    score = {
        "coerencia_com_metricas": 2,
        "clareza_para_medico": 2,
        "foco_em_fn": foco_fn,
        "acao_acionavel": acao,
        "aviso_segurança": aviso,
    }
    score["total"] = sum(score.values())
    return score

def call_gpt(prompt: str, model: str = "gpt-4.1-mini", temperature: float = 0.2) -> str:
   
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrada. Defina como variável de ambiente.")

    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
    )
    return resp.output_text

