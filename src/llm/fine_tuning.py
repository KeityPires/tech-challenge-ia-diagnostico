from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_finetuning_dataset(
    df: pd.DataFrame,
    instruction: str = (
        "Responda à pergunta médica de forma clara, objetiva, cautelosa e "
        "informativa, sem prescrever tratamento definitivo e sem substituir "
        "avaliação médica profissional."
    ),
) -> pd.DataFrame:
    """Converte um DataFrame com colunas question/answer em dataset textual para SFT."""
    required_columns = {"question", "answer"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"O DataFrame precisa conter as colunas {required_columns}. "
            f"Colunas ausentes: {missing}"
        )

    if df.empty:
        raise ValueError("O DataFrame está vazio. Não há dados para fine-tuning.")

    dataset_df = df[["question", "answer"]].copy()
    dataset_df = dataset_df.dropna(subset=["question", "answer"])
    dataset_df["question"] = dataset_df["question"].astype(str).str.strip()
    dataset_df["answer"] = dataset_df["answer"].astype(str).str.strip()
    dataset_df = dataset_df[
        (dataset_df["question"] != "") & (dataset_df["answer"] != "")
    ].reset_index(drop=True)

    dataset_df["text"] = dataset_df.apply(
        lambda row: (
            f"### Instrução:\n{instruction}\n\n"
            f"### Pergunta:\n{row['question']}\n\n"
            f"### Resposta:\n{row['answer']}"
        ),
        axis=1,
    )
    return dataset_df[["question", "answer", "text"]]



def split_finetuning_dataset(
    dataset_df: pd.DataFrame,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Realiza o split treino/validação de forma reprodutível."""
    if dataset_df.empty:
        raise ValueError("O dataset formatado está vazio.")

    train_df, val_df = train_test_split(
        dataset_df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)



def save_finetuning_dataset(
    dataset_df: pd.DataFrame,
    output_path: str,
) -> str:
    """Salva o dataset formatado em JSONL para auditoria e reuso."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_json(output, orient="records", lines=True, force_ascii=False)
    return str(output)



def run_medical_finetuning(
    df_medquad: pd.DataFrame,
    base_model: str = "google/gemma-2b-it",
    output_dir: str = "./artifacts/mistral-medquad-lora",
    instruction: str = (
        "Responda à pergunta médica de forma clara, objetiva, cautelosa e "
        "informativa, sem prescrever tratamento definitivo nem substituir "
        "avaliação médica profissional."
    ),
    test_size: float = 0.1,
    random_state: int = 42,
    num_train_epochs: int = 2,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 1024,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    use_4bit: bool = True,
    save_merged_model: bool = False,
    merged_model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Executa fine-tuning supervisionado com LoRA/QLoRA.

    Dependências necessárias:
    transformers, datasets, peft, trl, accelerate, bitsandbytes, sentencepiece
    """
    # Imports pesados aqui para não quebrar o restante do projeto quando o treino não for usado.
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    dataset_df = prepare_finetuning_dataset(df_medquad, instruction=instruction)
    train_df, val_df = split_finetuning_dataset(
        dataset_df=dataset_df,
        test_size=test_size,
        random_state=random_state,
    )

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = TrainingArguments(
    output_dir=str(output_path),
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=False,
    fp16=True,
    report_to="none",
)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )

    trainer.train()

    trainer.model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    result: Dict[str, Any] = {
        "status": "success",
        "base_model": base_model,
        "output_dir": str(output_path),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }

    if save_merged_model:
        if merged_model_dir is None:
            merged_model_dir = f"{output_dir}-merged"

        merged_path = Path(merged_model_dir)
        merged_path.mkdir(parents=True, exist_ok=True)

        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))

        result["merged_model_dir"] = str(merged_path)

    return result
