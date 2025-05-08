


# import os
# import sys

# # ----------------- Setup -----------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(BASE_DIR))

# import torch
# import pandas as pd
# import numpy as np
# from torch import nn
# from datasets import Dataset, DatasetDict
# from transformers import (
#     AutoTokenizer,
#     AutoModelForTokenClassification,
#     TrainingArguments,
#     Trainer,
#     AutoConfig,
#     pipeline,
# )
# from seqeval.metrics import classification_report
# import ast



# DATA_PATH = "data/IT jobs for training.csv"
# SPLIT_DIR = "data/split"
# os.makedirs(SPLIT_DIR, exist_ok=True)

# # ----------------- Annotation -----------------
# def annotate_skills(text, skills):
#     entities = []
#     if pd.isna(skills) or pd.isna(text):
#         return {'text': text, 'entities': []}

#     skill_list = [s.strip() for s in skills.split(',')]
#     text_lower = text.lower()

#     for skill in skill_list:
#         skill_lower = skill.lower()
#         start = 0
#         while True:
#             start = text_lower.find(skill_lower, start)
#             if start == -1:
#                 break
#             end = start + len(skill)
#             entities.append({'start': start, 'end': end, 'label': 'SKILL'})
#             start = end

#     return {'text': text, 'entities': entities}

# # ----------------- Load or Create Splits -----------------
# if os.path.exists(f"{SPLIT_DIR}/train.csv") and os.path.exists(f"{SPLIT_DIR}/test.csv"):
#     print("✅ Using previously saved train/test splits...")
#     train_df = pd.read_csv(f"{SPLIT_DIR}/train.csv")
#     test_df = pd.read_csv(f"{SPLIT_DIR}/test.csv")

#     train_df['entities'] = train_df['entities'].apply(ast.literal_eval)
#     test_df['entities'] = test_df['entities'].apply(ast.literal_eval)
# else:
#     print("🔄 Splitting and saving dataset for the first time...")
#     df = pd.read_csv(DATA_PATH)
#     annotated_data = [annotate_skills(row['job_description'], row['skills']) for _, row in df.iterrows()]
#     full_df = pd.DataFrame(annotated_data)
#     split = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
#     train_df = split.iloc[:int(0.8 * len(split))]
#     test_df = split.iloc[int(0.8 * len(split)):]
#     train_df.to_csv(f"{SPLIT_DIR}/train.csv", index=False)
#     test_df.to_csv(f"{SPLIT_DIR}/test.csv", index=False)

# # Convert to Hugging Face datasets
# dataset = DatasetDict({
#     "train": Dataset.from_pandas(train_df),
#     "test": Dataset.from_pandas(test_df)
# })

# # ----------------- Tokenization -----------------
# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# label_names = ["O", "B-SKILL", "I-SKILL"]

# def tokenize_and_align_labels(examples):
#     tokenized = tokenizer(examples["text"], truncation=True, padding=True)
#     labels = []

#     for i, entities in enumerate(examples["entities"]):
#         word_ids = tokenized.word_ids(batch_index=i)
#         label_ids = [-100] * len(word_ids)
#         for entity in entities:
#             for idx, word_id in enumerate(word_ids):
#                 if word_id is None:
#                     continue
#                 char_span = tokenized.token_to_chars(i, idx)
#                 if char_span and entity["start"] <= char_span.start < entity["end"]:
#                     label_ids[idx] = 1 if char_span.start == entity["start"] else 2
#         labels.append(label_ids)

#     tokenized["labels"] = labels
#     return tokenized

# tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# # ----------------- Model -----------------
# config = AutoConfig.from_pretrained(
#     model_name,
#     num_labels=3,
#     id2label={0: "O", 1: "B-SKILL", 2: "I-SKILL"},
#     label2id={"O": 0, "B-SKILL": 1, "I-SKILL": 2},
#     hidden_dropout_prob=0.3,
#     attention_probs_dropout_prob=0.3
# )

# model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs["logits"]
#         loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 5.0]).to(model.device))
#         loss = loss_fct(logits.view(-1, 3), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)
#     true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
#     true_preds = [[label_names[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
#     report = classification_report(true_labels, true_preds, output_dict=True)
#     return {
#         "precision": report["macro avg"]["precision"],
#         "recall": report["macro avg"]["recall"],
#         "f1": report["macro avg"]["f1-score"],
#         "accuracy": np.mean([float(t == p) for t, p in zip(true_labels, true_preds)]),
#     }

# # ----------------- Training -----------------
# training_args = TrainingArguments(
#     output_dir="./ner_results",
#     eval_strategy="epoch",
#     learning_rate=2e-6,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=10,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,
# )

# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     compute_metrics=compute_metrics,
# )

# print("🚀 Starting training...")
# trainer.train()

# model.save_pretrained("./ner_model")
# tokenizer.save_pretrained("./ner_tokenizer")

# eval_results = trainer.evaluate()
# print("\n✅ Final Evaluation Results:")
# print(f"Precision: {eval_results['eval_precision']:.4f}")
# print(f"Recall: {eval_results['eval_recall']:.4f}")
# print(f"F1 Score: {eval_results['eval_f1']:.4f}")
# print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")

# # ----------------------------------------
# # 🆕 NEW Inference Code
# # ----------------------------------------

# # Pre-load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("./ner_tokenizer")
# model = AutoModelForTokenClassification.from_pretrained("./ner_model")

# # Better Pipeline
# ner_pipeline = pipeline(
#     "ner",
#     model=model,
#     tokenizer=tokenizer,
#     aggregation_strategy="max"  # Better aggregation
# )

# def clean_text(text):
#     text = text.replace('\n', ' ')
#     text = text.replace('\t', ' ')
#     text = " ".join(text.split())
#     return text

# def extract_job_title_and_skills(text: str):
#     text = clean_text(text)
#     ner_results = ner_pipeline(text)

#     job_title = None
#     skills = []

#     for entity in ner_results:
#         label = entity["entity_group"]
#         word = entity["word"]

#         if label == "JOB_TITLE" and not job_title:
#             job_title = word
#         elif label == "SKILL":
#             skills.append(word)

#     if not job_title:
#         job_title = "Not detected"

#     return job_title, list(set(skills))  # Remove duplicate skills




# import os
# import sys
# import torch
# import pandas as pd
# import numpy as np
# from torch import nn
# from datasets import Dataset, DatasetDict
# from transformers import (
#     AutoTokenizer,
#     AutoModelForTokenClassification,
#     TrainingArguments,
#     Trainer,
#     AutoConfig,
#     pipeline,
# )
# from seqeval.metrics import classification_report
# import ast
# import re

# # ----------------- Setup -----------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(BASE_DIR))

# # Corrected DATA_PATH to point to Backend\data
# PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Points to Backend
# DATA_PATH = os.path.join(PROJECT_ROOT, "data/IT jobs for training.csv")
# SPLIT_DIR = os.path.join(PROJECT_ROOT, "data/split")
# MODEL_DIR = os.path.join(BASE_DIR, "ner_model")
# TOKENIZER_DIR = os.path.join(BASE_DIR, "ner_tokenizer")
# os.makedirs(SPLIT_DIR, exist_ok=True)
# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(TOKENIZER_DIR, exist_ok=True)

# # Verify dataset exists
# if not os.path.exists(DATA_PATH):
#     raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

# # ----------------- Annotation -----------------
# def annotate_skills_and_job_title(text, skills, job_title):
#     entities = []
#     if pd.isna(skills) or pd.isna(text) or pd.isna(job_title):
#         return {'text': text, 'entities': []}

#     text_lower = text.lower()
    
#     # Annotate job title
#     job_title_lower = job_title.lower().strip()
#     start = text_lower.find(job_title_lower)
#     if start != -1:
#         end = start + len(job_title_lower)
#         entities.append({'start': start, 'end': end, 'label': 'JOB_TITLE'})
    
#     # Annotate skills
#     skill_list = [s.strip() for s in skills.split(',')]
#     for skill in skill_list:
#         skill_lower = skill.lower()
#         start = 0
#         while True:
#             start = text_lower.find(skill_lower, start)
#             if start == -1:
#                 break
#             end = start + len(skill)
#             entities.append({'start': start, 'end': end, 'label': 'SKILL'})
#             start = end

#     # Sort entities by start position to avoid overlap issues
#     entities.sort(key=lambda x: (x['start'], -x['end']))
#     return {'text': text, 'entities': entities}

# # ----------------- Load or Create Splits -----------------
# if os.path.exists(f"{SPLIT_DIR}/train.csv") and os.path.exists(f"{SPLIT_DIR}/test.csv"):
#     print("✅ Using previously saved train/test splits...")
#     train_df = pd.read_csv(f"{SPLIT_DIR}/train.csv")
#     test_df = pd.read_csv(f"{SPLIT_DIR}/test.csv")

#     train_df['entities'] = train_df['entities'].apply(ast.literal_eval)
#     test_df['entities'] = test_df['entities'].apply(ast.literal_eval)
# else:
#     print("🔄 Splitting and saving dataset for the first time...")
#     df = pd.read_csv(DATA_PATH)
#     annotated_data = [
#         annotate_skills_and_job_title(row['job_description'], row['skills'], row['job_title'])
#         for _, row in df.iterrows()
#     ]
#     full_df = pd.DataFrame(annotated_data)
#     split = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
#     train_df = split.iloc[:int(0.8 * len(split))]
#     test_df = split.iloc[int(0.8 * len(split)):]
#     train_df.to_csv(f"{SPLIT_DIR}/train.csv", index=False)
#     test_df.to_csv(f"{SPLIT_DIR}/test.csv", index=False)

# # Convert to Hugging Face datasets
# dataset = DatasetDict({
#     "train": Dataset.from_pandas(train_df),
#     "test": Dataset.from_pandas(test_df)
# })

# # ----------------- Tokenization -----------------
# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# label_names = ["O", "B-SKILL", "I-SKILL", "B-JOB_TITLE", "I-JOB_TITLE"]
# label2id = {label: idx for idx, label in enumerate(label_names)}
# id2label = {idx: label for label, idx in label2id.items()}

# def tokenize_and_align_labels(examples):
#     tokenized = tokenizer(examples["text"], truncation=True, padding=True)
#     labels = []

#     for i, entities in enumerate(examples["entities"]):
#         word_ids = tokenized.word_ids(batch_index=i)
#         label_ids = [-100] * len(word_ids)
#         for entity in entities:
#             for idx, word_id in enumerate(word_ids):
#                 if word_id is None:
#                     continue
#                 char_span = tokenized.token_to_chars(i, idx)
#                 if char_span and entity["start"] <= char_span.start < entity["end"]:
#                     label = entity["label"]
#                     is_begin = char_span.start == entity["start"]
#                     if label == "SKILL":
#                         label_ids[idx] = label2id["B-SKILL"] if is_begin else label2id["I-SKILL"]
#                     elif label == "JOB_TITLE":
#                         label_ids[idx] = label2id["B-JOB_TITLE"] if is_begin else label2id["I-JOB_TITLE"]
#         labels.append(label_ids)

#     tokenized["labels"] = labels
#     return tokenized

# tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# # ----------------- Model -----------------
# config = AutoConfig.from_pretrained(
#     model_name,
#     num_labels=len(label_names),
#     id2label=id2label,
#     label2id=label2id,
#     hidden_dropout_prob=0.3,
#     attention_probs_dropout_prob=0.3
# )

# model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs["logits"]
#         loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 5.0, 10.0, 5.0]).to(model.device))
#         loss = loss_fct(logits.view(-1, len(label_names)), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)
#     true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
#     true_preds = [[label_names[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
#     report = classification_report(true_labels, true_preds, output_dict=True)
#     return {
#         "precision": report["macro avg"]["precision"],
#         "recall": report["macro avg"]["recall"],
#         "f1": report["macro avg"]["f1-score"],
#         "accuracy": np.mean([float(t == p) for t, p in zip(true_labels, true_preds)]),
#     }

# # ----------------- Training -----------------
# training_args = TrainingArguments(
#     output_dir="./ner_results",
#     eval_strategy="epoch",
#     learning_rate=2e-6,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=10,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,
# )

# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     compute_metrics=compute_metrics,
# )

# print("🚀 Starting training...")
# trainer.train()

# model.save_pretrained(MODEL_DIR)
# tokenizer.save_pretrained(TOKENIZER_DIR)

# eval_results = trainer.evaluate()
# print("\n Final Evaluation Results:")
# print(f"Precision: {eval_results['eval_precision']:.4f}")
# print(f"Recall: {eval_results['eval_recall']:.4f}")
# print(f"F1 Score: {eval_results['eval_f1']:.4f}")
# print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")

# # ----------------- Inference -----------------
# # Pre-load model and tokenizer for testing
# try:
#     tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
#     model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, local_files_only=True)
# except Exception as e:
#     print(f"Error loading model/tokenizer: {str(e)}")
#     raise

# # Better Pipeline
# ner_pipeline = pipeline(
#     "ner",
#     model=model,
#     tokenizer=tokenizer,
#     aggregation_strategy="simple"
# )

# def clean_text(text):
#     text = text.replace('\n', ' ').replace('\t', ' ')
#     text = re.sub(r'[^\w\s,]', '', text)
#     text = " ".join(text.split())
#     return text

# def extract_job_title_and_skills(text: str):
#     text = clean_text(text)
#     ner_results = ner_pipeline(text)
    
#     job_title = None
#     skills = []
    
#     for entity in ner_results:
#         word = entity["word"].strip()
#         if word.startswith("##") or len(word) < 2 or word.isdigit():
#             continue
#         if entity["entity_group"] == "JOB_TITLE" and not job_title:
#             job_title = word
#         elif entity["entity_group"] == "SKILL":
#             skills.append(word)
    
#     return job_title if job_title else "Not detected", list(set(skills))


# import os
# import sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(BASE_DIR))

# import torch
# import pandas as pd
# import numpy as np
# from torch import nn
# from datasets import Dataset, DatasetDict
# from transformers import (
#     AutoTokenizer,
#     AutoModelForTokenClassification,
#     TrainingArguments,
#     Trainer,
#     AutoConfig,
#     pipeline,
# )
# from seqeval.metrics import classification_report
# import ast
# import re
# import json



# PROJECT_ROOT = os.path.dirname(BASE_DIR)
# DATA_PATH = os.path.join(PROJECT_ROOT, "data/IT jobs for training.csv")
# CV_PATH = os.path.join(PROJECT_ROOT, "data/annotated_cv_data.json")
# SPLIT_DIR = os.path.join(PROJECT_ROOT, "data/split")
# MODEL_DIR = os.path.join(BASE_DIR, "ner_model")
# TOKENIZER_DIR = os.path.join(BASE_DIR, "ner_tokenizer")
# os.makedirs(SPLIT_DIR, exist_ok=True)
# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(TOKENIZER_DIR, exist_ok=True)

# # ----------------- Annotation Function -----------------
# def annotate_skills_and_job_title(text, skills, job_title):
#     entities = []
#     if pd.isna(skills) or pd.isna(text) or pd.isna(job_title):
#         return {'text': text, 'entities': []}

#     text_lower = text.lower()
#     job_title_lower = job_title.lower().strip()
#     start = text_lower.find(job_title_lower)
#     if start != -1:
#         end = start + len(job_title_lower)
#         entities.append({'start': start, 'end': end, 'label': 'JOB_TITLE'})

#     skill_list = [s.strip() for s in skills.split(',')]
#     for skill in skill_list:
#         skill_lower = skill.lower()
#         start = 0
#         while True:
#             start = text_lower.find(skill_lower, start)
#             if start == -1:
#                 break
#             end = start + len(skill)
#             entities.append({'start': start, 'end': end, 'label': 'SKILL'})
#             start = end

#     entities.sort(key=lambda x: (x['start'], -x['end']))
#     return {'text': text, 'entities': entities}

# # ----------------- Load and Annotate Datasets -----------------
# print("📦 Loading and processing datasets...")

# # Load job description dataset
# df = pd.read_csv(DATA_PATH)
# job_annotated_data = [
#     annotate_skills_and_job_title(row['job_description'], row['skills'], row['job_title'])
#     for _, row in df.iterrows()
# ]
# job_df = pd.DataFrame(job_annotated_data)

# # Load and reformat annotated CV dataset
# with open(CV_PATH, "r", encoding="utf-8") as f:
#     cv_annotated = json.load(f)

# for item in cv_annotated:
#     item["entities"] = [
#         {"start": start, "end": end, "label": label}
#         for (start, end, label) in item["entities"]
#     ]
# cv_df = pd.DataFrame(cv_annotated)

# # Combine and shuffle datasets
# combined_df = pd.concat([job_df, cv_df], ignore_index=True)
# combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Train/test split
# train_size = int(0.8 * len(combined_df))
# train_df = combined_df.iloc[:train_size]
# test_df = combined_df.iloc[train_size:]

# # Save for reproducibility
# train_df.to_csv(f"{SPLIT_DIR}/train_combined.csv", index=False)
# test_df.to_csv(f"{SPLIT_DIR}/test_combined.csv", index=False)

# # Convert to Hugging Face datasets
# dataset = DatasetDict({
#     "train": Dataset.from_pandas(train_df),
#     "test": Dataset.from_pandas(test_df)
# })

# # ----------------- Tokenization -----------------
# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# label_names = ["O", "B-SKILL", "I-SKILL", "B-JOB_TITLE", "I-JOB_TITLE"]
# label2id = {label: idx for idx, label in enumerate(label_names)}
# id2label = {idx: label for label, idx in label2id.items()}

# def tokenize_and_align_labels(examples):
#     tokenized = tokenizer(
#         examples["text"],
#         truncation=True,
#         padding="max_length",   # Use fixed length padding
#         max_length=512,         # Set max length for consistency
#         return_offsets_mapping=True
#     )
    
#     labels = []

#     for i, entities in enumerate(examples["entities"]):
#         entity_labels = ["O"] * len(tokenized["input_ids"][i])
#         offsets = tokenized["offset_mapping"][i]

#         for entity in entities:
#             start, end, label = entity["start"], entity["end"], entity["label"]

#             for idx, (token_start, token_end) in enumerate(offsets):
#                 if token_start is None or token_end is None:
#                     continue
#                 if token_start >= end:
#                     break
#                 if token_start >= start and token_end <= end:
#                     prefix = "B-" if token_start == start else "I-"
#                     entity_labels[idx] = prefix + label

#         label_ids = [label2id.get(lbl, 0) for lbl in entity_labels]
#         labels.append(label_ids)

#     tokenized["labels"] = labels
#     tokenized.pop("offset_mapping")  # Remove offset before training
#     return tokenized


# tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# # ----------------- Model & Trainer -----------------
# config = AutoConfig.from_pretrained(
#     model_name,
#     num_labels=len(label_names),
#     id2label=id2label,
#     label2id=label2id,
#     hidden_dropout_prob=0.3,
#     attention_probs_dropout_prob=0.3
# )
# model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs["logits"]
#         loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 5.0, 10.0, 5.0]).to(model.device))
#         loss = loss_fct(logits.view(-1, len(label_names)), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)
#     true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
#     true_preds = [[label_names[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
#     report = classification_report(true_labels, true_preds, output_dict=True)
#     return {
#         "precision": report["macro avg"]["precision"],
#         "recall": report["macro avg"]["recall"],
#         "f1": report["macro avg"]["f1-score"],
#         "accuracy": np.mean([float(t == p) for t, p in zip(true_labels, true_preds)]),
#     }

# training_args = TrainingArguments(
#     output_dir="./ner_results",
#     eval_strategy="epoch",
#     learning_rate=2e-6,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     gradient_accumulation_steps=4,
#     num_train_epochs=10,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,
# )

# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     compute_metrics=compute_metrics,
# )

# # ----------------- Training -----------------
# print("🚀 Starting training...")
# trainer.train()

# model.save_pretrained(MODEL_DIR)
# tokenizer.save_pretrained(TOKENIZER_DIR)

# # ----------------- Evaluation -----------------
# eval_results = trainer.evaluate()
# print("\n Final Evaluation Results:")
# print(f"Precision: {eval_results['eval_precision']:.4f}")
# print(f"Recall: {eval_results['eval_recall']:.4f}")
# print(f"F1 Score: {eval_results['eval_f1']:.4f}")
# print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")

# # ----------------- Inference -----------------
# try:
#     tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
#     model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, local_files_only=True)
# except Exception as e:
#     print(f"Error loading model/tokenizer: {str(e)}")
#     raise

# ner_pipeline = pipeline(
#     "ner",
#     model=model,
#     tokenizer=tokenizer,
#     aggregation_strategy="simple"
# )

# def clean_text(text):
#     text = text.replace('\n', ' ').replace('\t', ' ')
#     text = re.sub(r'[^\w\s,]', '', text)
#     return " ".join(text.split())

# def extract_job_title_and_skills(text: str):
#     text = clean_text(text)
#     ner_results = ner_pipeline(text)
#     job_title = None
#     skills = []
#     for entity in ner_results:
#         word = entity["word"].strip()
#         if word.startswith("##") or len(word) < 2 or word.isdigit():
#             continue
#         if entity["entity_group"] == "JOB_TITLE" and not job_title:
#             job_title = word
#         elif entity["entity_group"] == "SKILL":
#             skills.append(word)
#     return job_title if job_title else "Not detected", list(set(skills))


# 


# import os
# import sys

# # ----------------- Setup -----------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(BASE_DIR))


# import torch
# import pandas as pd
# import numpy as np
# from torch import nn
# from datasets import Dataset, DatasetDict
# from transformers import (
#     AutoTokenizer,
#     AutoModelForTokenClassification,
#     TrainingArguments,
#     Trainer,
#     AutoConfig,
#     pipeline,
#     AutoModelForSeq2SeqLM
# )
# from seqeval.metrics import classification_report
# import ast
# import re


# PROJECT_ROOT = os.path.dirname(BASE_DIR)
# DATA_PATH = os.path.join(PROJECT_ROOT, "data/IT jobs for training.csv")
# SPLIT_DIR = os.path.join(PROJECT_ROOT, "data/split")
# MODEL_DIR = os.path.join(BASE_DIR, "ner_model")
# TOKENIZER_DIR = os.path.join(BASE_DIR, "ner_tokenizer")
# os.makedirs(SPLIT_DIR, exist_ok=True)
# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(TOKENIZER_DIR, exist_ok=True)

# if not os.path.exists(DATA_PATH):
#     raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

# # ----------------- Annotation -----------------
# def annotate_skills_and_job_title(text, skills, job_title):
#     entities = []
#     if pd.isna(skills) or pd.isna(text) or pd.isna(job_title):
#         return {'text': text, 'entities': []}

#     text_lower = text.lower()
#     job_title_lower = job_title.lower().strip()
#     start = text_lower.find(job_title_lower)
#     if start != -1:
#         end = start + len(job_title_lower)
#         entities.append({'start': start, 'end': end, 'label': 'JOB_TITLE'})

#     skill_list = [s.strip() for s in skills.split(',')]
#     for skill in skill_list:
#         skill_lower = skill.lower()
#         start = 0
#         while True:
#             start = text_lower.find(skill_lower, start)
#             if start == -1:
#                 break
#             end = start + len(skill)
#             entities.append({'start': start, 'end': end, 'label': 'SKILL'})
#             start = end

#     entities.sort(key=lambda x: (x['start'], -x['end']))
#     return {'text': text, 'entities': entities}

# # ----------------- Load or Create Splits -----------------
# if os.path.exists(f"{SPLIT_DIR}/train.csv") and os.path.exists(f"{SPLIT_DIR}/test.csv"):
#     print("✅ Using previously saved train/test splits...")
#     train_df = pd.read_csv(f"{SPLIT_DIR}/train.csv")
#     test_df = pd.read_csv(f"{SPLIT_DIR}/test.csv")

#     train_df['entities'] = train_df['entities'].apply(ast.literal_eval)
#     test_df['entities'] = test_df['entities'].apply(ast.literal_eval)
# else:
#     print("🔄 Splitting and saving dataset for the first time...")
#     df = pd.read_csv(DATA_PATH)
#     annotated_data = [
#         annotate_skills_and_job_title(row['job_description'], row['skills'], row['job_title'])
#         for _, row in df.iterrows()
#     ]
#     full_df = pd.DataFrame(annotated_data)
#     split = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
#     train_df = split.iloc[:int(0.8 * len(split))]
#     test_df = split.iloc[int(0.8 * len(split)):]
#     train_df.to_csv(f"{SPLIT_DIR}/train.csv", index=False)
#     test_df.to_csv(f"{SPLIT_DIR}/test.csv", index=False)

# # ----------------- HuggingFace Dataset -----------------
# dataset = DatasetDict({
#     "train": Dataset.from_pandas(train_df),
#     "test": Dataset.from_pandas(test_df)
# })

# # ----------------- Tokenization -----------------
# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# label_names = ["O", "B-SKILL", "I-SKILL", "B-JOB_TITLE", "I-JOB_TITLE"]
# label2id = {label: idx for idx, label in enumerate(label_names)}
# id2label = {idx: label for label, idx in label2id.items()}

# def tokenize_and_align_labels(examples):
#     tokenized = tokenizer(examples["text"], truncation=True, padding=True)
#     labels = []

#     for i, entities in enumerate(examples["entities"]):
#         word_ids = tokenized.word_ids(batch_index=i)
#         label_ids = [-100] * len(word_ids)
#         for entity in entities:
#             for idx, word_id in enumerate(word_ids):
#                 if word_id is None:
#                     continue
#                 char_span = tokenized.token_to_chars(i, idx)
#                 if char_span and entity["start"] <= char_span.start < entity["end"]:
#                     label = entity["label"]
#                     is_begin = char_span.start == entity["start"]
#                     if label == "SKILL":
#                         label_ids[idx] = label2id["B-SKILL"] if is_begin else label2id["I-SKILL"]
#                     elif label == "JOB_TITLE":
#                         label_ids[idx] = label2id["B-JOB_TITLE"] if is_begin else label2id["I-JOB_TITLE"]
#         labels.append(label_ids)

#     tokenized["labels"] = labels
#     return tokenized

# tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# # ----------------- Model -----------------
# config = AutoConfig.from_pretrained(
#     model_name,
#     num_labels=len(label_names),
#     id2label=id2label,
#     label2id=label2id,
#     hidden_dropout_prob=0.3,
#     attention_probs_dropout_prob=0.3
# )

# model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs["logits"]
#         loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 5.0, 10.0, 5.0]).to(model.device))
#         loss = loss_fct(logits.view(-1, len(label_names)), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)
#     true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
#     true_preds = [[label_names[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
#     report = classification_report(true_labels, true_preds, output_dict=True)
#     return {
#         "precision": report["macro avg"]["precision"],
#         "recall": report["macro avg"]["recall"],
#         "f1": report["macro avg"]["f1-score"],
#         "accuracy": np.mean([float(t == p) for t, p in zip(true_labels, true_preds)]),
#     }

# # ----------------- Training -----------------
# training_args = TrainingArguments(
#     output_dir="./ner_results",
#     eval_strategy="epoch",
#     learning_rate=2e-6,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=10,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,
# )

# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     compute_metrics=compute_metrics,
# )

# print("\n Starting training...")
# trainer.train()

# model.save_pretrained(MODEL_DIR)
# tokenizer.save_pretrained(TOKENIZER_DIR)

# eval_results = trainer.evaluate()
# print("\n Final Evaluation Results:")
# print(f"Precision: {eval_results['eval_precision']:.4f}")
# print(f"Recall: {eval_results['eval_recall']:.4f}")
# print(f"F1 Score: {eval_results['eval_f1']:.4f}")
# print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")

# # ----------------- FLAN-T5 Inference -----------------
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# def clean_text(text):
#     text = text.replace('\n', ' ').replace('\t', ' ')
#     text = re.sub(r'[^\w\s,]', '', text)
#     return " ".join(text.split())

# def load_flan_t5():
#     model_name = "google/flan-t5-base"  # Can also use 'google/flan-t5-large' for better performance
#     device = 0 if torch.cuda.is_available() else -1
    
#     print(f"\n 🔄 Loading FLAN-T5 model from {model_name}...")
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#         device_map="auto"
#     )
    
#     flan_pipeline = pipeline(
#         "text2text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         device=device,
#         max_length=512
#     )
    
#     print(f" ✅ FLAN-T5 model loaded successfully!")
#     return flan_pipeline

# def extract_with_flan_t5(cv_text: str, flan_pipeline):
#     # Clean and normalize text
#     cleaned_text = clean_text(cv_text)
    
#     # Create a specific prompt for job title extraction
#     job_title_prompt = f"Extract the job title from this CV: {cleaned_text}"
#     job_title_result = flan_pipeline(job_title_prompt, max_length=50)[0]["generated_text"].strip()
    
#     # Create a specific prompt for skills extraction
#     skills_prompt = f"List all technical and professional skills from this CV as comma-separated values: {cleaned_text}"
#     skills_result = flan_pipeline(skills_prompt, max_length=150)[0]["generated_text"]
    
#     # Clean up the skills list
#     skills = [skill.strip() for skill in skills_result.split(",")]
#     skills = [skill for skill in skills if len(skill) > 1]  # Filter out very short items
    
#     return job_title_result, list(set(skills))

# # ----------------- BERT NER Inference -----------------
# def load_bert_ner():
#     print(f"\n 🔄 Loading trained BERT NER model from {MODEL_DIR}...")
    
#     ner_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
#     ner_model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    
#     ner_pipeline = pipeline(
#         "ner",
#         model=ner_model,
#         tokenizer=ner_tokenizer,
#         aggregation_strategy="simple"
#     )
    
#     print(f" ✅ BERT NER model loaded successfully!")
#     return ner_pipeline

# def extract_with_bert_ner(cv_text: str, ner_pipeline):
#     # Get NER results
#     ner_results = ner_pipeline(cv_text)
    
#     # Extract job titles and skills
#     job_titles = []
#     skills = []
    
#     for entity in ner_results:
#         if entity['entity_group'] in ('B-JOB_TITLE', 'I-JOB_TITLE'):
#             job_titles.append(entity['word'])
#         elif entity['entity_group'] in ('B-SKILL', 'I-SKILL'):
#             skills.append(entity['word'])
    
#     # Join job titles if multiple were found
#     job_title = " ".join(job_titles) if job_titles else "Not detected"
    
#     return job_title, list(set(skills))

# # ----------------- Main Function -----------------
# def analyze_cv(cv_text: str, use_bert=True, use_flan=True):
#     """
#     Analyze a CV using both BERT NER and FLAN-T5 extraction models.
    
#     Args:
#         cv_text: The CV text to analyze
#         use_bert: Whether to use BERT NER model
#         use_flan: Whether to use FLAN-T5 model
        
#     Returns:
#         A dictionary with extracted job titles and skills from both models
#     """
#     results = {
#         "bert_ner": {"job_title": None, "skills": []},
#         "flan_t5": {"job_title": None, "skills": []}
#     }
    
#     if use_bert:
#         try:
#             ner_pipeline = load_bert_ner()
#             job_title, skills = extract_with_bert_ner(cv_text, ner_pipeline)
#             results["bert_ner"]["job_title"] = job_title
#             results["bert_ner"]["skills"] = skills
#             print(f"\n 🔍 BERT NER Results:")
#             print(f"   Job Title: {job_title}")
#             print(f"   Skills: {skills}")
#         except Exception as e:
#             print(f"\n ❌ Error with BERT NER extraction: {str(e)}")
            
#     if use_flan:
#         try:
#             flan_pipeline = load_flan_t5()
#             job_title, skills = extract_with_flan_t5(cv_text, flan_pipeline)
#             results["flan_t5"]["job_title"] = job_title
#             results["flan_t5"]["skills"] = skills
#             print(f"\n 🔍 FLAN-T5 Results:")
#             print(f"   Job Title: {job_title}")
#             print(f"   Skills: {skills}")
#         except Exception as e:
#             print(f"\n ❌ Error with FLAN-T5 extraction: {str(e)}")
    
#     # Combine results (preferring FLAN-T5 for job title, combining skills)
#     combined_job_title = results["flan_t5"]["job_title"] if use_flan else results["bert_ner"]["job_title"]
#     combined_skills = list(set(results["bert_ner"]["skills"] + results["flan_t5"]["skills"]))
    
#     results["combined"] = {
#         "job_title": combined_job_title,
#         "skills": combined_skills
#     }
    
#     print(f"\n 🎯 Combined Results:")
#     print(f"   Job Title: {combined_job_title}")
#     print(f"   Skills: {combined_skills}")
    
#     return results

# if __name__ == "__main__":
#     # Example CV for testing
#     example_cv_text = """
#     John Doe
#     Email: john.doe@example.com
#     Phone: (123) 456-7890
    
#     Professional Summary:
#     Experienced Software Developer with 5 years of experience in Python, Django, and React.
#     Passionate about building scalable web applications and mentoring junior developers.
    
#     Work Experience:
#     Backend Engineer
#     ABC Tech - New York, NY
#     2020-Present
#     - Developed RESTful APIs using Django and Django REST Framework
#     - Containerized applications using Docker and orchestrated with Kubernetes
#     - Implemented CI/CD pipelines using Jenkins
    
#     Junior Developer
#     XYZ Solutions - Boston, MA
#     2018-2020
#     - Built responsive web applications using React and Redux
#     - Designed and maintained SQL databases
    
#     Skills:
#     Python, Django, React, JavaScript, Docker, Kubernetes, Jenkins, SQL, REST APIs, Git
    
#     Education:
#     BS in Computer Science
#     Boston University, 2018
#     """
    
#     # Run the analysis
#     results = analyze_cv(example_cv_text, use_bert=True, use_flan=True)


import os
import sys


# ----------------- Setup -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

import torch
import pandas as pd
import numpy as np
from torch import nn
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    pipeline,
    AutoModelForSeq2SeqLM
)
from seqeval.metrics import classification_report
import ast
import re


PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data/IT jobs for training.csv")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data/split")
MODEL_DIR = os.path.join(BASE_DIR, "ner_model")
TOKENIZER_DIR = os.path.join(BASE_DIR, "ner_tokenizer")
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

# ----------------- Annotation -----------------
def annotate_skills_and_job_title(text, skills, job_title):
    entities = []
    if pd.isna(skills) or pd.isna(text) or pd.isna(job_title):
        return {'text': text, 'entities': []}

    text_lower = text.lower()
    job_title_lower = job_title.lower().strip()
    start = text_lower.find(job_title_lower)
    if start != -1:
        end = start + len(job_title_lower)
        entities.append({'start': start, 'end': end, 'label': 'JOB_TITLE'})

    skill_list = [s.strip() for s in skills.split(',')]
    for skill in skill_list:
        skill_lower = skill.lower()
        start = 0
        while True:
            start = text_lower.find(skill_lower, start)
            if start == -1:
                break
            end = start + len(skill)
            entities.append({'start': start, 'end': end, 'label': 'SKILL'})
            start = end

    entities.sort(key=lambda x: (x['start'], -x['end']))
    return {'text': text, 'entities': entities}

# ----------------- Load or Create Splits -----------------
if os.path.exists(f"{SPLIT_DIR}/train.csv") and os.path.exists(f"{SPLIT_DIR}/test.csv"):
    print("✅ Using previously saved train/test splits...")
    train_df = pd.read_csv(f"{SPLIT_DIR}/train.csv")
    test_df = pd.read_csv(f"{SPLIT_DIR}/test.csv")
    train_df['entities'] = train_df['entities'].apply(ast.literal_eval)
    test_df['entities'] = test_df['entities'].apply(ast.literal_eval)
else:
    print("🔄 Splitting and saving dataset for the first time...")
    df = pd.read_csv(DATA_PATH)
    annotated_data = [
        annotate_skills_and_job_title(row['job_description'], row['skills'], row['job_title'])
        for _, row in df.iterrows()
    ]
    full_df = pd.DataFrame(annotated_data)
    split = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = split.iloc[:int(0.8 * len(split))]
    test_df = split.iloc[int(0.8 * len(split)):]
    train_df.to_csv(f"{SPLIT_DIR}/train.csv", index=False)
    test_df.to_csv(f"{SPLIT_DIR}/test.csv", index=False)

# ----------------- HuggingFace Dataset -----------------
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# ----------------- Tokenization -----------------
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
label_names = ["O", "B-SKILL", "I-SKILL", "B-JOB_TITLE", "I-JOB_TITLE"]
label2id = {label: idx for idx, label in enumerate(label_names)}
id2label = {idx: label for label, idx in label2id.items()}

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding=True)
    labels = []

    for i, entities in enumerate(examples["entities"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = [-100] * len(word_ids)
        for entity in entities:
            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                char_span = tokenized.token_to_chars(i, idx)
                if char_span and entity["start"] <= char_span.start < entity["end"]:
                    label = entity["label"]
                    is_begin = char_span.start == entity["start"]
                    if label == "SKILL":
                        label_ids[idx] = label2id["B-SKILL"] if is_begin else label2id["I-SKILL"]
                    elif label == "JOB_TITLE":
                        label_ids[idx] = label2id["B-JOB_TITLE"] if is_begin else label2id["I-JOB_TITLE"]
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# ----------------- Model -----------------
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)

model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 5.0, 10.0, 5.0]).to(model.device))
        loss = loss_fct(logits.view(-1, len(label_names)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_preds = [[label_names[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    report = classification_report(true_labels, true_preds, output_dict=True)
    return {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
        "accuracy": np.mean([float(t == p) for t, p in zip(true_labels, true_preds)]),
    }

# ----------------- Training -----------------
NER_RESULTS_DIR = os.path.join(BASE_DIR, "ner_results")
os.makedirs(NER_RESULTS_DIR, exist_ok=True)
training_args = TrainingArguments(
    #output_dir="./ner_results",
    output_dir=NER_RESULTS_DIR,
    eval_strategy="epoch",
    learning_rate=2e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
trainer.train()

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(TOKENIZER_DIR)

eval_results = trainer.evaluate()
print("\nFinal Evaluation Results:")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")

# ----------------- FLAN-T5 Inference -----------------
def clean_text(text):
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'[^\w\s,]', '', text)
    return " ".join(text.split())

def load_flan_t5():
    model_name = "google/flan-t5-base"
    device = 0 if torch.cuda.is_available() else -1

    print(f"\n🔄 Loading FLAN-T5 model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    flan_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=512
    )

    print(f"✅ FLAN-T5 model loaded successfully!")
    return flan_pipeline

def extract_with_flan_t5(cv_text: str, flan_pipeline):
    cleaned_text = clean_text(cv_text)

    job_title_prompt = f"Extract the job title from this CV: {cleaned_text}"
    job_title_result = flan_pipeline(job_title_prompt, max_length=50)[0]["generated_text"].strip()

    skills_prompt = f"List all technical and professional skills from this CV as comma-separated values: {cleaned_text}"
    skills_result = flan_pipeline(skills_prompt, max_length=150)[0]["generated_text"]

    skills = [skill.strip() for skill in skills_result.split(",") if len(skill.strip()) > 1]
    return job_title_result, list(set(skills))
