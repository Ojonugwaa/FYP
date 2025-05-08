# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers import TokenClassificationPipeline

# # Load trained model and tokenizer
# MODEL_DIR = "./ner_model"
# TOKENIZER_DIR = "./ner_tokenizer"

# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
# model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

# # Label map must match training
# label_map = {
#     0: "O",
#     1: "B-SKILL",
#     2: "I-SKILL",
#     3: "B-JOB_TITLE",
#     4: "I-JOB_TITLE"
# }


# # Create a pipeline
# ner_pipeline = TokenClassificationPipeline(
#     model=model,
#     tokenizer=tokenizer,
#     aggregation_strategy="simple",
#     device=0 if torch.cuda.is_available() else -1,
# )

# def extract_skills_from_text(text: str) -> list:
#     predictions = ner_pipeline(text)
#     skills = {pred['word'] for pred in predictions if pred['entity_group'] == 'SKILL'}
#     return list(skills)

# def extract_job_title_and_skills(text: str):
#     predictions = ner_pipeline(text)

#     job_title = None
#     skills = set()

#     for pred in predictions:
#         entity = pred['entity_group']
#         word = pred['word']

#         if entity == 'SKILL':
#             skills.add(word)
#         elif entity == 'JOB_TITLE' and job_title is None:
#             job_title = word

#     return job_title, list(skills)


import os
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Define base directory and model/tokenizer paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ner_model")
TOKENIZER_DIR = os.path.join(BASE_DIR, "ner_tokenizer")

# Verify directories exist
if not os.path.exists(TOKENIZER_DIR):
    raise FileNotFoundError(f"Tokenizer directory not found at: {TOKENIZER_DIR}")
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model directory not found at: {MODEL_DIR}")

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, local_files_only=True)
except Exception as e:
    raise Exception(f"Error loading model/tokenizer: {str(e)}")

# Set up NER pipeline
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

def clean_text(text):
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'[^\w\s,]', '', text)
    text = " ".join(text.split())
    return text

def extract_job_title_and_skills(text: str):
    text = clean_text(text)
    ner_results = ner_pipeline(text)
    
    job_title = None
    skills = []
    
    for entity in ner_results:
        word = entity["word"].strip()
        # Skip subword tokens, numbers, and short strings
        if word.startswith("##") or len(word) < 2 or word.isdigit():
            continue
        if entity["entity_group"] == "JOB_TITLE" and not job_title:
            job_title = word
        elif entity["entity_group"] == "SKILL":
            skills.append(word)
    
    return job_title if job_title else "Not detected", list(set(skills))