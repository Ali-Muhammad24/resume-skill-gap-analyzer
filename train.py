# Importing libraries
import random
import numpy as np

# PyTorch for model training 
import torch 

# Hugging Face Datasets
from datasets import load_dataset 

# Sentence Transformers for Cross-Encoders
from sentence_transformers import CrossEncoder, InputExample 

# Evaluation for binary classification
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator 

# PyTorch DataLoader for batching
from torch.utils.data import DataLoader 

random.seed(42)                 # fixed random seed 
np.random.seed(42)              # fixed numpy seed    

BASE_MODEL   = "cross-encoder/stsb-roberta-base"    # base pretrained model
OUTPUT_DIR   = "./finetuned_cross_encoder"          # final model save path
NUM_EPOCHS   = 3
BATCH_SIZE   = 16      
WARMUP_LEARNING_RATE = 10

# GPU Check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device.upper()}")

# Load Dataset 
def load_hf_dataset():
    ds = load_dataset("cnamuangtoun/resume-job-description-fit")

    train_samples, val_samples = [], []

    def get_label(label_val):
        if isinstance(label_val, (int, float)):
            return float(label_val)
        label_str = str(label_val).strip().lower()
        if "no fit" in label_str:
            return 0.0
        elif "fit" in label_str:
            return 1.0
        return 0.0

    # Processing Train Split
    for row in ds["train"]:
        resume = str(row.get("resume_text", ""))
        jd     = str(row.get("job_description_text", ""))
        label  = get_label(row["label"])
        train_samples.append(InputExample(texts=[resume, jd], label=label))

    # Target Split (validation or test)
    target_split = "validation" if "validation" in ds else "test"
    if target_split in ds:
        for row in ds[target_split]:
            resume = str(row.get("resume_text", ""))
            jd     = str(row.get("job_description_text", ""))
            label  = get_label(row["label"])
            val_samples.append(InputExample(texts=[resume, jd], label=label))
    else:
        split_idx     = int(len(train_samples) * 0.9)
        val_samples   = train_samples[split_idx:]
        train_samples = train_samples[:split_idx]
    return train_samples, val_samples


# Fine Tuning Function
def train():
    train_samples, val_samples = load_hf_dataset()
    model = CrossEncoder(BASE_MODEL, num_labels=1, device=device)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)

    val_texts  = [(s.texts[0], s.texts[1]) for s in val_samples]
    val_labels = [int(s.label) for s in val_samples]
    evaluator  = CEBinaryClassificationEvaluator(val_texts, val_labels)

    model.fit(
        train_dataloader  = train_dataloader,
        evaluator         = evaluator,
        epochs            = NUM_EPOCHS,
        warmup_steps      = WARMUP_LEARNING_RATE,
        output_path       = OUTPUT_DIR,
        save_best_model   = True,
        show_progress_bar = True,
    )

    model.save(OUTPUT_DIR)

# Run the training
train()