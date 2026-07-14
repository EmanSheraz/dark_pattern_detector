import pandas as pd
import numpy as np
import torch
import random
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# ============================================================
# RANDOM SEED
# ============================================================
RANDOM_STATE = 42

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_STATE)

# ============================================================
# DEVICE
# ============================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE INFO")
print("=" * 60)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# LOAD DATASET
# ============================================================
print("\nLoading dataset...")

df = pd.read_csv("data/dark_patterns_dataset_clean.csv")

df = df[["text", "label"]].dropna()

# ============================================================
# REMOVE DUPLICATES
# ============================================================
before = len(df)

df = df.drop_duplicates(subset=["text"])

after = len(df)

print(f"Removed duplicates: {before - after}")

print(f"Final dataset size: {len(df)}")

# ============================================================
# LABEL ENCODING
# ============================================================
label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["label"])

CLASS_NAMES = sorted(label_encoder.classes_.tolist())

NUM_CLASSES = len(CLASS_NAMES)

print("\nClass Mapping:")

for idx, name in enumerate(CLASS_NAMES):

    count = (df["label"] == idx).sum()

    print(f"{idx} -> {name} ({count})")

# ============================================================
# TRAIN TEST SPLIT
# ============================================================
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    stratify=df["label"],
    random_state=RANDOM_STATE
)

print(f"\nTraining samples: {len(train_texts)}")
print(f"Testing samples : {len(test_texts)}")

# ============================================================
# TOKENIZER
# ============================================================
print("\nLoading tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

MAX_LENGTH = 64

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)

# ============================================================
# DATASET CLASS
# ============================================================
class DarkPatternDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):

        return len(self.labels)

train_dataset = DarkPatternDataset(
    train_encodings,
    train_labels
)

test_dataset = DarkPatternDataset(
    test_encodings,
    test_labels
)

# ============================================================
# LOAD MODEL
# ============================================================
print("\nLoading model...")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_CLASSES
)

# ============================================================
# FREEZE LOWER 8 LAYERS
# ============================================================
for i, layer in enumerate(model.bert.encoder.layer):

    if i < 8:

        for param in layer.parameters():
            param.requires_grad = False

model.to(device)

# ============================================================
# METRICS
# ============================================================
def compute_metrics(pred):

    labels = pred.label_ids

    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1
    }

# ============================================================
# TRAINING ARGUMENTS
# ============================================================
training_args = TrainingArguments(

    output_dir="./results",

    num_train_epochs=8,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    learning_rate=2e-5,

    warmup_ratio=0.1,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1_macro",

    greater_is_better=True,

    logging_dir="./logs",

    logging_steps=10,

    report_to="none",

    seed=RANDOM_STATE,

    use_cpu=not torch.cuda.is_available()
)

# ============================================================
# TRAINER
# ============================================================
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)

# ============================================================
# TRAIN
# ============================================================
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

trainer.train()

# ============================================================
# EVALUATE
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

results = trainer.evaluate()

print(f"\nAccuracy        : {results['eval_accuracy']:.4f}")
print(f"Macro Precision : {results['eval_precision_macro']:.4f}")
print(f"Macro Recall    : {results['eval_recall_macro']:.4f}")
print(f"Macro F1        : {results['eval_f1_macro']:.4f}")

# ============================================================
# PREDICTIONS
# ============================================================
predictions = trainer.predict(test_dataset)

preds = predictions.predictions.argmax(-1)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(
    test_labels,
    preds,
    target_names=CLASS_NAMES,
    digits=4
))

# ============================================================
# CONFUSION MATRIX
# ============================================================
print("\nGenerating confusion matrix...")

cm = confusion_matrix(test_labels, preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(8, 6))

disp.plot(ax=ax)

plt.title("Confusion Matrix - Dark Pattern Detection")

os.makedirs("plots", exist_ok=True)

plt.savefig(
    "plots/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("✅ Confusion matrix saved to:")
print("plots/confusion_matrix.png")

# ============================================================
# SAVE MODEL
# ============================================================
print("\nSaving model...")

SAVE_PATH = "./model"

os.makedirs(SAVE_PATH, exist_ok=True)

model.save_pretrained(SAVE_PATH)

tokenizer.save_pretrained(SAVE_PATH)

torch.save(
    CLASS_NAMES,
    os.path.join(SAVE_PATH, "class_names.pt")
)

print("✅ Model saved successfully!")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)