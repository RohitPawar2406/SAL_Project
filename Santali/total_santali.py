import os
import random
import json
import numpy as np
import torch
import evaluate
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Union, Any

from datasets import load_from_disk, Audio, concatenate_datasets, Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)

# ----------------------------------------------------
# 0️⃣ GLOBAL CONFIGURATION
# ----------------------------------------------------
DATASET_PATH = "/scratch/rohit.pawar/sal_dir_santali"
BASE_MODEL = "facebook/wav2vec2-large-xlsr-53"

# --- Experiment Sample & Split ---
DATASET_SAMPLE_FRACTION = 0.01  # use 1% for quick testing
TEST_SPLIT_SIZE = 0.1            # 10% for evaluation

# --- Main Output Directory ---
MAIN_OUTPUT_DIR = "/scratch/rohit.pawar/runs_sal/sal_santali_experiment"

# --- Training Hyperparameters ---
NUM_EPOCHS = 4
BATCH_SIZE = 4
GRAD_ACCUMULATION = 8
LEARNING_RATE = 1e-4
NUM_PROC = 4  # CPU cores for preprocessing

# ----------------------------------------------------
# 1️⃣ SHARED HELPER CLASSES & FUNCTIONS
# ----------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def prepare_audio(batch):
    batch["audio"] = batch["audio_filepath"]
    return batch

# ----------------------------------------------------
# 2️⃣ THE EXPERIMENT ENGINE
# ----------------------------------------------------
def run_experiment(exp_config: Dict[str, Any], train_ds: Dataset, eval_ds: Dataset) -> List[Dict]:
    exp_name = exp_config["name"]
    text_column = exp_config["text_column"]
    base_model = exp_config["base_model"]

    output_dir = os.path.join(exp_config["main_output_dir"], exp_name)
    vocab_path = os.path.join(output_dir, "vocab.json")
    logging_dir = os.path.join(exp_config["main_output_dir"], "tensorboard", exp_name)

    print(f"\n--- STARTING EXPERIMENT: {exp_name} ---")
    print(f"Output Dir: {output_dir}")

    # --- Create vocabulary ---
    full_text_ds = concatenate_datasets([train_ds, eval_ds])
    def lowercase_text(batch):
        if batch[text_column]:
            batch[text_column] = batch[text_column].lower()
        return batch
    full_text_ds = full_text_ds.map(lowercase_text, num_proc=NUM_PROC)

    vocab_set = set()
    def extract_chars(batch):
        all_text = " ".join(batch[text_column])
        vocab_set.update(list(all_text))
    full_text_ds.map(extract_chars, batched=True, batch_size=1000, num_proc=NUM_PROC)

    vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab_set)))}
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f)

    # --- Processor ---
    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # --- Preprocess datasets ---
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        text = batch[text_column].lower()
        with processor.as_target_processor():
            batch["labels"] = processor(text).input_ids
        return batch

    train_ds_proc = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names, num_proc=NUM_PROC)
    eval_ds_proc = eval_ds.map(prepare_dataset, remove_columns=eval_ds.column_names, num_proc=NUM_PROC)

    # --- Collator and Metrics ---
    data_collator = DataCollatorCTCWithPadding(processor)
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    # --- Model ---
    model = Wav2Vec2ForCTC.from_pretrained(
        base_model, ctc_loss_reduction="mean", ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(vocab_dict), ignore_mismatched_sizes=True
    )
    model.freeze_feature_extractor()

    # --- Trainer ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=LEARNING_RATE,
        warmup_steps=400,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir=logging_dir
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_proc,
        eval_dataset=eval_ds_proc,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor
    )

    trainer.train()

    # --- Save ---
    final_path = os.path.join(output_dir, "checkpoint-final")
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    return trainer.state.log_history

# ----------------------------------------------------
# 3️⃣ MAIN EXECUTION
# ----------------------------------------------------
def main():
    print("Loading Santali dataset...")
    ds = load_from_disk(DATASET_PATH)

    sample_size = int(len(ds) * DATASET_SAMPLE_FRACTION)
    ds_sample = ds.shuffle(seed=42).select(range(sample_size))
    split_ds = ds_sample.train_test_split(test_size=TEST_SPLIT_SIZE, seed=42)

    train_ds = split_ds['train'].map(prepare_audio, remove_columns=["audio_filepath"], num_proc=NUM_PROC)
    eval_ds = split_ds['test'].map(prepare_audio, remove_columns=["audio_filepath"], num_proc=NUM_PROC)
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16_000))
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16_000))

    print(f"Dataset ready: {len(train_ds)} train, {len(eval_ds)} eval")

    # Experiment 1: Latin text
    exp_config = {
        "name": "latin_run",
        "text_column": "latin_text",
        "base_model": BASE_MODEL,
        "main_output_dir": MAIN_OUTPUT_DIR
    }

    history = run_experiment(exp_config, train_ds, eval_ds)

if __name__ == "__main__":
    main()