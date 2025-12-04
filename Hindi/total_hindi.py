import os
import json
import re
import numpy as np
import torch
import evaluate
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

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
DATASET_PATH = "/scratch/rohit.pawar/sal_dir"
BASE_MODEL = "ai4bharat/indicwav2vec-hindi"

# --- Experiment Sample & Split ---
# This is shared for both experiments to ensure a fair comparison
DATASET_SAMPLE_FRACTION = 0.001 # Use 60% of the data
TEST_SPLIT_SIZE = 0.1         # Use 10% of that sample for evaluation

# --- Main Output Directory ---
MAIN_OUTPUT_DIR = "/scratch/rohit.pawar/runs_sal/sal_hindi_experiment"

# --- Training Hyperparameters ---
NUM_EPOCHS = 4               # Set high for a real run, 2-3 for a test
BATCH_SIZE = 8                # This is for FULL fine-tuning, VRAM-heavy!
GRAD_ACCUMULATION = 8         # Increase if BATCH_SIZE is low (e.g., 4*8=32)
LEARNING_RATE = 1e-4          # Good starting point for full fine-tuning
NUM_PROC = 8                  # Number of cores for data preprocessing

# ----------------------------------------------------
# 1️⃣ SHARED HELPER CLASSES & FUNCTIONS
# ----------------------------------------------------

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that dynamically pads the inputs received.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def create_vocab_from_dataset(dataset: Union[Dataset, concatenate_datasets],
                              text_column: str,
                              vocab_path: str) -> Dict:
    """
    Scans the dataset's text column to create a vocab.json file.
    """
    print(f"Creating vocabulary from '{text_column}' column...")
    vocab_set = set()
    
    def extract_chars_batch(batch):
        all_text = " ".join(batch[text_column])
        vocab_set.update(list(all_text.lower()))
        
    dataset.map(
        extract_chars_batch,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC
    )
    
    vocab_set = {char for char in vocab_set if char}
    vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab_set)))}
    
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    
    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f)
        
    print(f"Vocabulary saved to {vocab_path} with {len(vocab_dict)} tokens.")
    return vocab_dict


def prepare_audio(batch):
    """Loads and resamples audio."""
    batch["audio"] = batch["audio_filepath"]
    return batch


def create_comparison_plots(history_devanagari: List[Dict], 
                            history_latin: List[Dict], 
                            output_dir: str):
    """
    Generates and saves comparison plots for Loss and CER.
    """
    print("Generating comparison plots...")
    
    # --- Parse Histories ---
    def parse_history(history):
        metrics = {
            "epochs": [], "train_loss": [], "eval_loss": [], "eval_cer": []
        }
        for log in history:
            if "loss" in log: # Training log
                metrics["epochs"].append(log["epoch"])
                metrics["train_loss"].append(log["loss"])
            if "eval_loss" in log: # Eval log
                metrics["eval_loss"].append(log["eval_loss"])
                metrics["eval_cer"].append(log["eval_cer"])
        # Ensure eval epochs align if no training logs are present at that step
        eval_epochs = [log["epoch"] for log in history if "eval_loss" in log]
        return metrics, eval_epochs

    metrics_dev, eval_epochs_dev = parse_history(history_devanagari)
    metrics_lat, eval_epochs_lat = parse_history(history_latin)

    # --- Plot 1: Loss Comparison (Side-by-Side) ---
    try:
        plt.figure(figsize=(15, 6))
        
        # Subplot 1: Devanagari Loss
        plt.subplot(1, 2, 1)
        plt.plot(metrics_dev["epochs"], metrics_dev["train_loss"], label="Devanagari Train Loss", marker='o', alpha=0.8)
        plt.plot(eval_epochs_dev, metrics_dev["eval_loss"], label="Devanagari Eval Loss", marker='o', linestyle='--')
        plt.title("Experiment 1: Devanagari Head")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Latin Loss
        plt.subplot(1, 2, 2)
        plt.plot(metrics_lat["epochs"], metrics_lat["train_loss"], label="Latin Train Loss", marker='o', color='green', alpha=0.8)
        plt.plot(eval_epochs_lat, metrics_lat["eval_loss"], label="Latin Eval Loss", marker='o', linestyle='--', color='red')
        plt.title("Experiment 2: Latin Head")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        loss_plot_path = os.path.join(output_dir, "loss_comparison.png")
        plt.savefig(loss_plot_path)
        print(f"✅ Loss comparison plot saved to {loss_plot_path}")
        plt.close()

    except Exception as e:
        print(f"⚠️ Could not generate loss plot: {e}")

    # --- Plot 2: CER Comparison (Combined) ---
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(eval_epochs_dev, metrics_dev["eval_cer"], label="Devanagari Head CER", marker='o', linewidth=2)
        plt.plot(eval_epochs_lat, metrics_lat["eval_cer"], label="Latin Head CER", marker='o', linewidth=2)
        plt.title("CER Comparison (Same Audio, Different Heads)")
        plt.xlabel("Epoch")
        plt.ylabel("CER (%)")
        plt.legend()
        plt.grid(True)
        
        cer_plot_path = os.path.join(output_dir, "cer_comparison.png")
        plt.savefig(cer_plot_path)
        print(f"✅ CER comparison plot saved to {cer_plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Could not generate CER plot: {e}")

# ----------------------------------------------------
# 2️⃣ THE EXPERIMENT "ENGINE"
# ----------------------------------------------------

def run_experiment(exp_config: Dict[str, Any], 
                   train_ds: Dataset, 
                   eval_ds: Dataset) -> List[Dict]:
    """
    Runs a single, complete fine-tuning experiment.
    """
    
    # --- 1. Unpack Config ---
    exp_name = exp_config["name"]
    text_column = exp_config["text_column"]
    base_model = exp_config["base_model"]
    
    output_dir = os.path.join(exp_config["main_output_dir"], exp_name)
    vocab_path = os.path.join(output_dir, "vocab.json")
    logging_dir = os.path.join(exp_config["main_output_dir"], "tensorboard", exp_name)
    
    print(f"\n--- STARTING EXPERIMENT: {exp_name} ---")
    print(f"  Base Model: {base_model}")
    print(f"  Text Column: {text_column}")
    print(f"  Output Dir: {output_dir}")
    
    # --- 2. Create Vocab ---
    full_text_ds = concatenate_datasets([train_ds, eval_ds])
    
    # Lowercase the text *before* creating the vocab
    def lowercase_text(batch):
        if batch[text_column]:
            batch[text_column] = batch[text_column].lower()
        return batch
    full_text_ds = full_text_ds.map(lowercase_text, num_proc=NUM_PROC)
    
    vocab_dict = create_vocab_from_dataset(full_text_ds, text_column, vocab_path)
    vocab_size = len(vocab_dict)

    # --- 3. Initialize Processor ---
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token=" "
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model)
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    # --- 4. Preprocess Datasets ---
    print("Preprocessing data for this experiment...")
    
    def prepare_dataset_for_model(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        
        text = batch[text_column].lower() # Ensure lowercase
        with processor.as_target_processor():
            batch["labels"] = processor(text).input_ids
        return batch

    train_ds_processed = train_ds.map(
        prepare_dataset_for_model,
        remove_columns=train_ds.column_names,
        num_proc=NUM_PROC
    )
    eval_ds_processed = eval_ds.map(
        prepare_dataset_for_model,
        remove_columns=eval_ds.column_names,
        num_proc=NUM_PROC
    )
    
    # --- 5. Collator and Metrics ---
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    # --- 6. Load Model (The "Head Swap") ---
    print("Loading model and performing 'head swap'...")
    model = Wav2Vec2ForCTC.from_pretrained(
        base_model,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=vocab_size,
        ignore_mismatched_sizes=True, # This is the "swap"
        use_safetensors=True
    )
    model.freeze_feature_extractor()
    print("✅ Model 'ear' frozen, new 'hand' attached.")

    # --- 7. Train ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        eval_strategy="epoch", # Use 'eval_strategy' for older transformers
        logging_strategy="epoch",
        learning_rate=LEARNING_RATE,
        warmup_steps=400,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir=logging_dir,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_processed,
        eval_dataset=eval_ds_processed,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print(f"🚀 Starting training for {exp_name}...")
    trainer.train()
    print(f"✅ Training complete for {exp_name}.")

    # --- 8. Save Final Model ---
    final_checkpoint_path = os.path.join(output_dir, "checkpoint-final")
    print(f"Saving final model and processor to {final_checkpoint_path}")
    trainer.save_model(final_checkpoint_path)
    processor.save_pretrained(final_checkpoint_path)
    
    return trainer.state.log_history

# ----------------------------------------------------
# 3️⃣ MAIN SCRIPT EXECUTION
# ----------------------------------------------------

def main():
    print("=============================================")
    print("STARTING SAL HINDI ORTHOGRAPHY EXPERIMENT")
    print("=============================================")

    # --- PHASE 1: Load and Prepare Shared Data ---
    print("\n--- PHASE 1: Loading and Splitting Shared Data ---")
    ds = load_from_disk(DATASET_PATH)

    print(f"Shuffling and sampling {DATASET_SAMPLE_FRACTION*100}% of data...")
    ds_sample = ds.shuffle(seed=42).select(range(int(len(ds) * DATASET_SAMPLE_FRACTION)))

    print("Splitting into train and evaluation sets...")
    split_dataset = ds_sample.train_test_split(test_size=TEST_SPLIT_SIZE, seed=42)
    
    # These two datasets will be re-used for both experiments
    train_ds = split_dataset['train']
    eval_ds = split_dataset['test']
    
    # Load and resample audio
    train_ds = train_ds.map(prepare_audio, remove_columns=["audio_filepath"], num_proc=NUM_PROC)
    eval_ds = eval_ds.map(prepare_audio, remove_columns=["audio_filepath"], num_proc=NUM_PROC)
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16_000))
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16_000))
    
    print(f"✅ Shared data is ready: {len(train_ds)} train, {len(eval_ds)} eval samples.")

    # --- PHASE 2: Run Experiment 1 (Devanagari) ---
    exp1_config = {
        "name": "devanagari_run",
        "text_column": "text",
        "base_model": BASE_MODEL,
        "main_output_dir": MAIN_OUTPUT_DIR,
    }
    history_devanagari = run_experiment(exp1_config, train_ds, eval_ds)

    # --- PHASE 3: Run Experiment 2 (Latin) ---
    exp2_config = {
        "name": "latin_run",
        "text_column": "transliterated_text",
        "base_model": BASE_MODEL,
        "main_output_dir": MAIN_OUTPUT_DIR,
    }
    history_latin = run_experiment(exp2_config, train_ds, eval_ds)

    # --- PHASE 4: Analysis and Plotting ---
    print("\n--- PHASE 4: Generating Final Comparison Plots ---")
    create_comparison_plots(history_devanagari, history_latin, MAIN_OUTPUT_DIR)

    print("\n=============================================")
    print("✅ EXPERIMENT SUITE COMPLETE")
    print(f"  Find all outputs in: {MAIN_OUTPUT_DIR}")
    print(f"  Find TensorBoard logs in: {MAIN_OUTPUT_DIR}/tensorboard")
    print("=============================================")

if __name__ == "__main__":
    main()