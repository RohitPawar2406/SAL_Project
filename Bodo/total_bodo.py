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
# 0️⃣ GLOBAL CONFIGURATION (BODO)
# ----------------------------------------------------
DATASET_PATH = "/scratch/rohit.pawar/sal_dir_bodo"  # <--- Bodo dataset saved previously
BASE_MODEL = "wav2vec2-large-xlsr-53"       # Replace with a Bodo-pretrained model if available

# --- Experiment Sample & Split ---
DATASET_SAMPLE_FRACTION = 0.001  # Fraction of dataset to use (tweak for real runs)
TEST_SPLIT_SIZE = 0.1            # 10% of the sample for eval

# --- Main Output Directory ---
MAIN_OUTPUT_DIR = "/scratch/rohit.pawar/runs_sal/sal_bodo_experiment"

# --- Training Hyperparameters ---
NUM_EPOCHS = 4
BATCH_SIZE = 8
GRAD_ACCUMULATION = 8
LEARNING_RATE = 1e-4
NUM_PROC = 8

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

    # Reserve tokens
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f)

    print(f"Vocabulary saved to {vocab_path} with {len(vocab_dict)} tokens.")
    return vocab_dict


def prepare_audio(batch):
    """Loads and resamples audio. Expects an "audio_filepath" column previously saved as in your pipeline."""
    # Keep the same key name used in the original pipeline (match your dataset)
    batch["audio"] = batch.get("audio_filepath") or batch.get("audio") or batch.get("path")
    # If your dataset already has an 'audio' column with arrays, skip this mapping.
    return batch


def create_comparison_plots(history_devanagari: List[Dict], 
                            history_latin: List[Dict], 
                            output_dir: str):
    """
    Generates and saves comparison plots for Loss and CER.
    """
    print("Generating comparison plots...")

    def parse_history(history):
        metrics = {"epochs": [], "train_loss": [], "eval_loss": [], "eval_cer": []}
        for log in history:
            if "loss" in log:
                metrics["epochs"].append(log.get("epoch", len(metrics["epochs"]) + 1))
                metrics["train_loss"].append(log["loss"])
            if "eval_loss" in log:
                metrics["eval_loss"].append(log["eval_loss"])
                metrics["eval_cer"].append(log.get("eval_cer", None))
        eval_epochs = [log["epoch"] for log in history if "eval_loss" in log]
        return metrics, eval_epochs

    metrics_dev, eval_epochs_dev = parse_history(history_devanagari)
    metrics_lat, eval_epochs_lat = parse_history(history_latin)

    # Loss Comparison (side-by-side)
    try:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.plot(metrics_dev["epochs"], metrics_dev["train_loss"], label="Devanagari Train Loss", marker='o')
        if eval_epochs_dev:
            plt.plot(eval_epochs_dev, metrics_dev["eval_loss"], label="Devanagari Eval Loss", marker='o', linestyle='--')
        plt.title("Experiment 1: Devanagari Head")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(metrics_lat["epochs"], metrics_lat["train_loss"], label="Latin Train Loss", marker='o')
        if eval_epochs_lat:
            plt.plot(eval_epochs_lat, metrics_lat["eval_loss"], label="Latin Eval Loss", marker='o', linestyle='--')
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

    # CER Comparison
    try:
        plt.figure(figsize=(10, 6))
        if eval_epochs_dev and metrics_dev["eval_cer"]:
            plt.plot(eval_epochs_dev, metrics_dev["eval_cer"], label="Devanagari Head CER", marker='o')
        if eval_epochs_lat and metrics_lat["eval_cer"]:
            plt.plot(eval_epochs_lat, metrics_lat["eval_cer"], label="Latin Head CER", marker='o')
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
# 2️⃣ THE EXPERIMENT "ENGINE" (adapted for Bodo)
# ----------------------------------------------------

def run_experiment(exp_config: Dict[str, Any], 
                   train_ds: Dataset, 
                   eval_ds: Dataset) -> List[Dict]:
    """
    Runs a single, complete fine-tuning experiment.
    """
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

    # Create Vocab
    full_text_ds = concatenate_datasets([train_ds, eval_ds])

    def lowercase_text(batch):
        if batch.get(text_column):
            batch[text_column] = [t.lower() for t in batch[text_column]]
        return batch

    full_text_ds = full_text_ds.map(lowercase_text, batched=True, num_proc=NUM_PROC)

    vocab_dict = create_vocab_from_dataset(full_text_ds, text_column, vocab_path)
    vocab_size = len(vocab_dict)

    # Initialize Processor
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

    # Preprocess Datasets
    print("Preprocessing data for this experiment...")

    def prepare_dataset_for_model(batch):
        # If dataset has audio filepaths, load as array using the "audio" column when casted
        audio = batch["audio"]
        # audio may be a dict when cast to Audio type
        if isinstance(audio, dict):
            array = audio.get("array")
            sr = audio.get("sampling_rate")
            batch["input_values"] = processor(array, sampling_rate=sr).input_values[0]
        else:
            # If already pre-extracted input_values are present
            batch["input_values"] = processor(audio, sampling_rate=16_000).input_values[0]

        text = batch[text_column].lower() if isinstance(batch[text_column], str) else batch[text_column]
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

    # Collator and Metrics
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

    # Load Model (Head swap)
    print("Loading model and performing 'head swap'...")
    model = Wav2Vec2ForCTC.from_pretrained(
        base_model,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=vocab_size,
        ignore_mismatched_sizes=True,
        use_safetensors=True
    )
    model.freeze_feature_extractor()
    print("✅ Frozen feature extractor, new classifier attached.")

    # Training arguments
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

    # Save final model
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
    print("STARTING SAL BODO ORTHOGRAPHY EXPERIMENT")
    print("=============================================")

    # PHASE 1: Load and Prepare Shared Data
    print("\n--- PHASE 1: Loading and Splitting Shared Data ---")
    ds = load_from_disk(DATASET_PATH)

    print(f"Shuffling and sampling {DATASET_SAMPLE_FRACTION*100}% of data...")
    ds_sample = ds.shuffle(seed=42).select(range(int(len(ds) * DATASET_SAMPLE_FRACTION)))

    print("Splitting into train and evaluation sets...")
    split_dataset = ds_sample.train_test_split(test_size=TEST_SPLIT_SIZE, seed=42)

    train_ds = split_dataset['train']
    eval_ds = split_dataset['test']

    # Load and resample audio -- expect audio_filepath column or already audio column
    train_ds = train_ds.map(prepare_audio, remove_columns=[c for c in train_ds.column_names if c in ("audio_filepath",)], num_proc=NUM_PROC)
    eval_ds = eval_ds.map(prepare_audio, remove_columns=[c for c in eval_ds.column_names if c in ("audio_filepath",)], num_proc=NUM_PROC)

    # Cast to Audio so 'audio' column becomes a dict with array and sampling_rate
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16_000))
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16_000))

    print(f"✅ Shared data is ready: {len(train_ds)} train, {len(eval_ds)} eval samples.")

    # EXPERIMENT 1: Devanagari (use original text column if present)
    exp1_config = {
        "name": "devanagari_run",
        "text_column": "text",  # change if your Bodo dataset uses 'sentence' or another field
        "base_model": BASE_MODEL,
        "main_output_dir": MAIN_OUTPUT_DIR,
    }
    history_devanagari = run_experiment(exp1_config, train_ds, eval_ds)

    # EXPERIMENT 2: Latin (use the transliterated_text field added earlier)
    exp2_config = {
        "name": "latin_run",
        "text_column": "transliterated_text",
        "base_model": BASE_MODEL,
        "main_output_dir": MAIN_OUTPUT_DIR,
    }
    history_latin = run_experiment(exp2_config, train_ds, eval_ds)

    # PHASE 4: Analysis and Plotting
    print("\n--- PHASE 4: Generating Final Comparison Plots ---")
    create_comparison_plots(history_devanagari, history_latin, MAIN_OUTPUT_DIR)

    print("\n=============================================")
    print("✅ EXPERIMENT SUITE COMPLETE")
    print(f"  Find all outputs in: {MAIN_OUTPUT_DIR}")
    print(f"  Find TensorBoard logs in: {MAIN_OUTPUT_DIR}/tensorboard")
    print("=============================================")


if __name__ == "__main__":
    main()
