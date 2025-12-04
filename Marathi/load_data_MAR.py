# This file downloads the Marathi dataset and saves it to the specified path

import os
import random
from datasets import load_dataset
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# -------------------------------
# 1️⃣ Confirm HF_HOME setup
# -------------------------------
hf_home = os.environ.get("HF_HOME", None)
print(f"✅ HF_HOME currently set to: {hf_home if hf_home else '~/.cache/huggingface'}")

# -------------------------------
# 2️⃣ Load full Marathi dataset
# -------------------------------
print("🔹 Loading full Marathi IndicVoices dataset...")
# Note: data_dir for Marathi is usually "marathi" in IndicVoices
marathi_ds = load_dataset(
    "ai4bharat/IndicVoices",
    data_dir="marathi", 
    split="train",     # full dataset
)

print(f"✅ Dataset loaded with {len(marathi_ds)} samples.\n")
print("Sample before transliteration:")
print(marathi_ds[0])

# -------------------------------
# 3️⃣ Transliteration function
# -------------------------------
def add_transliteration(batch):
    # 'sentence' is often the key, but fallback to 'text' just in case
    text = batch.get("sentence") or batch.get("text")
    if not text:
        batch["transliterated_text"] = ""
        return batch
    # Marathi uses Devanagari script, so the source script remains DEVANAGARI
    batch["transliterated_text"] = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    return batch

# -------------------------------
# 4️⃣ Apply transliteration
# -------------------------------
print("🔹 Applying transliteration (Devanagari → Latin)...")
marathi_ds = marathi_ds.map(add_transliteration)

# -------------------------------
# 5️⃣ Print a single random sample (Moved before save for verification)
# -------------------------------
idx = random.randint(0, len(marathi_ds) - 1)
print("\n✅ Example sample:")
# print("Devanagari :", marathi_ds[idx]["sentence"])
print("Latinized  :", marathi_ds[idx]["transliterated_text"])

# -------------------------------
# 6️⃣ Save to disk
# -------------------------------
# Changed directory to 'sal_dir_marathi' to avoid overwriting Hindi data
save_path = os.path.expanduser("/scratch/rohit.pawar/sal_dir_marathi")
marathi_ds.save_to_disk(save_path)
print(f"\n✅ Transliterated Marathi dataset saved at: {save_path}")

# breakpoint()