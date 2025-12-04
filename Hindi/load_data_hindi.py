# This file download dataset and save where you mentioned HF_HOME env variable


import os
from datasets import load_dataset
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# -------------------------------
# 1️⃣ Confirm HF_HOME setup
# -------------------------------
hf_home = os.environ.get("HF_HOME", None)
print(f"✅ HF_HOME currently set to: {hf_home if hf_home else '~/.cache/huggingface'}")

# # You can inspect where datasets are cached:
# from datasets import config
# print(f"Datasets cache dir: {config.HF_DATASETS_CACHE}")
# # print(f"Transformers cache dir: {config.TRANSFORMERS_CACHE}")
# # print(f"Metrics cache dir: {config.HF_METRICS_CACHE}\n")

# -------------------------------
# 2️⃣ Load full Hindi dataset
# -------------------------------
print("🔹 Loading full Hindi IndicVoices dataset...")
hindi_ds = load_dataset(
    "ai4bharat/IndicVoices",
    data_dir="hindi",
    split="train",     # full dataset (remove [:1%])
)

print(f"✅ Dataset loaded with {len(hindi_ds)} samples.\n")
print("Sample before transliteration:")
print(hindi_ds[0])

# -------------------------------
# 3️⃣ Transliteration function
# -------------------------------
def add_transliteration(batch):
    text = batch.get("sentence") or batch.get("text")
    if not text:
        batch["transliterated_text"] = ""
        return batch
    batch["transliterated_text"] = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    return batch

# -------------------------------
# 4️⃣ Apply transliteration
# -------------------------------
print("🔹 Applying transliteration (Devanagari → Latin)...")
hindi_ds = hindi_ds.map(add_transliteration)

# -------------------------------
# 6️⃣ Save to disk
# -------------------------------
save_path = os.path.expanduser("/scratch/rohit.pawar/sal_dir")
hindi_ds.save_to_disk(save_path)
print(f"\n✅ Transliterated dataset saved at: {save_path}")

breakpoint()
# -------------------------------
# 5️⃣ Print a single random sample
# -------------------------------
import random
idx = random.randint(0, len(hindi_ds) - 1)
print("\n✅ Example sample:")
#print("Devanagari :", hindi_ds[idx]["sentence"])
print("Latinized  :", hindi_ds[idx]["transliterated_text"])





########################## HINDI #####################
# num_rows: 333256 max duration : 29 sec and min is 0.17 sec

######################################################

########################## HINDI #####################
# num_rows: 333256 max duration : 29 sec and min is 0.17 sec

######################################################