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
# 2️⃣ Load full Bodo dataset
# -------------------------------
print("🔹 Loading full Bodo IndicVoices dataset...")
# "bodo" is the config name in IndicVoices for Bodo language
bodo_ds = load_dataset(
    "ai4bharat/IndicVoices",
    data_dir="bodo", 
    split="train",
)

print(f"✅ Dataset loaded with {len(bodo_ds)} samples.\n")

# -------------------------------
# 3️⃣ Transliteration function
# -------------------------------
def add_transliteration(batch):
    text = batch.get("sentence") or batch.get("text")
    if not text:
        batch["transliterated_text"] = ""
        return batch
    # Bodo uses Devanagari, so we use the same Source Script
    batch["transliterated_text"] = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    return batch

# -------------------------------
# 4️⃣ Apply transliteration
# -------------------------------
print("🔹 Applying transliteration (Devanagari → Latin)...")
bodo_ds = bodo_ds.map(add_transliteration)

# -------------------------------
# 5️⃣ Print a random sample
# -------------------------------
idx = random.randint(0, len(bodo_ds) - 1)
print("\n✅ Example sample:")
#print("Devanagari (Bodo) :", bodo_ds[idx]["sentence"])
#print("Latinized         :", bodo_ds[idx]["transliterated_text"])

# -------------------------------
# 6️⃣ Save to disk
# -------------------------------
save_path = os.path.expanduser("/scratch/rohit.pawar/sal_dir_bodo")
bodo_ds.save_to_disk(save_path)
print(f"\n✅ Transliterated Bodo dataset saved at: {save_path}")