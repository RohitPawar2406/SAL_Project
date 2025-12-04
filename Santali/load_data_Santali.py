import os
import random
from datasets import load_dataset
from aksharamukha import transliterate

# -------------------------------
# 1️⃣ Confirm HF_HOME setup
# -------------------------------
hf_home = os.environ.get("HF_HOME", None)
print(f"HF_HOME currently set to: {hf_home if hf_home else '~/.cache/huggingface'}")

# -------------------------------
# 2️⃣ Load full Santali dataset
# -------------------------------
print("🔹 Loading full Santali IndicVoices dataset...")

santali_ds = load_dataset(
    "ai4bharat/IndicVoices",
    data_dir="santali",
    split="train",
)

print(f"✅ Dataset loaded with {len(santali_ds)} samples.\n")

# -------------------------------
# 3️⃣ Transliteration function (Ol Chiki → Latin)
# -------------------------------
def add_transliteration(batch):
    text = batch.get("sentence") or batch.get("text")
    if not text:
        batch["latin_text"] = ""
        return batch

    # Use Aksharamukha converter
    batch["latin_text"] = transliterate.process("Ol Chiki", "Latin", text)
    return batch

# -------------------------------
# 4️⃣ Apply transliteration
# -------------------------------
print("🔹 Applying Aksharamukha transliteration (Ol Chiki → Latin)...")
santali_ds = santali_ds.map(add_transliteration)

# -------------------------------
# 5️⃣ Print sample
# -------------------------------
idx = random.randint(0, len(santali_ds) - 1)
print("\nExample:")
print("Ol Chiki :", santali_ds[idx]["sentence"])
print("Latin    :", santali_ds[idx]["latin_text"])

# -------------------------------
# 6️⃣ Save to disk
# -------------------------------
save_path = os.path.expanduser("/scratch/rohit.pawar/sal_dir_santali")
santali_ds.save_to_disk(save_path)

print(f"\n✅ Transliterated Santali dataset saved at: {save_path}")
