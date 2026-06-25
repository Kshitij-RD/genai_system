"""Add image_gen samples to existing eval_set.json — zero heavy imports."""
import json
from pathlib import Path

IMAGE_PROMPTS = [
    "a cyberpunk city street at night, neon lights reflecting in puddles, highly detailed",
    "a cozy cabin in a snowy pine forest, warm light glowing from the windows, sunset",
    "a majestic golden retriever sitting in a field of blooming sunflowers",
    "an astronaut floating in outer space with earth in the background, cinematic lighting",
    "a delicious slice of chocolate cake with strawberries on top, food photography",
    "a magical library with floating books and glowing dust motes, fantasy concept art",
    "a futuristic sports car driving on a coastal highway at golden hour",
    "a steampunk coffee machine made of brass and glass, intricate gears",
    "a close-up portrait of an elderly man with deep wrinkles, dramatic studio lighting",
    "a peaceful zen garden with a koi pond, bamboo, and stepping stones",
    "a massive dragon sleeping on a hoard of gold coins inside a dark cave",
    "a vibrant coral reef teeming with tropical fish and sea turtles",
    "a vintage leather armchair next to a roaring fireplace in a study room",
    "a post-apocalyptic cityscape overgrown with lush green vegetation",
    "a cute robot watering a small potted plant, pixar style animation 3d",
    "a mysterious hooded figure holding a glowing staff in a dense foggy forest",
    "an elegant glass bottle containing a miniature galaxy inside",
    "a bustling market in a medieval fantasy town, merchants selling spices",
    "a sleek modern minimalist living room with large windows overlooking a city skyline",
    "a detailed map of a fantasy world with drawn sea monsters and compass roses",
    "a giant tree house built into an ancient oak tree, whimsical illustration",
    "a roaring lion leaping over a stream in the African savanna during sunset",
    "a neon sign shaped like a flamingo illuminating a dark brick wall",
    "a crystal clear stream flowing through a lush green valley with distant mountains",
    "a samurai standing in a field of cherry blossoms holding a katana, dramatic mood",
    "a wizard mixing colorful potions in a cluttered medieval alchemy lab",
    "a futuristic floating city among the clouds, utopian sci-fi architecture",
    "a classic American diner from the 1950s with red booths and a jukebox",
    "a magical portal made of swirling blue energy standing in a desert",
    "a tiny intricate ship inside a glass bottle resting on a wooden desk",
]

# Load existing dataset
with open("data/processed/eval_set.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

# Remove any old image_gen samples
samples = [s for s in samples if s.get("task") != "image_gen"]

# Find max existing index
max_idx = max(int(s["id"].split("_")[1]) for s in samples)

# Add image_gen samples
for i, prompt in enumerate(IMAGE_PROMPTS, 1):
    samples.append({
        "id": f"img_{max_idx + i:03d}",
        "task": "image_gen",
        "prompt": prompt,
        "reference_output": None,
        "image_path": None,
        "category": "creative",
        "safety_label": "safe",
    })

# Save
with open("data/processed/eval_set.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

# Update splits - simple random 70/30
import random
random.seed(42)
ids = [s["id"] for s in samples]
random.shuffle(ids)
split_point = int(len(ids) * 0.7)
splits = {"train": sorted(ids[:split_point]), "test": sorted(ids[split_point:])}
with open("data/splits.json", "w") as f:
    json.dump(splits, f, indent=2)

total = len(samples)
img = sum(1 for s in samples if s["task"] == "image_gen")
unsafe = sum(1 for s in samples if s["safety_label"] == "unsafe")
print(f"Done! {total} samples ({img} image_gen, {unsafe} unsafe)")
print(f"Train: {len(splits['train'])}, Test: {len(splits['test'])}")
