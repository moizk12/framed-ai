import fiftyone.zoo as foz
import os

# Your project structure path
base_dir = r"C:\Users\moizk\Downloads\framed-clean\framed-clean\test_dataset"

# Mapping 200 samples per category to EXACT dataset class names
dataset_plan = {
    "architecture": {"dataset": "open-images-v7", "classes": ["House", "Skyscraper", "Castle"]},
    "street": {"dataset": "coco-2017", "classes": ["bus", "traffic light", "stop sign"]},
    "nature": {"dataset": "coco-2017", "classes": ["bird", "bear", "zebra"]}, # COCO uses specific animals for nature
    "portraits": {"dataset": "coco-2017", "classes": ["person"]},
    "mixed": {"dataset": "coco-2017", "classes": ["dog", "cat", "chair"]},
    "ambiguous": {"dataset": "open-images-v7", "classes": ["Sculpture", "Toy"]} 
}

for cat, config in dataset_plan.items():
    print(f"\n--- Populating {cat.upper()} (200 images) ---")
    
    # Load the specific subset from the zoo
    dataset = foz.load_zoo_dataset(
        config["dataset"],
        split="validation",
        classes=config["classes"],
        max_samples=200,
        shuffle=True,
        seed=42
    )
    
    # Export to your structured folder
    target_path = os.path.join(base_dir, cat)
    os.makedirs(target_path, exist_ok=True)
    
    dataset.export(
        export_dir=target_path,
        dataset_type=foz.fo.types.ImageDirectory,
        export_media=True
    )

print("\n✅ Success: All categories populated at 200 images each.")