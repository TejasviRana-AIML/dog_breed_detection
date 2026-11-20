
import torch

files = [
    "dog_breed_model.pth",
    "dog_breed_model_ft.pth"
]

for file in files:
    print("\nChecking:", file)

    checkpoint = torch.load(
        file,
        map_location="cpu",
        weights_only=True
    )

    print("Type:", type(checkpoint))

    if isinstance(checkpoint, dict):
        print("Keys:", list(checkpoint.keys())[:15])
        print("Total keys:", len(checkpoint))

    else:
        print("Checkpoint is not a dictionary")
