
import torch

files = [
    "dog_breed_model.pth",
    "dog_breed_model_ft.pth"
]

for file in files:
    print("\nModel:", file)

    state = torch.load(
        file,
        map_location="cpu",
        weights_only=True
    )

    for key, value in state.items():
        if "fc" in key:
            print(key, value.shape)
