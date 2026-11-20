
import json

notebook_path = "dog_breed_detection_model.ipynb"

with open(notebook_path, "r", encoding="utf-8") as file:
    notebook = json.load(file)

for i, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])

        print(f"\n{'=' * 60}")
        print(f"CELL {i + 1}")
        print("=" * 60)
        print(source)
