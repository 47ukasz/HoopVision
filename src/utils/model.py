from pathlib import Path
import re

def create_next_model_folder(base_dir="models", prefix="hoopvision_v"):
    models_dir = Path(base_dir)
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)$")

    existing = []
    for path in models_dir.iterdir():
        if path.is_dir():
            match = pattern.match(path.name)
            if match:
                existing.append(int(match.group(1)))

    last_num = max(existing) if existing else 0
    new_num = last_num + 1
    new_name = f"{prefix}{new_num}"
    new_dir = models_dir / new_name
    new_dir.mkdir(exist_ok=True)

    return new_dir  
