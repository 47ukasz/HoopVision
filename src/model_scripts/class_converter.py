import os

LABELS_DIR = "data/labels"   # katalog z etykietami YOLO (train/ i val/ w środku)

# Mapowanie:
# 0 -> 0 (basketball)
# 2 -> 1 (rim)
# 1 i 3 są usuwane
CLASS_MAP = {0: 0, 2: 1}

for root, dirs, files in os.walk(LABELS_DIR):
    for file in files:
        if not file.endswith(".txt"):
            continue

        path = os.path.join(root, file)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])

            if cls not in CLASS_MAP:
                continue  # pomijamy player i rim_moved

            new_cls = CLASS_MAP[cls]
            new_line = " ".join([str(new_cls)] + parts[1:])
            new_lines.append(new_line)

        # ZAPIS NADPISUJĄCY
        with open(path, "w") as f:
            for l in new_lines:
                f.write(l + "\n")

print("Gotowe! Wszystkie etykiety zostały przerobione na 2 klasy.")