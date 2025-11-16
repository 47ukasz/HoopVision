import yaml
from pathlib import Path

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):

    if not Path(path_to_classes_txt).exists():
        print(f'Plik classes.txt nie istnieje! Stwórz plik z klasami i przenieś go do ścieżki: {path_to_classes_txt}')
        return

    with open(path_to_classes_txt, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    number_of_classes = len(classes)

    data = {
        'path': '/data',
        'train': 'images/train',
        'val': 'images/val',
        'nc': number_of_classes,
        'names': classes
    }

    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f'Stworzono plik konfiguracyjny w: {path_to_data_yaml}')


project_cwd = Path.cwd()
print(project_cwd)
classes_path = project_cwd / "data/classes.txt"
yaml_path = project_cwd / "data.yaml"

create_data_yaml(classes_path, yaml_path)
