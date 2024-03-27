import json


def load_dataset(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    # Assuming your JSON data has a specific structure, e.g.:
    # {"males": [...], "females": [...]}
    # Where each sub-list contains feature vectors

    males = data["males"]
    females = data["females"]

    dataset = []
    for features in males:
        dataset.append((features, "male"))
    for features in females:
        dataset.append((features, "female"))

    return dataset
