from tqdm.auto import tqdm
import os
from datasets import load_dataset


def loaddataset(datasetname):
    try:
        dataset = load_dataset(datasetname, "20220301.en", cache_dir="./dataset_cache")
    except:
        raise ValueError("unable to download dataset")
    return dataset


def get_data(datasetname):
    text_data = []
    file_count = 0
    dataset = loaddataset(datasetname)
    if not os.path.exists("wikipedia"):
        os.makedirs("wikipedia")

    for sample in tqdm(dataset["train"]):
        sample = sample["text"].replace("\n", "")
        text_data.append(sample)
        if len(text_data) == 10_000:
            file_name = str(file_count).zfill(4)
            with open(f"wikipedia/text_{file_name}.txt", "w", encoding="utf-8") as file:
                file.write("\n".join(text_data))
                text_data = []
                file_count += 1

if __name__ == "__main__":
    output_dir = "./dataset_cache"
    datasetname = "wikipedia"
    os.makedirs(output_dir, exist_ok=True)
    get_data(datasetname)
