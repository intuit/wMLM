from pathlib import Path
import os
from tqdm.auto import tqdm


paths = [str(x) for x in Path("wikipedia").glob("**/*.txt")]

doc_size = 96
doc_per_file = 50000
text_data = []
file_count = 0
dir_name = "wiki_" + str(doc_size)

if not os.path.exists(dir_name):

    os.makedirs(dir_name)

for path in tqdm(paths):

    with open(path, "r") as file:

        for line in file:

            tokens = line.strip().split()

            if len(tokens) % doc_size == 0:
                segments = len(tokens) // doc_size
            else:
                segments = len(tokens) // doc_size + 1

            for i in range(segments):

                text_data.append(" ".join(tokens[doc_size * i : doc_size * (i + 1)]))

            if len(text_data) >= doc_per_file:

                file_name = str(file_count).zfill(5)

                with open(
                    f"{dir_name}/text_{file_name}.txt", "w", encoding="utf-8"
                ) as fp:
                    fp.write("\n".join(text_data[:doc_per_file]))

                text_data = text_data[doc_per_file:]
                file_count += 1
