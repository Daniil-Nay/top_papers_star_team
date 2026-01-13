# test workflow
from datetime import datetime
import os

import api

print("Start stuff")

with open("test.txt", "w") as fout:
    fout.write(f"{datetime.now()} some logs...\n")

# test LLM API call (Gigachat)
text = api.get_text(
    "Привет, дай краткий тост на русском, одно предложение.",
    api="gigachat",
    model=api.GIGACHAT_MODEL,
    temperature=0.3,
)

with open("test.txt", "a") as fout:
    fout.write(f"{datetime.now()} {text}\n")


def try_rename_file(fpath, new_name=None):
    if not new_name:
        new_name = fpath
    if os.path.isfile(fpath):
        print("Renaming")
        date = datetime.now().strftime("%Y-%m-%d")
        os.rename(fpath, f"{date}_{new_name}")
    else:
        print("No file to rename")


try_rename_file("test.txt")
