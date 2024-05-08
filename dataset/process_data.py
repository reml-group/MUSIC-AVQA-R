import yaml
import json
import os
import ast

def fill_blank(paths: list):
    for path in paths:
        data = []
        with open(path, "r") as f:
            samples = json.load(f)
            for s in samples:
                # s["type"] = ast.literal_eval(s["type"])
                if s["templ_values"] == "[]":
                    del s["templ_values"]
                    data.append(s)
                else:
                    que = s["question_content"].split(" ")
                    tmp = ast.literal_eval(s["templ_values"])
                    i = 0
                    for j, word in enumerate(que):
                        if word.startswith("<"):
                            que[j] = tmp[i]
                            i += 1
                    s["question_content"] = " ".join(que)
                    del s["templ_values"]
                    data.append(s)

        with open(os.path.join("./", path.split("/")[-1]), "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    with open("../option.yaml", "r") as f:
        config = yaml.load(stream=f, Loader=yaml.FullLoader)

    # all = config["path"]["test_data"]["extend"]
    # head = config["path"]["test_data"]["extend-head"]
    # tail = config["path"]["test_data"]["extend-tail"]
    # fill_blank([all, head, tail])

    # train = config["path"]["train_data"]
    # val = config["path"]["val_data"]
    test = config["path"]["test_data"]["original"]
    fill_blank([test])