import json

if __name__ == '__main__':

    files = ["train", "dev", "test"]

    for file in files:
        path = f"./json_data/{file}.json"

        en_lines = []
        zh_lines = []

        en_path = "./corpus/" + file + ".en"
        zh_path = "./corpus/" + file + ".zh"

        with open(path, "r", encoding="utf-8") as f:
            lines = json.load(f)

            for line in lines:
                en_lines.append(line[0] + "\n")
                zh_lines.append(line[1] + "\n")

        with open(en_path, "w", encoding="utf-8") as fen:
            fen.writelines(en_lines)
        with open(zh_path, "w", encoding="utf-8") as fzh:
            fzh.writelines(zh_lines)