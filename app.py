from bert_model import apply_model
import pandas as pd


def get_csv(file_path):
    df = pd.read_csv(
        file_path,
        header=None,
        names=["model", "type", "review"],
        encoding="utf-8",
        skiprows=1,
    )
    df["type"] = df["type"].str.replace("_x0000_", "")
    df = df.dropna()
    df["review"] = df["review"].apply(lambda x: x[:510] if len(x) > 512 else x)

    df = df.groupby(["model", "type"])

    return df


def output_to_txt(dic, the_type):
    with open(f"output/result_{the_type}.txt", "w", encoding="utf-8") as f:
        for k, v in dic.items():
            output_str = f"{k[0]},{k[1]},{v['positive']},{v['negative']},{v['total']}\n"
            f.write(output_str)


if __name__ == "__main__":
    # 对应高端中端低端
    types = ["high", "mid", "low"]

    for the_type in types:
        path = f"data/{the_type}_review.csv"
        dfs = get_csv(path)
        # 修改模型路径
        model_path = "model/bert_epoch4_lr2e-05_batch16_wd0.0001.pt"

        review_dic = {}

        for (
            name,
            df,
        ) in dfs:
            try:
                positive, negative = apply_model(model_path, df["review"])
            except:
                print(df["review"].shape)
                raise Exception("error")
            review_dic[name] = {
                "positive": positive,
                "negative": negative,
                "total": positive + negative,
            }

        output_to_txt(review_dic, the_type)

        print(review_dic)
