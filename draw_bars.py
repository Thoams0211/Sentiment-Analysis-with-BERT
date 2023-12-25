import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置汉字格式
plt.rcParams["font.sans-serif"] = ["STSong"]
# 设置全局字号
plt.rcParams["font.size"] = 10


def parse_txt(path):
    res = {}
    remove_lis = []
    max_review = 850
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")
            phone_type = line[0]
            ev_type = line[1]
            positive = int(line[2])
            negative = int(line[3])
            tot = int(line[4])

            if ev_type not in res:
                res[ev_type] = {"positive": positive, "negative": negative}

            else:
                res[ev_type]["positive"] += positive
                res[ev_type]["negative"] += negative

        for key, value in res.items():
            # 选好评少的
            if (
                res[key]["positive"] + res[key]["negative"] < max_review
                or res[key]["positive"] >= res[key]["negative"]
            ):
                remove_lis.append(key)

        for key in remove_lis:
            del res[key]

    return res


def parse_pie_txt(path):
    res = {"positive": 0, "negative": 0}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")
            phone_type = line[0]
            ev_type = line[1]
            positive = int(line[2])
            negative = int(line[3])
            tot = int(line[4])

            res["positive"] += positive
            res["negative"] += negative

    return res


def plot_reviews(data):
    # 提取评价标准和对应的数据

    criteria = [key for key in data.keys()]

    positive_reviews = [value["positive"] for value in data.values()]
    negative_reviews = [value["negative"] for value in data.values()]

    # 设置柱状图的宽度
    bar_width = 0.3

    # 设置每个柱状图的位置
    r1 = range(len(criteria))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # 绘制柱状图
    plt.bar(
        r1,
        positive_reviews,
        color="green",
        width=bar_width,
        edgecolor="grey",
        label="好评",
    )
    plt.bar(
        r3, negative_reviews, color="red", width=bar_width, edgecolor="grey", label="差评"
    )

    # 添加标签和标题
    plt.xlabel("评价标准")
    plt.ylabel("评价数量")
    plt.title("手机评价统计")
    plt.xticks([r + bar_width for r in range(len(criteria))], criteria)

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()


def plot_pie_chart(data):
    labels = data.keys()
    values = data.values()

    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=["green", "red"],
    )
    plt.axis("equal")  # 使饼状图保持圆形
    plt.title("Sentiment Analysis")

    plt.show()


if __name__ == "__main__":
    types = ["high"]
    for the_type in types:
        path = f"output/result_{the_type}.txt"
        bar_data = parse_txt(path)
        pie_data = parse_pie_txt(path)
        print(bar_data)
        # plot_reviews(bar_data)
        plot_pie_chart(pie_data)
        break
