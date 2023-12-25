import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import transformers as tfs
import warnings
from transformers import logging
import matplotlib.pyplot as plt
import logging as log
import re

logging.set_verbosity_error()
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Config:
    def __init__(self) -> None:
        self.split = 0.8
        self.epoch = 5
        # [5e-5, 3e-5, 2e-5]
        self.lr = 2e-5
        self.batch_size = 8
        self.weight_decay = 2e-4


class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        model_class, tokenizer_class = (tfs.BertModel, tfs.BertTokenizer)

        self.tokenizer = tokenizer_class.from_pretrained("bert-base-chinese")
        self.bert = model_class.from_pretrained("bert-base-chinese")

        self.dense = nn.Linear(768, 2)  # bert默认的隐藏单元数是768， 输出单元是2，表示二分类

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(
            batch_sentences, add_special_tokens=True, max_len=66, pad_to_max_length=True
        )  # tokenize、add special token、pad

        input_ids = torch.tensor(batch_tokenized["input_ids"]).to(device)
        attention_mask = torch.tensor(batch_tokenized["attention_mask"]).to(device)

        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]  # 提取[CLS]对应的隐藏状态
        linear_output = self.dense(bert_cls_hidden_state)
        return linear_output


def get_train_test(file_path, config):
    df = pd.read_csv(
        file_path,
        usecols=[1, 2],
        header=None,
        names=["label", "comment"],
        encoding="gb18030",
        skiprows=1,
    )

    # 将第二列的字符串类型转换为整数类型
    df["label"] = df["label"].astype(int)
    df = df.dropna()
    df = df[df["comment"].apply(lambda x: len(str(x)) <= 512)]
    # 打乱df顺序
    df = df.sample(frac=1).reset_index(drop=True)

    df_len = len(df)

    split_len = int(df_len * config.split)

    _trian_X = df["comment"][:split_len]
    _train_y = df["label"][:split_len]
    _test_X = df["comment"][split_len:]
    _test_y = df["label"][split_len:]

    print(_trian_X.shape)
    print(_train_y.shape)
    print(_test_X.shape)
    print(_test_y.shape)

    return _trian_X, _train_y, _test_X, _test_y


def train_model(_train_X, _train_y, config):
    bert_classification_model = BertClassificationModel()
    optimizer = optim.AdamW(
        bert_classification_model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    batch_count = int(len(_train_y) / config.batch_size)

    bert_classification_model = bert_classification_model.to(device)

    for epoch in range(config.epoch):
        epoch_loss = 0
        cnt = 0
        loss_lis = []
        for i in tqdm(range(batch_count), desc=f"Epoch {epoch} Training"):
            cnt += 1
            batch_X = _train_X[i * config.batch_size : (i + 1) * config.batch_size]
            batch_y = _train_y[i * config.batch_size : (i + 1) * config.batch_size]
            tensor_batch_y = torch.tensor(batch_y.values).to(device)
            optimizer.zero_grad()
            output = bert_classification_model(batch_X).to(device)
            loss = criterion(output, tensor_batch_y)
            loss_lis.append(round(loss.item(), 5))
            loss.backward()
            optimizer.step()

            epoch_loss = loss.item()

            torch.cuda.empty_cache()

        else:
            print(f"\n Epoch: {epoch}, Loss: {epoch_loss}")
            # 生成横坐标（迭代次数，例如每个 epoch）
            iterations = list(range(1, len(loss_lis) + 1))

            # 绘制损失曲线图
            # plt.plot(iterations, loss_lis, label="Loss")
            # plt.title("Loss Curve")
            # plt.xlabel("Iterations (or Epochs)")
            # plt.ylabel("Loss Value")
            # plt.legend()
            # # 保存损失曲线图，替换 'loss_curve.png' 为你想要的文件名和格式
            # plt.savefig(
            #     "result/loss_curve_"
            #     + f"epoch{epoch}_lr{config.lr}_batch{config.batch_size}_wd{config.weight_decay}"
            #     + ".png"
            # )

            # # 保存模型
            # train_model_path = (
            #     "checkpoints/bert_"
            #     + f"epoch{epoch}_lr{config.lr}_batch{config.batch_size}_wd{config.weight_decay}"
            #     + ".pt"
            # )
            # torch.save(bert_classification_model.state_dict(), train_model_path)

            torch.cuda.empty_cache()

    return bert_classification_model


def test_model(test_model_path, _test_X, _test_y):
    # 加载模型
    bert_classification_model = BertClassificationModel()
    bert_classification_model = bert_classification_model.to(device)
    bert_classification_model.load_state_dict(torch.load(test_model_path))

    # 使用正则表达式从文件名中提取信息
    pattern = r"checkpoints/bert_epoch(\d+)_lr([\d.e-]+)_batch(\d+)_wd([\d.e-]+).pt"
    match = re.search(pattern, test_model_path)

    if match:
        # 提取匹配到的信息
        epoch = int(match.group(1))
        lr = float(match.group(2))
        batch_size = int(match.group(3))
        weight_decay = float(match.group(4))

        # 返回一个包含信息的字典
        test_config = {
            "epoch": epoch + 1,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
        }

    else:
        raise Exception("Filename format doesn't match the expected pattern.")

    # 测试模型
    bert_classification_model.eval()

    print(id(bert_classification_model))

    batch_count = int(len(_test_y) / test_config["batch_size"])
    hit = 0
    with torch.no_grad():
        for i in tqdm(range(batch_count), desc="Testing"):
            batch_X = _test_X[
                i * test_config["batch_size"] : (i + 1) * test_config["batch_size"]
            ]
            batch_y = _test_y[
                i * test_config["batch_size"] : (i + 1) * test_config["batch_size"]
            ]
            tensor_batch_y = torch.tensor(batch_y.values).to(device)
            output = bert_classification_model(batch_X)
            output = torch.argmax(output, dim=1)
            hit += torch.sum(output == tensor_batch_y)

        log.basicConfig(
            filename="result/test.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        log.info(f"Accuracy: {hit/len(test_y)} Config: {test_config}")
        print(f"Accuracy: {hit/len(test_y)} Config: {test_config}")


def apply_model(model_path, comment_df):
    # 加载模型
    bert_classification_model = BertClassificationModel()
    bert_classification_model = bert_classification_model.to(device)
    bert_classification_model.load_state_dict(torch.load(model_path))

    # 测试模型
    bert_classification_model.eval()

    # 使用正则表达式从文件名中提取信息
    pattern = r"checkpoints/bert_epoch(\d+)_lr([\d.e-]+)_batch(\d+)_wd([\d.e-]+).pt"
    match = re.search(pattern, model_path)

    if match:
        # 提取匹配到的信息
        epoch = int(match.group(1))
        lr = float(match.group(2))
        batch_size = int(match.group(3))
        weight_decay = float(match.group(4))

        # 返回一个包含信息的字典
        apply_config = {
            "epoch": epoch + 1,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
        }

    else:
        raise Exception("Filename format doesn't match the expected pattern.")

    batch_count = (
        int(comment_df.shape[0] / 16)
        if comment_df.shape[0] % 16 == 0
        else int(comment_df.shape[0] / 16) + 1
    )
    positive = 0
    with torch.no_grad():
        for i in tqdm(range(batch_count), desc="Testing"):
            # 选取16为一个batch
            batch_X = comment_df[i * 16 : (i + 1) * 16]
            output = bert_classification_model(batch_X)
            output = torch.argmax(output, dim=1)
            positive += torch.sum(output == 1).item()

    negative = comment_df.shape[0] - positive

    return positive, negative


if __name__ == "__main__":
    # 数据集路径
    path = "data/online_shopping_10_cats.csv"

    # 超参
    config = Config()

    # 获得训练集与测试集
    train_X, train_y, test_X, test_y = get_train_test(path, config)

    # 训练模型
    bert_classification_model = train_model(train_X, train_y, config)

    # 释放显存
    torch.cuda.empty_cache()

    # 测试模型
    # test_model("checkpoints/bert_epoch2_lr2e-05_batch8_wd0.0002.pt", test_X, test_y)
