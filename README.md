# Sentiment Analysis with BERT


## Author 🛰️

- [Yidan Sun of Nankai University](https://github.com/Thoams0211)

You can contact me by sending email to syd20021134@gmail.com or 2111438@mail.nankai.edu.cn

## Introduction 🌈
Welcome to Sentiment Analysis of Mobile User Reviews!

Sentiment Analysis of Mobile User Reviews is a comprehensive open-source project that focuses on analyzing and understanding the sentiment expressed in mobile user reviews. Leveraging advanced natural language processing techniques(BERT), this project aims to provide insights into how users perceive and evaluate different aspects of mobile devices. What's more, producers can also use this project to understand the strengths and weaknesses of their products and make improvements accordingly.


## Dataset 🌏

We use the training dataset with the test dataset as `data/online_shopping_10_cats.csv`. The model can be applied to datasets `data/high_review.csv`, `data/mid_review.csv`, and `data/low_review.csv` to perceive the emotions embedded in the reviews of different users for different cell phone model ratings.

## Dependencies 🎇
We recommend using `Nvidia RTX4090` or `Nvidia RTX3080` to run the code. We deploy our code on `PyTorch  2.0.0`, `Python  3.8(Ubuntu 20.4)`, and `Cuda  11.8`. 

## Getting Started 🚀
Before running the code, you need to install the required packages in `requirements.txt` by running the following command:
```bash
pip install -r requirements.txt
```

After installing the required packages, you should download the pre-trained BERT model from [Hugging Face](https://huggingface.co/bert-base-chinese) and put it in `bert-base-chinese/`. Then you should download my trained model from [BaiduPan](https://pan.baidu.com/s/1HrH4X6M7JNCzvm4OaVimAw?pwd=78ud) and put it in folder `checkpoints/`.

You can run the code by running the following command:
```bash
python app.py
```
Our model has an accuracy of 0.98319 on the test set. The model will predict the sentiment of the input text and output the result to the files in folder `output/`. There are some result in this folder. You can delete them if you want to get the result of your own input text.
## Model 🌟
The model you used was already trained. If you want to retrain your own model, you can run the following command:
```bash
python bert_model.py
```
Your model will be saved in `model/checkpoints`.If you want to
verify the model, you should comment and uncomment two lines of code in line 273 and line 279 of `bert_model.py`
```python
...
# bert_classification_model = train_model(train_X, train_y, config)
...
test_model("checkpoints/bert_epoch2_lr2e-05_batch8_wd0.0001.pt", test_X, test_y)
...
```
Log information will be saved in `result/test.log`.

