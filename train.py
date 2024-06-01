import os
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoFeatureExtractor, AutoModelForSequenceClassification
import evaluate
import argparse
from module.mydatasets import *
from module.dataConstant import SEX, Dialect, SPEAKER_NUM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="microsoft/wavlm-base")
    parser.add_argument("--manifest_path", type=str, default="./data")
    parser.add_argument("--dataset_path", type=str, default="/root/DialectDataset/Datatang-Dialect")
    parser.add_argument("--model_name", type=str, default="wavlm-base-dialect")
    return parser.parse_args()

# 评估指标
acc_metric = evaluate.load("./metrics/accuracy")
f1_metric = evaluate.load("./metrics/f1")
re_metric = evaluate.load("./metrics/recall")
def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    accuracy = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    recall = re_metric.compute(predictions=predictions, references=labels)
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "recall": recall["recall"]
    }

def collate_fn(batch):
    """
    batch -> speech_feature, input_lengths, label, sex, speaker
    Return:
        inputs : [batch, max_time, dim]
        input_lengths: [batch]
        label: [batch]
        sex: [batch]
        speaker: [batch]
    """
    speech_feature = [i[0].numpy() for i in batch]
    label = torch.LongTensor([i[2] for i in batch])

    speech_feature = feature_extractor(
        speech_feature, 
        sampling_rate=feature_extractor.sampling_rate,
        padding=True,
        return_tensors="pt"
    )["input_values"]

    return {
            "input_values": speech_feature,
            "labels": label,
        }

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    model.to(device)

    train_dataset = DialectDataset(
        manifest_path=os.path.join(manifest_path,"dialects_train.tsv"), dataset_path=dataset_path, speed_perturb=True)

    dev_dataset = DialectDataset(
        manifest_path=os.path.join(manifest_path,"dialects_test.tsv"), dataset_path=dataset_path)

    train_args = TrainingArguments(output_dir=output_dir,  # 输出文件夹
                                auto_find_batch_size="power2",  # 自动寻找batch_size
                                    gradient_accumulation_steps=2,  # 梯度累积
                                logging_steps=10,                # log 打印的频率
                                evaluation_strategy="epoch",     # 评估策略
                                num_train_epochs = 5,            # 训练epoch数
                                save_strategy="epoch",           # 保存策略
                                save_total_limit=1,              # 最大保存数
                                learning_rate=2e-5,              # 学习率
                                weight_decay=0.01,               # weight_decay
                                metric_for_best_model="accuracy",      # 设定评估指标
                                load_best_model_at_end=True
                                )     # 训练完成后加载最优模型
    
    # TODO add coustom optimizers for Trainer
    trainer = Trainer(model = model, # 训练模型
                    args = train_args, # 训练参数
                    train_dataset = train_dataset, # 训练集
                    eval_dataset = dev_dataset, # 测试集
                    data_collator=collate_fn, # 数据处理
                    compute_metrics = eval_metric) # 评估函数
    # 模型训练
    trainer.train()
    # 模型评估
    trainer.evaluate(eval_dataset=dev_dataset)
    # 模型测试
    # trainer.predict(tokenized_datasets["test"])
    print("All done!")

if __name__ == "__main__":
    args = get_args()
    model_path = args.model_path
    manifest_path = args.manifest_path
    dataset_path = args.dataset_path
    model_name = args.model_name
    output_dir = os.path.join("./exp", model_name)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(Dialect))

    main()