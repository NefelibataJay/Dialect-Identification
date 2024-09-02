import torch
from transformers import AutoFeatureExtractor,AutoConfig, Wav2Vec2ForSequenceClassification, HubertForSequenceClassification,WavLMForSequenceClassification
from transformers import Wav2Vec2Config, HubertConfig, WavLMConfig, AutoConfig
from torch.utils.data import DataLoader
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model.grl_classification import GRLClassification
from module.mydatasets import MyDataset

root_file = f"./analysis_res/wavlm-base-FT-Dialect-GRL"
model_path = "exp/wavlm-base-FT-Dialect-GRL"
dataset_path = "/root/KeSpeech/"
manifest_path = "./data/dialect"
dataset = MyDataset(manifest_path=os.path.join(manifest_path,"test_balance.tsv"), dataset_path=dataset_path, label_path=os.path.join(manifest_path,"labels.txt"))

if not os.path.exists(root_file):
    os.makedirs(root_file)

with open(os.path.join(root_file, "label2dialect"), "w", encoding="utf-8") as f:
    for key,value in dataset.labels_dict.items():
        f.write(f"{value} {key}\n")


feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
model_path = os.path.join(os.getcwd(), model_path)

config = AutoConfig.from_pretrained(model_path)
model = GRLClassification.from_pretrained(model_path,config)

model.eval()

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
    label = torch.LongTensor([i[1] for i in batch])
    speaker = torch.LongTensor([i[2] for i in batch])
    sex = torch.LongTensor([i[3] for i in batch])

    # TODO add FBANK and MFCC feature

    speech_feature = feature_extractor(
        speech_feature, 
        sampling_rate=feature_extractor.sampling_rate,
        padding=True,
        return_tensors="pt"
    )

    return {
            "input_values": speech_feature["input_values"],
            "labels": label,
            "speaker_labels": sex,
        }

dataloaders = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# 记录模型权重
with open(os.path.join(root_file, "layer_weights"), "w", encoding="utf-8") as f:
    f.write(f"{torch.nn.functional.softmax(model.layer_weights, dim=-1).detach()}\n")

for batch in tqdm(dataloaders):
    outputs = model(input_values=batch["input_values"], labels=batch["labels"])

    for idx in range(len(model.layer_weights)):
        hidden_state = outputs.hidden_states[idx].detach() # [batch, time, dim]
        layer_weight = model.layer_weights[idx].detach()

        hidden_state = hidden_state * torch.nn.functional.softmax(layer_weight, dim=-1)
        hidden_state = hidden_state.mean(dim=1)

        # 写入到对应的文件中
        path = os.path.join(root_file, f"layer_{12-idx}")
        if not os.path.exists(path):
            os.makedirs(path)
        
        with open(os.path.join(path, "hidden_state.txt"), "a") as f:
            f.write(f"{batch['labels'][0].detach()}\t{batch['speaker_labels'][0].detach()}\t{str(hidden_state.squeeze(0).detach().tolist())}\n")

        


        


