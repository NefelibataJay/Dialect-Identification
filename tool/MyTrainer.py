# !! unfinished !!

import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoFeatureExtractor, AutoModelForSequenceClassification, WavLMForSequenceClassification, HubertForSequenceClassification, Wav2Vec2ForSequenceClassification,WhisperForAudioClassification
import evaluate
import argparse
from module.mydatasets import *

class Trainer:
    def __init__(self,
                 model,
                 optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler,
                 num_epoch: int,
                 valid_interval: int,
                 gradient_accumulation_steps: int,
                 grad_clip: float,
                 metric: function,
                 logger: SummaryWriter,
                 device) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger

        self.num_epoch = num_epoch
        self.valid_interval = valid_interval

        self.accum_grad = gradient_accumulation_steps
        self.grad_clip = grad_clip

        self.metric = metric

    
    def train(self, train_dataloader, valid_dataloader):
        self.model.to(self.device)
        print("========================= Start Training =========================")
        for epoch in range(1,self.num_epoch + 1):
            torch.cuda.empty_cache()
            self.train_loop(train_dataloader, epoch)
            if (epoch + 1) % self.valid_interval == 0 or epoch == self.num_epoch:
                self.validate_loop(valid_dataloader, epoch)

    def train_loop(self, train_dataloader, epoch):
        print(f"========================= {epoch} Start Training =========================")
        model.train()
        train_loss = 0
        self.optimizer.zero_grad()

        bar = tqdm(enumerate(train_dataloader), desc=f"Training Epoch {epoch}")
        for idx, batch in bar:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            result = self.model(inputs, labels=targets)
            loss = result.loss

            loss /= self.accum_grad
            loss.backward()

            if (idx + 1) % self.accum_grad == 0 or (idx + 1) == len(train_dataloader):
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                max_norm=self.grad_clip, norm_type=2)
                self.optimizer.step()
                self.optimizer.zero_grad()
            bar.set_postfix(loss='{:.4f}'.format(loss.item() * self.accum_grad))
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        self.logger.add_scalar("train_loss_epoch", train_loss, epoch)
        self.logger.add_scalar("train_lr", self.scheduler.get_last_lr()[0], epoch)
        train_loss += loss.item()

    @torch.no_grad()
    def validate_loop(self, valid_dataloader, epoch):
        self.model.eval()
        print(f"========================= {epoch} Eval=========================")
        valid_loss = 0
        valid_acc = 0
        for batch in valid_dataloader:
            inputs,targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            result = self.model(inputs, labels=targets)
            loss = result.loss

            logits = result.logits

            acc = self.metric(logits, targets)
            valid_acc += acc
            valid_loss += loss.item()

        valid_loss /= len(valid_dataloader)
        valid_acc /= len(valid_dataloader)

        self.logger.add_scalar("valid_acc", valid_acc, epoch)
        self.logger.add_scalar("valid_loss", valid_loss, epoch)

        print("valid_acc:", valid_acc)
        print("valid_loss:", valid_loss)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/wav2vec2-base", help="The path or name of the pre-trained model")
    parser.add_argument("--manifest_path", type=str, default="./data", help="The path of the manifest file")
    parser.add_argument("--dataset_path", type=str, default="/root/DialectDataset/Datatang-Dialect", help="The path of the dataset")
    parser.add_argument("--model_name", type=str, default="hubert-base-dialect", help="The name of your trained model")
    parser.add_argument("--num_eopch", type=int, default=5, help="The number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="The number of gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-5, help="The learning rate of the optimizer")
    parser.add_argument("--freeze_feature_encoder", action="store_true", help="Whether to freeze the feature encoder")
    parser.add_argument("--batch_size",type=int, default=12, help="The number of training batch size")
    return parser.parse_args()

acc_metric = evaluate.load("./metrics/accuracy")
def eval_metric(logits, targets):
    predictions, labels = logits.argmax(axis=-1), targets
    predictions = predictions[0].argmax(axis=-1)
    accuracy = acc_metric.compute(predictions=predictions, references=labels)
    return accuracy["accuracy"]

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

    # TODO add FBANK and MFCC feature

    speech_feature = feature_extractor(
        speech_feature, 
        sampling_rate=feature_extractor.sampling_rate,
        padding=True,
        return_tensors="pt"
    )["input_values"]

    return speech_feature, label

if __name__ == "__main__":
    args = get_args()
    model_path = args.model_path
    manifest_path = args.manifest_path
    dataset_path = args.dataset_path
    model_name = args.model_name
    output_dir = os.path.join("./exp", model_name)

    train_dataset = MyDataset(
        manifest_path=os.path.join(manifest_path,"train.tsv"), dataset_path=dataset_path, speed_perturb=True)

    dev_dataset = MyDataset(
        manifest_path=os.path.join(manifest_path,"dev.tsv"), dataset_path=dataset_path)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    if not os.path.exists(model_path):
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        if model_path.startswith("microsoft/wavlm"):
            model = WavLMForSequenceClassification.from_pretrained(model_path, num_labels=len(train_dataset.labels_dict))
        elif model_path.startswith("facebook/wav2vec2"):
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path, num_labels=len(train_dataset.labels_dict))
        elif model_path.startswith("facebook/hubert"):
            model = HubertForSequenceClassification.from_pretrained(model_path, num_labels=len(train_dataset.labels_dict))
        elif model_path.startswith("openai/whisper"):
            model = WhisperForAudioClassification.from_pretrained(model_path, num_labels=len(train_dataset.labels_dict))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(train_dataset.labels_dict))
    else:
        # !! please change the code below to match your model
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        # model = ***.from_pretrained(model_path)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path, num_labels=len(train_dataset.labels_dict))
        # raise ValueError("You may be using a local directory to load models, but these models have different initializers, so you'll need to change the initializer in your code to match the model you need.")
    
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    step_size = (len(train_dataloader) * args.num_eopch)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, last_epoch=args.num_eopch)


