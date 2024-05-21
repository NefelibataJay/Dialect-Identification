from dialect_dataset import DialectDataset
from torch.utils.data import DataLoader
from dataConstant import SEX, Dialect, SPEAKER_NUM
from torch.optim import AdamW
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
from transformers import get_scheduler
from tqdm import tqdm
import evaluate

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base")

model_path = "facebook/wav2vec2-base"
num_epochs = 50
eval_step = 3


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
    input_lengths = torch.LongTensor([i[1] for i in batch])
    label = torch.LongTensor([i[2] for i in batch])
    sex = torch.LongTensor([i[3] for i in batch])
    speaker = torch.LongTensor([i[4] for i in batch])

    # speech_feature = torch.nn.utils.rnn.pad_sequence(speech_feature, batch_first=True, padding_value=0)

    speech_feature = feature_extractor(
        speech_feature, 
        sampling_rate=feature_extractor.sampling_rate,
        padding=True,
        return_tensors="pt"
    )["input_values"]

    return speech_feature, input_lengths, label, sex, speaker


train_dataset = DialectDataset(
    manifest_path="/data_disk/code/TransformersForAduioClassification/data/dialects_train.tsv", dataset_path="/data_disk/datasets/Datatang-Dialect", speed_perturb=True)

dev_dataset = DialectDataset(
    manifest_path="/data_disk/code/TransformersForAduioClassification/data/dialects_test.tsv", dataset_path="/data_disk/datasets/Datatang-Dialect")

train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn)
eval_dataloader = DataLoader(dev_dataset, batch_size=8, collate_fn=collate_fn)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    model_path, num_labels=len(Dialect))

# output_hidden_states  返回所有层的输出
# output_attentions  返回所有注意力层的输出

optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

accuracy = evaluate.load("./metrics/accuracy")

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    tarin_bar = tqdm(range(len(train_dataloader)), desc=f"Training Epoch {epoch}")
    for batch in train_dataloader:
        input_values, input_lengths, labels, sex, speaker = batch
        input_values = input_values.to(device)
        labels = labels.to(device)
        outputs = model(input_values=input_values, labels=labels, output_hidden_states=True)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        tarin_bar.set_postfix(loss='{:.4f}'.format(loss))
        tarin_bar.update(1)
    
    if (epoch % 3 == 0):
        model.eval()
        eval_bar = tqdm(len(eval_dataloader), desc=f"Training Eval")
        for batch in eval_dataloader:
            input_values, input_lengths, labels, sex, speaker = batch
            input_values = input_values.to(device)
            labels = labels.to(device)
            with torch.no_grad:
                outputs = model(input_values=input_values, labels=labels, output_hidden_states=True)
            loss = outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            acc = accuracy.compute(references=labels, predictions=predictions)
            eval_bar.set_postfix(loss='{:.4f}'.format(loss), accuracy='{:.4f}'.format(acc))
            eval_bar.update(1)
            
        model.save_pretrained(f"./exp/wav2vec2/wav2vec2-base-{epoch}")
            

model.eval()
print("====== TEST ======")
for batch in eval_dataloader:
    input_values, input_lengths, labels, sex, speaker = batch
    input_values = input_values.to(device)
    labels = labels.to(device)
    with torch.no_grad:
        outputs = model(input_values=input_values, labels=labels, output_hidden_states=True)
    loss = outputs.loss
model.save_pretrained(f"./exp/wav2vec2/wav2vec2-base-final")