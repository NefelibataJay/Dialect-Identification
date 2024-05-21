from dialect_dataset import DialectDataset
from torch.utils.data import DataLoader
from dataConstant import SEX, Dialect, SPEAKER_NUM
from torch.optim import AdamW
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from transformers import get_scheduler
from tqdm import tqdm

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base", return_tensors="pt")


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
    speech_feature = [i[0] for i in batch]
    input_lengths = torch.LongTensor([i[1] for i in batch])
    label = torch.LongTensor([i[2] for i in batch])
    sex = torch.LongTensor([i[3] for i in batch])
    speaker = torch.LongTensor([i[4] for i in batch])

    speech_feature = torch.nn.utils.rnn.pad_sequence(
        speech_feature, batch_first=True, padding_value=0)

    speech_feature = feature_extractor(
        speech_feature, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )

    return speech_feature, input_lengths, label, sex, speaker


train_dataset = DialectDataset(
    stage="train", manifest_path="./data/dialect_train.tsv", dataset_path="E:/datasets/Datatang-Dialect", speed_perturb=True)

dev_dataset = DialectDataset(
    stage="dev", manifest_path="./data/dialects_test.tsv", dataset_path="E:/datasets/Datatang-Dialect")

train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn)
eval_dataloader = DataLoader(dev_dataset, batch_size=8, collate_fn=collate_fn)

model = Wav2Vec2ForSequenceClassification(
    "facebook/wav2vec2-base", num_labels=len(Dialect))

# output_hidden_states  返回所有层的输出
# output_attentions  返回所有注意力层的输出

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()

    for batch in train_dataloader:
        input_values, input_lengths, labels, sex, speaker = batch
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
