from module.mydatasets import *
import torch
from transformers import Trainer, TrainingArguments, AutoFeatureExtractor, AutoModelForSequenceClassification, WavLMForSequenceClassification, HubertForSequenceClassification, Wav2Vec2ForSequenceClassification,WhisperForAudioClassification
import evaluate
from torch.utils.data import DataLoader

acc_metric = evaluate.load("./metrics/accuracy")
def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions[0].argmax(axis=-1)
    accuracy = acc_metric.compute(predictions=predictions, references=labels)
    return {
        "accuracy": accuracy["accuracy"],
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

    return  speech_feature["input_values"],label

if __name__ == "__main__":
    manifest_path = "./data/dialect"
    dataset_path = "/root/KeSpeech/"
    model_path = "./exp/wavlm-base-FT-Dialect"
    dataset = MyDataset(os.path.join(manifest_path,"test_balance.tsv"), dataset_path=dataset_path, label_path=os.path.join(manifest_path,"labels.txt"))
    data_loader = DataLoader(dataset=dataset, batch_size=8, collate_fn=collate_fn,shuffle=False)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    model = WavLMForSequenceClassification.from_pretrained(model_path, num_labels=len(dataset.labels_dict))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    
    lbs = []
    preds = []
    model.eval()
    for batch in data_loader:
        input_values,labels= batch
        input_values = input_values.to(device)
        labels = labels.to(device)
        output = model(input_values=input_values, labels=labels)

        # batch = {k: v.to(device) for k,v in batch.items()}
        # output = model(**batch)

        lbs.extend(labels.cpu())
        preds.extend(output.logits.cpu().argmax(axis=-1))
        torch.cuda.empty_cache()

    accuracy = acc_metric.compute(predictions=preds, references=lbs)

    print(f"=== accuracy: {accuracy} ===")


