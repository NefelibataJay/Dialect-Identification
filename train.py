import os
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoFeatureExtractor, AutoModelForSequenceClassification, WavLMForSequenceClassification, HubertForSequenceClassification, Wav2Vec2ForSequenceClassification
from transformers import Wav2Vec2Config, HubertConfig, WavLMConfig, AutoConfig
import evaluate
import argparse
from model.grl_classification import GRLClassification
from module.mydatasets import *

def get_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model_path", type=str, default="./exp/wav2vec2-base", help="The path or name of the pre-trained model")
    parser.add_argument("--manifest_path", type=str, default="./data/dialect", help="The path of the manifest file")
    parser.add_argument("--dataset_path", type=str, default="/root/KeSpeech/", help="The path of the dataset")
    parser.add_argument("--model_name", type=str, default="wavlm-large-FT-Dialect", help="The name of your trained model")
    parser.add_argument("--num_eopch", type=int, default=10, help="The number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="The number of gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="The learning rate of the optimizer")
    parser.add_argument("--freeze_feature_encoder", action="store_true", help="Whether to freeze the feature encoder")
    parser.add_argument("--grl", action="store_true", help="Whether to freeze the feature encoder")
    return parser.parse_args()

acc_metric = evaluate.load("./metrics/accuracy")
def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
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

    return {
            "input_values": speech_feature["input_values"],
            "labels": label,
            # "speaker_labels": sex,
        }

def main(args):
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model.to(device)
    train_args = TrainingArguments(output_dir=output_dir, 
                                auto_find_batch_size=True,
                                # per_device_train_batch_size=16,
                                # per_device_eval_batch_size=8,
                                logging_steps=50,
                                evaluation_strategy="epoch",
                                save_strategy="epoch",
                                num_train_epochs = args.num_eopch,
                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                learning_rate=args.lr,
                                warmup_ratio=0.1,
                                metric_for_best_model="accuracy",
                                eval_accumulation_steps=10,
                                gradient_checkpointing=True,
                                load_best_model_at_end=True
                                )
    
    # TODO add coustom optimizers for Trainer
    trainer = Trainer(model = model, 
                    args = train_args,
                    train_dataset = train_dataset,
                    eval_dataset = dev_dataset,
                    data_collator= collate_fn,
                    compute_metrics = eval_metric)
    print("Start training...")
    trainer.train()
    # trainer.evaluate()
    trainer.save_model()
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    # test(trainer)
    print("All done!")

if __name__ == "__main__":
    args = get_args()
    model_path = args.model_path
    manifest_path = args.manifest_path
    dataset_path = args.dataset_path
    model_name = args.model_name
    output_dir = os.path.join("./exp", model_name)

    train_dataset = MyDataset(manifest_path=os.path.join(manifest_path,"train.tsv"), label_path=os.path.join(manifest_path,"labels.txt"), dataset_path=dataset_path, speed_perturb=True)
    dev_dataset = MyDataset(manifest_path=os.path.join(manifest_path,"dev.tsv"), label_path=os.path.join(manifest_path,"labels.txt"), dataset_path=dataset_path)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)


    # !! please change the code below to match your model
    # model = ***.from_pretrained(config)
    # config.num_labels = len(train_dataset.labels_dict)
    # config.lamda = 0.1
    # config.num_speaker = len(train_dataset.sex_dict)
    model = GRLClassification.from_pretrained(model_path,num_labels=len(train_dataset.labels_dict))
    # model = WavLMForSequenceClassification.from_pretrained(model_path,num_labels=len(train_dataset.labels_dict))
    
    if args.freeze_feature_encoder:
        print("==========freeze_feature_encoder===========")
        model.freeze_feature_encoder()

    # model.labels_dict = train_dataset.labels_dict
    # model.speaker_dict = train_dataset.speaker_dict

    main(args)
 