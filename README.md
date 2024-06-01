# Using Transformers for speech classification

## Custom dataset

You need to prepare two `tsv` files, called `train.tsv` and `dev.tsv`.

Preferably a file in the following format, Separated by "\t":

```txt
id audio_path label speaker sex text"
```

**If you don't have that many parameters you'll have to change the code in two places:**

#### (Optional) Modify MyDataset class

#### (Optional) Modify the `collate_fn` function to support the pre-trained models you need

## Install

``` Shell
pip install -r requirements.txt
```

## Train

```shell
python train.py --model_path facebook/wav2vec2-base --dataset_path /path/your/dataset --manifest_path /path/your/manifest --model_name your_out_model_name --num_eopch 10
```

