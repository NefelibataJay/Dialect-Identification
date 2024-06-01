# Using Transformers for speech classification

## Custom dataset

You need to prepare two `tsv` files, called `train.tsv` and `dev.tsv`.

### Modify MyDataset class

### (Optional) Modify the `collate_fn` function to support the pre-trained models you need

## Install

``` Shell
pip install -r requirements.txt
```

## Train

```shell
python train.py --model_path facebook/wav2vec2-base --dataset_path /path/your/dataset --manifest_path /path/your/manifest --model_name your_out_model_name --num_eopch 10
```

