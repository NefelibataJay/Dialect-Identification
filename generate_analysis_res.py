from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import ast
from tqdm import tqdm

tsne = TSNE(n_components=2, random_state=42, learning_rate=200, init='pca',)

def plot_xy(logits, label, idx, state):
    """绘图"""
    x_values = tsne.fit_transform(logits)
    df = pd.DataFrame(x_values, columns=['x', 'y'])
    for(i, l) in enumerate(label):
        df.loc[i, 'label'] = label_dict[l]
    # df['label'] = label
    sns.scatterplot(x="x", y="y", hue="label", data=df, palette='deep', s=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',)
    plt.tight_layout()
    plt.title(f"T-SNE Layer {idx}")

    # 保存图片
    out_path = os.path.join(root_file, f"layer_{idx}", f"tsne_layer_{state}.png")
    plt.savefig(out_path,dpi=300)
    plt.close()

def save_tsne_res(logits, labels, idx, state):
    # 进行TSNE降维
    tsne_result = tsne.fit_transform(logits)

    # 绘制散点图
    plt.figure()
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap="viridis",s=3)
    plt.title(f"T-SNE Layer {idx}")

    # 保存图片
    out_path = os.path.join(root_file, f"layer_{idx}", f"tsne_layer_{state}.png")
    plt.savefig(out_path,dpi=300)
    plt.close()

if __name__ == "__main__":
    root_file = f"./analysis_res/hubert-base-FT-Dialect/"

    label_dict = {}

    with open(os.path.join(root_file, "label2dialect"), "r", encoding="utf-8") as f:
        for line in f.readlines():
            label, dialect = line.strip().split(" ")
            label_dict[int(label)] = dialect

    for idx in tqdm(range(13)):
        path = os.path.join(root_file, f"layer_{idx}")

        hidden_states = []
        labels = []
        speakers = []
        with open(os.path.join(path, "hidden_state.txt"), "r") as f:
            for line in f.readlines():
                label, speaker, hidden_state = line.strip().split("\t")
                hidden_states.append(np.array(ast.literal_eval(hidden_state)))
                labels.append(int(label))
                speakers.append(int(speaker))
            
        hidden_states = np.array(hidden_states)
        labels = np.array(labels)
        speakers = np.array(speakers)
        # save_tsne_res(hidden_states, labels, idx, "dialect")
        plot_xy(hidden_states, labels, idx, "dialect")
        # save_tsne_res(hidden_states, speakers, idx, "speaker")

