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

# custom_palette = sns.color_palette("tab10", 10)
custom_palette = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 黄色
    # '#17becf'   # 青色
]

def plot_xy(logits, label, idx, state):
    """绘图"""
    x_values = tsne.fit_transform(logits)
    df = pd.DataFrame(x_values, columns=['x', 'y'])
    if state == "dialect":
        for(i, l) in enumerate(label):
            df.loc[i, 'label'] = label_dict[l]
    elif state == "speaker":
        for(i, l) in enumerate(label):
            df.loc[i, 'label'] = sex_dict[l]
    # df['label'] = label
    sns.scatterplot(x="x", y="y", hue="label", data=df, palette=custom_palette, s=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.title(f"T-SNE Layer {idx}")

    # 保存图片
    out_path = os.path.join(root_file, f"layer_{idx}", f"tsne_layer_{state}.png")
    plt.savefig(out_path,dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    root_file = f"./analysis_res/wavlm-base-FT-Dialect-GRL"
    sex_dict = {1:"Male", 2:"Female"}
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
        plot_xy(hidden_states, labels, idx, "dialect")

        plot_xy(hidden_states, speakers, idx, "speaker")

