from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
from tqdm import tqdm

def save_tsne_res(logits, labels, idx, state):
    # 进行TSNE降维
    tsne = TSNE(n_components=2, random_state=42, learning_rate=200)
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
    root_file = f"./analysis_res/wav2vec2-base-FT-Dialect/"

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
        save_tsne_res(hidden_states, labels, idx, "dialect")
        save_tsne_res(hidden_states, speakers, idx, "speaker")

