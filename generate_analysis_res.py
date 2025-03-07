import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import ast
from tqdm import tqdm
import seaborn as sns

from matplotlib import font_manager

# 指定字体文件路径
font_path = './SimHei.ttf'  # 替换为字体文件的实际路径
font_prop = font_manager.FontProperties(fname=font_path)

tsne = TSNE(n_components=2, random_state=42, learning_rate=200, init='pca',)

sex_dict = {1:"男", 2:"女"}
color_map = {'男': 'blue', '女': 'red'}

label_dict = {1:"普通话",
                2:"西南",
                3:"江淮",
                4:"东北",
                5:"胶辽",
                6:"北京",
                7:"中原",
                8:"冀鲁",
                9:"兰银"}

def draw_xy(file_path, hue,output_path):
    data = pd.read_csv(file_path, sep="\t")  # 确保文件路径和编码格式正确

    sns.scatterplot(x="x", y="y", hue=hue, data=data, palette=color_map, s=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=font_prop)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.title(f"T-SNE 可视化第{idx}层", fontproperties=font_prop)
    # plt.xlabel('X', fontproperties=font_prop)
    # plt.ylabel('Y', fontproperties=font_prop)

    img_path = os.path.join(output_path, f"tsne_layer_{hue}.png")
    plt.savefig(img_path,dpi=300, bbox_inches='tight')
    plt.close()

def save_data(path):
    hidden_states = []
    labels = []
    speakers = []
    genders = []

    with open(os.path.join(path, "hidden_state.txt"), "r") as f:
        for line in f.readlines():
            label, speaker, gender, hidden_state = line.strip().split("\t")
            hidden_states.append(np.array(ast.literal_eval(hidden_state)))
            labels.append(int(label))
            speakers.append(int(speaker))
            genders.append(int(gender))

    hidden_states = np.array(hidden_states)
    labels = np.array(labels)
    random.shuffle(speakers)
    speakers = np.array(speakers)
    genders = np.array(genders)

    x_values = tsne.fit_transform(hidden_states)
    df = pd.DataFrame(x_values, columns=['x', 'y'])

    if labels is not None:
        for(i, l) in enumerate(labels):
            df.loc[i, 'dialect'] = label_dict[l]
    if speakers is not None:
        for(i, l) in enumerate(speakers):
            df.loc[i, 'speaker'] = f"speaker-{int(l)}"
    if genders is not None:
        for(i, l) in enumerate(genders):
            df.loc[i, 'gender'] =  sex_dict[l]
    
    csv_path = os.path.join(path, f"tsne_layer.csv")
    df.to_csv(csv_path, index=False, sep="\t")  # 不保存索引列
    

if __name__ == "__main__":
    root_file = f"./analysis_res"
    path_list = ["wav2vec2-base"]
    # for sub_file in os.listdir(root_file):
    for sub_file in path_list:
        for idx in tqdm(range(13)):
            path = os.path.join(root_file,sub_file, f"layer_{idx}")
            # save_data(path)

            # draw_xy(os.path.join(path, "tsne_layer.csv"), "dialect", path)
            # draw_xy(os.path.join(path, "tsne_layer.csv"), "speaker", path)
            draw_xy(os.path.join(path, "tsne_layer.csv"), "gender", path)


