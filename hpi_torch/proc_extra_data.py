#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/1/4 16:54
# @Author  : Eric Ching
import pandas as pd
import os
import numpy as np
from config import config as cfg
from tqdm import tqdm
import glob
from itertools import chain
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns

extra_data_dir = 'V:/data_repos/hpi/extra'
extra_data_csv = os.path.join(extra_data_dir, 'subcellular_location.tsv')
subcellular_location = pd.read_csv(extra_data_csv, sep="\t", index_col=None)
all_label_names = cfg.label_names.copy()
all_label_names.update({
    # new classes
    28: "Vesicles",
    29: "Nucleus",
    30: "Midbody",
    31: "Cell Junctions",
    32: "Midbody ring",
    33: "Cleavage furrow"
})

print(all_label_names)
all_names = []
for j in tqdm(range(len(subcellular_location))):
    names = np.array(subcellular_location[['Enhanced', 'Supported', 'Approved', 'Uncertain']].values[j])
    names = [name for name in names if str(name) != 'nan']
    split_names = []
    for i in range(len(names)):
        split_names = split_names + (names[i].split(';'))
    all_names.append(split_names)
subcellular_location['names'] = all_names
extra_image_dir = os.path.join(extra_data_dir, 'external_data')
imgs_path_list = glob.glob(os.path.join(extra_image_dir, '*'))
# only old names
data = []
for i in tqdm(range(len(subcellular_location))):
    im_name = subcellular_location['Gene'].values[i] + '-' + subcellular_location['Gene name'].values[i]
    for im in glob.glob(os.path.join(extra_image_dir, im_name + '*')):
        labels = []
        for name in subcellular_location['names'].values[i]:
            try:
                if name == 'Rods & Rings': name = "Rods & rings"
                labels.append(list(cfg.label_names.values()).index(name))
            except:
                pass
        if len(labels) > 0:

            str_labels = ''
            for item in labels:
                str_labels += str(item) + ' '

            data.append([os.path.split(im)[-1].split('.png')[0], str_labels.strip()])

df = pd.DataFrame(data, columns=['Id', 'Target'])
df.to_csv(os.path.join(extra_data_dir, 'extra_data.csv'), index=False)

count_labels = Counter(list(chain.from_iterable(df['Target'].values)))
plt.figure(figsize=(16, 10))
sns.barplot(list(cfg.label_names), [count_labels[k] for k in list(cfg.label_names)], )
plt.xticks(list(cfg.label_names), list(cfg.label_names.values()), rotation=90, size=15)
for i in count_labels:
    plt.text(i - 0.4, count_labels[i], count_labels[i], size=12)
plt.show()