import os
import numpy as np
import pandas as pd
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch

from torchvision import datasets as datasets
from PIL import Image

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

path_karpathy_coco_splits = '/home/brandon/Documents/datasets/karpathy_splits/dataset_coco.json'
path_coco_dataset= '/home/brandon/Documents/datasets/mscoco'
folder_karpathy_splits = '/home/brandon/Documents/datasets/mscoco/karpathy_splits'

path_keywords_data = '/home/brandon/Documents/git/phd/phd-image-captioning/keywords-predictor/data'
#path_keywords_file = '/home/brandon/Documents/git/phd/phd-image-captioning/keywords-predictor/cnn/vocabularies/vocabulary_350_pos_hist.csv'
#path_keywords_file = '/home/brandon/Documents/git/phd/phd-image-captioning/keywords-predictor/cnn/vocabularies/vocabulary_cleaned_lemma_pos_filtered_dist_1000.csv'
path_keywords_file = '/home/brandon/Documents/git/phd/phd-image-captioning/keywords-predictor/openimages/openimages_full_intersection_mscoco_dist.csv'

coco_splits = {}
data = []

def read_karpathy_splits(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data

def img_captions(img_id):
        img = data['images'][img_id]
        captions = img['sentences']
        captions_list = []
        for i in range(len(captions)):
            captions_list += [captions[i]['raw']]
            
        return captions_list

def show_image(img_id=None, path=None, captions=True):
    if path is not None:
        img_path = path
    else:
        img = data['images'][img_id]
        img_path = path_coco_dataset + '/' + img['filepath'] + '/' + img['filename']
        if captions:
            img_captions(img_id)
    
    print(img_id, img_path)
    img = mpimg.imread(img_path)
    channels = len(img.shape)

    plt.figure(figsize = (7,7))
    
    if channels == 2:    
        imgplot = plt.imshow(img, cmap='gray')
    else:
        imgplot = plt.imshow(img, cmap='gray')

    plt.axis('off')
    plt.show
    print(img.shape)


def load_splits(data):
    def update_dic(dic, key, value):
        if(key not in dic.keys()):
            dic[key] = []
        dic[key] += [value]
        
        return dic

    splits = {}
    filepaths_dic = {}

    for img in data['images']:
        update_dic(splits, img['split'], img)
        update_dic(filepaths_dic, img['filepath'], img['imgid'])
    
    fulltrain = splits['train'] + splits['restval']
    splits['fulltrain'] = fulltrain
    
    return splits, filepaths_dic

def load_keywords(path_file):
    keywords = list(pd.read_csv(path_file)['keyword'])
    return keywords


def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = []

    nltk_tagged = nltk.pos_tag(tokens)
    for t in nltk_tagged:
        word = t[0]
        tag = nltk_pos_tagger(t[1])
        if tag is not None:
            lemmatized_tokens += [lemmatizer.lemmatize(word, tag)]
        else:
            lemmatized_tokens += [word]
    
    return lemmatized_tokens

def get_ds_num_of_keywords_per_img(ds):
    keywords_sizes = []

    keywords = ds.keywords

    for idx, img in enumerate(ds.images):
        if idx % 1000 == 0:
            print(idx, len(ds.images))
        
        sentences = img['sentences']
        img_keywords = []
        target = np.zeros(len(keywords))

        for s in sentences:
            tokens = s['tokens']
            tokens_lemmatized = lemmatize_tokens(tokens)

            for token in tokens_lemmatized:
                if token in keywords and token not in img_keywords:
                    img_keywords += [token]
                    target[keywords.index(token)] = 1

        keywords_sizes += [int(np.sum(target))]

    return keywords_sizes

def convert_target_to_keywords(target, keywords):
    return sorted(list(np.array(keywords)[np.array((target==1).tolist())]))

def get_distribution_of_keywords(ds):
    dist = {}
    for idx, ds_obj in enumerate(ds):
        if idx % 1000 == 0:
            print(idx, len(ds.images))
            print(len(dist.keys()))
        
        target = ds_obj[1]
        keywords = convert_target_to_keywords(target, ds.keywords)
        for k in keywords:
            if k in dist.keys():
                dist[k] += 1
            else:
                dist[k] = 1

    return dist


class KarpathySplits(datasets.coco.CocoDetection):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = root
        data = read_karpathy_splits(path_karpathy_coco_splits)
        splits, self.filepaths_dic = load_splits(data)
        self.keywords = load_keywords(path_keywords_file)
        
        self.transform = transform
        self.target_transform = target_transform
        
        if split == 'train':
            self.images = splits['fulltrain']
        elif split == 'val':
            self.images = splits['val']
        elif split == 'test':
            self.images = splits['test']


    def __getitem__(self, index):
        img = self.images[index]
        keywords = self.keywords
        sentences = img['sentences']
        img_keywords = []
        
        target = torch.zeros(len(keywords), dtype=torch.long)

        for s in sentences:
            tokens = s['tokens']
            tokens_lemmatized = lemmatize_tokens(tokens)

            for token in tokens_lemmatized:
                if token in keywords and token not in img_keywords:
                    img_keywords += [token]
                    target[keywords.index(token)] = 1

        file_path = img['filepath']
        file_name = img['filename']
        full_file_path = os.path.join(self.root, file_path, file_name)

        raw_img = Image.open(full_file_path).convert('RGB')

        if self.transform is not None:
            raw_img = self.transform(raw_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return raw_img, target, full_file_path

    def __len__(self) -> int:
        return len(self.images)

        



    

        
