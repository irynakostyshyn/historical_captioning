import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import cv2
import torchvision
import argparse
import os
import numpy as np
from collections import defaultdict
import torch
from PIL import Image


class DataSet(object):
    def __init__(self, root, annotation, vocab, train_img,  transform):
        self.root = root
        self.annotation = get_captions(annotation)
        self.transform = transform
        self.vocab = vocab
        self.images_path = list(set(open(train_img, 'r').read().strip().split('\n')))
        self.captions = {}
        print(len(self.images_path))
        for img in self.images_path:
            self.captions[img] = self.annotation[img]

    def __getitem__(self, idx):

        caption = self.captions[self.images_path[idx]]
        img = self.images_path[idx]
        path = os.path.join(self.root, img)
        vocab = self.vocab
        
        try:
            image = cv2.imread(path)
            image = Image.fromarray(image, 'RGB')
        except Exception:
            print("PATH", path)
            raise

        if self.transform is not None:
            image = self.transform(image)
        tokens = nltk.tokenize.word_tokenize(' '.join(caption).lower())
        caption = []
        caption.append(vocab('<start>'))

        for token in tokens:
            
            caption.append(vocab(token))
        caption.append(vocab('<end>'))

        target = torch.Tensor(caption)

        return image, target

    def __len__(self):
        return len(self.images_path)


def collate_fn(data):

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths


def get_captions(annotations):

    caption_annotations = open(annotations, 'r').read().strip().split('\n')
    captions = defaultdict(list)
    for row in caption_annotations:
        title_text = row.split("^")
        title = title_text[0]

        text = '^'.join(title_text[1:])

        captions[title].append(text)
    return captions


def get_loader(root, annotation, vocab, train_img, transform, batch_size, shuffle, num_workers):

    data = DataSet(root=root, annotation=annotation, vocab=vocab, train_img=train_img, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader

