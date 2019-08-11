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
import torch
from PIL import Image


class Flickr8k(object):
    def __init__(self, root, annotation, vocab, train_img,  transform):
        self.root = root
        self.annotation = get_captions(annotation)
        self.transform = transform
        self.vocab = vocab
        # TODO: naming of variables
        self.imgs = list(set(open(train_img, 'r').read().strip().split('\n')))
        self.captions = {}
        print(len(self.imgs))
        for img in self.imgs:
            self.captions[img] = self.annotation[img]

    def __getitem__(self, idx):

        caption = self.captions[self.imgs[idx]]

        img = self.imgs[idx]
        # TODO: use `os.path.join` inplace here
        path = get_full_path_to_img(self.root, img)
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
        #caption.extend([vocab(token) for token in tokens])
        for token in tokens:
            
            caption.append(vocab(token))
        caption.append(vocab('<end>'))

        target = torch.Tensor(caption)

        return image, target

    def __len__(self):
        return len(self.imgs)


def get_full_path_to_img(root, img_title):
    return root + img_title


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
    # TODO: bad variable name
    captions_tmp = open(annotations, 'r').read().strip().split('\n')

    captions = {}
    for row in captions_tmp:
        title = row.split("^")[0]
        # TODO: could be bug here, join all after first index
        text = row.split("^")[1]
        # TODO: use collections.defaultdict(list)
        if not (title in captions):
            captions[title] = []

        captions[title].append(text)
    return captions


# TODO: fix naming of variable `ann`
def get_loader(root, ann, vocab, train_img, transform, batch_size, shuffle, num_workers):

    data = Flickr8k(root=root, annotation=ann, vocab=vocab, train_img=train_img, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader

