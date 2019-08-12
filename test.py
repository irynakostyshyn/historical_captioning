# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import pickle
import os
from torchvision import transforms
from build_vocab import Vocabulary
from model.model import EncoderCNN, DecoderRNN
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    print(image_path)
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def test(image_path, state_path, vocab_path, embed_size, hidden_size, num_layers, img_size):

    transform = transforms.Compose([

        transforms.Resize((img_size, img_size)),

        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(embed_size).eval()
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    state = torch.load(state_path, map_location=device)
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])

    # Prepare an image
    image = load_image(image_path, transform)
    image_tensor = image.to(device)

    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)

    sampled_ids = sampled_ids.cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)


    return sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--img_size', type=int, default=298, help='resizing size')
    parser.add_argument('--state_path', type=str,
                        default='./model/historical/encoder-epoch-20-loss-0.4047403931617737.ckpt',
                        help='path for trained encoder and decoder')

    parser.add_argument('--vocab_path', type=str, default='./vocab_historical2.pkl', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    args = parser.parse_args()
    result = test(image_path=args.image,
         state_path=args.state_path,
         vocab_path=args.vocab_path,
         embed_size=args.embed_size,
         hidden_size=args.hidden_size,
         num_layers=args.num_layers,
         img_size=args.img_size)
    print(result)