import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer


class CFG:
    debug = False
    # image_path = image_path
    # captions_train_path = captions_t_path
    # captions_val_path = "/kaggle/working/content/UIT-ViLC/val"
    batch_size = 90
    num_workers = 2
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'efficientnet_b0.ra4_e3600_r224_in1k'
    image_embedding = 1280
    # text_encoder_model = "distilbert-base-uncased"
    text_encoder_model = "vinai/phobert-base-v2"
    text_embedding = 768
    # text_tokenizer = "distilbert-base-uncased"
    text_tokenizer = "vinai/phobert-base-v2"
    max_length = 500

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1

