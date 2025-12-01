import torch
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE
from networks.roomlayout.RoomLayoutAutoRegressiveNet import RoomLayoutAutoRegressiveNet
from networks.roomlayout.quant import TokenSequentializer
from datasets.Threed_front_dataset import ThreeDFrontDataset
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses.recon_loss import ObjTokenReconstructionLoss
from config import parse_arguments
import os
import wandb
import matplotlib.pyplot as plt

class ARSceneTrainer:
    def __init__(
        self, device, t_scales,
        encoder_local: RoomLayoutVQVAE, ar_scene_model: RoomLayoutAutoRegressiveNet 
    ):
        self.sar, self.encoder_local, self.quantize_local = ar_scene_model, encoder_local,encoder_local.token_sequentializer
        self.quantize_local: TokenSequentializer


    def train_step(self, inp_obj_tokens, room_mask=None, text_condition=None):
        gt_code = self.encoder_local.
