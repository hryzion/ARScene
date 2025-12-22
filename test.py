import torch.nn.functional as F
import torch
from torchvision import models
import torch.nn as nn
feat_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feat_extractor.fc = nn.Identity()  # 去
a = torch.tensor(
    [[1, 2, 3], [1, 2, 3], [1, 2, 6]],
    dtype=torch.float32
)
b = torch.tensor([2, 1, 0])


print(F.cross_entropy(a, b, reduction='none'))