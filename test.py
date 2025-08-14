import torch.nn.functional as F
import torch
a = torch.tensor(
    [[1, 2, 3], [1, 2, 3], [1, 2, 6]],
    dtype=torch.float32
)
b = torch.tensor([2, 1, 0])


print(F.cross_entropy(a, b, reduction='none'))