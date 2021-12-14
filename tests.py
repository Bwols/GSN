import torch
import torch.nn.functional as F
from torch import  nn
a = torch.tensor([3,1,3])


k = torch.nn.functional.one_hot(a,10)
print(k)
#print(k)
z = input = torch.randn(3, 5)

#loss = F.nll_loss(k, k)
loss3 = F.nll_loss()
#loss = F.cross_entropy(z,z)
loss2 = nn.BCEWithLogitsLoss()
target = torch.randint(0, 10, (10,))
print(target)
one_hot = torch.nn.functional.one_hot(target)
loss2(one_hot.float(), one_hot.float())
print(one_hot)