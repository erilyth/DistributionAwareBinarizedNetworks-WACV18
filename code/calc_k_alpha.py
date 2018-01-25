import torch
import torch.nn as nn
from utils import optimalK
from models.alexnetwbin import Net

def give_vals(convNode):

    s = convNode.weight.data.size()
    thresh = optimalK(convNode)

    thresh = thresh.view(-1, 1, 1, 1).repeat(1, s[1], s[2], s[3])

    pbinWeights = convNode.weight.data.clone()
    pbinWeights[convNode.weight.data.le(thresh)] = 0 # pbinWeights is W+ now
    nbinWeights = convNode.weight.data.clone()
    nbinWeights[convNode.weight.data.gt(thresh)] = 0 # nbinWeights is W- now

    k = torch.sign(pbinWeights).abs().sum(3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True)# this is the number of positive elements
    nk = torch.mul(torch.add(k,s[1]*s[2]*s[3]*-1),-1) # this is the number of negative elements

    pmW = torch.div(pbinWeights.norm(1, 3, keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True), k) #L1 norm of W+/K

    nmW = torch.div(nbinWeights.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True), nk) #L1 norm of W-/(N-K)
    nmW = torch.mul(nmW,-1)

    # mW = torch.add(pmW,1,nmW)
    return pmW, nmW, k, nk

model = Net(1000).cuda()
# model = torch.load(opt.pretrained_file)

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        output = give_vals(m)
        print(m)
        print('alpha is ', output[0])
        print('beta is', output[1])
        print('k is', output[2])
        print('nk is', output[3])
        print("---------")
