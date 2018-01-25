import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import models.resnet as resnet
import PIL.ImageOps

def optimalK(convNode):

    x = convNode.clone()
    nval = x.size(1) * x.size(2) * x.size(3)

    x = x.view(x.size()[0], -1)
    totx = x.sum(1,keepdim=True).repeat(1,x.size(1))

    x1 = torch.sort(x, 1, descending=True)[0]
    csumx1 = torch.cumsum(x1, dim = 1)

    karr1 = torch.arange(1, nval + 1, 1)
    nkarr1 = torch.arange(nval, 0, -1)

    den1 = karr1.cuda()
    denval1 = nkarr1.cuda()

    numer1 = torch.mul(csumx1, csumx1)
    output1 = torch.div(numer1, den1)
    csumxval1 = totx.sub(1,csumx1)
    numerval1 = torch.mul(csumxval1, csumxval1)
    outputval1 = torch.div(numerval1, denval1)
    output1 = torch.add(output1,outputval1)

    maxval1, maxk1 = output1.max(1)
    maxk1 = maxk1.float()

    indx1 = torch.arange(0, x1.size(0))
    indx1 = indx1.long().cpu().numpy().tolist()
    kindx1 = maxk1.long().cpu().numpy().tolist()
    thresh1 = x1[indx1, kindx1]

    x2 = torch.sort(x, 1)[0]
    csumx2 = torch.cumsum(x2, dim = 1)

    karr2 = torch.arange(1, nval + 1, 1)
    nkarr2 = torch.arange(nval, 0, -1)

    den2 = karr2.cuda()
    denval2 = nkarr2.cuda()

    numer2 = torch.mul(csumx2, csumx2)
    output2 = torch.div(numer2, den2)

    csumxval2 = totx.sub(1,csumx2)
    numerval2 = torch.mul(csumxval2, csumxval2)
    outputval2 = torch.div(numerval2, denval2)
    output2 = torch.add(output2,outputval2)

    maxval2, maxk2 = output2.max(1)
    maxk2 = maxk2.float()

    indx2 = torch.arange(0, x2.size(0))
    indx2 = indx2.long().cpu().numpy().tolist()
    kindx2 = maxk2.long().cpu().numpy().tolist()
    thresh2 = x2[indx2, kindx2]

    maxk = torch.max(maxk1, maxk2)
    selection_max = (maxk1 == maxk)
    selection_max = selection_max.float()
    final_k = torch.mul(thresh1, selection_max) + torch.mul(thresh2, (1 - selection_max))

    return final_k

def binarizeConvParams(convNode):
    s = convNode.size()
    n = s[1]*s[2]*s[3]

    m = convNode.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n)
    convNode[convNode.eq(0)] = -1e-6
    convNode = convNode.sign().mul(m.repeat(1, s[1], s[2], s[3]))
    return convNode

def DualBinarizeConvParams(convNode):

    s = convNode.size()
    thresh = optimalK(convNode)

    thresh = thresh.view(-1, 1, 1, 1).repeat(1, s[1], s[2], s[3])

    pbinWeights = convNode.clone()
    pbinWeights[convNode.le(thresh)] = 0 # pbinWeights is W+ now
    nbinWeights = convNode.clone()
    nbinWeights[convNode.gt(thresh)] = 0 # nbinWeights is W- now

    k = torch.sign(pbinWeights).abs().sum(3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True)# this is the number of positive elements
    nk = torch.mul(torch.add(k,s[1]*s[2]*s[3]*-1),-1) # this is the number of negative elements

    pmW = torch.div(pbinWeights.norm(1, 3, keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True), k) #L1 norm of W+/K
    pmW = pmW.repeat(1, pbinWeights.size(1), pbinWeights.size(2), pbinWeights.size(3)) #This is expanded for the hadamard
    pmW[convNode.le(thresh)] = 0

    nmW = torch.div(nbinWeights.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True), nk) #L1 norm of W-/(N-K)
    nmW = nmW.repeat(1, nbinWeights.size(1), nbinWeights.size(2), nbinWeights.size(3)) #This is expanded for the hadamard
    nmW[convNode.gt(thresh)] = 0
    nmW = torch.mul(nmW,-1)

    mW = torch.add(pmW,1,nmW)
    return mW

resume = "/home/vishalapr/WACV17FinalModels/resnet18_sketchyrecognition_best.pth.tar"
model = torch.load(resume)
# model = torchvision.models.alexnet(pretrained=True)

wts = model['state_dict']['conv1.weight']

image1 = Image.open('1.jpg').convert('L')
image1 = PIL.ImageOps.invert(image1)
image1 = np.asarray(image1)

finalwts0 = wts.cpu().numpy()[1].squeeze()
finalwts1 = DualBinarizeConvParams(wts).squeeze().cpu().numpy()[1]
finalwts2 = binarizeConvParams(wts).squeeze().cpu().numpy()[1]

print(finalwts2)
plt.imshow(finalwts2, cmap='coolwarm')
plt.show()
ct = 1



#for wt in finalwts1:

#    out1 = ndimage.convolve(image1, wt, mode='constant', cval=0.0)

#    plt.imshow(out1)
#    plt.savefig('animalpic/xnor/' + str(ct) + '.jpg')

#    ct += 1
