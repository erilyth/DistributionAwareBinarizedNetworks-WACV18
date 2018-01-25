import torch
import torch.nn as nn
from torch.nn import init
import copy
import random
import math
from PIL import Image

from torchvision import transforms

class AverageMeter():
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, opt):
    """
    Accuracy given the predicted output and target
    """
    x = (output == target)
    x = sum(x)
    x = x#[0]
    return x*1.0 / len(output)

def precision(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Invert(object):
    """
    Transform class to invert images
    Used in TUBerlin loader
    """
    def __call__(self, img):
        img = 1.0 - img
        return img


class RandomRotate(object):
    """
    Transform class to randomly rotate images in a range
    """
    def __init__(self, rrange):
        self.rrange = rrange

    def __call__(self, img):
        size = img.size
        angle = random.randint(-self.rrange, self.rrange)
        img = img.rotate(angle, resample=Image.BICUBIC)
        img = img.resize(size, Image.ANTIALIAS)
        return img


class TenCrop(object):
    """
    Modify input and outputs to perform ten-crop
    """
    def __init__(self, size, opt):
        self.size = size
        self.opt = opt

    def __call__(self, img):
        centerCrop = transforms.CenterCrop(self.size)
        toPILImage = transforms.ToPILImage()
        toTensor = transforms.ToTensor()
        if self.opt.dataset == 'tuberlin':
            normalize = transforms.Normalize(mean=[0.06,], std=[0.93])
        if self.opt.dataset == 'sketchyrecognition':
            normalize = transforms.Normalize(mean=[0.0465,], std=[0.9])
        w, h = img.size(2), img.size(1)
        temp_output = []
        output = torch.FloatTensor(10, img.size(0), self.size, self.size)
        img = toPILImage(img)
        for img_cur in [img, img.transpose(Image.FLIP_LEFT_RIGHT)]:
            temp_output.append(centerCrop(img_cur))
            temp_output.append(img_cur.crop([0, 0, self.size, self.size]))
            temp_output.append(img_cur.crop([w-self.size, 0, w, self.size]))
            temp_output.append(img_cur.crop([0, h-self.size, self.size, h]))
            temp_output.append(img_cur.crop([w-self.size, h-self.size, w, h]))

        for img_idx in range(10):
            img_cur = temp_output[img_idx]
            img_cur = toTensor(img_cur)
            img_cur = normalize(img_cur)
            output[img_idx] = img_cur.view(img_cur.size(0), img_cur.size(1), img_cur.size(2))

        return output

def binarizeConvParams(convNode, bnnModel):
    """
    Binarize the parameters of a conv layer (Standard method of binarization)
    """
    s = convNode.weight.data.size()
    n = s[1]*s[2]*s[3]
    if bnnModel:
      convNode.weight.data[convNode.weight.data.eq(0)] = -1e-6
      convNode.weight.data = convNode.weight.data.sign()
    else:
      m = convNode.weight.data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n)
      convNode.weight.data[convNode.weight.data.eq(0)] = -1e-6
      convNode.weight.data = convNode.weight.data.sign().mul(m.repeat(1, s[1], s[2], s[3]))

def updateBinaryGradWeight(convNode, bnnModel):
    """
    Update the parameters of a conv layer (Standard method of update)
    """
    s = convNode.weight.data.size()
    n = s[1]*s[2]*s[3]
    m = convNode.weight.data.clone()
    if bnnModel:
        m = convNode.weight.data.clone().fill(1)
        m[convNode.weight.data.le(-1)] = 0
        m[convNode.weight.data.ge(1)] = 0
        m = torch.mul(1 - 1.0/s[1])
    else:
        m = convNode.weight.data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).repeat(1, s[1], s[2], s[3])
        m[convNode.weight.data.le(-1)] = 0
        m[convNode.weight.data.ge(1)] = 0
        m = torch.add(m, 1.0/n)
        m = torch.mul(m, 1.0 - 1.0/s[1])
        m = torch.mul(m, n)
    convNode.weight.grad.data.mul_(m)

def meancenterConvParams(convNode):
    """
    Mean center parameters of a conv layer
    """
    s = convNode.weight.data.size()
    negMean = torch.mul(convNode.weight.data.mean(1,keepdim=True),-1).repeat(1,1,1,1)
    #print(negMean.size())
    negMean = negMean.repeat(1, s[1], 1, 1)
    #print(negMean.size(),convNode.weight.data.size())
    convNode.weight.data.add_(negMean)

def optimalK(convNode):
    """
    Method to select optimal K for binarization threshold to binarize weights for DABN
    """
    x = convNode.weight.data.clone()

    nval = x.size(1) * x.size(2) * x.size(3)

    x = x.view(x.size()[0], -1)
    totx = x.sum(1,keepdim=True).repeat(1,x.size(1))

    # Compute kth value in descending order
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

    # Compute kth value in ascending order
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

    # Select the maximum of the two
    maxk = torch.max(maxk1, maxk2)
    selection_max = (maxk1 == maxk)

    selection_max = selection_max.float()

    final_k = torch.mul(thresh1, selection_max) + torch.mul(thresh2, (1 - selection_max))

    return final_k


def DualBinarizeConvParams(convNode, bnnModel):
    """
    Binarize the parameters of a conv layer (DABN)
    """
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

    pmW = pmW.repeat(1, pbinWeights.size(1), pbinWeights.size(2), pbinWeights.size(3)) #This is expanded for the hadamard

    pmW[convNode.weight.data.le(thresh)] = 0


    nmW = torch.div(nbinWeights.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True), nk) #L1 norm of W-/(N-K)
    nmW = nmW.repeat(1, nbinWeights.size(1), nbinWeights.size(2), nbinWeights.size(3)) #This is expanded for the hadamard
    nmW[convNode.weight.data.gt(thresh)] = 0

    nmW = torch.mul(nmW,-1)

    mW = torch.add(pmW,1,nmW)

    convNode.weight.data = mW


def updateDualGradWeight(convNode, bnnModel):
    """
    Update parameters of a conv layer (DABN)
    """
    s = convNode.weight.data.size()

    thresh = optimalK(convNode)

    thresh = thresh.view(-1, 1, 1, 1).repeat(1, s[1], s[2], s[3])

    pbinWeights = convNode.weight.data.clone()
    pbinWeights[convNode.weight.data.le(thresh)] = 0
    nbinWeights = convNode.weight.data.clone()
    nbinWeights[convNode.weight.data.gt(thresh)] = 0

    k = torch.sign(pbinWeights).abs().sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True)
    nk = torch.mul(torch.add(k,s[1]*s[2]*s[3]*-1),-1)

    ones = torch.ones(k.size()).cuda()

    kinv = torch.div(ones, k)
    kinv = kinv.repeat(1, pbinWeights.size(1), pbinWeights.size(2), pbinWeights.size(3))
    nkinv = torch.div(ones, nk)
    nkinv = nkinv.repeat(1, nbinWeights.size(1), nbinWeights.size(2), nbinWeights.size(3))

    pmW = torch.div(pbinWeights.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True), k)
    pmW = pmW.repeat(1, pbinWeights.size(1), pbinWeights.size(2), pbinWeights.size(3)) #This is expanded for the hadamard
    pmW[convNode.weight.data.le(-1)] = 0
    pmW[convNode.weight.data.ge(1)] = 0
    pmW = torch.add(pmW, kinv)
    pmW[convNode.weight.data.le(thresh)] = 0

    nmW = torch.div(nbinWeights.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True), nk)
    nmW = nmW.repeat(1, nbinWeights.size(1), nbinWeights.size(2), nbinWeights.size(3)) #This is expanded for the hadamard
    nmW[convNode.weight.data.le(-1)] = 0
    nmW[convNode.weight.data.ge(1)] = 0
    #nmW = torch.mul(nmW,-1)
    nmW = torch.add(nmW, nkinv)
    nmW[convNode.weight.data.gt(thresh)] = 0

    mW = torch.add(pmW,1,nmW)
    mW = torch.mul(mW, 1.0 - 1.0/s[1])
    convNode.weight.grad.data.mul_(mW)


def clampConvParams(convNode):
    """
    Clamp the parameters of a conv layer
    """
    convNode.weight.data.clamp_(-1, 1)

def adjust_learning_rate(opt, optimizer, epoch):
    """
    Learning rate scheduler
    """
    epoch = copy.deepcopy(epoch)
    lr = opt.maxlr
    wd = opt.weightDecay
    if opt.learningratescheduler == 'decayschedular':
        while epoch >= opt.decayinterval:
            lr = lr/opt.decaylevel
            epoch = epoch - opt.decayinterval
    lr = max(lr,opt.minlr)
    opt.lr = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = wd

def get_mean_and_std(dataloader):
    '''
    Compute the mean and std value of dataset
    '''
    mean = torch.zeros(3)
    std = torch.zeros(3)
    len_dataset = 0
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        len_dataset += 1
        for i in range(len(inputs[0])):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len_dataset)
    std.div_(len_dataset)
    return mean, std

def weights_init(model, opt):
    '''
    Perform weight initializations
    '''
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine == True:
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            c =  math.sqrt(2.0 / m.weight.data.size(1));
            if m.bias is not None:
                init.constant(m.bias, 0)
