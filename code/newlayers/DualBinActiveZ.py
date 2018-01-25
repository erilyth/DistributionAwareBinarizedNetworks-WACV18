import torch

class DualBinActiveZ(torch.autograd.Function):
    def __init__(self):
        super(DualBinActiveZ, self).__init__()

    def clampInput(self, input):
        return torch.clamp(input, -1, 1)

    def meancenterInput(self, input):
        s = input.size()
        negMean = torch.mul(input.mean(1,keepdim=True),-1).repeat(1,1,1,1)
        negMean = negMean.repeat(1, s[1], 1, 1)
        return torch.add(input, 1, negMean)

    def forward(self, input):
        self.save_for_backward(input)
        s = input.size()
        input_current = self.meancenterInput(input.clone())
        #input_current = self.clampInput(input_current)

        pbin_input = input_current.clone()
        pbin_input[pbin_input.le(0)] = 0
        nbin_input = input_current.clone()
        nbin_input[nbin_input.gt(0)] = 0

        k = torch.sign(pbin_input).abs().sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True)
        nk = torch.mul(torch.add(k,s[1]*s[2]*s[3]*-1),-1)

        pmW = torch.div(pbin_input.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True), k)
        pmW = pmW.repeat(1, pbin_input.size(1), pbin_input.size(2), pbin_input.size(3))
        pmW[input_current.le(0)] = 0

        nmW = torch.div(nbin_input.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True), nk)
        nmW = nmW.repeat(1, nbin_input.size(1), nbin_input.size(2), nbin_input.size(3))
        nmW[input_current.gt(0)] = 0

        nmW = torch.mul(nmW, -1)

        #combine
        mW = torch.add(pmW, 1, nmW)

        #print('Input:')
        #print(input[0][0])
        #print('mW')
        #print(mW[0][0])
        return mW

    def backward(self, gradOutput):
        input = self.saved_tensors[0]
        s = input.size()
        input_current = self.meancenterInput(input.clone())
        #input_current = self.clampInput(input_current)

        s = gradOutput.size()

        #positive part
        pbin_input = input_current.clone()
        pbin_input[pbin_input.le(0)] = 0
        nbin_input = input_current.clone()
        nbin_input[nbin_input.gt(0)] = 0

        k = torch.sign(pbin_input).abs().sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True)
        nk = torch.mul(torch.add(k,s[1]*s[2]*s[3]*-1),-1)

        ones = torch.ones(k.size()).cuda()

        kinv = torch.div(ones, k)
        kinv = kinv.repeat(1, pbin_input.size(1), pbin_input.size(2), pbin_input.size(3))
        nkinv = torch.div(ones, nk)
        nkinv = nkinv.repeat(1, nbin_input.size(1), nbin_input.size(2), nbin_input.size(3))

        pmW = torch.div(pbin_input.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True), k)
        pmW = pmW.repeat(1, pbin_input.size(1), pbin_input.size(2), pbin_input.size(3))
        pmW[pbin_input.le(-1)] = 0
        pmW[pbin_input.ge(1)] = 0
        pmW = torch.add(pmW, 1, kinv)
        pmW[pbin_input.le(0)] = 0

        nmW = torch.div(nbin_input.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True), nk)
        nmW = nmW.repeat(1, nbin_input.size(1), nbin_input.size(2), nbin_input.size(3))
        nmW[nbin_input.le(-1)] = 0
        nmW[nbin_input.ge(1)] = 0
        nmW = torch.add(nmW, 1, nkinv)
        nmW[nbin_input.gt(0)] = 0

        mW = torch.add(pmW, 1, nmW)

        #print('GradOutput')
        #print(gradOutput[0][0])
        out = torch.mul(mW, gradOutput)
        #print('FinalGradient')
        #print(out[0][0])

        return out

class DualActive(torch.nn.Module):
    def __init__(self):
        super(DualActive, self).__init__()

    def forward(self, dataInput):
        return DualBinActiveZ()(dataInput)
