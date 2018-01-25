import torch

class BinActiveZ(torch.autograd.Function):
    def __init__(self):
        super(BinActiveZ, self).__init__()

    def forward(self, input):
        self.save_for_backward(input)
        output = input.clone()
        return output.sign()

    def backward(self, gradOutput):
        input = self.saved_tensors[0]

        # Image backward pass - Calculate STE
        gradInput = gradOutput.clone()
        gradInput[input.le(-1)] = 0
        gradInput[input.ge(1)] = 0

        return gradInput

class Active(torch.nn.Module):
    def __init__(self):
        super(Active, self).__init__()

    def forward(self, dataInput):
        return BinActiveZ()(dataInput)
