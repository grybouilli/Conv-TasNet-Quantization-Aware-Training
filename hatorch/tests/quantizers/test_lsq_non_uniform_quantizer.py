import pickle
import os
import torch
from torch.autograd import Variable
from hatorch.quantizers.autograd.lsq_step_driven_quantizer import LsqStepDrivenQuantizer

test_device = "cpu"

def test_rscmlsq_non_uniform_forwards():
    nbits = 3
    Qn = -2 ** (nbits - 1)
    Qp = 2 ** (nbits - 1) - 1
    g = 1
    targets = torch.tensor([-20, -13, -8, -6, -5, -3, -2, -1, 0, 1, 2, 4, 5, 7, 12, 19], dtype=torch.float).to("cpu")
    wqs = []

    # load precomputed results
    path = os.path.dirname(os.path.abspath(__file__)) + "/wsq_rscmlsq.pkl"
    with open(path, "rb") as input_file:
        precomputed = pickle.load(input_file)

    for i in range(1000):
        inputs = Variable(torch.Tensor([(i * 0.01) - 5]))
        scale = torch.ones(1, requires_grad=True)
        is_init = torch.Tensor([True])
        w_q = LsqStepDrivenQuantizer.apply(inputs, scale, targets, Qn, Qp, g, is_init)
        wqs.append(w_q.data[0])

    assert precomputed == wqs


def test_rscmlsq_uniform_backward():
    nbits = 3
    Qn = -2 ** (nbits - 1)
    Qp = 2 ** (nbits - 1) - 1
    g = 1
    targets = torch.tensor([-20, -13, -8, -6, -5, -3, -2, -1, 0, 1, 2, 4, 5, 7, 12, 19], dtype=torch.float).to("cpu")

    path = os.path.dirname(os.path.abspath(__file__)) + "/alpha_grad_rscmlsq.pkl"
    with open(path, "rb") as input_file:
        precomputed = pickle.load(input_file)

    scale_grads = []
    for i in range(1000):
        inputs = Variable(torch.Tensor([(i * 0.01) - 5]), requires_grad=True)
        scale = torch.ones(1, requires_grad=True)
        is_init = torch.Tensor([True])
        w_q = LsqStepDrivenQuantizer.apply(inputs, scale, targets, Qn, Qp, g, is_init)
        w_q.backward(retain_graph=True)
        scale_grads.append(scale.grad)

    assert precomputed == scale_grads
