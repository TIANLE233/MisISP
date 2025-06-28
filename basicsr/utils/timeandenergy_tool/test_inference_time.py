import os
import numpy as np
import time, datetime
import torch
import argparse
import math
import shutil
from collections import OrderedDict
from thop import profile
from basicsr.utils.timeandenergy_tool.gpu_energy_estimation import GPUEnergyEvaluator
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed


def time(model, idle, gpu=4):
    cudnn.benchmark = True
    cudnn.enabled = True

    # gpu = args.gpu
    # device = torch.device(gpu)
    # model = eval(arch)
    print(model.net_g)
    # model = model.to(device)

    input_image_size_a = 256
    input_image_size_b = 256
    # input_image = torch.randn(1, 4, input_image_size, input_image_size).to(model.device)
    # flops, params = profile(model.net_g, inputs=(input_image,))
    # print('Params: %.2f M' % (params / 1e6))
    # print('Flops: %.2f G' % (flops / 1e9))

    # model = eval(args.arch)(sparsity=sparsity).cuda()

    # load training data
    print('==> Preparing data..')
    batch_size = 1

    model.net_g.eval()
    times = []

    input = torch.randn(batch_size, 4, input_image_size_a, input_image_size_b, dtype=torch.float32)
    noise = torch.tensor(0.0, dtype=torch.float32)
    input = input.to(model.device)
    # noise = noise.to(model.device)
    with torch.no_grad():
        for i in range(40):
            output = model.net_g(input)
            torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    evaluator = GPUEnergyEvaluator(watts_offset=idle,gpuid=gpu)

    model.net_g.eval()
    input = input.to(model.device)
    evaluator.start()
    with torch.no_grad():
        for i in range(1000):
            start_evt.record()
            output = model.net_g.forward(input)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_time = start_evt.elapsed_time(end_evt)
            times.append(elapsed_time)
    power = evaluator.end()

    # print("Energy used (J/image)", energy_used / (100 * batch_size))
    print("average time:", sum(times) / (10000 * batch_size), "ms")
    print("power:", power * 1000 / sum(times), "W")
    # print("FPS:", batch_size * 1e+3 / np.mean(times))