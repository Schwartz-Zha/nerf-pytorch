import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

from fvcore.nn import FlopCountAnalysis

from run_nerf_helpers import *


if __name__ == "__main__":

    with torch.cuda.device(0):
        # net = NeRF
        # macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, 
        #                 print_per_layer_stat=False, verbose=False)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        flops = FlopCountAnalysis(net, torch.randn(1, 3, 224, 224))
        print(flops.total())