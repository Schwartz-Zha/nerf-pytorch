# Usage for vig_pytorch_demo


## Command
CUDA_VISIBLE_DEVICES=X python -m torch.distributed.launch --nproc_per_node=1 vig_pytorch_demo.py /path/imagenet --model pvig_s_224_gelu --sched cosine --epochs 300 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 2e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output ./output

## Expected Output
```
Model is successfully built!
input_shape :  torch.Size([15, 64, 90])
output_shape :  torch.Size([15, 64, 4])
```

## How to adapt this code?
* Some tunable parameters at *pyramid_vig.py* 
```
def pvig_s_224_gelu(pretrained=False, **kwargs):
    ...
    self.k = 3 # The neareat neighbor for constructing graph
    self.channels = [90, 90, 90, 90] # The channel size
    ...
```

* Design your own dataloader and loss function! 

* 