

def is_contain_chinese(check_str):
    
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


files = [
    '/ssddata/jzhaaa/Projects/jingye/code/release/nerf-pytorch/vig_pytorch/gcn_lib/pos_embed.py',
    '/ssddata/jzhaaa/Projects/jingye/code/release/nerf-pytorch/vig_pytorch/gcn_lib/torch_edge.py',
    '/ssddata/jzhaaa/Projects/jingye/code/release/nerf-pytorch/vig_pytorch/gcn_lib/torch_nn.py',
    '/ssddata/jzhaaa/Projects/jingye/code/release/nerf-pytorch/vig_pytorch/gcn_lib/torch_vertex.py',
    '/ssddata/jzhaaa/Projects/jingye/code/release/nerf-pytorch/vig_pytorch/pyramid_vig.py',
    '/ssddata/jzhaaa/Projects/jingye/code/release/nerf-pytorch/run_nerf.py',
    '/ssddata/jzhaaa/Projects/jingye/code/release/nerf-pytorch/run_nerf_helpers.py'
]
for file in files:
    lines = open(file, 'r')
    for index, line in enumerate(lines):
        for c in line:
            if is_contain_chinese(c):
                print(file, line)