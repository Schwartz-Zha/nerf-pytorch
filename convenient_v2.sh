CUDA_VISIBLE_DEVICES=$GPU_ID python run_nerf.py --config configs/lego_transformer.txt ;
CUDA_VISIBLE_DEVICES=$GPU_ID python run_nerf.py --config configs/lego_transformer_intdim32;

while true; do
    rm -rf logs/blender_lego_transformer_debug/
    CUDA_VISIBLE_DEVICES=$GPU_ID python run_nerf.py --config configs/lego_transformer_debug.txt
    sleep 2
done