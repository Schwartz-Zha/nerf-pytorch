CUDA_VISIBLE_DEVICES=$GPU_ID python run_nerf.py --config configs/lego_transformer_b3.txt;
CUDA_VISIBLE_DEVICES=$GPU_ID python run_nerf.py --config configs/lego_transformer_b2.txt ;

while true; do
    rm -rf logs/blender_lego_transformer_debug/
    CUDA_VISIBLE_DEVICES=$GPU_ID python run_nerf.py --config configs/lego_transformer_debug.txt
    sleep 2
done