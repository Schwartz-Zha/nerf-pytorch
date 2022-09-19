
while true; do
    rm -rf logs/blender_ficus/
    CUDA_VISIBLE_DEVICES=$GPU_ID python run_nerf.py --config configs/ficus.txt
    sleep 2
done