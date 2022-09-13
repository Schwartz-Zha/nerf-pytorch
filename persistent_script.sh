
while true; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python run_nerf.py --config configs/ficus.txt
    sleep 2
done