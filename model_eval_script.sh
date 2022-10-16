GPU_ID=0

for d in logs/*/ ; do
    echo "$d"
    CUDA_VISIBLE_DEVICES=$GPU_ID python run_nerf.py --config $d'config.txt' --render_only --render_test
done