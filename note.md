
python scripts/zero_agent.py --task "Template-Ur5e-v0" --num_envs 2
python scripts/test.py --task "Template-Ur5e-v0" --num_envs 2

# stl转usd
python scripts/stl_to_usd_visualizer.py --stl_path /home/mrmeng/work_space/rl/ur5e_isaac_lab/10.STL

# 使用 python 启动训练脚本
# 可视化
python scripts/rsl_rl/train.py --task Template-Ur5e-v0  --num_envs 16

# 训练
python scripts/rsl_rl/train.py --task Template-Ur5e-v0  --num_envs 1024 --headless --checkpoint logs/rsl_rl/cartpole_direct/2026-03-06_16-37-43/model_1450.pt
# 
python scripts/rsl_rl/play.py --task Template-Ur5e-v0 --num_envs 4  --checkpoint logs/rsl_rl/cartpole_direct/2026-03-08_10-47-31/model_3200.pt

# 有效果

python scripts/rsl_rl/train.py \
    --task Template-Ur5e-v0 \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run 2026-03-06_16-37-43 \
    --checkpoint model_1450.pt