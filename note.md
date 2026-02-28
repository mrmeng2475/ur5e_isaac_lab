
python /home/bygpu/work_space/ur5e_isaac_lab/ur5e/scripts/zero_agent.py --task "Template-Ur5e-v0" --num_envs 2

# stl转usd
python scripts/stl_to_usd_visualizer.py --stl_path source/ur5e/ur5e/tasks/manager_based/usd/assemble_part/1.stl

# 使用 python 启动训练脚本
# 可视化
python scripts/rsl_rl/train.py --task Template-Ur5e-v0  --num_envs 16
# 训练
python scripts/rsl_rl/train.py --task Template-Ur5e-v0  --num_envs 4096 --headless
# 
python scripts/rsl_rl/play.py --task Template-Ur5e-v0 --num_envs 32  --checkpoint "model_50.pt"

# 有效果

2026-02-26_13-42-44 (跟踪效果比较好)