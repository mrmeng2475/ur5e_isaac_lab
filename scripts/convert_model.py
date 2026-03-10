import torch

# 1. 这里填入你最原始的那个老模型路径 (包含 model_state_dict 的那个)
original_old_model_path = "/home/wuyou/work_space/ur5e_isaac_lab/logs/rsl_rl/cartpole_direct/2026-03-09_01-39-38/model_5999.pt"  
# 2. 最终转换好的模型路径
final_model_path = "final_converted_model.pt"

print(f"正在读取原始模型: {original_old_model_path}")
old_dict = torch.load(original_old_model_path, map_location="cpu")

if 'model_state_dict' in old_dict:
    old_state = old_dict['model_state_dict']
    
    new_actor_state = {}
    new_critic_state = {}

    # 遍历老模型的所有层，进行改名并分配给对应的网络
    for key, value in old_state.items():
        # 处理 Actor 层
        if key.startswith('actor.'):
            new_key = key.replace('actor.', 'mlp.')
            new_actor_state[new_key] = value
        # 处理 Critic 层
        elif key.startswith('critic.'):
            new_key = key.replace('critic.', 'mlp.')
            new_critic_state[new_key] = value
        # 处理动作分布标准差参数
        elif key == 'std':
            new_actor_state['distribution.std_param'] = value
        else:
            print(f"跳过未识别的键: {key}")

    # 组装全新的、完全符合当前 RSL-RL 架构的字典
    new_dict = {
        'actor_state_dict': new_actor_state,
        'critic_state_dict': new_critic_state,
        'optimizer_state_dict': old_dict.get('optimizer_state_dict', {}),
        'iter': old_dict.get('iter', 0),
        'infos': old_dict.get('infos', {})
    }

    torch.save(new_dict, final_model_path)
    print(f"\n✅ 深度转换成功！")
    print(f"文件已保存至: {final_model_path}")
    print(f"Actor 包含层数: {len(new_actor_state)}")
    print(f"Critic 包含层数: {len(new_critic_state)}")
else:
    print("❌ 错误：在原始模型中未找到 'model_state_dict'。")