import json
import random

# 定义基础机型静态能力图谱 (用于强化LLM的常识)
CAPABILITY_PROFILES = {
    "UAV": "空中飞行器。优势：视场极广、速度快(最大20m/s)；局限：续航短(30分钟)、无法入水、无法穿透深水通信。",
    "USV": "水面无人艇。优势：长航时(12小时)、可作为通信中继和定位基站；局限：无法上岸、无法潜水。",
    "AUV": "自主水下机器人。优势：大范围自主航行、挂载侧扫声呐与DVL；局限：水下无法直接连无线电，需要USV进行声学中继，或每20分钟浮出水面同步。",
    "ROV": "遥控水下机器人。优势：定点悬停、配备多自由度机械手、高清摄像机；局限：有缆连接、移动速度慢(最大1m/s)。"
}

TEMPLATES = [
    {
        "scenario": "A区域发生海底原油管道泄漏，需要精确定位漏点并关闭阀门。",
        "logic": "1. 派USV前往泄漏区域上方建立通信与基站。\n2. 派AUV携带侧扫声呐进行管道大范围排查定位漏点。\n3. AUV发现漏点后，将坐标通过USV回传。\n4. 派ROV下潜至漏点坐标，使用机械臂关闭阀门。",
        "plan": [
            {"task_id": 1, "agent": "USV_1", "action": "Navigate_To_Area", "params": {"target": "A_Surface"}, "depends": []},
            {"task_id": 2, "agent": "AUV_1", "action": "Sonar_Scan_Pipeline", "params": {"target": "A_Pipeline_Sea_Bed"}, "depends": [1]},
            {"task_id": 3, "agent": "ROV_1", "action": "Subsea_Manipulator_Operation", "params": {"target": "Leak_Valve_Position", "sub_action": "close_valve"}, "depends": [2]}
        ]
    },
    {
        "scenario": "搜寻海域中不幸坠毁的飞机黑匣子，并对其进行水下定点拍照取证。",
        "logic": "1. 派UAV进行水面漂浮碎片的快速广域搜索，缩小黑匣子可能掉落的区域。\n2. 派遣USV移动至该区域，部署声学网关。\n3. 派遣AUV下潜，利用水听器与侧扫声呐搜索黑匣子的信标信号。\n4. AUV定位后，ROV下潜进行高清拍照取证。",
        "plan": [
            {"task_id": 1, "agent": "UAV_1", "action": "Aerial_Wide_Search", "params": {"target": "Debris_Area"}, "depends": []},
            {"task_id": 2, "agent": "USV_1", "action": "Establish_Acoustic_Gateway", "params": {"target": "Debris_Center_Surface"}, "depends": [1]},
            {"task_id": 3, "agent": "AUV_1", "action": "Beacon_Locating", "params": {"target": "Sea_Floor_Search"}, "depends": [2]},
            {"task_id": 4, "agent": "ROV_1", "action": "Close_up_Photography", "params": {"target": "Blackbox_Position"}, "depends": [3]}
        ]
    },
    {
        "scenario": "对沿海大坝的水下基础安全结构进行一年一度的例行健康巡检，检查是否有裂缝。",
        "logic": "1. 派USV携带多波束测深仪对大坝外围地形进行扫描，防止大障碍物影响潜器。\n2. 派AUV沿大坝壁面进行贴壁自主扫描，记录大范围裂纹数据。\n3. 若发现疑似严重结构损伤裂缝，ROV定点悬停，使用机械臂探针进行无损检测。",
        "plan": [
            {"task_id": 1, "agent": "USV_1", "action": "Bathymetric_Survey", "params": {"target": "Dam_Front_Water"}, "depends": []},
            {"task_id": 2, "agent": "AUV_1", "action": "Wall_Following_Scan", "params": {"target": "Dam_Wall_Surface"}, "depends": [1]},
            {"task_id": 3, "agent": "ROV_1", "action": "NDT_Crack_Inspection", "params": {"target": "Crack_Candidate_Position"}, "depends": [2]}
        ]
    }
]

def generate_synthetic_data(num_samples=10000):
    dataset = []
    for i in range(num_samples):
        # 随机选择一个场景模板
        tpl = random.choice(TEMPLATES)
        
        # 随机生成动态状态参数，增加数据多样性
        uav_bat = random.randint(35, 98)
        usv_fuel = random.randint(40, 95)
        auv_bat = random.randint(30, 95)
        rov_status = random.choice(["Nominal", "Ready"])
        
        user_command = f"当前任务：{tpl['scenario']}"
        
        system_prompt = (
            "你是一个多异构机器人系统的全局指挥官（Central LLM）。你的职责是将用户的自然语言指令拆解为多机协同、"
            "时序正确的子任务拓扑图。你必须严格遵守机器人的物理和通信约束（例如：水下潜器AUV需要水面USV进行声学中继）。"
            "你必须百分之百输出合法的 JSON，不允许夹带任何闲聊文本。"
        )
        
        fleet_status = (
            f"【当前可用设备状态】\n"
            f"- UAV_1 (无人机): 电量 {uav_bat}%, 载荷 [高清可见光相机]\n"
            f"- USV_1 (无人艇): 燃油 {usv_fuel}%, 载荷 [声学基阵, RTK GPS]\n"
            f"- AUV_1 (自主水下潜器): 电量 {auv_bat}%, 载荷 [侧扫声呐, DVL]\n"
            f"- ROV_1 (有缆遥控潜器): 状态 {rov_status}, 载荷 [5自由度机械臂, 高清微光摄像机]"
        )
        
        # 构造CoT（思维链），帮助模型学习推理逻辑
        cot_thought = (
            f"思维分析：\n"
            f"1. 针对任务：'{tpl['scenario']}'，需要跨域协同。\n"
            f"2. 通信约束考量：由于AUV_1潜航深度涉及声学通信，必须首先派遣USV_1到达指定海域建立中继网关（任务1）。\n"
            f"3. 路径与角色匹配：{tpl['logic']}\n"
            f"4. 动力评估：当前AUV电量为{auv_bat}%，USV燃油为{usv_fuel}%，满足安全裕度，可以执行该任务。"
        )
        
        assistant_response = {
            "thought": cot_thought,
            "planning": {
                "total_tasks": len(tpl["plan"]),
                "tasks": tpl["plan"]
            }
        }
        
        # ChatML 结构
        chatml_instance = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{fleet_status}\n\n{user_command}"},
                {"role": "assistant", "content": json.dumps(assistant_response, ensure_ascii=False, indent=2)}
            ]
        }
        dataset.append(chatml_instance)
        
    return dataset

if __name__ == "__main__":
    generated_data = generate_synthetic_data()
    with open("robot_planning_train.jsonl", "w", encoding="utf-8") as f:
        for item in generated_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"成功生成 {len(generated_data)} 条多机任务规划微调数据集，已保存至 robot_planning_train.jsonl")