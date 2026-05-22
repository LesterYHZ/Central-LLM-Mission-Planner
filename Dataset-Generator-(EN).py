# -*- coding: utf-8 -*-
"""
HoloOcean 多机协同任务规划 - 英文数据集生成器
本脚本用于程序化生成 ChatML 格式的训练数据集。
生成的 Prompt、CoT 推理和 JSON 规划全部为英文，有利于 2B~3B 小模型的高精度结构化输出。
代码注释和控制台输出已适配为中文，便于理解。
"""

import json
import random

# 静态能力图谱定义（提供给大模型微调的基础常识库）
CAPABILITY_PROFILES = {
    "UAV": "Aerial drone. High speed (max 20m/s), wide field of view. Limitation: Short endurance (30 mins), cannot submerge, no underwater communication.",
    "USV": "Unmanned surface vessel. High endurance (12 hours). Serves as an acoustic/radio gateway and GPS base station. Limitation: Cannot operate on land or underwater.",
    "AUV": "Autonomous underwater vehicle. Long underwater range, side-scan sonar, DVL. Limitation: No direct RF link underwater; requires USV acoustic relay, or surfacing every 20 mins to sync.",
    "ROV": "Remotely operated vehicle. Tethered, high-precision station keeping, multi-DOF robotic arm, HD cameras. Limitation: Slow motion (max 1m/s), limited tether radius."
}

# 预设的英文跨域协同与应急响应模板库
TEMPLATES = [
    {
        "scenario": "A subsea oil pipeline leak is detected in sector A. Locate the leak point precisely and shut down the emergency valve.",
        "logic": "1. Deploy USV_1 to sector A surface to establish communication and GPS baseline.\n2. Dispatch AUV_1 to perform wide-area side-scan sonar search along the seabed pipeline to pinpoint the leak coordinates.\n3. AUV_1 transmits leak coordinates back to Central via USV_1 acoustic relay.\n4. Deploy ROV_1 to the designated coordinates and use its manipulator arm to close the valve.",
        "plan_valid": True,
        "plan": [
            {"task_id": 1, "agent": "USV_1", "action": "Navigate_To_Area", "params": {"target": "Sector_A_Surface"}, "depends": []},
            {"task_id": 2, "agent": "AUV_1", "action": "Sonar_Scan_Pipeline", "params": {"target": "Subsea_Pipeline_Path"}, "depends": [1]},
            {"task_id": 3, "agent": "ROV_1", "action": "Subsea_Manipulator_Operation", "params": {"target": "Leak_Valve_Coordinates", "sub_action": "close_valve"}, "depends": [2]}
        ]
    },
    {
        "scenario": "Locate the flight data recorder (blackbox) of a downed aircraft in deep water, and perform close-up photographic inspection.",
        "logic": "1. Dispatch UAV_1 for rapid wide-area aerial visual search to narrow down floating debris field.\n2. Navigate USV_1 to the center of the debris field to deploy acoustic transponder arrays.\n3. Deploy AUV_1 to search for underwater pinger acoustic signals and map the seabed.\n4. Once located, submerge ROV_1 to perform high-definition visual inspection and photography.",
        "plan_valid": True,
        "plan": [
            {"task_id": 1, "agent": "UAV_1", "action": "Aerial_Wide_Search", "params": {"target": "Debris_Spread_Zone"}, "depends": []},
            {"task_id": 2, "agent": "USV_1", "action": "Establish_Acoustic_Gateway", "params": {"target": "Acoustic_Relay_Center"}, "depends": [1]},
            {"task_id": 3, "agent": "AUV_1", "action": "Beacon_Locating", "params": {"target": "Seabed_Search_Area"}, "depends": [2]},
            {"task_id": 4, "agent": "ROV_1", "action": "Close_up_Photography", "params": {"target": "Blackbox_Position"}, "depends": [3]}
        ]
    },
    {
        "scenario": "Perform the annual structural integrity inspection on the underwater foundation of the coastal concrete dam.",
        "logic": "1. Navigate USV_1 to bathymetric survey the dam's front water zone to check for massive structural obstacles.\n2. Direct AUV_1 to execute a wall-following vertical bathymetric scan of the dam's concrete face.\n3. If structural anomalies are detected, lock ROV_1 onto the target coordinate to perform Non-Destructive Testing (NDT).",
        "plan_valid": True,
        "plan": [
            {"task_id": 1, "agent": "USV_1", "action": "Bathymetric_Survey", "params": {"target": "Dam_Frontal_Basin"}, "depends": []},
            {"task_id": 2, "agent": "AUV_1", "action": "Wall_Following_Scan", "params": {"target": "Concrete_Dam_Wall"}, "depends": [1]},
            {"task_id": 3, "agent": "ROV_1", "action": "NDT_Crack_Inspection", "params": {"target": "Anomaly_Hotspot"}, "depends": [2]}
        ]
    },
    {
        "scenario": "Investigate offshore oil spill boundary and track underwater plume expansion near the marine conservation sanctuary.",
        "logic": "1. Deploy UAV_1 to map the surface oil slick boundary using multispectral imaging.\n2. Command USV_1 to patrol the slick perimeter and measure surface water quality parameters.\n3. Direct AUV_1 to dive beneath the slick to measure dissolved oxygen and hydrocarbon concentration profile.",
        "plan_valid": True,
        "plan": [
            {"task_id": 1, "agent": "UAV_1", "action": "Multispectral_Slick_Mapping", "params": {"target": "Conservation_Border"}, "depends": []},
            {"task_id": 2, "agent": "USV_1", "action": "Water_Quality_Profiling", "params": {"target": "Slick_Perimeter"}, "depends": [1]},
            {"task_id": 3, "agent": "AUV_1", "action": "Subsea_Plume_Tracking", "params": {"target": "Sub_Slick_Water_Column"}, "depends": [2]}
        ]
    }
]

# 无法执行（安全拒绝）的异常场景库（Out-of-Distribution, OOD）
IMPOSSIBLE_SCENARIOS = [
    {
        "scenario": "Perform subsea valve closing operation immediately.",
        "reason": "Active Fleet lacks ROV capability (only UAV and USV are available). Subsea manipulator operations cannot be executed by aerial or surface vessels.",
        "error_code": "INSUFFICIENT_EQUIPMENT_CAPABILITY"
    },
    {
        "scenario": "Conduct side-scan sonar search on the seabed in Sector B.",
        "reason": "Active Fleet has no underwater agents (AUV/ROV). Underwater sensing cannot be performed by UAV and USV.",
        "error_code": "INSUFFICIENT_EQUIPMENT_CAPABILITY"
    }
]

def generate_synthetic_data(num_samples=10000):
    """
    生成英文 ChatML 格式的训练数据集
    """
    dataset = []
    
    # 英文系统级 Prompt
    system_prompt = (
        "You are the Central LLM, the global coordinator of a heterogeneous multi-robot team. "
        "Your job is to decompose high-level natural language user commands into a sequence of coordinated, "
        "dependency-aware sub-tasks. You must strictly obey physical and communication constraints "
        "(e.g., underwater AUVs require a surface USV for acoustic relay). "
        "Output your response strictly in the specified JSON format containing a logical Chain of Thought ('thought') "
        "and the task sequence ('planning'). Do not include any conversational filler."
    )
    
    for _ in range(num_samples):
        # 85% 概率生成可行规划，15% 概率生成因物理约束无法执行的安全拒绝规划
        if random.random() > 0.15:
            # 可行案例
            tpl = random.choice(TEMPLATES)
            uav_bat = random.randint(40, 99)
            usv_fuel = random.randint(45, 99)
            auv_bat = random.randint(35, 99)
            rov_status = random.choice(["Nominal", "Ready", "Standby"])
            
            fleet_status = (
                f"Active Fleet Telemetry:\n"
                f"- UAV_1: Battery {uav_bat}%, Payload [HD_Optical_Camera, Multispectral_Imager]\n"
                f"- USV_1: Fuel {usv_fuel}%, Payload [Acoustic_Transceiver_Array, RTK_GPS_Base]\n"
                f"- AUV_1: Battery {auv_bat}%, Payload [Side_Scan_Sonar, DVL, Depth_Sensor]\n"
                f"- ROV_1: Status {rov_status}, Payload [5DOF_Hydraulic_Manipulator, Low_Light_HD_Camera]"
            )
            
            user_command = f"Mission Objective: {tpl['scenario']}"
            
            cot_thought = (
                f"Analyzing mission constraints and fleet state. "
                f"1. Communication constraints check: Since the mission requires deep-water actions with AUV_1, "
                f"we must first position USV_1 near the action zone to act as an acoustic telemetry gateway. "
                f"2. Temporal sequence planning: {tpl['logic']}. "
                f"3. Resource check: UAV_1 battery is {uav_bat}%, AUV_1 battery is {auv_bat}%, and USV_1 fuel is {usv_fuel}%, "
                f"which are above safety margins. Initiating task dispatch."
            )
            
            assistant_json = {
                "thought": cot_thought,
                "planning": {
                    "plan_valid": True,
                    "total_tasks": len(tpl["plan"]),
                    "tasks": tpl["plan"]
                }
            }
        else:
            # 安全拒绝案例（OOD 无解环境）
            tpl = random.choice(IMPOSSIBLE_SCENARIOS)
            uav_bat = random.randint(40, 99)
            usv_fuel = random.randint(45, 99)
            
            # 强行排除 AUV 和 ROV，使其由于硬件缺失而报错
            fleet_status = (
                f"Active Fleet Telemetry:\n"
                f"- UAV_1: Battery {uav_bat}%, Payload [HD_Optical_Camera]\n"
                f"- USV_1: Fuel {usv_fuel}%, Payload [RTK_GPS_Base]"
            )
            
            user_command = f"Mission Objective: {tpl['scenario']}"
            
            cot_thought = (
                f"Analyzing fleet status against mission requirements. "
                f"The mission requires underwater manipulation or subsea sensing, but the active fleet list only contains UAV_1 and USV_1. "
                f"No subsea assets (AUV/ROV) are registered or healthy. Thus, this request is physically impossible to execute."
            )
            
            assistant_json = {
                "thought": cot_thought,
                "planning": {
                    "plan_valid": False,
                    "error_code": tpl["error_code"],
                    "reason": tpl["reason"],
                    "total_tasks": 0,
                    "tasks": []
                }
            }
            
        chat_instance = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{fleet_status}\n\n{user_command}"},
                {"role": "assistant", "content": json.dumps(assistant_json, indent=2)}
            ]
        }
        dataset.append(chat_instance)
        
    return dataset

if __name__ == "__main__":
    print(">>> 启动 Central LLM 英文多机训练数据集生成流程...")
    raw_data = generate_synthetic_data(10000)
    
    output_file = "robot_planning_train_raw.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in raw_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"🎉 原始数据集程序化生成成功！共计 {len(raw_data)} 条样本。")
    print(f"已导出至：{output_file} (请在训练前运行数据校验器进行净化)")