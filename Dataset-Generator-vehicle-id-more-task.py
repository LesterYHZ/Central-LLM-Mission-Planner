# -*- coding: utf-8 -*-
"""
HoloOcean 跨域多机任务规划 - 算法级思维链（Algorithmic CoT）数据生成器 (满血扩容版)
核心升级：
1. 场景扩容：从 3 个基础场景扩展到 7 个跨域复杂场景（包含纯水面与多层水下联合作业）。
2. 精准负样本：在制造负样本时定向选择需要潜水器的任务，产出极高质量的拒绝（Abort）思维链。
3. 数据量级：30,000 条算法级排查日志，彻底巩固 LLM 的泛化推理能力。
"""

import json
import random

ALLOWED_ACTIONS = [
    "Navigate_To_Relay_Station", "Establish_Acoustic_Gateway", "Bathymetric_Survey", "Water_Quality_Profiling", 
    "Aerial_Wide_Search", "Multispectral_Slick_Mapping", "Air_Radio_Relay", 
    "Sonar_Scan_Pipeline", "Beacon_Locating", "Wall_Following_Scan", "Subsea_Plume_Tracking", 
    "Subsea_Manipulator_Operation", "Close_up_Photography", "NDT_Crack_Inspection" 
]

AGENT_SPECIFICATIONS = {
    "UAV": {"domain": "air", "required_payloads": ["HD_Optical_Camera", "Multispectral_Imager"]},
    "USV": {"domain": "surface", "required_payloads": ["Acoustic_Transceiver_Array", "RTK_GPS_Base", "Water_Quality_Sensor"]},
    "AUV": {"domain": "underwater", "required_payloads": ["Side_Scan_Sonar", "DVL", "Depth_Sensor"]},
    "ROV": {"domain": "underwater", "required_payloads": ["5DOF_Hydraulic_Manipulator", "Low_Light_HD_Camera", "NDT_Crack_Sensor"]}
}

ENVIRONMENTS = [
    {"name": "Sector Alpha Offshore Wind Farm", "desc": "near the rotating turbine foundations"},
    {"name": "Dam Basin Sector Seven", "desc": "close to the vertical concrete dam walls"},
    {"name": "Marine Sanctuary Border Zone", "desc": "near the coral reef restoration shelf"},
    {"name": "Deep Subsea Pipeline Corridor Delta", "desc": "along the high-pressure gas bypass line"},
    {"name": "Coastal Estuary Spill Boundary", "desc": "adjacent to the chemical terminal jetty"},
    {"name": "Deepwater Horizon Platform", "desc": "around the massive submerged steel pylons"},
    {"name": "Subsea Communication Cable Alpha", "desc": "along the transatlantic fiber optic trench"}
]

# 扩充为 7 个经典跨域任务模板
TASK_TEMPLATES = [
    {
        "type": "subsea_leak",
        "required_agents": ["USV", "AUV", "ROV"],
        "description": "Detect the subsea pressure drop anomaly, locate the oil plume source, and secure the underwater valve."
    },
    {
        "type": "blackbox_rescue",
        "required_agents": ["UAV", "USV", "AUV", "ROV"],
        "description": "Locate the acoustic pinger of the downed aircraft and secure high-definition visual confirmation of the blackbox."
    },
    {
        "type": "dam_inspection",
        "required_agents": ["USV", "AUV", "ROV"],
        "description": "Conduct structural integrity scanning of the concrete dam face to identify macro-fractures and perform non-destructive testing."
    },
    {
        "type": "oil_slick_tracking",
        "required_agents": ["UAV", "USV", "AUV"],
        "description": "Map the boundary of the surface chemical spill and trace the subsurface hydrocarbon dispersion column."
    },
    {
        "type": "surface_patrol", # 纯空面任务
        "required_agents": ["UAV", "USV"],
        "description": "Perform wide-area aerial visual sweep for the missing vessel and deploy surface vessel for water quality profiling."
    },
    {
        "type": "underwater_cable_inspection",
        "required_agents": ["USV", "AUV"],
        "description": "Perform rapid acoustic scanning along the subsea communication cable to identify potential trenching damage."
    },
    {
        "type": "offshore_platform_maintenance",
        "required_agents": ["UAV", "USV", "ROV"],
        "description": "Inspect topside structures via aerial feed, deploy acoustic gateway, and execute underwater crack NDT on platform pylons."
    }
]

def generate_dynamic_fleet(force_negative=False):
    """
    动态生成 1 到 9 之间任意 ID 的机队。
    若 force_negative=True，则死锁池子只产出空面载具（UAV/USV），故意让深海任务失败。
    """
    fleet = {}
    agent_types = ["UAV", "USV"] if force_negative else ["UAV", "USV", "AUV", "ROV"]
    
    for atype in agent_types:
        num_agents = random.randint(1, 4) if not force_negative else random.randint(1, 3)
        if not force_negative and random.random() < 0.1:
            continue
            
        indices = random.sample(range(1, 10), num_agents)
        for idx in indices:
            aid = f"{atype}_{idx}"
            specs = AGENT_SPECIFICATIONS[atype]
            battery = random.randint(5, 100)
            status = random.choices(["Nominal", "Ready", "Degraded", "Maintenance"], weights=[50, 20, 10, 20])[0]
            
            fleet[aid] = {
                "type": atype, "battery": battery, "status": status, "payloads": specs["required_payloads"]
            }
    return fleet

def generate_algorithmic_cot_and_plan(task_meta, fleet):
    """
    生成算法级状态排查日志与拓扑 DAG 任务链
    """
    required = task_meta["required_agents"]
    thought_steps = [f"Analyzing fleet. The mission requires {required}."]
    missing = []
    assigned_agents = {}
    
    for req_type in required:
        thought_steps.append(f"Checking {req_type} availability:")
        type_agents = {k: v for k, v in fleet.items() if v["type"] == req_type}
        
        if not type_agents:
            thought_steps.append(f"No {req_type} present in fleet.")
            missing.append(req_type)
            continue
            
        valid_candidate = None
        best_battery = -1
        
        for aid in sorted(type_agents.keys()):
            info = type_agents[aid]
            if info["status"] == "Maintenance":
                thought_steps.append(f"- {aid}: Status is Maintenance (Invalid).")
            elif info["battery"] < 15:
                thought_steps.append(f"- {aid}: Battery {info['battery']}% is below 15% threshold (Invalid).")
            else:
                thought_steps.append(f"- {aid}: Status {info['status']}, Battery {info['battery']}% (Valid).")
                if info["battery"] > best_battery:
                    best_battery = info["battery"]
                    valid_candidate = aid
                    
        if valid_candidate:
            thought_steps.append(f"Selected optimal agent: {valid_candidate}.")
            assigned_agents[req_type] = valid_candidate
        else:
            thought_steps.append(f"No healthy {req_type} found.")
            missing.append(req_type)

    if missing:
        thought_steps.append(f"Physical constraints block mission. Generating abort packet.")
        return " ".join(thought_steps), False, {}, assigned_agents
    else:
        thought_steps.append("Fleet validation passed. Deploying optimal agents to execute tasks sequence.")
        
        plan_tasks = []
        # 新版扩容 DAG 构建器
        if task_meta["type"] == "dam_inspection":
            plan_tasks.append({"task_id": 1, "agent": assigned_agents["USV"], "action": "Bathymetric_Survey", "params": {"target": "Dam_Basin"}, "depends": []})
            plan_tasks.append({"task_id": 2, "agent": assigned_agents["AUV"], "action": "Wall_Following_Scan", "params": {"target": "Dam_Face"}, "depends": [1]})
            plan_tasks.append({"task_id": 3, "agent": assigned_agents["ROV"], "action": "NDT_Crack_Inspection", "params": {"target": "Crack_Hotspot"}, "depends": [2]})
            
        elif task_meta["type"] == "subsea_leak":
            plan_tasks.append({"task_id": 1, "agent": assigned_agents["USV"], "action": "Navigate_To_Relay_Station", "params": {"target": "Surface"}, "depends": []})
            plan_tasks.append({"task_id": 2, "agent": assigned_agents["AUV"], "action": "Sonar_Scan_Pipeline", "params": {"target": "Seabed"}, "depends": [1]})
            plan_tasks.append({"task_id": 3, "agent": assigned_agents["ROV"], "action": "Subsea_Manipulator_Operation", "params": {"target": "Valve"}, "depends": [2]})
            
        elif task_meta["type"] == "blackbox_rescue":
            plan_tasks.append({"task_id": 1, "agent": assigned_agents["UAV"], "action": "Aerial_Wide_Search", "params": {"target": "Surface"}, "depends": []})
            plan_tasks.append({"task_id": 2, "agent": assigned_agents["USV"], "action": "Establish_Acoustic_Gateway", "params": {"target": "Area"}, "depends": [1]})
            plan_tasks.append({"task_id": 3, "agent": assigned_agents["AUV"], "action": "Beacon_Locating", "params": {"target": "Seabed"}, "depends": [2]})
            plan_tasks.append({"task_id": 4, "agent": assigned_agents["ROV"], "action": "Close_up_Photography", "params": {"target": "Wreck"}, "depends": [3]})
            
        elif task_meta["type"] == "oil_slick_tracking":
            plan_tasks.append({"task_id": 1, "agent": assigned_agents["UAV"], "action": "Multispectral_Slick_Mapping", "params": {"target": "Spill_Border"}, "depends": []})
            plan_tasks.append({"task_id": 2, "agent": assigned_agents["USV"], "action": "Water_Quality_Profiling", "params": {"target": "Slick_Center"}, "depends": [1]})
            plan_tasks.append({"task_id": 3, "agent": assigned_agents["AUV"], "action": "Subsea_Plume_Tracking", "params": {"target": "Subsurface_Plume"}, "depends": [2]})
            
        elif task_meta["type"] == "surface_patrol":
            plan_tasks.append({"task_id": 1, "agent": assigned_agents["UAV"], "action": "Aerial_Wide_Search", "params": {"target": "Search_Zone"}, "depends": []})
            plan_tasks.append({"task_id": 2, "agent": assigned_agents["USV"], "action": "Water_Quality_Profiling", "params": {"target": "Anomaly_Spot"}, "depends": [1]})
            
        elif task_meta["type"] == "underwater_cable_inspection":
            plan_tasks.append({"task_id": 1, "agent": assigned_agents["USV"], "action": "Navigate_To_Relay_Station", "params": {"target": "Cable_Path_Surface"}, "depends": []})
            plan_tasks.append({"task_id": 2, "agent": assigned_agents["AUV"], "action": "Sonar_Scan_Pipeline", "params": {"target": "Subsea_Cable"}, "depends": [1]})
            
        elif task_meta["type"] == "offshore_platform_maintenance":
            plan_tasks.append({"task_id": 1, "agent": assigned_agents["UAV"], "action": "Aerial_Wide_Search", "params": {"target": "Topside_Structure"}, "depends": []})
            plan_tasks.append({"task_id": 2, "agent": assigned_agents["USV"], "action": "Establish_Acoustic_Gateway", "params": {"target": "Platform_Base"}, "depends": [1]})
            plan_tasks.append({"task_id": 3, "agent": assigned_agents["ROV"], "action": "NDT_Crack_Inspection", "params": {"target": "Submerged_Pylons"}, "depends": [2]})
            
        return " ".join(thought_steps), True, plan_tasks, assigned_agents

def generate_dataset_scale(num_samples=30000):
    actions_str = ", ".join(ALLOWED_ACTIONS)
    system_prompt = (
        "You are the Central LLM, the global coordinator of a heterogeneous multi-robot team. "
        f"CRITICAL RULE 1: You can ONLY choose actions from this allowed list: [{actions_str}]. DO NOT invent new actions. "
        "CRITICAL RULE 2: Underwater tasks REQUIRE an AUV or ROV. "
        "CRITICAL RULE 3: AUV/ROV subsea tasks REQUIRE a USV for acoustic relay. "
        "If the fleet lacks required assets for the task, you MUST reject by outputting {\"plan_valid\": false, \"error_code\": \"INSUFFICIENT_EQUIPMENT_CAPABILITY\"}. "
        "Output your response strictly in the specified JSON format containing 'thought' and 'planning'."
    )
    
    dataset = []
    for i in range(num_samples):
        # 控制 30% 为强负样本
        force_hard_negative = random.random() < 0.30
        
        # 精准负样本匹配：如果强制负样本，定向分配水下深海任务，确保触发缺失 AUV/ROV 的硬拒绝逻辑
        if force_hard_negative:
            subsea_tasks = [t for t in TASK_TEMPLATES if "AUV" in t["required_agents"] or "ROV" in t["required_agents"]]
            tpl = random.choice(subsea_tasks)
        else:
            tpl = random.choice(TASK_TEMPLATES)
            
        env = random.choice(ENVIRONMENTS)
        fleet = generate_dynamic_fleet(force_negative=force_hard_negative)
        
        fleet_telemetry_lines = ["Active Fleet Telemetry:"]
        for aid, info in fleet.items():
            fleet_telemetry_lines.append(f"- {aid}: Battery {info['battery']}%, Status {info['status']}, Payload [{', '.join(info['payloads'])}]")
        fleet_status_str = "\n".join(fleet_telemetry_lines)
        
        user_command = f"Mission Objective: {tpl['description']} Execute the mission at {env['name']} ({env['desc']})."
        
        cot_thought, is_valid, plan_tasks, _ = generate_algorithmic_cot_and_plan(tpl, fleet)
        
        if not is_valid:
            assistant_json = {
                "thought": cot_thought,
                "planning": {
                    "plan_valid": False,
                    "error_code": "INSUFFICIENT_EQUIPMENT_CAPABILITY",
                    "reason": "Refer to thought log: Physical constraints block mission.",
                    "total_tasks": 0,
                    "tasks": []
                }
            }
        else:
            assistant_json = {
                "thought": cot_thought,
                "planning": {
                    "plan_valid": True,
                    "total_tasks": len(plan_tasks),
                    "tasks": plan_tasks
                }
            }
            
        chat_instance = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{fleet_status_str}\n\n{user_command}"},
                {"role": "assistant", "content": json.dumps(assistant_json, indent=2)}
            ]
        }
        dataset.append(chat_instance)
        
    return dataset

if __name__ == "__main__":
    print(">>> 启动 广度扩容版 算法级思维链 数据生成器...")
    # 一次性生成 30,000 条高质量训练语料
    scale_dataset = generate_dataset_scale(30000)
    output_filename = "robot_planning_train_more_raw.jsonl"
    with open(output_filename, "w", encoding="utf-8") as f:
        for item in scale_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"🎉 成功生成 30,000 条算法级排查训练数据至 {output_filename}")