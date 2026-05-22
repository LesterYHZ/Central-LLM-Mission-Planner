# -*- coding: utf-8 -*-
import json
import random

# 1. 严格锁死的合法动作空间 (Action Space)
ALLOWED_ACTIONS = [
    "Navigate_To_Relay_Station", "Establish_Acoustic_Gateway", "Bathymetric_Survey", "Water_Quality_Profiling",  # USV
    "Aerial_Wide_Search", "Multispectral_Slick_Mapping", "Air_Radio_Relay",  # UAV
    "Sonar_Scan_Pipeline", "Beacon_Locating", "Wall_Following_Scan", "Subsea_Plume_Tracking",  # AUV
    "Subsea_Manipulator_Operation", "Close_up_Photography", "NDT_Crack_Inspection"  # ROV
]

AGENT_SPECIFICATIONS = {
    "UAV": {"domain": "air", "required_payloads": ["HD_Optical_Camera", "Multispectral_Imager"]},
    "USV": {"domain": "surface",
            "required_payloads": ["Acoustic_Transceiver_Array", "RTK_GPS_Base", "Water_Quality_Sensor"]},
    "AUV": {"domain": "underwater", "required_payloads": ["Side_Scan_Sonar", "DVL", "Depth_Sensor"]},
    "ROV": {"domain": "underwater",
            "required_payloads": ["5DOF_Hydraulic_Manipulator", "Low_Light_HD_Camera", "NDT_Crack_Sensor"]}
}

ENVIRONMENTS = [
    {"name": "Sector Alpha Offshore Wind Farm", "desc": "near the rotating turbine foundations"},
    {"name": "Dam Basin Sector Seven", "desc": "close to the vertical concrete dam walls"},
    {"name": "Marine Sanctuary Border Zone", "desc": "near the coral reef restoration shelf"},
    {"name": "Deep Subsea Pipeline Corridor Delta", "desc": "along the high-pressure gas bypass line"}
]

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
    }
]


def generate_combinatorial_fleet(force_negative_subsea=False):
    fleet = {}
    agents_pool = ["UAV_1", "UAV_2", "USV_1", "USV_2", "AUV_1", "AUV_2", "ROV_1", "ROV_2"]

    # 【核心修复】：如果强制制造负样本，就把所有水下设备从池子里剔除！
    if force_negative_subsea:
        agents_pool = ["UAV_1", "UAV_2", "USV_1", "USV_2"]

    selected_agents = random.sample(agents_pool, k=random.randint(2, len(agents_pool)))

    for agent_id in selected_agents:
        agent_type = agent_id.split("_")[0]
        specs = AGENT_SPECIFICATIONS[agent_type]
        battery = random.randint(8, 99)
        status = random.choices(["Nominal", "Ready", "Degraded", "Maintenance"], weights=[70, 15, 10, 5])[0]

        fleet[agent_id] = {
            "type": agent_type,
            "battery": battery,
            "status": status,
            "payloads": specs["required_payloads"]
        }
    return fleet


def build_concrete_planning(task_meta, fleet):
    def select_best_agent(agent_type):
        candidates = [aid for aid, info in fleet.items() if
                      info["type"] == agent_type and info["battery"] >= 15 and info["status"] != "Maintenance"]
        candidates.sort(key=lambda x: fleet[x]["battery"], reverse=True)
        return candidates[0]

    assigned = {t: select_best_agent(t) for t in task_meta["required_agents"]}
    plan_tasks = []

    if task_meta["type"] == "subsea_leak":
        plan_tasks.append({"task_id": 1, "agent": assigned["USV"], "action": "Navigate_To_Relay_Station",
                           "params": {"target": "Surface_Anchor"}, "depends": []})
        plan_tasks.append(
            {"task_id": 2, "agent": assigned["AUV"], "action": "Sonar_Scan_Pipeline", "params": {"target": "Seabed"},
             "depends": [1]})
        plan_tasks.append({"task_id": 3, "agent": assigned["ROV"], "action": "Subsea_Manipulator_Operation",
                           "params": {"target": "Valve"}, "depends": [2]})
    elif task_meta["type"] == "blackbox_rescue":
        plan_tasks.append(
            {"task_id": 1, "agent": assigned["UAV"], "action": "Aerial_Wide_Search", "params": {"target": "Surface"},
             "depends": []})
        plan_tasks.append({"task_id": 2, "agent": assigned["USV"], "action": "Establish_Acoustic_Gateway",
                           "params": {"target": "Area"}, "depends": [1]})
        plan_tasks.append(
            {"task_id": 3, "agent": assigned["AUV"], "action": "Beacon_Locating", "params": {"target": "Seabed"},
             "depends": [2]})
        plan_tasks.append(
            {"task_id": 4, "agent": assigned["ROV"], "action": "Close_up_Photography", "params": {"target": "Wreck"},
             "depends": [3]})

    return plan_tasks


def generate_dataset_scale(num_samples=15000):
    # 【核心修复】：将动作锁死和硬规则刻在 System Prompt 里！
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
        tpl = random.choice(TASK_TEMPLATES)
        env = random.choice(ENVIRONMENTS)

        # 强制制造 35% 左右的极度缺乏设备的恶劣情况
        force_hard_negative = random.random() < 0.35
        fleet = generate_combinatorial_fleet(force_negative_subsea=force_hard_negative)

        fleet_telemetry_lines = ["Active Fleet Telemetry:"]
        for aid, info in fleet.items():
            fleet_telemetry_lines.append(
                f"- {aid}: Battery {info['battery']}%, Status {info['status']}, Payload [{', '.join(info['payloads'])}]")
        fleet_status_str = "\n".join(fleet_telemetry_lines)

        user_command = f"Mission Objective: {tpl['description']} Execute the mission at {env['name']} ({env['desc']})."

        # 判断可行性
        missing_types = [t for t in tpl["required_agents"] if not any(
            v["type"] == t and v["battery"] >= 15 and v["status"] != "Maintenance" for k, v in fleet.items())]

        if missing_types:
            cot_thought = f"Analyzing fleet. The mission requires {tpl['required_agents']}. Active Fleet lacks healthy units for: {missing_types}. Subsea/deep-water operations are physically impossible. Generating abort packet."
            assistant_json = {
                "thought": cot_thought,
                "planning": {
                    "plan_valid": False,
                    "error_code": "INSUFFICIENT_EQUIPMENT_CAPABILITY",
                    "reason": f"Fleet lacks healthy critical assets: {missing_types}.",
                    "total_tasks": 0,
                    "tasks": []
                }
            }
        else:
            plan_tasks = build_concrete_planning(tpl, fleet)
            assistant_json = {
                "thought": "Fleet validation passed. USV deployed for relay, Subsea assets deployed successfully.",
                "planning": {"plan_valid": True, "total_tasks": len(plan_tasks), "tasks": plan_tasks}
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
    print(">>> 启动加强版数据生成器 (35% 强制负样本 + 锁死动作空间)...")
    scale_dataset = generate_dataset_scale(15000)
    output_filename = "robot_planning_train_raw.jsonl"
    with open(output_filename, "w", encoding="utf-8") as f:
        for item in scale_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"🎉 成功生成 15000 条高质量训练数据至 {output_filename}")