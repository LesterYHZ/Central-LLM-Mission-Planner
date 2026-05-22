# -*- coding: utf-8 -*-
import os
import sys
import json
import re

if sys.platform.startswith("win"):
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "C:\\triton_cache\\inductor"
    os.environ["TRITON_CACHE_DIR"] = "C:\\triton_cache\\triton"
    os.makedirs("C:\\triton_cache\\inductor", exist_ok=True)
    os.makedirs("C:\\triton_cache\\triton", exist_ok=True)

import torch
from unsloth import FastLanguageModel

MODEL_DIR = "central_llm_lora_model_v3"
MAX_SEQ_LENGTH = 2048

load_in_4bit = True

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    print("🎉 Central LLM 决策大脑装载成功！\n")
except Exception as e:
    print(f"❌ 模型装载失败。具体错误: {e}")
    sys.exit(1)


def robust_json_load(raw_str):
    cleaned = raw_str.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        try:
            parts = cleaned.split('"')
            for i in range(1, len(parts), 2):
                parts[i] = parts[i].replace('\n', '\\n').replace('\r', '\\r')
            fixed_str = '"'.join(parts)
            return json.loads(fixed_str)
        except Exception:
            raise e


def parse_task_object(task_obj, index):
    task_id = task_obj.get("task_id", task_obj.get("id", index + 1))
    agent = task_obj.get("agent", None)
    action = task_obj.get("action", task_obj.get("task", "N/A"))
    params = task_obj.get("params", task_obj.get("args", {}))
    depends = task_obj.get("depends", [])

    if not agent:
        match = re.search(r'(UAV_\d|USV_\d|AUV_\d|ROV_\d)', str(action) + " " + str(task_id), re.IGNORECASE)
        agent = match.group(1).upper() if match else "N/A"

    return {
        "task_id": task_id,
        "agent": str(agent).upper(),
        "action": action,
        "params": params,
        "depends": depends
    }


def parse_and_display_plan(raw_response):
    print("\n--- [ 大模型原始输出响应 ] ---")
    print(raw_response)
    print("-" * 30)

    try:
        data = robust_json_load(raw_response)
        print("\n✅ JSON 语法校验：成功解析！")

        thought = data.get("thought", "No thought provided.")
        print(f"\n🧠 【大模型决策思维链 (Chain of Thought)】:\n{thought}")

        planning = data.get("planning", {})
        tasks_list = []
        plan_valid = True

        if isinstance(planning, list):
            tasks_list = planning
        elif isinstance(planning, dict):
            plan_valid = planning.get("plan_valid", True)
            if "tasks" in planning and isinstance(planning["tasks"], list):
                tasks_list = planning["tasks"]
            elif "planning" in planning and isinstance(planning["planning"], list):
                tasks_list = planning["planning"]

        if not plan_valid:
            print(f"\n🚨 【安全红线触发：模型主动拒绝规划】:")
            print(f"    错误代码 (Error Code): {planning.get('error_code')}")
            print(f"    拒绝原因 (Reason): {planning.get('reason')}")
            return

        print(f"\n📋 【规划任务有向无环图 (DAG)】: 任务总数: {len(tasks_list)}")
        for idx, raw_task in enumerate(tasks_list):
            parsed = parse_task_object(raw_task, idx)
            print(f"    ├─ [任务 ID: {parsed['task_id']}] 指派给: {parsed['agent']}")
            print(f"    │  动作原语: {parsed['action']}")
            print(f"    │  执行参数: {parsed['params']}")
            print(f"    │  依赖先决: {parsed['depends']}")
            print(f"    └──────────────────────────────────────")

    except Exception as e:
        print(f"\n❌ 解析发生异常: {e}")


def query_central_llm(user_command, fleet_status):
    # 【核心修复】：Prompt 必须和训练时一模一样！一字不差！
    system_prompt = (
        "You are the Central LLM, the global coordinator of a heterogeneous multi-robot team. "
        "CRITICAL RULE 1: You can ONLY choose actions from this allowed list: [Navigate_To_Relay_Station, Establish_Acoustic_Gateway, Bathymetric_Survey, Water_Quality_Profiling, Aerial_Wide_Search, Multispectral_Slick_Mapping, Air_Radio_Relay, Sonar_Scan_Pipeline, Beacon_Locating, Wall_Following_Scan, Subsea_Plume_Tracking, Subsea_Manipulator_Operation, Close_up_Photography, NDT_Crack_Inspection]. DO NOT invent new actions. "
        "CRITICAL RULE 2: Underwater tasks REQUIRE an AUV or ROV. "
        "CRITICAL RULE 3: AUV/ROV subsea tasks REQUIRE a USV for acoustic relay. "
        "If the fleet lacks required assets for the task, you MUST reject by outputting {\"plan_valid\": false, \"error_code\": \"INSUFFICIENT_EQUIPMENT_CAPABILITY\"}. "
        "Output your response strictly in the specified JSON format containing 'thought' and 'planning'."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{fleet_status}\n\n{user_command}"}
    ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            use_cache=True,
            temperature=0.1,
            top_p=0.95
        )
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)


def run_predefined_tests():
    print("\n================================================================")
    print(">>> 启动：预设典型多机协同场景自动测试 (Batch Tests)")
    print("================================================================")

    # 🌟 场景 1：标准跨域协同 (黑匣子打捞 - 完美对齐训练集格式)
    print("\n🌟 [测试场景一]：经典多机跨域协同 (黑匣子打捞)")
    fleet_1 = (
        "Active Fleet Telemetry:\n"
        "- UAV_1: Battery 95%, Status Nominal, Payload [HD_Optical_Camera, Multispectral_Imager]\n"
        "- USV_1: Battery 88%, Status Nominal, Payload [Acoustic_Transceiver_Array, RTK_GPS_Base, Water_Quality_Sensor]\n"
        "- AUV_1: Battery 85%, Status Nominal, Payload [Side_Scan_Sonar, DVL, Depth_Sensor]\n"
        "- ROV_1: Battery 90%, Status Nominal, Payload [5DOF_Hydraulic_Manipulator, Low_Light_HD_Camera, NDT_Crack_Sensor]"
    )
    cmd_1 = "Mission Objective: Locate the acoustic pinger of the downed aircraft and secure high-definition visual confirmation of the blackbox. Execute the mission at Sector Alpha Offshore Wind Farm."

    resp_1 = query_central_llm(cmd_1, fleet_1)
    parse_and_display_plan(resp_1)

    # 🌟 场景 1.5：标准跨域协同 (黑匣子打捞 - 完美对齐训练集格式 - 低电量)
    print("\n🌟 [测试场景一点五]：经典多机跨域协同 (黑匣子打捞) - 低电量")
    fleet_1_5 = (
        "Active Fleet Telemetry:\n"
        "- UAV_1: Battery 95%, Status Nominal, Payload [HD_Optical_Camera, Multispectral_Imager]\n"
        "- USV_1: Battery 8%, Status Nominal, Payload [Acoustic_Transceiver_Array, RTK_GPS_Base, Water_Quality_Sensor]\n"
        "- AUV_1: Battery 85%, Status Nominal, Payload [Side_Scan_Sonar, DVL, Depth_Sensor]\n"
        "- ROV_1: Battery 10%, Status Nominal, Payload [5DOF_Hydraulic_Manipulator, Low_Light_HD_Camera, NDT_Crack_Sensor]"
    )
    cmd_1_5 = "Mission Objective: Locate the acoustic pinger of the downed aircraft and secure high-definition visual confirmation of the blackbox. Execute the mission at Sector Alpha Offshore Wind Farm."

    resp_1_5 = query_central_llm(cmd_1_5, fleet_1_5)
    parse_and_display_plan(resp_1_5)

    # 🌟 场景 2：安全拒绝测试 (缺失水下机器人，强行指派水下关闭阀门)
    print("\n🌟 [测试场景二]：异常边界与越权任务检测 (缺失水下机器人，强行指派水下关闭阀门)")
    fleet_2 = (
        "Active Fleet Telemetry:\n"
        "- UAV_1: Battery 90%, Status Nominal, Payload [HD_Optical_Camera]\n"
        "- USV_1: Battery 85%, Status Nominal, Payload [RTK_GPS_Base, Acoustic_Transceiver_Array]"
    )
    cmd_2 = "Mission Objective: Detect the subsea pressure drop anomaly, locate the oil plume source, and secure the underwater valve. Execute the mission at Marine Sanctuary Border Zone."

    resp_2 = query_central_llm(cmd_2, fleet_2)
    parse_and_display_plan(resp_2)

    # 🌟 场景 3：标准跨域协同 ()
    print("\n🌟 [测试场景三]：经典多机跨域协同 (海上搜救)")
    fleet_3 = (
        "Active Fleet Telemetry:\n"
        "- UAV_1: Battery 5%, Status Nominal, Payload [HD_Optical_Camera, Multispectral_Imager]\n"
        "- UAV_2: Battery 65%, Status Nominal, Payload [HD_Optical_Camera, Multispectral_Imager]\n"
        "- USV_1: Battery 58%, Status Nominal, Payload [Acoustic_Transceiver_Array, RTK_GPS_Base, Water_Quality_Sensor]\n"
        "- AUV_1: Battery 85%, Status Nominal, Payload [Side_Scan_Sonar, DVL, Depth_Sensor]\n"
        "- AUV_2: Battery 10%, Status Nominal, Payload [Side_Scan_Sonar, DVL, Depth_Sensor]\n"
        "- ROV_1: Battery 100%, Status Nominal, Payload [5DOF_Hydraulic_Manipulator, Low_Light_HD_Camera, NDT_Crack_Sensor]"
    )
    cmd_3 = "Mission Objective: Perform wide-area aerial visual sweep for the missing vessel and deploy surface vessel for water quality profiling. Execute the mission at Coastal Open Water."

    resp_3 = query_central_llm(cmd_3, fleet_3)
    parse_and_display_plan(resp_3)

    # 🌟 场景 4：标准跨域协同 ()
    print("\n🌟 [测试场景四]：经典多机跨域协同 (大坝检测)")
    fleet_4 = (
        "Active Fleet Telemetry:\n"
        "- UAV_1: Battery 5%, Status Nominal, Payload [HD_Optical_Camera, Multispectral_Imager]\n"
        "- UAV_2: Battery 65%, Status Nominal, Payload [HD_Optical_Camera, Multispectral_Imager]\n"
        "- USV_1: Battery 38%, Status Nominal, Payload [Acoustic_Transceiver_Array, RTK_GPS_Base, Water_Quality_Sensor]\n"
        "- USV_2: Battery 98%, Status Nominal, Payload [Acoustic_Transceiver_Array, RTK_GPS_Base, Water_Quality_Sensor]\n"
        "- AUV_1: Battery 8%, Status Nominal, Payload [Side_Scan_Sonar, DVL, Depth_Sensor]\n"
        "- AUV_2: Battery 10%, Status Nominal, Payload [Side_Scan_Sonar, DVL, Depth_Sensor]\n"
        "- ROV_1: Battery 80%, Status Nominal, Payload [5DOF_Hydraulic_Manipulator, Low_Light_HD_Camera, NDT_Crack_Sensor]"
    )
    cmd_4 = "Mission Objective: Conduct structural integrity scanning of the concrete dam face to identify macro-fractures and perform non-destructive testing. Execute the mission at Dam Basin Sector Seven."

    resp_4 = query_central_llm(cmd_4, fleet_4)
    parse_and_display_plan(resp_4)

    # 🌟 场景 4.5：标准跨域协同 ()
    print("\n🌟 [测试场景四点五]：经典多机跨域协同 (大坝检测)")
    fleet_4_5 = (
        "Active Fleet Telemetry:\n"
        "- UAV_1: Battery 5%, Status Nominal, Payload [HD_Optical_Camera, Multispectral_Imager]\n"
        "- UAV_2: Battery 65%, Status Nominal, Payload [HD_Optical_Camera, Multispectral_Imager]\n"
        "- USV_1: Battery 38%, Status Nominal, Payload [Acoustic_Transceiver_Array, RTK_GPS_Base, Water_Quality_Sensor]\n"
        "- USV_2: Battery 98%, Status Nominal, Payload [Acoustic_Transceiver_Array, RTK_GPS_Base, Water_Quality_Sensor]\n"
        "- AUV_4: Battery 85%, Status Maintenance, Payload [Side_Scan_Sonar, DVL, Depth_Sensor]\n"
        "- AUV_3: Battery 10%, Status Nominal, Payload [Side_Scan_Sonar, DVL, Depth_Sensor]\n"
        "- AUV_1: Battery 30%, Status Nominal, Payload [Side_Scan_Sonar, DVL, Depth_Sensor]\n"
        "- AUV_8: Battery 100%, Status Nominal, Payload [Side_Scan_Sonar, DVL, Depth_Sensor]\n"
        "- ROV_1: Battery 80%, Status Nominal, Payload [5DOF_Hydraulic_Manipulator, Low_Light_HD_Camera, NDT_Crack_Sensor]"
    )
    cmd_4_5 = "Mission Objective: Conduct structural integrity scanning of the concrete dam face to identify macro-fractures and perform non-destructive testing. Execute the mission at Dam Basin Sector Seven."

    resp_4_5 = query_central_llm(cmd_4_5, fleet_4_5)
    parse_and_display_plan(resp_4_5)


if __name__ == "__main__":
    run_predefined_tests()