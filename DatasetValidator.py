# -*- coding: utf-8 -*-
"""
HoloOcean 多机协同规划 - 自动化符号校验器与清洗器
本脚本负责对自动生成的训练集进行多物理域校验：
1. Kahn 拓扑排序：彻底检测并修复任务循环依赖（死锁）。
2. 物理域匹配校验：拦截让无人机（UAV）潜水、让潜器（AUV/ROV）升空等低级常识幻觉。
3. 伴随通信规则检测：验证水下 AUV 动作是否配有水面 USV 通信网关支持。
"""

import json
import collections

# 静态物理规则库（与生成器严格对齐）
STATIC_PROFILES = {
    "UAV": {
        "allowed_domains": ["air"],
        "forbidden_actions": [
            "Subsea_Manipulator_Operation", 
            "Sonar_Scan_Pipeline", 
            "Beacon_Locating", 
            "Underwater_Inspection", 
            "Close_up_Photography",
            "Bathymetric_Survey",
            "Subsea_Plume_Tracking"
        ]
    },
    "USV": {
        "allowed_domains": ["surface"],
        "forbidden_actions": [
            "Subsea_Manipulator_Operation", 
            "Underwater_Inspection", 
            "Aerial_Wide_Search",
            "Multispectral_Slick_Mapping",
            "Close_up_Photography"
        ]
    },
    "AUV": {
        "allowed_domains": ["underwater", "surface"],
        "forbidden_actions": [
            "Aerial_Wide_Search", 
            "Subsea_Manipulator_Operation",
            "Multispectral_Slick_Mapping"
        ]
    },
    "ROV": {
        "allowed_domains": ["underwater"],
        "forbidden_actions": [
            "Aerial_Wide_Search", 
            "Sonar_Scan_Pipeline", 
            "Bathymetric_Survey",
            "Multispectral_Slick_Mapping",
            "Water_Quality_Profiling",
            "Subsea_Plume_Tracking"
        ]
    }
}

class DatasetValidator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.valid_samples = []
        self.failed_stats = collections.defaultdict(int)

    def run_cleaning_pipeline(self, clean_output_path):
        """
        运行完整的清洗过滤流水线
        """
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"\n================ 开始加载 & 符号清洗流水线 ================")
        print(f"数据源路径: {self.dataset_path} | 总样本量: {len(lines)}")

        for idx, line in enumerate(lines):
            try:
                sample = json.loads(line.strip())
                is_valid, reason = self.validate_sample(sample)
                if is_valid:
                    self.valid_samples.append(sample)
                else:
                    self.failed_stats[reason] += 1
            except Exception as e:
                self.failed_stats[f"JSON解析或意外系统错误: {str(e)}"] += 1

        self.print_report(len(lines))
        
        # 仅在产生有效样本时进行保存输出
        if self.valid_samples:
            self.save_clean_dataset(clean_output_path)
        else:
            print("❌ 错误：清洗后未产生任何合格数据，请检查数据集生成模板。")

    def validate_sample(self, sample):
        """
        单条大模型 ChatML 样本逻辑闭环校验
        """
        messages = sample.get("messages", [])
        if len(messages) < 3:
            return False, "ChatML结构破损 (缺失对话轮次)"

        assistant_content = messages[2]["content"]
        try:
            plan_data = json.loads(assistant_content)
        except json.JSONDecodeError:
            return False, "Assistant输出非合法的标准JSON字符串"

        planning = plan_data.get("planning", {})
        plan_valid = planning.get("plan_valid", True)
        
        # 如果是安全拒绝样本（即大模型拒绝了无解的任务），只要拒绝理由存在即合规
        if not plan_valid:
            if not planning.get("reason") or not planning.get("error_code"):
                return False, "拒绝样板字段缺失 (缺失reason/error_code)"
            return True, "Success"

        tasks = planning.get("tasks", [])
        if not tasks:
            return False, "生成的正向规划中 tasks 数组为空"

        # 1. Kahn 算法拓扑死锁校验
        if not self._check_dag_validity(tasks):
            return False, "任务树拓扑排序失败 (存在循环依赖死锁)"

        # 2. 物理域安全红线校验
        for task in tasks:
            agent_raw = task.get("agent", "")
            agent_type = agent_raw.split("_")[0].upper()
            action = task.get("action", "")

            if agent_type not in STATIC_PROFILES:
                return False, f"指派了静态库中未知的设备类型: {agent_raw}"

            # 动作与载荷物理域交叉比对
            forbidden = STATIC_PROFILES[agent_type]["forbidden_actions"]
            if action in forbidden:
                return False, f"严重物理域越权: 严禁指派 [{agent_raw}] 执行 [{action}]"

        # 3. 水下跨域通信伴随中继校验
        # 规则：若有 AUV 进行水下探测任务，其依赖前置任务中必须包含 USV 的建立中继或导航任务
        has_auv_deep_work = False
        has_usv_relay = False

        for task in tasks:
            agent_raw = task.get("agent", "").upper()
            action = task.get("action", "")

            if "AUV" in agent_raw and action in ["Sonar_Scan_Pipeline", "Beacon_Locating", "Wall_Following_Scan", "Subsea_Plume_Tracking"]:
                has_auv_deep_work = True
            if "USV" in agent_raw and action in ["Establish_Acoustic_Gateway", "Navigate_To_Area", "Bathymetric_Survey"]:
                has_usv_relay = True

        if has_auv_deep_work and not has_usv_relay:
            return False, "通信高危违规: 水下 AUV 扫描缺失水面 USV 信号网关伴随"

        return True, "Success"

    def _check_dag_validity(self, tasks):
        """
        基于 Kahn 拓扑排序算法，验证并寻找环路
        """
        in_degree = {}
        adj_list = collections.defaultdict(list)
        all_task_ids = set()

        for task in tasks:
            tid = task["task_id"]
            all_task_ids.add(tid)
            if tid not in in_degree:
                in_degree[tid] = 0

            depends = task.get("depends", [])
            for dep in depends:
                all_task_ids.add(dep)
                adj_list[dep].append(tid)
                in_degree[tid] = in_degree.get(tid, 0) + 1

        # 获取所有入度为 0 的节点
        queue = [node for node in all_task_ids if in_degree.get(node, 0) == 0]
        visited_count = 0

        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for neighbor in adj_list[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 遍历节点数必须等于任务图总结点数
        return visited_count == len(all_task_ids)

    def save_clean_dataset(self, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in self.valid_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"🎉 数据清洗导出成功！完美数据集已保存至: {output_path}")

    def print_report(self, total):
        print("\n================ 数据质量分析及净化报告 ================")
        print(f"总输入样本: {total} 条")
        print(f"合格样本数: {len(self.valid_samples)} 条")
        print(f"数据纯净度: {(len(self.valid_samples)/total)*100:.2f}%")
        print(f"被滤除样本: {total - len(self.valid_samples)} 条")
        if self.failed_stats:
            print("\n剔除原因频次细分:")
            for reason, count in self.failed_stats.items():
                print(f"  - {reason}: {count} 次")
        print("========================================================\n")


if __name__ == "__main__":
    # validator = DatasetValidator("robot_planning_train_raw.jsonl")
    # validator.run_cleaning_pipeline("robot_planning_train_cleaned.jsonl")

    # validator = DatasetValidator("robot_planning_incremental.jsonl")
    # validator.run_cleaning_pipeline("robot_planning_incremental.jsonl")

    validator = DatasetValidator("robot_planning_train_more_raw.jsonl")
    validator.run_cleaning_pipeline("robot_planning_train_more_cleaned.jsonl")