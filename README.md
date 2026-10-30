# Central LLM Mission Planner

A dataset-generation, validation, fine-tuning, and inference toolkit for coordinating heterogeneous marine robots with a central language model. The project targets missions involving:

- **UAVs** for aerial search and multispectral mapping
- **USVs** for surface navigation, GPS support, water-quality sensing, and acoustic relay
- **AUVs** for autonomous underwater sonar and plume surveys
- **ROVs** for tethered underwater inspection, photography, NDT, and manipulation

The planner converts a natural-language mission objective and live fleet telemetry into a dependency-aware JSON task graph. When the fleet cannot safely execute a mission, the model is trained to return a structured refusal instead of inventing an assignment.

## Repository contents

| File | Purpose |
| --- | --- |
| `Dataset-Generator-(EN).py` | Generates baseline English ChatML/JSONL training data. |
| `Dataset-Generator-(CN).py` | Generates baseline Chinese training data. |
| `Dataset-Generator-Broad-Scenarios.py` | Generates a broader mixture of surface, aerial, and subsea scenarios. |
| `Dataset-Generator-more-vehicles.py` | Generates larger fleets with multiple UAV, USV, AUV, and ROV units. |
| `Dataset-Generator-vehicle-id-more-task.py` | Generates expanded scenarios, dynamic vehicle IDs, and algorithmic fleet checks. |
| `DatasetValidator.py` | Validates and filters JSONL data using DAG, physical-domain, and communication rules. |
| `Unsloth-4070-Training-Script.py` | Performs the initial Qwen2.5-3B LoRA fine-tuning run. |
| `Incremental-4070-Training-Script.py` | Continues training from an existing LoRA model with a lower learning rate. |
| `Central-LLM-Direct-Test-Script.py` | Loads the final LoRA model and runs predefined mission tests. |

Generated datasets and model directories are not committed to this repository. They are produced locally by the scripts.

## Planning contract

A valid assistant response is JSON with a `thought` field and a `planning` object:

```json
{
  "thought": "Fleet validation passed.",
  "planning": {
    "plan_valid": true,
    "total_tasks": 2,
    "tasks": [
      {
        "task_id": 1,
        "agent": "USV_1",
        "action": "Establish_Acoustic_Gateway",
        "params": {"target": "Mission_Area"},
        "depends": []
      },
      {
        "task_id": 2,
        "agent": "AUV_1",
        "action": "Beacon_Locating",
        "params": {"target": "Seabed_Search_Area"},
        "depends": [1]
      }
    ]
  }
}
```

For missions that cannot be executed safely, the planner returns a refusal packet:

```json
{
  "thought": "The required subsea asset is unavailable.",
  "planning": {
    "plan_valid": false,
    "error_code": "INSUFFICIENT_EQUIPMENT_CAPABILITY",
    "reason": "No healthy AUV or ROV is available.",
    "total_tasks": 0,
    "tasks": []
  }
}
```

The expanded generators constrain the action vocabulary to the actions declared in their `ALLOWED_ACTIONS` list. Task dependencies form a directed acyclic graph (DAG); each dependency refers to a preceding task ID.

## Safety and validation rules

`DatasetValidator.py` rejects samples that violate the project’s planning constraints, including:

1. Invalid ChatML structure or malformed assistant JSON.
2. Cyclic task dependencies, checked with Kahn’s topological-sort algorithm.
3. Assigning actions to an incompatible vehicle domain, such as aerial vehicles performing subsea work.
4. Underwater AUV work without a surface USV relay/gateway task.
5. Refusal samples missing `error_code` or `reason`.

These checks should be run before training so the model learns from the cleaned dataset rather than raw generated samples.

## Recommended workflow

### 1. Create a dataset

Run one of the generators from the repository root. For example:

```bash
python Dataset-Generator-vehicle-id-more-task.py
```

That generator writes `robot_planning_train_more_raw.jsonl` by default. Other generators write the output filenames shown in their `__main__` blocks.

### 2. Validate and clean the dataset

Update the input and output paths in the `__main__` block of `DatasetValidator.py`, then run:

```bash
python DatasetValidator.py
```

The validator writes a cleaned JSONL file only when valid samples are available.

### 3. Fine-tune the base model

Install a CUDA-compatible PyTorch/Unsloth environment and the libraries used by the training scripts:

```bash
pip install torch unsloth datasets trl transformers
```

Then ensure the expected cleaned dataset path exists and run:

```bash
python Unsloth-4070-Training-Script.py
```

The initial training script uses `Qwen/Qwen2.5-3B-Instruct`, 4-bit loading, LoRA adapters, a maximum sequence length of 2048, and saves the adapter to `central_llm_lora_model`.

### 4. Perform incremental training (optional)

Place the previous adapter at the directory expected by `Incremental-4070-Training-Script.py` (currently `central_llm_lora_model_v2`) and provide `robot_planning_train_more_cleaned.jsonl`. Then run:

```bash
python Incremental-4070-Training-Script.py
```

The incremental script saves the new adapter as `central_llm_lora_model_v3`.

### 5. Run inference tests

`Central-LLM-Direct-Test-Script.py` expects the final adapter directory `central_llm_lora_model_v3` and a CUDA-capable PyTorch runtime:

```bash
python Central-LLM-Direct-Test-Script.py
```

It exercises normal multi-robot missions, low-battery cases, missing-equipment refusals, and expanded fleet identifiers. The script prints the raw model output, parses the JSON response, and displays the resulting task DAG.

## Runtime notes

- The training and inference scripts are designed for GPU execution and use 4-bit quantization. The training configuration is tuned around an RTX 4070-class Windows setup, but compatible CUDA hardware may work with adjusted settings.
- The inference script loads tensors onto `cuda`; CPU-only execution is not configured.
- Model adapters, generated JSONL files, training outputs, and caches are local artifacts and must be created separately.
- The system prompt used during inference is intentionally aligned with the training format. Changes to the prompt or output schema should be reflected in the data generators and validator.

## Project status

This repository currently provides the core data and model workflow rather than a deployed vehicle-control system. Generated plans should be validated and reviewed by the mission-execution layer before being sent to physical robots.