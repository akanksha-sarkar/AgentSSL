from __future__ import annotations
import json
from pathlib import Path


def load_results(results_path: str):
    with open(results_path, "r") as f:
        return json.load(f)


def select_best_result(results: list[dict], key: str = "fitness") -> dict:
    if not results:
        raise ValueError("Results JSON is empty.")
    if key not in results[0]:
        raise ValueError(f"Key '{key}' not found in result entries.")
    return max(results, key=lambda x: x[key])


def inject_template(
    template_text: str,
    *,
    net_name: str,
    train_epochs: int,
    num_classes: int,
    peft_config: dict,
    fitness: float,
) -> str:
    # json.dumps works well here because:
    # - strings become valid Python string literals
    # - dicts become valid Python dict literals
    # - ints remain ints
    replacements = {
        "{NET_NAME_INPUT}": json.dumps(net_name),
        "{TRAIN_EPOCHS_INPUT}": str(train_epochs),
        "{NUM_CLASSES_INPUT}": str(num_classes),
        "{PEFT_CONFIG_INPUT}": json.dumps(peft_config, indent=4),
        "{FITNESS_INPUT}": str(round(fitness, 4)),
    }

    out = template_text
    for placeholder, value in replacements.items():
        out = out.replace(placeholder, value)

    return out


def build_program_from_best_result(
    results_json_path: str,
    template_py_path: str,
    output_py_path: str,
    num_classes: int,
    score_key: str = "fitness",
):
    results = load_results(results_json_path)
    best = select_best_result(results, key=score_key)

    template_text = Path(template_py_path).read_text()

    rendered = inject_template(
        template_text,
        net_name=best["net_name"],
        train_epochs=best["train_epochs"],
        num_classes=num_classes,
        peft_config=best["peft_config"],
        fitness=round(best["fitness"], 4),
    )

    Path(output_py_path).write_text(rendered)

    print("Wrote:", output_py_path)
    print("Best result selected:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    for dataset in ["retino"]:
        if dataset == "clevr_count":
            num_classes = 8
        elif dataset == "dtd":
            num_classes = 47
        elif dataset == "resisc45":
            num_classes = 45
        elif dataset == "kitti":
            num_classes = 4
        elif dataset == "sun397":
            num_classes = 397
        elif dataset == "retino":
            num_classes = 5
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        for results_json_path in Path(f"setup/warmstart/{dataset}").rglob(f"*.json"):    
            build_program_from_best_result(
                results_json_path=str(results_json_path),
                template_py_path="setup/warmstart/warm_start_template.py",
                output_py_path=str(results_json_path.parent / "warm_start_program.py"),
                num_classes=num_classes,
                score_key="fitness",
            )