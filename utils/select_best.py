import json

def select_best(results_path, metric='fitness'):
    with open(results_path, "r") as f:
        results = json.load(f)

    best_result = max(results, key=lambda x: x[metric])
    return best_result


if __name__ == "__main__":
    results_path = "/home/eyl45/Sun/AgentSSL/dtd/aSSL_backbone_1804/k3/seed0/results.json"
    best_result = select_best(results_path)
    print(best_result)