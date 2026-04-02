# AgentSSL

This repo is scalable framework designed to easily and robustly evaluate **data efficiency** across multiple domains.  

We present a diverse set of domains, settings, and shots to comprehensively evaluate agent approaches on the semi-supervised learning task. 

---

## Directory Structure

```
AgentSSL/
├── metrics
├── pytorch_adapt
├── domain1/ # Ex. CUB or RESIC
│ ├── annotations/
│ │ ├── val/
│ │ ├── test/
| ├── setSL/ (supervised agent setting)
│ │ ├── build/ # folder to build the shot and seed folders
│ │ │ ├── evaluate.py # Setting specific evaluation function
│ │ │ ├── description.md # Setting specific task description
│ | │ └── build.py # File to build the rest of the folders
│ │ ├── k1/ # 1 shot 
│ │ │ ├── seed0/ # Random seed
│ │ │ │ ├── annotations/ # Annotation files
│ │ │ │ │ ├── train.json # train labels
│ │ │ │ │ ├── unlabeled.json # index unlabeled data
│ │ │ │ ├── src/ # API & dataset setup functions
│ │ │ │ ├── evaluate.py # Evaluation function
│ | │ │ └── description.md # Task description and notes
│ │ │ ├── seed26/ # Random seed
│ │ │ └── seed42/ # Random seed
| ├── setSSL/ (semi supervised agent setting)
| └── setC/ (setting)
├── domain2
├── domain3
├── domain4
└── Other domains...
```

## Data Storage (Not sure how this looks right now?)

## Apptainer Environment

Inside the container, the file system visible to the agent is organized as follows:
```
root_dir/
├── data/ # All dataset files, read only
├── work/ # Read-only
│ ├── metrics/ # Metric files accessible as "metrics.*"
│ ├── pytorch_adapt/ # pytorch_adapt library accessible as "pytorch_adapt.*"
│ ├── annotations/ # Task-specific training annotations
│ ├── src/ # Source files accessible as "src.*"
│ ├── evaluate.py # Evaluation script
│ └── description.md # Task description
├── run_tmp/ # Read-only
│ ├── program.py # PROGRAM AGENT WRITES!
└── tmp/ # Writable directory for outputs
```

## `evaluate.py`

The **evaluation script** is responsible for running the agent’s program on the dataset and producing a performance score. This file is run in the agent's isolated Apptainer environment. As a result, *all code should be run with respect to the Apptainer file structure, not the SciDUC structure.*

### Expected Behavior

1. **Input**  
   Accepts a `program path` (available as environment variable `PROGRAM_PATH` in Apptainer).

2. **Execution**  
   Evaluates the provided program on the current dataset subset (e.g., `k1`, `k5`, etc.).

3. **Output**  
   Print a dictionary with the optimization metric under the key `"fitness"`. The dictionary should come directly after a printed "METRICS: " string. 
   Example:
   ```python
   print("METRICS: ", {"fitness": 0.823, "others": {...}})
4. **Logging**  
   Print relevant information for debugging and monitoring — the agent can read stdout logs.

### Recommended Header Snippet

To ensure that src imports work correctly, add the following to the top of your evaluate.py file: (This might not be 100% necessary).
```python
import sys, os
# Add the Apptainer bind mount path to Python's module search
sys.path.insert(0, "/work")

print("✅ Added /work to sys.path")
print("Working dir:", os.getcwd())
```

## description.md

This file describes the task and any relevant context or debugging hints.
It should include:

1. The dataset’s purpose and structure

2. The evaluation goal (e.g., mAP, accuracy, IoU, etc.)

3. Notes on known issues (e.g., library incompatibilities)