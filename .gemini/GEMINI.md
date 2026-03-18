# Project Setup

## Tech Stack

- **Language/Runtime:** Python
- **Environment:** Conda

## Coding Standards & Rules

- **Strict Environment Enforcement:** ALL python, pip, or other project-related terminal commands MUST be executed within the `crypto` conda environment.
- **Terminal Execution:** Every single time you run a command via the `run_command` tool, you must explicitly use `conda run -n crypto <command>` or chain the activation `source $(conda info --base)/etc/profile.d/conda.sh && conda activate crypto && <command>`.
- **Do not** use the base Python environment, the system Python environment, or any arbitrary virtual environment. ALWAYS use the `crypto` conda environment for this project.
