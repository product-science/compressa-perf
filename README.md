# Compressa Performance Measurement Tool

This tool is designed to measure the performance of Compressa models.  
It uses the OpenAI API to run inference tasks and stores the results in a SQLite database.

## Installation

```bash
git clone https://github.com/compressa-ai/compressa-perf.git
cd compressa-perf
poetry install
```

## Install with Pip

```bash
pip install compressa-perf
```

## Usage

### 1. Run experiment with prompts from a file

```bash
❯ compressa-perf measure \
    --node_url http://example.node.url:8545 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --account_address 0x1234567890abcdef1234567890abcdef12345678 \
    --private_key_hex 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef \
    --experiment_name "File Prompts Run" \
    --prompts_file resources/prompts.csv \
    --num_tasks 1000 \
    --num_runners 100
```

### 2. Run experiment with generated prompts

```bash
❯ compressa-perf measure \
    --node_url http://example.node.url:8545 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --account_address 0x1234567890abcdef1234567890abcdef12345678 \
    --private_key_hex 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef \
    --experiment_name "Generated Prompts Run" \
    --num_tasks 2 \
    --num_runners 2 \
    --generate_prompts \
    --num_prompts 1000 \
    --prompt_length 5000
```

Full parameter list can be obtained with `compressa-perf measure -h`.

### 3. Run set of experiments from YAML file

You can describe set of experiments in YAML file and run them on different services in one command:

```bash
❯ compressa-perf measure-from-yaml \
    --private_key_hex 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef \
    config.yml
```

You can also override values from the YAML file using command-line options:

```bash
# Override node_url
❯ compressa-perf measure-from-yaml \
    --node_url http://override.node.url:8545 \
    --private_key_hex 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef \
    config.yml

# Override account_address
❯ compressa-perf measure-from-yaml \
    --account_address 0x9876543210abcdef9876543210abcdef98765432 \
    --private_key_hex 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef \
    config.yml

# Override model_name
❯ compressa-perf measure-from-yaml \
    --model_name Qwen/Qwen2.5-14B-Instruct \
    --private_key_hex 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef \
    config.yml
```

Example of YAML file:

```yaml
- model_name: Qwen/Qwen2.5-7B-Instruct
  experiment_name: "File Prompts Run 1"
  description: "Experiment using prompts from a file with 10 tasks and 5 runners"
  prompts_file: resources/prompts.csv
  num_tasks: 10
  num_runners: 5
  generate_prompts: false
  num_prompts: 0
  prompt_length: 0
  max_tokens: 1000
  node_url: "http://example.node.url:8545"
  account_address: "0x1234567890abcdef1234567890abcdef12345678"

- model_name: Qwen/Qwen2.5-7B-Instruct
  experiment_name: "Qwen2-7B Long Input / Short Output"
  description: "Experiment using prompts from a file with 20 tasks and 10 runners"
  prompts_file: resources/prompts.csv
  num_tasks: 20
  num_runners: 10
  generate_prompts: true
  num_prompts: 10
  prompt_length: 10000
  max_tokens: 100
  node_url: "http://example.node.url:8545"
  account_address: "0x1234567890abcdef1234567890abcdef12345678"
```

### 4. List experiments

You can select experiments by name, parameters or metrics (or substrings in these fields) via `compressa-perf list` command.

For example:
```
❯ compressa-perf list \
    --show-metrics \
    --param-filter node_url=example \
    --param-filter avg_n_input=30

List of Experiments:
+----+----------------------------------------------------------------------------+---------------------+--------+-----------------------+
|    | ID                                                                         | Name                | Date   | Description           |
+====+============================================================================+=====================+========+=======================+
| 25 | Compressa-Qwen2.5-14B-Instruct-Int4 Long Input / Short Output | 5 runners  | 2024-10-03 06:21:45 |        | ttft: 25.0960         |
|    |                                                                            |                     |        | latency: 52.5916      |
|    |                                                                            |                     |        | tpot: 0.5530          |
|    |                                                                            |                     |        | throughput: 2891.0323 |
+----+----------------------------------------------------------------------------+---------------------+--------+-----------------------+
| 23 | Compressa-Qwen2.5-14B-Instruct-Int4 Long Input / Short Output | 4 runners  | 2024-10-03 06:14:57 |        | ttft: 17.1862         |
|    |                                                                            |                     |        | latency: 37.9612      |
|    |                                                                            |                     |        | tpot: 0.3954          |
|    |                                                                            |                     |        | throughput: 3230.8769 |
+----+----------------------------------------------------------------------------+---------------------+--------+-----------------------+
```

Full parameter list:
```bash
❯ compressa-perf list -h
usage: compressa-perf list [-h] [--db DB] [--show-parameters] [--show-metrics] [--name-filter NAME_FILTER] [--param-filter PARAM_FILTER]

options:
  -h, --help            show this help message and exit
  --db DB               Path to the SQLite database
  --show-parameters     Show all parameters for each experiment
  --show-metrics        Show metrics for each experiment
  --name-filter NAME_FILTER
                        Filter experiments by substring in the name
  --param-filter PARAM_FILTER
                        Filter experiments by parameter value (e.g., paramkey=value_substring)
```


### 5. Generate a report for an experiment

```bash
❯ compressa-perf report <EXPERIMENT_ID>
```

Output example:

```
❯ compressa-perf report 3

Experiment Details:
ID: 3
Name: My First Run
Date: 2024-09-24 07:10:39
Description: None

Experiment Parameters:
╒══════════════╤═══════════════════════════════════════════╕
│    Parameter │                                     Value │
╞══════════════╪═══════════════════════════════════════════╡
│  num_workers │                                         2 │
├──────────────┼───────────────────────────────────────────┤
│    num_tasks │                                         2 │
├──────────────┼───────────────────────────────────────────┤
│    node_url  │ http://example.node.url:8545          │
├──────────────┼───────────────────────────────────────────┤
│  avg_n_input │                                        32 │
├──────────────┼───────────────────────────────────────────┤
│  std_n_input │                                    2.8284 │
├──────────────┼───────────────────────────────────────────┤
│ avg_n_output │                                  748.5000 │
├──────────────┼───────────────────────────────────────────┤
│ std_n_output │                                    2.1213 │
╘══════════════╧═══════════════════════════════════════════╛

Experiment Metrics:
╒═══════════════════════╤══════════╕
│ Metric                │    Value │
╞═══════════════════════╪══════════╡
│ MetricName.TTFT       │   0.7753 │
├───────────────────────┼──────────┤
│ MetricName.LATENCY    │   7.5016 │
├───────────────────────┼──────────┤
│ MetricName.TPOT       │   0.01   │
├───────────────────────┼──────────┤
│ MetricName.THROUGHPUT │ 207.84   │
╘═══════════════════════╧══════════╛
```

For more information on available commands and options, run:

```bash
compressa-perf --help
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
