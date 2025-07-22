import argparse
import signal
import sys
from typing import List
from compressa.perf.cli.tools import (
    run_experiment,
    report_experiment,
    list_experiments,
    run_experiments_from_yaml,
    run_continuous_stress_test,
    check_balances,
    DEFAULT_DB_PATH,
)
from compressa.perf.db.setup import (
    stop_db_writer,
    get_db_writer,
)


def handle_stop_signals(signum, frame):
    print(f"Received signal {signum}, stopping DB writer...")
    db_writer = get_db_writer()
    db_writer.wait_for_write()
    stop_db_writer()
    sys.exit(0)


def run_experiment_args(args):
    # node_url is always required - either for direct connection or testnet account creation
    if not args.node_url:
        raise ValueError("--node_url is required")
    
    run_experiment(
        db=args.db,
        node_url=args.node_url,
        model_name=args.model_name,
        account_address=args.account_address,
        private_key_hex=args.private_key_hex,
        experiment_name=args.experiment_name,
        description=args.description,
        prompts_file=args.prompts_file,
        num_tasks=args.num_tasks,
        num_runners=args.num_runners,
        generate_prompts=args.generate_prompts,
        num_prompts=args.num_prompts,
        prompt_length=args.prompt_length,
        max_tokens=args.max_tokens,
        no_sign=args.no_sign,
        old_sign=args.old_sign,
        create_account_testnet=args.create_account_testnet,
        account_name=args.account_name,
        inferenced_path=args.inferenced_path
    )


def report_experiment_args(args):
    report_experiment(
        experiment_id=args.experiment_id,
        db=args.db,
        recompute=args.recompute
    )


def list_experiments_args(args):
    list_experiments(
        db=args.db,
        show_parameters=args.show_parameters,
        show_metrics=args.show_metrics,
        name_filter=args.name_filter,
        param_filters=args.param_filter,
        recompute=args.recompute,
        csv_file=args.csv_file,
    )


def run_experiments_from_yaml_args(args):
    run_experiments_from_yaml(
        yaml_file=args.yaml_file,
        db=args.db,
        node_url=args.node_url,
        account_address=args.account_address,
        private_key_hex=args.private_key_hex,
        model_name=args.model_name,
        no_sign=args.no_sign,
        old_sign=args.old_sign,
        create_account_testnet=args.create_account_testnet,
        account_name=args.account_name,
        inferenced_path=args.inferenced_path
    )


def run_continuous_stress_test_args(args):
    # node_url is always required - either for direct connection or testnet account creation
    if not args.node_url:
        raise ValueError("--node_url is required")
    
    run_continuous_stress_test(
        db=args.db,
        node_url=args.node_url,
        model_name=args.model_name,
        account_address=args.account_address,
        private_key_hex=args.private_key_hex,
        experiment_name=args.experiment_name,
        description=args.description,
        prompts_file=args.prompts_file,
        num_runners=args.num_runners,
        generate_prompts=args.generate_prompts,
        num_prompts=args.num_prompts,
        prompt_length=args.prompt_length,
        max_tokens=args.max_tokens,
        report_freq_min=args.report_freq_min,
        no_sign=args.no_sign,
        old_sign=args.old_sign,
        create_account_testnet=args.create_account_testnet,
        account_name=args.account_name,
        inferenced_path=args.inferenced_path,
        account_pool_size=args.account_pool_size
    )


def check_balances_args(args):
    check_balances(
        node_url=args.node_url,
    )



def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for running and analyzing experiments",
        epilog="""
IMPORTANT EXAMPLES (most common use cases):

1. Quick testnet experiment with generated prompts (RECOMMENDED FOR TESTING):
    compressa-perf measure \\
        --node-url http://testnet.node.url:8545 \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --experiment-name "My Test Run" \\
        --create-account-testnet \\
        --generate-prompts \\
        --num-tasks 10 \\
        --num-runners 2

2. Continuous stress test with account rotation (RECOMMENDED FOR LOAD TESTING):
    compressa-perf stress \\
        --node-url http://testnet.node.url:8545 \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --experiment-name "Stress Test" \\
        --create-account-testnet \\
        --account-pool-size 5 \\
        --num-runners 20 \\
        --generate-prompts \\
        --report-freq-min 1

3. Run multiple experiments from YAML config:
    compressa-perf measure-from-yaml \\
        --node-url http://testnet.node.url:8545 \\
        --create-account-testnet \\
        config.yml

4. Check network participant balances:
    compressa-perf check-balances --node-url http://testnet.node.url:8545

OTHER EXAMPLES:

5. Production experiment with existing credentials:
    compressa-perf measure \\
        --seed-url http://production.node.url:8545 \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --experiment-name "Production Test" \\
        --account-address 0x1234... \\
        --private-key-hex 0xabcd... \\
        --prompts-file prompts.csv \\
        --num-tasks 1000

6. List experiments and generate reports:
    compressa-perf list --show-metrics
    compressa-perf report <EXPERIMENT_ID>
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')

    parser_run = subparsers.add_parser(
        "measure",
        help="Run an experiment",
    )
    # Core required parameters (most important)
    parser_run.add_argument(
        "--node_url", "--node-url", "--seed_url", "--seed-url", 
        type=str, required=True, dest="node_url",
        help="Node/seed URL for blockchain connection and testnet account creation"
    )
    parser_run.add_argument(
        "--model_name", "--model-name", 
        type=str, required=True, dest="model_name",
        help="Model name (e.g., Qwen/Qwen2.5-7B-Instruct)"
    )
    parser_run.add_argument(
        "--experiment_name", "--experiment-name", 
        type=str, required=True, dest="experiment_name",
        help="Name of the experiment"
    )
    
    # Authentication options (high priority)
    parser_run.add_argument(
        "--account_address", "--account-address", 
        type=str, required=False, dest="account_address",
        help="Account address (not required if using --create-account-testnet)"
    )
    parser_run.add_argument(
        "--private_key_hex", "--private-key-hex", 
        type=str, required=False, dest="private_key_hex",
        help="Private key hex (not required if using --create-account-testnet or --no-sign)"
    )
    parser_run.add_argument(
        "--create-account-testnet", "--create_account_testnet", 
        action="store_true", dest="create_account_testnet",
        help="[TESTNET] Automatically create account and export key using node URL (recommended for testing)"
    )
    parser_run.add_argument(
        "--account-name", "--account_name", 
        type=str, required=False, dest="account_name",
        help="Account name for testnet account creation (optional, defaults to 'testnetuser')"
    )
    parser_run.add_argument(
        "--no-sign", "--no_sign", 
        action="store_true", dest="no_sign",
        help="Send requests without signing (for testing unsigned mode)"
    )
    
    # Experiment configuration
    parser_run.add_argument(
        "--num_tasks", "--num-tasks", 
        type=int, default=100, dest="num_tasks",
        help="Number of requests to send (default: 100)"
    )
    parser_run.add_argument(
        "--num_runners", "--num-runners", 
        type=int, default=10, dest="num_runners",
        help="Number of concurrent runners (default: 10)"
    )
    
    # Advanced options (lower priority)
    parser_run.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database (default: compressa-perf-db.sqlite)",
    )
    parser_run.add_argument(
        "--description", 
        type=str, 
        help="Description of the experiment"
    )
    parser_run.add_argument(
        "--old-sign", "--old_sign", 
        action="store_true", dest="old_sign",
        help="Use legacy signing method for backward compatibility"
    )
    parser_run.add_argument(
        "--inferenced-path", "--inferenced_path", 
        type=str, default="./inferenced", dest="inferenced_path",
        help="Path to the inferenced binary (default: ./inferenced, fallback: inferenced in PATH)"
    )

    # Prompt configuration (important for experiment setup)
    parser_run.add_argument(
        "--prompts_file", "--prompts-file", 
        type=str, dest="prompts_file",
        help="Path to file containing prompts (alternative to --generate-prompts)"
    )
    parser_run.add_argument(
        "--generate_prompts", "--generate-prompts", 
        action="store_true", dest="generate_prompts",
        help="Generate random prompts instead of using a file (recommended for testing)"
    )
    parser_run.add_argument(
        "--num_prompts", "--num-prompts", 
        type=int, default=100, dest="num_prompts",
        help="Number of prompts to generate (if using --generate-prompts, default: 100)"
    )
    parser_run.add_argument(
        "--prompt_length", "--prompt-length", 
        type=int, default=100, dest="prompt_length",
        help="Length of each generated prompt (if using --generate-prompts, default: 100)"
    )
    parser_run.add_argument(
        "--max_tokens", "--max-tokens", 
        type=int, default=1000, dest="max_tokens",
        help="Maximum tokens for model to generate (default: 1000)"
    )
    parser_run.set_defaults(func=run_experiment_args)

    parser_report = subparsers.add_parser(
        "report",
        help="Generate a report for an experiment",
    )
    parser_report.add_argument(
        "experiment_id", type=int, help="ID of the experiment to report on"
    )
    parser_report.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database",
    )
    parser_report.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute metrics before generating the report",
    )
    parser_report.set_defaults(func=report_experiment_args)

    parser_list = subparsers.add_parser(
        "list",
        help="List all experiments",
    )
    parser_list.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database",
    )
    parser_list.add_argument(
        "--show-parameters",
        action="store_true",
        help="Show all parameters for each experiment"
    )
    parser_list.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show metrics for each experiment"
    )
    parser_list.add_argument(
        "--name-filter",
        type=str,
        help="Filter experiments by substring in the name"
    )
    parser_list.add_argument(
        "--param-filter",
        type=str,
        action="append",
        help="Filter experiments by parameter value (e.g., paramkey=value_substring)"
    )
    parser_list.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute metrics before listing experiments"
    )
    parser_list.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Path to the CSV file to save the experiments"
    )
    parser_list.set_defaults(func=list_experiments_args)

    parser_yaml = subparsers.add_parser(
        "measure-from-yaml",
        help="Run experiments from a YAML configuration file",
    )
    parser_yaml.add_argument(
        "yaml_file",
        help="YAML configuration file for experiments",
    )
    # Override parameters (most important)
    parser_yaml.add_argument(
        "--node_url", "--node-url", "--seed_url", "--seed-url",
        type=str, required=False, dest="node_url",
        help="Node/seed URL (overrides value in YAML config if provided)",
    )
    parser_yaml.add_argument(
        "--model_name", "--model-name",
        type=str, required=False, dest="model_name",
        help="Model name (overrides value in YAML config if provided)",
    )
    parser_yaml.add_argument(
        "--account_address", "--account-address",
        type=str, required=False, dest="account_address",
        help="Account address (overrides value in YAML config if provided)",
    )
    parser_yaml.add_argument(
        "--private_key_hex", "--private-key-hex",
        type=str, required=False, dest="private_key_hex",
        help="Private key hex (required if not using --create-account-testnet or --no-sign)",
    )
    
    # Authentication options
    parser_yaml.add_argument(
        "--create-account-testnet", "--create_account_testnet", 
        action="store_true", dest="create_account_testnet",
        help="[TESTNET] Automatically create account and export key using node URL (recommended for testing)"
    )
    parser_yaml.add_argument(
        "--account-name", "--account_name", 
        type=str, required=False, dest="account_name",
        help="Account name for testnet account creation (optional, defaults to 'testnetuser')"
    )
    parser_yaml.add_argument(
        "--no-sign", "--no_sign", 
        action="store_true", dest="no_sign",
        help="Send requests without signing (for testing unsigned mode)"
    )
    
    # Advanced options (lower priority)
    parser_yaml.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database (default: compressa-perf-db.sqlite)",
    )
    parser_yaml.add_argument(
        "--old-sign", "--old_sign", 
        action="store_true", dest="old_sign",
        help="Use legacy signing method for backward compatibility"
    )
    parser_yaml.add_argument(
        "--inferenced-path", "--inferenced_path", 
        type=str, default="./inferenced", dest="inferenced_path",
        help="Path to the inferenced binary (default: ./inferenced, fallback: inferenced in PATH)"
    )

    parser_yaml.set_defaults(func=run_experiments_from_yaml_args)

    parser_stress = subparsers.add_parser(
        "stress",
        help="Run a continuous stress test (infinite requests, windowed metrics).",
    )
    # Core required parameters (most important)
    parser_stress.add_argument(
        "--node_url", "--node-url", "--seed_url", "--seed-url", 
        type=str, required=True, dest="node_url",
        help="Node/seed URL for blockchain connection and testnet account creation"
    )
    parser_stress.add_argument(
        "--model_name", "--model-name", 
        type=str, required=True, dest="model_name",
        help="Model name (e.g., Qwen/Qwen2.5-7B-Instruct)"
    )
    parser_stress.add_argument(
        "--experiment_name", "--experiment-name", 
        type=str, required=True, dest="experiment_name",
        help="Name of the stress test experiment"
    )
    
    # Stress test specific (high priority)
    parser_stress.add_argument(
        "--num_runners", "--num-runners", 
        type=int, default=10, dest="num_runners",
        help="Number of concurrent runners for stress test (default: 10)"
    )
    parser_stress.add_argument(
        "--report_freq_min", "--report-freq-min", 
        type=float, default=1, dest="report_freq_min",
        help="Report frequency in minutes for windowed metrics (default: 1)"
    )
    parser_stress.add_argument(
        "--account-pool-size", "--account_pool_size", 
        type=int, default=1, dest="account_pool_size",
        help="[ADVANCED] Number of accounts for random selection to avoid rate limiting (default: 1, set >1 to enable rotation)"
    )
    
    # Authentication options (high priority)
    parser_stress.add_argument(
        "--account_address", "--account-address", 
        type=str, required=False, dest="account_address",
        help="Account address (not required if using --create-account-testnet)"
    )
    parser_stress.add_argument(
        "--private_key_hex", "--private-key-hex", 
        type=str, required=False, dest="private_key_hex",
        help="Private key hex (not required if using --create-account-testnet or --no-sign)"
    )
    parser_stress.add_argument(
        "--create-account-testnet", "--create_account_testnet", 
        action="store_true", dest="create_account_testnet",
        help="[TESTNET] Automatically create account(s) and export key using node URL (recommended for testing)"
    )
    parser_stress.add_argument(
        "--account-name", "--account_name", 
        type=str, required=False, dest="account_name",
        help="Account name for testnet account creation (optional, defaults to 'stresstest')"
    )
    parser_stress.add_argument(
        "--no-sign", "--no_sign", 
        action="store_true", dest="no_sign",
        help="Send requests without signing (for testing unsigned mode)"
    )
    
    # Prompt configuration
    parser_stress.add_argument(
        "--prompts_file", "--prompts-file", 
        type=str, dest="prompts_file",
        help="File containing prompts (alternative to --generate-prompts)"
    )
    parser_stress.add_argument(
        "--generate_prompts", "--generate-prompts", 
        action="store_true", dest="generate_prompts",
        help="Generate random prompts instead of using a file (recommended for testing)"
    )
    parser_stress.add_argument(
        "--num_prompts", "--num-prompts", 
        type=int, default=100, dest="num_prompts",
        help="Number of prompts to generate (if using --generate-prompts, default: 100)"
    )
    parser_stress.add_argument(
        "--prompt_length", "--prompt-length", 
        type=int, default=100, dest="prompt_length",
        help="Length of each generated prompt (if using --generate-prompts, default: 100)"
    )
    parser_stress.add_argument(
        "--max_tokens", "--max-tokens", 
        type=int, default=1000, dest="max_tokens",
        help="Maximum tokens for generation (default: 1000)"
    )
    
    # Advanced options (lower priority)
    parser_stress.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database (default: compressa-perf-db.sqlite)",
    )
    parser_stress.add_argument(
        "--description", 
        type=str, 
        help="Description of the stress test experiment"
    )
    parser_stress.add_argument(
        "--old-sign", "--old_sign", 
        action="store_true", dest="old_sign",
        help="Use legacy signing method for backward compatibility"
    )
    parser_stress.add_argument(
        "--inferenced-path", "--inferenced_path", 
        type=str, default="./inferenced", dest="inferenced_path",
        help="Path to the inferenced binary (default: ./inferenced, fallback: inferenced in PATH)"
    )
    parser_stress.set_defaults(func=run_continuous_stress_test_args)

    parser_balances = subparsers.add_parser(
        "check-balances",
        help="Check the balance of the account on the specified node.",
    )
    parser_balances.add_argument(
        "--node_url", "--node-url", "--seed_url", "--seed-url",
        type=str, required=True, dest="node_url",
        help="Node/seed URL to check participant balances on",
    )
    parser_balances.set_defaults(func=check_balances_args)



    def default_function(args):
        parser.print_help()

    parser.set_defaults(func=default_function)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
