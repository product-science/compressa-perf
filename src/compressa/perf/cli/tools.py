import sqlite3
from tabulate import tabulate
from typing import Dict, List, Tuple, Union
import json

import pandas as pd

from compressa.perf.experiment.inference import ExperimentRunner
from compressa.perf.experiment.analysis import Analyzer
from compressa.perf.data.models import Experiment
from compressa.perf.db.operations import (
    fetch_metrics_by_experiment,
    fetch_parameters_by_experiment,
    fetch_experiment_by_id,
    fetch_all_experiments,
    clear_metrics_by_experiment,
)
from compressa.perf.db.db_inserts import direct_insert_experiment as insert_experiment
from compressa.perf.db.setup import (
    create_tables,
    start_db_writer,
    stop_db_writer,
    get_db_writer,
)
import datetime
import sys
import random
import string
from compressa.perf.experiment.config import (
    load_yaml_configs,
)

from compressa.perf.experiment.continuous_stress import ContinuousStressTestRunner

from compressa.utils import get_logger
import requests
import subprocess
import shlex
import time
import re
import sys
import os

DEFAULT_DB_PATH = "compressa-perf-db.sqlite"


def _wait_for_db_writer(db_writer, context: str) -> None:
    if db_writer is None:
        return
    if not db_writer.wait_for_write():
        raise TimeoutError(f"Timed out waiting for DB writer after {context}")


def _count_experiment_rows(conn: sqlite3.Connection, table: str, experiment_id: int) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()[0]


def _create_testnet_account(
    account_name: str,
    seed_url: str,
    inferenced_path: str = "./inferenced"
) -> Tuple[str, str]:
    """
    Create a testnet account and export its private key.
    
    Args:
        account_name: Name for the account
        seed_url: Seed node URL for testnet
        inferenced_path: Path to the inferenced binary
        
    Returns:
        Tuple of (account_address, private_key_hex)
        
    Raises:
        ValueError: If account creation fails or required parameters are missing
    """
    if not account_name:
        raise ValueError("account_name is required for testnet account creation")
    if not seed_url:
        raise ValueError("seed_url is required for testnet account creation")
    
    # Validate seed_url format
    if not seed_url.startswith(('http://', 'https://')):
        raise ValueError(f"seed_url must start with http:// or https://, got: {seed_url}")
    
    # Determine inferenced binary path
    if os.path.exists(inferenced_path):
        inferenced_bin = inferenced_path
    else:
        # Try to find inferenced in PATH
        try:
            subprocess.run(["which", "inferenced"], check=True, capture_output=True)
            inferenced_bin = "inferenced"
        except subprocess.CalledProcessError:
            raise ValueError(f"inferenced binary not found at '{inferenced_path}' or in PATH")
    
    print(f"[Testnet] Using inferenced binary: {inferenced_bin}")
    print(f"[Testnet] Using seed URL: {seed_url}")
    print(f"[Testnet] Creating account '{account_name}'...")
    
    # Step 1: Create client - use array format to avoid shell quoting issues
    try:
        client_cmd = ["sh", "-c", f'echo "y" | {shlex.quote(inferenced_bin)} create-client {shlex.quote(account_name)} --node-address {shlex.quote(seed_url)}']
        print(f"[Testnet] Running command: {' '.join(client_cmd)}")
        client_output = subprocess.check_output(client_cmd, text=True, stderr=subprocess.STDOUT)
        print(client_output)
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to create client. Command failed with exit code {e.returncode}.\n"
        error_msg += f"Output: {e.output if hasattr(e, 'output') else str(e)}\n"
        error_msg += f"Seed URL used: {seed_url}\n"
        error_msg += f"Account name: {account_name}"
        raise ValueError(error_msg)
    
    # Extract account address from output
    match = re.search(r'(gonka[a-z0-9]{39})', client_output)
    account_address = match.group(1) if match else None
    if not account_address:
        raise ValueError(f"Could not extract account address from client creation output:\n{client_output}")
    
    print(f"[Testnet] Account created: {account_address}")
    
    # Step 2: Export private key
    print("[Testnet] Exporting private key...")
    try:
        key_cmd = ["sh", "-c", f'echo "y" | {shlex.quote(inferenced_bin)} keys export {shlex.quote(account_name)} --unarmored-hex --unsafe']
        print(f"[Testnet] Running command: {' '.join(key_cmd)}")
        private_key_hex = subprocess.check_output(key_cmd, text=True, stderr=subprocess.STDOUT).strip()
        if not private_key_hex:
            raise ValueError("Private key export returned empty result")
        print("[Testnet] Private key exported successfully")
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to export private key. Command failed with exit code {e.returncode}.\n"
        error_msg += f"Output: {e.output if hasattr(e, 'output') else str(e)}\n"
        error_msg += f"Account name: {account_name}"
        raise ValueError(error_msg)
    
    # Brief pause to ensure account is ready
    print("[Testnet] Waiting for account to be ready...")
    time.sleep(5)
    
    return account_address, private_key_hex


def _create_testnet_account_pool(
    account_pool_size: int,
    base_account_name: str,
    seed_url: str,
    inferenced_path: str = "./inferenced"
) -> List[Tuple[str, str]]:
    """
    Create a pool of testnet accounts for rotation.
    
    Args:
        account_pool_size: Number of accounts to create
        base_account_name: Base name for accounts (will be suffixed with numbers)
        seed_url: Seed node URL for testnet
        inferenced_path: Path to the inferenced binary
        
    Returns:
        List of (account_address, private_key_hex) tuples
        
    Raises:
        ValueError: If account creation fails
    """
    if account_pool_size <= 0:
        raise ValueError("account_pool_size must be greater than 0")
    
    print(f"[Account Pool] Creating {account_pool_size} accounts for rotation...")
    accounts = []
    
    for i in range(account_pool_size):
        account_name = f"{base_account_name}_{i:03d}"  # e.g. "stress_000", "stress_001"
        try:
            account_address, private_key_hex = _create_testnet_account(
                account_name=account_name,
                seed_url=seed_url,
                inferenced_path=inferenced_path
            )
            accounts.append((account_address, private_key_hex))
            print(f"[Account Pool] Created account {i+1}/{account_pool_size}: {account_address}")
            
            # Wait after the last account to ensure all accounts are ready
            if i == account_pool_size - 1:  # Sleep only after the last account
                print("[Account Pool] Waiting for all accounts to be ready...")
                time.sleep(5)
                
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Failed to create account {account_name}: {e}")
            # Continue trying to create other accounts
            continue
    
    if not accounts:
        raise ValueError("Failed to create any accounts")
    
    print(f"[Account Pool] Successfully created {len(accounts)} accounts")
    return accounts


logger = get_logger(__name__)
DEFAULT_API_TOKEN_ENV_VAR = "GONKA_API_TOKEN"


def normalize_node_url(node_url: str) -> str:
    if not node_url:
        return node_url

    normalized = node_url.strip()
    if not normalized:
        return normalized

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", normalized):
        normalized = f"http://{normalized}"

    return normalized.rstrip("/")


def resolve_api_token(
    api_token: str = None,
    api_token_env_var: str = None,
) -> Tuple[str, str]:
    env_var_name = api_token_env_var or DEFAULT_API_TOKEN_ENV_VAR
    resolved_api_token = api_token if api_token is not None else os.getenv(env_var_name)
    return resolved_api_token, env_var_name


def format_value(value, precision=4):
    try:
        numeric_value = float(value)
        if numeric_value.is_integer():
            return f"{int(numeric_value):<{precision}}"
        elif numeric_value < 0.01:
            return f"{numeric_value:.{precision}e}"
        else:
            return f"{numeric_value:.{precision}f}"
    except ValueError:
        return str(value)


def ensure_db_initialized(conn):
    try:
        # Check if the Experiments table exists
        conn.execute("SELECT 1 FROM Experiments LIMIT 1")
    except sqlite3.OperationalError:
        # If the table doesn't exist, create the tables
        print("Database not initialized. Creating tables...")
        create_tables(conn)
        print("Tables created successfully.")


def generate_random_text(
    length: int,
    choice_generator: random.Random,
):
    date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    words = [date_string]
    current_length = len(date_string)
    while current_length < length:
        word_length = choice_generator.randint(1, 20)
        word = ''.join(choice_generator.choice(string.ascii_lowercase) for _ in range(word_length))
        words.append(word)
        current_length += len(word) + 1

    words.append(". Repeat this text at least 10 times. Number the repetitions.")
    return ' '.join(words)[:length]

def generate_prompts_list(
    num_prompts: int,
    prompt_length: int,
    seed: int = 42,
):
    logger.info(f"Generating {num_prompts} prompts with length {prompt_length} and seed {seed}")
    choice_generator = random.Random(seed)
    prompts = []
    for i in range(num_prompts):
        random_text = generate_random_text(prompt_length - len(str(i)) - 1, choice_generator)
        prompt = f"{i} {random_text}"
        prompts.append(prompt)
    return prompts

ChatMessage = Dict[str, str]
PromptInput = Union[str, List[ChatMessage]]


def _normalize_chat_messages(messages) -> List[ChatMessage]:
    normalized_messages = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"Chat message must be an object, got: {type(message).__name__}")

        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            raise ValueError("Each chat message must contain both 'role' and 'content'")

        normalized_messages.append(
            {
                "role": str(role),
                "content": str(content),
            }
        )

    if not normalized_messages:
        raise ValueError("Chat prompt cannot be empty")

    return normalized_messages


def _truncate_chat_messages(messages: List[ChatMessage], prompt_length: int) -> List[ChatMessage]:
    if not prompt_length or prompt_length <= 0:
        return messages

    total_length = sum(len(message["content"]) for message in messages)
    if total_length <= prompt_length:
        return messages

    truncated_messages = []
    remaining_length = prompt_length

    # Keep the newest turns first. This mirrors how long chat contexts are usually
    # trimmed in production systems when they approach the context window.
    for message in reversed(messages):
        if remaining_length <= 0:
            break

        content = message["content"]
        if len(content) <= remaining_length:
            truncated_messages.append(message)
            remaining_length -= len(content)
            continue

        truncated_messages.append(
            {
                "role": message["role"],
                "content": content[-remaining_length:],
            }
        )
        remaining_length = 0

    return list(reversed(truncated_messages))


def _read_structured_prompts(records, prompt_length: int) -> List[PromptInput]:
    prompts = []
    for index, record in enumerate(records, start=1):
        if isinstance(record, list):
            messages = record
        elif isinstance(record, dict):
            if "messages" in record:
                messages = record["messages"]
            elif "conversation" in record:
                messages = record["conversation"]
            else:
                raise ValueError(
                    f"Structured prompt record #{index} must contain 'messages' or 'conversation'"
                )
        else:
            raise ValueError(
                f"Structured prompt record #{index} must be a JSON object or array, got {type(record).__name__}"
            )

        normalized_messages = _normalize_chat_messages(messages)
        prompts.append(_truncate_chat_messages(normalized_messages, prompt_length))

    return prompts


def read_prompts_from_file(file_path, prompt_length):
    lower_path = file_path.lower()
    if lower_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]
        return _read_structured_prompts(records, prompt_length)

    if lower_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            records = [records]
        return _read_structured_prompts(records, prompt_length)

    df = pd.read_csv(file_path, header=None)
    return df[0].map(lambda x: x[:prompt_length]).tolist()

def run_experiment(
    db: str = DEFAULT_DB_PATH,
    node_url: str = None,
    model_name: str = None,
    account_address: str = None,
    private_key_hex: str = None,
    api_token: str = None,
    api_token_env_var: str = None,
    experiment_name: str = None,
    description: str = None,
    prompts_file: str = None,
    num_tasks: int = 100,
    num_runners: int = 10,
    generate_prompts: bool = False,
    num_prompts: int = 100,
    prompt_length: int = 100,
    max_tokens: int = 1000,
    min_tokens: int = None,
    seed: int = 42,
    no_sign: bool = False,
    old_sign: bool = False,
    transport: str = "requests",
    create_account_testnet: bool = False,
    account_name: str = None,
    inferenced_path: str = "./inferenced",
    **kwargs
):
    node_url = normalize_node_url(node_url)
    if create_account_testnet:
        account_name = account_name or "testnetuser"
        account_address, private_key_hex = _create_testnet_account(
            account_name=account_name,
            seed_url=node_url,
            inferenced_path=inferenced_path
        )
    api_token, _ = resolve_api_token(
        api_token=api_token,
        api_token_env_var=api_token_env_var,
    )
    if not node_url:
        raise ValueError("node_url is not set")
    if not no_sign and not api_token:
        if not account_address:
            raise ValueError("account_address is not set (required when --no-sign is not used)")
        if not private_key_hex:
            raise ValueError("private_key_hex is not set (required when --no-sign is not used)")

    with sqlite3.connect(db) as conn:
        create_tables(conn)
        start_db_writer(db)
        db_writer = get_db_writer()

        experiment_runner = ExperimentRunner(
            node_url=node_url,
            model_name=model_name,
            account_address=account_address,
            private_key_hex=private_key_hex,
            api_token=api_token,
            num_runners=num_runners,
            no_sign=no_sign,
            old_sign=old_sign,
            transport=transport,
        )

        experiment = Experiment(
            id=None,
            experiment_name=experiment_name,
            experiment_date=datetime.datetime.now(),
            description=description,
        )
        experiment.id = insert_experiment(conn, experiment)
        print(f"Experiment created: {experiment}")

        if generate_prompts:
            prompts = generate_prompts_list(num_prompts, prompt_length, seed)
        else:
            prompts = read_prompts_from_file(prompts_file, prompt_length)

        logger.info(f"Num of prompts: {len(prompts)}\nNum of tasks: {num_tasks}\nNum of runners: {num_runners}\nMax tokens: {max_tokens}")

        experiment_runner.run_experiment(
            experiment_id=experiment.id,
            prompts=prompts,
            num_tasks=num_tasks,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            seed=seed,
        )

        _wait_for_db_writer(db_writer, "experiment measurements")
        analyzer = Analyzer(conn)
        analyzer.compute_metrics(experiment.id)
        _wait_for_db_writer(db_writer, "metric computation")

        report_experiment(
            experiment_id=experiment.id,
            db=db,
            recompute=False
        )
        stop_db_writer()


def report_experiment(
    experiment_id: int,
    db: str = DEFAULT_DB_PATH,
    recompute: bool = False,
):
    with sqlite3.connect(db) as conn:
        ensure_db_initialized(conn)
        db_writer = None

        experiment = fetch_experiment_by_id(conn, experiment_id)
        if not experiment:
            print(f"Error: Experiment with ID {experiment_id} not found.")
            sys.exit(1)

        analyzer = Analyzer(conn)

        if recompute:
            start_db_writer(db)
            db_writer = get_db_writer()
            clear_metrics_by_experiment(conn, experiment_id)
            analyzer.compute_metrics(experiment_id)
            _wait_for_db_writer(db_writer, "metric recomputation")
            same_conn_metric_count = _count_experiment_rows(conn, "Metrics", experiment_id)
            same_conn_parameter_count = _count_experiment_rows(conn, "Parameters", experiment_id)
            with sqlite3.connect(db) as verify_conn:
                fresh_conn_metric_count = _count_experiment_rows(verify_conn, "Metrics", experiment_id)
                fresh_conn_parameter_count = _count_experiment_rows(verify_conn, "Parameters", experiment_id)
            logger.info(
                "Post-recompute row counts for experiment_id=%s: same_conn metrics=%s parameters=%s; fresh_conn metrics=%s parameters=%s",
                experiment_id,
                same_conn_metric_count,
                same_conn_parameter_count,
                fresh_conn_metric_count,
                fresh_conn_parameter_count,
            )

        parameters = fetch_parameters_by_experiment(conn, experiment_id)
        metrics = fetch_metrics_by_experiment(conn, experiment_id)
        logger.info(
            "Fetched report rows for experiment_id=%s: parameters=%s metrics=%s",
            experiment_id,
            len(parameters),
            len(metrics),
        )

        print(f"\nExperiment Details:")
        print(f"ID: {experiment.id}")
        print(f"Name: {experiment.experiment_name}")
        print(f"Date: {experiment.experiment_date}")
        print(f"Description: {experiment.description}")

        param_table = [[p.key, format_value(p.value)] for p in parameters]
        print("\nExperiment Parameters:")
        print(tabulate(
            param_table, 
            headers=["Parameter", "Value"], 
            tablefmt="fancy_grid", 
            stralign="right",
            numalign="right",
        ))

        # Prepare metrics table with better formatting
        metrics_table = [[m.metric_name, format_value(m.metric_value)] for m in metrics]
        print("\nExperiment Metrics:")
        print(tabulate(
            metrics_table, 
            headers=["Metric", "Value"], 
            tablefmt="fancy_grid", 
            numalign="decimal",
        ))
        if db_writer is not None:
            stop_db_writer()


def list_experiments(
    db: str = DEFAULT_DB_PATH,
    show_parameters: bool = False,
    show_metrics: bool = False,
    name_filter: str = None,
    param_filters: str = None,
    recompute: bool = False,
    csv_file: str = None,
):
    with sqlite3.connect(db) as conn:
        ensure_db_initialized(conn)

        experiments = fetch_all_experiments(conn)
        if recompute:
            start_db_writer(db)
            analyzer = Analyzer(conn)
            for exp in experiments:
                try:
                    clear_metrics_by_experiment(conn, exp.id)
                    analyzer.compute_metrics(exp.id)
                except Exception as e:
                    logger.error(f"Error computing metrics for experiment {exp.id}: {e}")
                finally:
                    logger.info(f"Metrics computed for experiment {exp.id}")

        if name_filter:
            experiments = [exp for exp in experiments if name_filter in exp.experiment_name]

        if param_filters:
            for param_filter in param_filters:
                param_key, _, param_value_substring = param_filter.partition('=')
                experiments = [exp for exp in experiments if any(
                    p.key == param_key and param_value_substring in format_value(p.value)
                    for p in fetch_parameters_by_experiment(conn, exp.id)
                )]

        if not experiments:
            print("No experiments found in the database.")
            return

        _print_experiments_cli(
            experiments=experiments,
            conn=conn,
            show_parameters=show_parameters,
            show_metrics=show_metrics,
        )
        if csv_file is not None:
            _export_experiments_csv(
                experiments=experiments,
                conn=conn,
                csv_file=csv_file,
            )


def _print_experiments_cli(
    experiments: List[Experiment],
    conn: sqlite3.Connection,
    show_parameters: bool = False,
    show_metrics: bool = False,
):

    table_data = []
    headers = ["ID", "Name", "Date", "Description"]

    if show_parameters:
        headers.extend(["Parameters"])

    desciptiont_length = 20 if show_parameters or show_metrics else 50
    for exp in experiments:
        row = [
            exp.id,
            exp.experiment_name,
            exp.experiment_date.strftime("%Y-%m-%d %H:%M:%S") if exp.experiment_date else "N/A",
            exp.description[:desciptiont_length] + "..." if exp.description and len(exp.description) > desciptiont_length else exp.description
        ]

        if show_parameters:
            parameters = fetch_parameters_by_experiment(conn, exp.id)
            param_str = "\n".join([
                f"{p.key}: {format_value(p.value, precision=2)[:10] + '...' if len(format_value(p.value, precision=2)) > 10 else format_value(p.value, precision=2)}" 
                for p in parameters
            ])
            row.append(param_str)

        if show_metrics:
            metrics = fetch_metrics_by_experiment(conn, exp.id)
            metrics_str = "\n".join([f"{m.metric_name}: {format_value(m.metric_value)}" for m in metrics])
            row.append(metrics_str)

        table_data.append(row)

    print("\nList of Experiments:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def _export_experiments_csv(
    experiments: List[Experiment],
    conn: sqlite3.Connection,
    csv_file: str,
):
    table_data = []
    metric_columns = set()

    for exp in experiments:
        item = {}
        item["id"] = exp.id
        item["name"] = exp.experiment_name
        item["date"] = exp.experiment_date.strftime("%Y-%m-%d %H:%M:%S") if exp.experiment_date else "N/A"
        item["description"] = exp.description
        item["parameters"] = {}

        parameters = fetch_parameters_by_experiment(conn, exp.id)
        for p in parameters:
            item["parameters"][p.key] = format_value(p.value, precision=2)

        metrics = fetch_metrics_by_experiment(conn, exp.id)
        for m in metrics:
            metric_column = f"M_{m.metric_name}"
            item[metric_column] = format_value(m.metric_value)
            metric_columns.add(metric_column)

        table_data.append(item)

    df = pd.DataFrame(table_data)
    df = df.reindex(columns=["id", "name", "date", "description", "parameters"] + list(metric_columns), fill_value=None)
    df.to_csv(csv_file, index=False)

def run_experiments_from_yaml(
    yaml_file: str,
    db: str = DEFAULT_DB_PATH,
    node_url: str = None,
    account_address: str = None,
    private_key_hex: str = None,
    api_token: str = None,
    api_token_env_var: str = None,
    model_name: str = None,
    no_sign: bool = False,
    old_sign: bool = False,
    transport: str = None,
    create_account_testnet: bool = False,
    account_name: str = None,
    inferenced_path: str = "./inferenced",
    **kwargs
):
    effective_account_address = account_address
    effective_private_key_hex = private_key_hex
    effective_node_url = normalize_node_url(node_url) if node_url else None
    if create_account_testnet:
        account_name = account_name or "testnetuser"
        effective_account_address, effective_private_key_hex = _create_testnet_account(
            account_name=account_name,
            seed_url=effective_node_url,
            inferenced_path=inferenced_path
        )

    configs = load_yaml_configs(yaml_file)

    for config in configs:
        # Use command-line node_url if provided, otherwise use from config
        config_node_url = effective_node_url if effective_node_url else normalize_node_url(config.node_url)
        if not config_node_url:
            raise ValueError("node_url is not set (neither in command line nor in config file)")

        config_api_token_env_var = api_token_env_var or getattr(config, "api_token_env_var", None)
        config_api_token, _ = resolve_api_token(
            api_token=api_token,
            api_token_env_var=config_api_token_env_var,
        )

        # Use command-line account_address if provided, otherwise use from config
        config_account_address = effective_account_address if effective_account_address else config.account_address

        # Use command-line private_key_hex if provided, otherwise use from config
        config_private_key_hex = effective_private_key_hex if effective_private_key_hex else getattr(config, 'private_key_hex', None)
        if not no_sign and not config_api_token:
            if not config_account_address:
                raise ValueError("account_address is not set (required when bearer auth is unavailable and --no-sign is not used)")
            if not config_private_key_hex:
                raise ValueError("private_key_hex is not set (required when bearer auth is unavailable and --no-sign is not used)")

        # Use command-line model_name if provided, otherwise use from config
        config_model_name = model_name if model_name else config.model_name
        if not config_model_name:
            raise ValueError("model_name is not set (neither in command line nor in config file)")

        run_experiment(
            db=db,
            node_url=config_node_url,
            model_name=config_model_name,
            account_address=config_account_address,
            private_key_hex=config_private_key_hex,
            api_token=config_api_token,
            api_token_env_var=config_api_token_env_var,
            experiment_name=config.experiment_name,
            description=config.description,
            prompts_file=config.prompts_file,
            num_tasks=config.num_tasks,
            num_runners=config.num_runners,
            generate_prompts=config.generate_prompts,
            num_prompts=config.num_prompts,
            prompt_length=config.prompt_length,
            max_tokens=config.max_tokens,
            min_tokens=config.min_tokens,
            seed=config.seed,
            no_sign=no_sign,
            old_sign=old_sign,
            transport=transport or getattr(config, "transport", "requests"),
        )

    list_experiments(db=db)



def run_continuous_stress_test(
    db: str,
    node_url: str,
    model_name: str,
    account_address: str,
    private_key_hex: str,
    experiment_name: str,
    description: str,
    prompts_file: str,
    num_runners: int,
    generate_prompts: bool,
    num_prompts: int,
    prompt_length: int,
    max_tokens: int,
    min_tokens: int = None,
    report_freq_min: float = 1.0,
    no_sign: bool = False,
    old_sign: bool = False,
    transport: str = "requests",
    create_account_testnet: bool = False,
    account_name: str = None,
    inferenced_path: str = "./inferenced",
    account_pool_size: int = 1,
    api_token: str = None,
    api_token_env_var: str = None,
    **kwargs
):
    node_url = normalize_node_url(node_url)
    # Handle account creation/rotation
    account_pool = None
    if create_account_testnet:
        account_name = account_name or "stresstest"
        if account_pool_size > 1:
            # Create pool of accounts for random selection
            print(f"[Account Pool] Creating {account_pool_size} accounts for random selection")
            account_pool = _create_testnet_account_pool(
                account_pool_size=account_pool_size,
                base_account_name=account_name,
                seed_url=node_url,
                inferenced_path=inferenced_path
            )
            # Use the first account initially
            account_address, private_key_hex = account_pool[0]
        else:
            # Create single account (default behavior)
            account_address, private_key_hex = _create_testnet_account(
                account_name=account_name,
                seed_url=node_url,
                inferenced_path=inferenced_path
            )
    api_token, _ = resolve_api_token(
        api_token=api_token,
        api_token_env_var=api_token_env_var,
    )
    """
    Creates an Experiment, loads or generates prompts, and starts
    an infinite stress test that computes windowed metrics in real time.
    Uses optimized HTTP client for high-performance concurrent requests.
    """
    if not node_url:
        raise ValueError("node_url is not set")
    if not no_sign and not api_token:
        if not account_address:
            raise ValueError("account_address is not set (required when --no-sign is not used)")
        if not private_key_hex:
            raise ValueError("private_key_hex is not set (required when --no-sign is not used)")

    with sqlite3.connect(db) as conn:
        create_tables(conn)
        start_db_writer(db)
        db_writer = get_db_writer()
        experiment = Experiment(
            id=None,
            experiment_name=experiment_name,
            experiment_date=datetime.datetime.now(),
            description=description,
        )
        experiment.id = insert_experiment(conn, experiment)
        print(f"Continuous Stress Experiment created: {experiment}")

        if generate_prompts:
            prompts = generate_prompts_list(num_prompts, prompt_length)
        else:
            if not prompts_file:
                raise ValueError("You must provide --prompts_file or use --generate_prompts")
            prompts = read_prompts_from_file(prompts_file, prompt_length)

        logger.info(f"Number of prompts: {len(prompts)}")

        runner = ContinuousStressTestRunner(
            db_path=db,
            node_url=node_url,
            model_name=model_name,
            account_address=account_address,
            private_key_hex=private_key_hex,
            api_token=api_token,
            experiment_id=experiment.id,
            prompts=prompts,
            num_runners=num_runners,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            report_freq_min=report_freq_min,
            no_sign=no_sign,
            old_sign=old_sign,
            transport=transport,
            account_pool=account_pool,
        )
        runner.start_test()

        _wait_for_db_writer(db_writer, "continuous stress writes")
        stop_db_writer()


def check_balances(node_url: str):
    """Check balances of all participants in the network and print as a table with URLs, weight, models, balance, and address. Sorted by weight descending."""
    node_url = normalize_node_url(node_url)
    try:
        response = requests.get(f"{node_url}/v1/epochs/current/participants", timeout=10)
        data = response.json()
        participants = data.get('active_participants', {}).get('participants', [])
        
        if not participants:
            print("No active participants found.")
            return

        table = []
        for participant in participants:
            address = participant.get('index')
            inference_url = participant.get('inference_url', '-')
            models = participant.get('models', [])
            models_str = ', '.join(models) if models else '-'
            weight = participant.get('weight', 0)
            if address:
                participant_url = f"{node_url}/v1/participants/{address}"
                balance_response = requests.get(participant_url, timeout=10)
                balance_data = balance_response.json()
                balance = balance_data.get('balance', 0)
                table.append([inference_url, weight, models_str, balance, address])

        # Sort by weight descending
        table.sort(key=lambda x: x[1], reverse=True)

        print(f"\nNode URL: {node_url}\n")
        print(tabulate(
            table,
            headers=["Inference URL", "Weight", "Models", "Balance", "Address"],
            tablefmt="github"
        ))
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to node: {e}")
    except Exception as e:
        print(f"Error checking balances: {e}")



