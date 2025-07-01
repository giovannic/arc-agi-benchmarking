import os
import argparse
import time
from typing import List, Optional

import sys
import logging

# Add the project root directory and the src directory to sys.path
# This allows cli/run_all.py to import 'main' and 'src' from the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root) # For importing main.py

if src_dir not in sys.path:
    sys.path.insert(0, src_dir) # For importing arc_agi_benchmarking from src

from main import BatchedARCTester
from arc_agi_benchmarking.utils.metrics import set_metrics_enabled, set_metrics_filename_prefix

logger = logging.getLogger(__name__)

# Default values
DEFAULT_MODEL_CONFIGS_TO_TEST: List[str] = [
    "gpt-4o-2024-11-20",
]

DEFAULT_DATA_DIR = "data/arc-agi/data/evaluation"
DEFAULT_SUBMISSIONS_ROOT = "submissions" # Changed from DEFAULT_SAVE_SUBMISSION_DIR_BASE
DEFAULT_OVERWRITE_SUBMISSION = False
DEFAULT_PRINT_SUBMISSION = False # ARCTester specific: whether it logs submission content
DEFAULT_NUM_ATTEMPTS = 2
DEFAULT_RETRY_ATTEMPTS = 2
# DEFAULT_PRINT_LOGS = False # This is now controlled by the global log level

def main(task_list_file: Optional[str],
         model_configs_to_test: List[str],
         data_dir: str, submissions_root: str,
         overwrite_submission: bool, print_submission: bool,
         num_attempts: int) -> int:
    
    start_time = time.perf_counter()
    logger.info("Starting Batched ARC Test Orchestrator...")
    logger.info(f"Testing with model configurations: {model_configs_to_test}")

    task_ids: List[str] = []
    try:
        if task_list_file:
            logger.info(f"Using task list file: {task_list_file}")
            with open(task_list_file, 'r') as f:
                task_ids = [line.strip() for line in f if line.strip()]
            if not task_ids:
                logger.error(f"No task IDs found in {task_list_file}. Exiting.")
                return 1 # Return an error code
            logger.info(f"Loaded {len(task_ids)} task IDs from {task_list_file}.")
        else:
            logger.info(f"No task list file provided. Inferring task list from data directory: {data_dir}")
            task_ids = [
                os.path.splitext(fname)[0] 
                for fname in os.listdir(data_dir) 
                if os.path.isfile(os.path.join(data_dir, fname)) and fname.endswith('.json')
            ]
            if not task_ids:
                logger.error(f"No task files (.json) found in {data_dir}. Exiting.")
                return 1 # Return an error code
            logger.info(f"Found {len(task_ids)} task IDs in {data_dir}.")

    except FileNotFoundError:
        if task_list_file:
            logger.error(f"Task list file not found: {task_list_file}. Exiting.")
        else: # Should not happen if data_dir is validated by argparse, but as a safeguard
            logger.error(f"Data directory not found: {data_dir}. Exiting.")
        return 1 # Return an error code
    except Exception as e:
        logger.error(f"Error loading tasks: {e}", exc_info=True)
        return 1 # Return an error code

    if not task_ids:
        logger.warning("No tasks to run. Exiting.")
        return 1
    
    logger.info(f"Total tasks to process: {len(task_ids)}")

    # Process each model config
    for config_name in model_configs_to_test:
        try:
            logger.info(f"Processing tasks with config: {config_name}")
            arc_solver = BatchedARCTester(
                config=config_name,
                save_submission_dir=submissions_root,
                overwrite_submission=overwrite_submission,
                print_submission=print_submission,
                num_attempts=num_attempts,
                retry_attempts=1  # Not used in BatchedARCTester but required by BaseARCTester
            )
            
            arc_solver.generate_batched_task_solutions(
                data_dir=data_dir,
                task_ids=task_ids
            )
            
            logger.info(f"Successfully processed all tasks with config: {config_name}")
            
        except Exception as e:
            logger.error(f"Failed to process tasks with config '{config_name}': {e}", exc_info=True)
            return 1

    end_time = time.perf_counter()
    total_duration = end_time - start_time
    logger.info("--- Orchestrator Summary ---")
    logger.info(f"✨ All tasks completed successfully.")
    logger.info(f"Total execution time: {total_duration:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARC tasks in batch using BatchedARCTester.")
    parser.add_argument(
        "--task_list_file", 
        type=str, 
        default=None,
        required=False,
        help="Optional path to a .txt file containing task IDs, one per line. If not provided, tasks are inferred from all .json files in --data_dir."
    )
    parser.add_argument(
        "--model_configs",
        type=str,
        default=",".join(DEFAULT_MODEL_CONFIGS_TO_TEST),
        help=f"Comma-separated list of model configuration names to test (from models.yml). Defaults to: {','.join(DEFAULT_MODEL_CONFIGS_TO_TEST)}"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Data set directory to run. If --task_list_file is not used, .json task files are inferred from here. Defaults to {DEFAULT_DATA_DIR}"
    )
    parser.add_argument(
        "--submissions-root",
        type=str,
        default=DEFAULT_SUBMISSIONS_ROOT,
        help=f"Root folder name to save submissions under. Subfolders per config will be created. Defaults to {DEFAULT_SUBMISSIONS_ROOT}"
    )
    parser.add_argument(
        "--overwrite_submission",
        action="store_true",
        help=f"Overwrite submissions if they already exist. Defaults to {DEFAULT_OVERWRITE_SUBMISSION}"
    )
    parser.add_argument(
        "--print_submission",
        action="store_true",
        help=f"Enable ARCTester to log final submission content (at INFO level). Defaults to {DEFAULT_PRINT_SUBMISSION}"
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=DEFAULT_NUM_ATTEMPTS,
        help=f"Number of attempts for each prediction by ARCTester. Defaults to {DEFAULT_NUM_ATTEMPTS}"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"],
        help="Set the logging level for the orchestrator and ARCTester (default: INFO). Use NONE to disable logging."
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        help="Enable metrics collection and dumping (disabled by default)."
    )

    args = parser.parse_args()

    # Set metrics enabled status based on CLI arg
    set_metrics_enabled(args.enable_metrics)

    # Configure logging for the entire application based on --log-level
    if args.log_level == "NONE":
        log_level_to_set = logging.CRITICAL + 1 
        logging.basicConfig(
            level=log_level_to_set,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)] 
        )
    else:
        log_level_to_set = getattr(logging, args.log_level.upper())
        logging.basicConfig(
            level=log_level_to_set,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )

    model_configs_list = [m.strip() for m in args.model_configs.split(',') if m.strip()]
    if not model_configs_list: 
        model_configs_list = DEFAULT_MODEL_CONFIGS_TO_TEST
        logger.info(f"No model_configs provided or empty, using default: {model_configs_list}")

    # Set metrics filename prefix based on the model config(s) being run
    if args.enable_metrics:
        config_identifier = model_configs_list[0] if len(model_configs_list) == 1 else f"{len(model_configs_list)}_configs"
        prefix = f"batched_{config_identifier}"
        set_metrics_filename_prefix(prefix)
        logger.info(f"Metrics enabled. Filename prefix set to: {prefix}")

    # Run the main function
    exit_code = main(
        task_list_file=args.task_list_file,
        model_configs_to_test=model_configs_list,
        data_dir=args.data_dir,
        submissions_root=args.submissions_root,
        overwrite_submission=args.overwrite_submission,
        print_submission=args.print_submission,
        num_attempts=args.num_attempts
    )
    
    sys.exit(exit_code) 
