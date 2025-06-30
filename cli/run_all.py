import asyncio
import os
import argparse
import time
from typing import List, Tuple, Dict, Any, Optional

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

from main import ARCTester
from arc_agi_benchmarking.utils.task_utils import read_models_config, read_provider_rate_limits
from arc_agi_benchmarking.utils.rate_limiter import AsyncRequestRateLimiter, AsyncSingleRequestLimiter, RateLimiter
from arc_agi_benchmarking.utils.metrics import set_metrics_enabled, set_metrics_filename_prefix

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log

logger = logging.getLogger(__name__)

# Attempt to import provider-specific exceptions for retrying
try:
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    AnthropicRateLimitError = None
    logger.warning("Anthropic SDK not installed or RateLimitError not found. Retries for Anthropic rate limits will not be specific.")

try:
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = None
    logger.warning("OpenAI SDK not installed or RateLimitError not found. Retries for OpenAI rate limits will not be specific.")

try:
    from google.api_core.exceptions import ResourceExhausted as GoogleResourceExhausted
except ImportError:
    GoogleResourceExhausted = None
    logger.warning("Google API Core SDK not installed or ResourceExhausted not found. Retries for Google rate limits will not be specific.")

_RETRYABLE_EXCEPTIONS_CLASSES = tuple(
    exc for exc in (AnthropicRateLimitError, OpenAIRateLimitError, GoogleResourceExhausted) if exc is not None
)

if not _RETRYABLE_EXCEPTIONS_CLASSES:
    logger.warning(
        "No specific retryable exception classes were successfully imported. "
        "Retries might not trigger as expected or might catch too broadly if fallback to general Exception is used."
    )
    EFFECTIVE_RETRYABLE_EXCEPTIONS = (Exception,)
else:
    EFFECTIVE_RETRYABLE_EXCEPTIONS = _RETRYABLE_EXCEPTIONS_CLASSES

# Default values
DEFAULT_RATE_LIMIT_RATE = 400
DEFAULT_RATE_LIMIT_PERIOD = 60

# --- Configuration ---
# Default model configurations to test if not provided via CLI.
# These are names from your models.yml file.
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

# --- Globals for Orchestrator ---
PROVIDER_RATE_LIMITERS: Dict[str, RateLimiter] = {}
MODEL_CONFIG_CACHE: Dict[str, Any] = {}

def get_model_config(config_name: str):
    if config_name not in MODEL_CONFIG_CACHE:
        MODEL_CONFIG_CACHE[config_name] = read_models_config(config_name)
    return MODEL_CONFIG_CACHE[config_name]

def get_or_create_rate_limiter(provider_name: str, all_provider_limits: Dict) -> RateLimiter:
    if provider_name not in PROVIDER_RATE_LIMITERS:
        if provider_name not in all_provider_limits:
            logger.warning(f"No rate limit configuration found for provider '{provider_name}' in provider_config.yml. Using default ({DEFAULT_RATE_LIMIT_RATE} req/{DEFAULT_RATE_LIMIT_PERIOD}s).")
            default_config_rate = DEFAULT_RATE_LIMIT_RATE
            default_config_period = DEFAULT_RATE_LIMIT_PERIOD
            actual_rate_for_limiter = default_config_rate / default_config_period
            actual_capacity_for_limiter = max(1.0, actual_rate_for_limiter)
        else:
            limits = all_provider_limits[provider_name]
            if 'sequential' in limits:
                logger.info(f"Provider '{provider_name}' is configured to be sequential. Using AsyncSingleRequestLimiter.")
                PROVIDER_RATE_LIMITERS[provider_name] = AsyncSingleRequestLimiter()
            else:
                config_rate = limits['rate']
                config_period = limits['period']
                if config_period <= 0:
                    actual_rate_for_limiter = float('inf')
                    actual_capacity_for_limiter = float('inf')
                    logger.warning(f"Provider '{provider_name}' has period <= 0 in config. Treating as unconstrained.")
                else:
                    calculated_rps = config_rate / config_period
                    actual_rate_for_limiter = calculated_rps
                    actual_capacity_for_limiter = max(1.0, calculated_rps)
                logger.info(f"Initializing rate limiter for provider '{provider_name}' with rate={actual_rate_for_limiter:.2f} req/s, capacity={actual_capacity_for_limiter:.2f}.")
                PROVIDER_RATE_LIMITERS[provider_name] = AsyncRequestRateLimiter(rate=actual_rate_for_limiter, capacity=actual_capacity_for_limiter)
    return PROVIDER_RATE_LIMITERS[provider_name]

async def run_single_test_wrapper(config_name: str, task_id: str, limiter: RateLimiter,
                                  data_dir: str, submissions_root: str, # Changed from save_submission_dir_base
                                  overwrite_submission: bool, print_submission: bool,
                                  num_attempts: int, retry_attempts: int) -> bool: # removed print_logs
    logger.info(f"[Orchestrator] Queuing task: {task_id}, config: {config_name}")

    # Apply tenacity retry decorator directly to the synchronous function
    # The logger passed to before_sleep_log is the module-level logger of cli.run_all
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(4),
        retry=retry_if_exception_type(EFFECTIVE_RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _synchronous_task_execution_attempt_with_tenacity():
        logger.debug(f"[Thread-{task_id}-{config_name}] Spawning ARCTester (Executing attempt)...")
        arc_solver = ARCTester(
            config=config_name,
            save_submission_dir=submissions_root,
            overwrite_submission=overwrite_submission,
            print_submission=print_submission, # This ARCTester arg controls if it logs submission content
            num_attempts=num_attempts,
            retry_attempts=retry_attempts # ARCTester's internal retries
            # print_logs removed from ARCTester instantiation
        )
        logger.debug(f"[Thread-{task_id}-{config_name}] Starting generate_task_solution...")
        arc_solver.generate_task_solution(
            data_dir=data_dir,
            task_id=task_id
        )
        logger.debug(f"[Thread-{task_id}-{config_name}] Task attempt completed successfully.")

    try:
        async with limiter:
            logger.info(f"[Orchestrator] Rate limiter acquired for: {config_name}. Executing task {task_id} with tenacity retries...")
            await asyncio.to_thread(_synchronous_task_execution_attempt_with_tenacity)
        
        logger.info(f"[Orchestrator] Successfully processed (with tenacity retries if any): {config_name} / {task_id}")
        return True
    except Exception as e:
        logger.error(f"[Orchestrator] Failed to process (after all tenacity retries or due to non-retryable error): {config_name} / {task_id}. Error: {type(e).__name__} - {e}", exc_info=True)
        return False

async def main(task_list_file: Optional[str], # Added Optional
               model_configs_to_test: List[str],
               data_dir: str, submissions_root: str,
               overwrite_submission: bool, print_submission: bool,
               num_attempts: int, retry_attempts: int) -> int: # Added return type hint
    # Basic logging setup is now done in if __name__ == "__main__"
    
    start_time = time.perf_counter()
    logger.info("Starting ARC Test Orchestrator...")
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

    all_jobs_to_run: List[Tuple[str, str]] = []
    for config_name in model_configs_to_test:
        for task_id in task_ids:
            all_jobs_to_run.append((config_name, task_id))
    
    if not all_jobs_to_run:
        logger.warning("No jobs to run (check model_configs_to_test and task list). Exiting.")
        return 1 # Return an error code
    
    logger.info(f"Total jobs to process: {len(all_jobs_to_run)}")

    try:
        all_provider_limits = read_provider_rate_limits()
        logger.info(f"Loaded rate limits from provider_config.yml for providers: {list(all_provider_limits.keys())}")
    except FileNotFoundError:
        logger.warning("provider_config.yml not found. Using default rate limits (400 req/60s per provider).")
        all_provider_limits = {}
    except Exception as e:
        logger.warning(f"Error reading or parsing provider_config.yml: {e}. Using default rate limits.")
        all_provider_limits = {}

    async_tasks_to_execute = []
    for config_name, task_id in all_jobs_to_run:
        try:
            model_config_obj = get_model_config(config_name)
            provider_name = model_config_obj.provider
            limiter = get_or_create_rate_limiter(provider_name, all_provider_limits)
            async_tasks_to_execute.append(run_single_test_wrapper(
                config_name, task_id, limiter,
                data_dir, submissions_root,
                overwrite_submission, print_submission, 
                num_attempts, retry_attempts
            ))
        except ValueError as e: # Specific error for model config issues
            logger.error(f"Skipping config '{config_name}' for task '{task_id}' due to model config error: {e}")
        except Exception as e: # General error for other setup issues
            logger.error(f"Unexpected error setting up task for '{config_name}', '{task_id}': {e}", exc_info=True)

    if not async_tasks_to_execute:
        logger.warning("No tasks could be prepared for execution. Exiting.")
        return 1 # Return an error code

    logger.info(f"Executing {len(async_tasks_to_execute)} tasks concurrently...")
    results = await asyncio.gather(*async_tasks_to_execute, return_exceptions=True)

    successful_runs = sum(1 for r in results if r is True)
    orchestrator_level_failures = sum(1 for r in results if r is False or isinstance(r, Exception))

    logger.info("--- Orchestrator Summary ---")
    exit_code = 0 # Default to success
    if orchestrator_level_failures == 0:
        logger.info(f"✨ All {successful_runs} test configurations completed successfully by the orchestrator.")
    else:
        logger.error(f"💥 {orchestrator_level_failures} out of {len(results)} test configurations failed or encountered errors during orchestration.")
        for i, res in enumerate(results):
            original_job_config, original_job_task_id = all_jobs_to_run[i] # Get original job details
            if isinstance(res, Exception):
                logger.error(f"  - Error for {original_job_config}/{original_job_task_id}: {type(res).__name__} - {str(res)}", exc_info=True)
            elif res is False: # Wrapper reported failure
                logger.warning(f"  - Failure reported by wrapper for {original_job_config}/{original_job_task_id} (check ARCTester logs for this task/config)")
        exit_code = 1 # Indicate failure
    
    logger.info("Note: Individual task success/failure is logged by ARCTester within its own logger (main.py's logger).")
    logger.info("Orchestrator failure indicates an issue with running the ARCTester task itself or an unhandled exception in the wrapper.")
    
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    logger.info("--- Orchestrator Timing ---")
    logger.info(f"Total execution time for cli/run_all.py: {total_duration:.2f} seconds")
    
    return exit_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARC tasks concurrently. Tasks can be specified via a task list file or inferred from a data directory.")
    parser.add_argument(
        "--task_list_file", 
        type=str, 
        default=None, # Default to None, indicating it's optional
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
        "--submissions-root", # Renamed from --save_submission_dir_base
        type=str,
        default=DEFAULT_SUBMISSIONS_ROOT,
        help=f"Root folder name to save submissions under. Subfolders per config will be created. Defaults to {DEFAULT_SUBMISSIONS_ROOT}"
    )
    parser.add_argument(
        "--overwrite_submission",
        action="store_true", # Defaults to False if not present
        help=f"Overwrite submissions if they already exist. Defaults to {DEFAULT_OVERWRITE_SUBMISSION}"
    )
    parser.add_argument(
        "--print_submission", # This flag is for ARCTester to log submission content
        action="store_true", # Defaults to False if not present
        help=f"Enable ARCTester to log final submission content (at INFO level). Defaults to {DEFAULT_PRINT_SUBMISSION}"
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=DEFAULT_NUM_ATTEMPTS,
        help=f"Number of attempts for each prediction by ARCTester. Defaults to {DEFAULT_NUM_ATTEMPTS}"
    )
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help=f"Number of internal retry attempts by ARCTester for failed predictions. Defaults to {DEFAULT_RETRY_ATTEMPTS}"
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
        action="store_true", # Defaults to False if not present
        help="Enable metrics collection and dumping (disabled by default)."
    )

    args = parser.parse_args()

    # Set default for boolean flags if not provided
    # No explicit default= needed for store_true/store_false in parser.add_argument if you rely on their inherent behavior
    # However, to be absolutely clear with our DEFAULT_ values, we can check args against them if needed,
    # but typically argparse handles this based on action. For `store_true`, if the flag is present, it's True, else False.
    # Our DEFAULT_OVERWRITE_SUBMISSION = False and DEFAULT_PRINT_SUBMISSION = False align with this.
    # args.enable_metrics will also be False by default.

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
        model_configs_list = DEFAULT_MODEL_CONFIGS_TO_TEST # Fallback to default
        logger.info(f"No model_configs provided or empty, using default: {model_configs_list}")

    # --- Set metrics filename prefix based on the model config(s) being run --- 
    if args.enable_metrics:
        config_identifier = model_configs_list[0] if len(model_configs_list) == 1 else f"{len(model_configs_list)}_configs"
        provider_name = "unknown_provider"
        try:
            if model_configs_list: # Ensure there's at least one config
                first_config_obj = get_model_config(model_configs_list[0])
                provider_name = first_config_obj.provider
        except Exception: 
            logger.warning(f"Could not determine provider for metrics filename from config: {model_configs_list[0] if model_configs_list else 'N/A'}")
        
        prefix = f"{provider_name}_{config_identifier}"
        set_metrics_filename_prefix(prefix)
        logger.info(f"Metrics enabled. Filename prefix set to: {prefix}")
    # ----------------------------------------------------------------------------

    # Ensure `main` returns an exit code which is then used by sys.exit
    exit_code_from_main = asyncio.run(main(
        task_list_file=args.task_list_file, # Pass the arg here
        model_configs_to_test=model_configs_list,
        data_dir=args.data_dir,
        submissions_root=args.submissions_root,
        overwrite_submission=args.overwrite_submission,
        print_submission=args.print_submission,
        num_attempts=args.num_attempts,
        retry_attempts=args.retry_attempts
    ))
    
    sys.exit(exit_code_from_main) 