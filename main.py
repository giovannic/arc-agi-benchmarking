import sys
import os
from typing import Tuple

# Added: Add the src directory to sys.path to allow direct execution of main.py
# This assumes main.py is in the project root and 'src' is a subdirectory.
_project_root = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import json
from arc_agi_benchmarking.adapters import (
    ProviderAdapter,
    AnthropicAdapter,
    OpenAIAdapter,
    DeepseekAdapter,
    GeminiAdapter,
    HuggingFaceFireworksAdapter,
    FireworksAdapter,
    GrokAdapter,
    OpenRouterAdapter,
    LocalLlamaAdapter
)
from dotenv import load_dotenv
import arc_agi_benchmarking.utils as utils
from arc_agi_benchmarking.utils.metrics import timeit, set_metrics_enabled
from arc_agi_benchmarking.schemas import ARCPair, Attempt
from arc_agi_benchmarking.prompts.prompt_manager import convert_task_pairs_to_prompt
from typing import List, Optional
import argparse
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

load_dotenv()

class BaseARCTester(ABC):
    def __init__(self, config: str, save_submission_dir: str, overwrite_submission: bool, print_submission: bool, num_attempts: int, retry_attempts: int):
        self.config = config
        self.model_config = utils.read_models_config(config)
        self.provider = self.init_provider(self.model_config.provider)
        self.save_submission_dir = save_submission_dir
        self.overwrite_submission = overwrite_submission
        self.print_submission = print_submission
        self.num_attempts = num_attempts
        self.retry_attempts = retry_attempts

    def init_provider(self, provider_name: str) -> ProviderAdapter:
        if provider_name == "anthropic":
            return AnthropicAdapter(self.config)
        elif provider_name == "openai":
            return OpenAIAdapter(self.config)
        elif provider_name == "deepseek":
            return DeepseekAdapter(self.config)
        elif provider_name == "gemini":
            return GeminiAdapter(self.config)
        elif provider_name == "huggingfacefireworks":
            return HuggingFaceFireworksAdapter(self.config)
        elif provider_name == "fireworks":
            return FireworksAdapter(self.config)
        elif provider_name == "grok":
            return GrokAdapter(self.config)
        elif provider_name == "openrouter":
            return OpenRouterAdapter(self.config)
        elif provider_name == "localllama":
            return LocalLlamaAdapter(self.config)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

class ARCTester(BaseARCTester):
    def predict_task_output(self, training_pairs: List[ARCPair], test_input: ARCPair, task_id: str, test_id: str, pair_index: int):
        prompt = convert_task_pairs_to_prompt(training_pairs, test_input)
        logger.info(f"Making prediction for task {task_id}, test {test_id}, pair_index {pair_index}")
        response: Attempt = self.provider.make_prediction(prompt, task_id=task_id, test_id=test_id, pair_index=pair_index)
        return response

    def get_task_prediction(self, training_pairs: List[ARCPair], test_input: ARCPair, task_id: str, test_id: str, pair_index: int) -> Attempt:
        attempt: Attempt = self.predict_task_output(training_pairs, test_input, task_id, test_id, pair_index)
        try:
            if isinstance(attempt.answer, str):
                parsed = self.provider.extract_json_from_response(attempt.answer)
                if parsed is None:
                    raise ValueError("Failed to parse answer")
                attempt.answer = parsed
            return attempt
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                f"Parsing/Validation failed for task {task_id}, test {test_id}, pair_index {pair_index}: {e}",
                exc_info=True,
            )
            raise

    @timeit
    def generate_task_solution(self, data_dir, task_id):
        """
        Generates and saves the solution for a specific ARC task.

        Args:
            data_dir: The directory containing the ARC task data (e.g., 'data/arc-agi/data/evaluation').
            task_id: The ID of the specific task to solve (e.g., '0a1d4ef5').

        Instance attributes used:
            self.config: The model configuration name being used.
            self.num_attempts: The number of prediction attempts per test pair.
            self.retry_attempts: The number of internal retries if a prediction attempt fails.
            self.save_submission_dir: Directory to save the final submission JSON.
            self.overwrite_submission: Whether to overwrite existing submission files.
            self.print_submission: Whether to log the final submission JSON content.
            self.provider: The initialized provider adapter.

        Returns:
            A list representing the submission structure if successful and saving is enabled,
            or None if no valid predictions were made or saving is disabled but run completes.
            Returns None immediately if submission exists and overwrite is False.
        """
        
        logger.info(f"Running task {task_id} with config {self.config}")
        utils.validate_data(data_dir, task_id)

        # Use the config name as the test_id
        test_id = self.config
        
        logger.info(f"Using model_config: {test_id} for task_id: {task_id}")

        # Logic for overwrite. If save_submission_dir is provided, check if the submission already exists
        if self.save_submission_dir and utils.submission_exists(self.save_submission_dir, task_id) and not self.overwrite_submission:
            logger.info(f"Submission for task {task_id} using {test_id} already exists, skipping")
            return
        
        task_attempts = []

        train_pairs = utils.get_train_pairs_from_task(data_dir, task_id)
        test_input_pairs = utils.get_test_input_from_task(data_dir, task_id)

        # Go through each test pair to get a prediction. 96% of challenges have 1 pair.
        for t, pair_input_obj in enumerate(test_input_pairs):
            pair_index = t
            logger.info(f"Starting task {task_id}, ModelConfig: {test_id}, Test Pair Index: {pair_index+1}/{len(test_input_pairs)}")
            
            pair_submission_attempts = {}

            # Run through each prediction attempt
            for attempt_num in range(1, self.num_attempts + 1):
                attempt_key = f"attempt_{attempt_num}"
                pair_submission_attempts[attempt_key] = None

                for retry_num in range(self.retry_attempts):
                    try:
                        logger.debug(f"    Task {task_id}, ModelConfig {test_id}, Pair {pair_index+1}, Predicting attempt #{attempt_num}, retry #{retry_num + 1}")
                        # Now storing the full attempt object with task_id and test_id
                        attempt_obj = self.get_task_prediction(
                            training_pairs=train_pairs,
                            test_input=pair_input_obj,
                            task_id=task_id,
                            test_id=test_id,
                            pair_index=pair_index
                        )

                        if attempt_obj is not None:
                            logger.debug(f"    Task {task_id}, ModelConfig {test_id}, Pair {pair_index+1}, Attempt #{attempt_num} successful. Prediction: {attempt_obj.answer}")
                            pair_submission_attempts[attempt_key] = attempt_obj.model_dump(mode='json')
                            break 
                    except Exception as e:
                        logger.warning(f"    Task {task_id}, ModelConfig {test_id}, Pair {pair_index+1}, Attempt #{attempt_num}, Retry #{retry_num + 1} failed. Error: {e}")

                    if retry_num == self.retry_attempts - 1:
                        logger.warning(f"    Task {task_id}, ModelConfig {test_id}, Pair {pair_index+1}, All {self.retry_attempts} retries failed for attempt #{attempt_num}")

            # Only append non-None attempts for this pair
            if any(v is not None for v in pair_submission_attempts.values()):
                task_attempts.append(pair_submission_attempts)

        if task_attempts:
            if self.print_submission:
                # Log the submission content; use json.dumps for potentially large structures
                logger.info(f"Final submission for task {task_id}, ModelConfig {test_id}:\n{json.dumps(task_attempts, indent=4)}")
            if self.save_submission_dir:
                utils.save_submission(self.save_submission_dir, task_id, task_attempts)
                logger.info(f"Submission for task {task_id}, ModelConfig {test_id} saved to {self.save_submission_dir}")
        else:
            logger.warning(f"No valid predictions for task {task_id}, ModelConfig {test_id} after all attempts. Skipping submission.")

        return task_attempts if task_attempts else None

class BatchedARCTester(BaseARCTester):

    def predict_batched_task_output(
        self,
        training_pairs: List[List[ARCPair]],
        test_inputs_per_task: List[List[ARCPair]],
        task_ids: List[str],
        test_ids: List[str],
        ) -> List[Attempt]:
        results: List[Tuple[str, str, str, int]] = [
            (
                convert_task_pairs_to_prompt(tp, ti), 
                task_id,
                test_id,
                pair_index
            )
            for tp, test_inputs, task_id, test_id
            in zip(
                training_pairs,
                test_inputs_per_task,
                task_ids,
                test_ids
            )
            for pair_index, ti in enumerate(test_inputs)
        ]
        prompts, task_ids, test_ids, pair_indices = zip(*results) # type: ignore
        if not hasattr(self.provider, "make_batch_prediction"):
            raise ValueError("Provider does not support batch prediction")
        response: List[Attempt] = self.provider.make_batch_prediction( # type: ignore
            prompts,
            task_ids=task_ids,
            test_ids=test_ids,
            pair_indices=pair_indices
        )
        return response


    def get_batched_task_prediction(
        self,
        training_pairs: List[List[ARCPair]],
        test_inputs: List[List[ARCPair]],
        task_ids: List[str],
        test_ids: List[str],
        ) -> List[Attempt]:
        attempts: List[Attempt] = self.predict_batched_task_output(
            training_pairs,
            test_inputs,
            task_ids,
            test_ids
        )
        for i, attempt in enumerate(attempts):
            try:
                if isinstance(attempt.answer, str):
                    parsed = self.provider.extract_json_from_response(attempt.answer)
                    if parsed is None:
                        raise ValueError(
                            f"Failed to parse answer for task {task_ids[i]}, test {test_ids[i]}, pair_index {pair_indices[i]}")
                    attempt.answer = parsed
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(
                    f"Parsing/Validation failed for task {task_ids[i]}, test {test_ids[i]}, pair_index {pair_indices[i]}: {e}",
                    exc_info=True,
                )
        return attempts

    def generate_batched_task_solutions(self, data_dir: str, task_ids: List[str]):
        for task_id in task_ids:
            utils.validate_data(data_dir, task_id)
        
        test_ids = [f'{self.config}_i' for i in range(len(task_ids))]

        # Logic for overwrite. If save_submission_dir is provided, check if the submission already exists
        if self.save_submission_dir and not self.overwrite_submission:
            task_ids = [
                task_id for task_id in task_ids
                if not utils.submission_exists(self.save_submission_dir, task_id)
            ]
        
        train_pairs = [
            utils.get_train_pairs_from_task(data_dir, task_id)
            for task_id in task_ids
        ]
        test_input_pairs = [
            utils.get_test_input_from_task(data_dir, task_id)
            for task_id in task_ids
        ]

        attempting_ids = task_ids

        for _ in range(self.num_attempts):
            attempts = self.get_batched_task_prediction(
                training_pairs=train_pairs,
                test_inputs=test_input_pairs,
                task_ids=attempting_ids,
                test_ids=test_ids,
            )
            task_ids = [
                task_id
                for attempt, task_id in zip(attempts, attempting_ids)
                if attempt.answer is None
            ]
            pair_indices = [
                pair_index
                for attempt, pair_index in zip(attempts, pair_indices)
                if attempt.answer is None
            ]
            test_ids = [
                test_id
                for attempt, test_id in zip(attempts, test_ids)
                if attempt.answer is None
            ]
            if not attempting_ids:
                break

        attempts = [
            attempt
            for attempt in attempts
            if attempt.answer is not None
        ]

        if attempts:
            for task_id, attempt in zip(task_ids, attempts):
                if self.print_submission:
                    # Log the submission content; use json.dumps for potentially large structures
                    logger.info(f"Final submission for task {task_id}, ModelConfig {test_ids[0]}:\n{json.dumps(attempt, indent=4)}")
                if self.save_submission_dir:
                    utils.save_submission(self.save_submission_dir, task_id, attempt)
                    logger.info(f"Submission for task {task_id}, ModelConfig {test_ids[0]} saved to {self.save_submission_dir}")
        else:
            logger.warning(f"No valid predictions for any tasks. Skipping submission.")

        return attempts if attempts else None

def main_cli(cli_args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Run ARC Tester")
    parser.add_argument("--data_dir", type=str, help="Data set to run. Configure in config/config.json")
    parser.add_argument("--task_id", type=str, help="Specific task ID to run")
    parser.add_argument("--config", type=str, required=True, help="Configuration name (e.g., 'o1_high', 'gemini_short_response')")
    parser.add_argument(
        "--save_submission_dir",
        type=str,
        metavar="FOLDER_NAME",
        help="Folder name to save the submissions under Ex: 'submissions/o1_high'"
    )
    parser.add_argument("--overwrite_submission", action="store_true", help="Overwrite the submission if it already exists")
    parser.add_argument("--print_submission", action="store_true", help="Print the submission to the console after each task")
    parser.add_argument("--task_set", type=str, default="public_eval", choices=["public_eval", "public_training"], help="Task set to run")
    parser.add_argument("--num_attempts", type=int, default=2, help="Number of attempts for each prediction")
    parser.add_argument("--retry_attempts", type=int, default=2, help="Number of retry attempts for failed predictions")
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        default=False,
        help="Enable metrics collection and dumping (disabled by default)."
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        help="Set the logging level (default: INFO)"
    )
    args = parser.parse_args(cli_args)

    # Set metrics enabled status based on CLI arg first
    set_metrics_enabled(args.enable_metrics)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    arc_solver = ARCTester(
        config=args.config,
        save_submission_dir=args.save_submission_dir, 
        overwrite_submission=args.overwrite_submission,
        print_submission=args.print_submission,
        num_attempts=args.num_attempts,
        retry_attempts=args.retry_attempts
    )
   
    arc_solver.generate_task_solution(
        data_dir=args.data_dir,
        task_id=args.task_id
    )
    # Optionally return the solver or a status for more detailed testing if needed
    # For this test, we'll just ensure it runs without error.

if __name__ == "__main__":
    main_cli()
