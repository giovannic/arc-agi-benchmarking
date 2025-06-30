from .provider import ProviderAdapter
from arc_agi_benchmarking.schemas import Attempt, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails
from datetime import datetime, timezone
from sal.config import Config
from vllm import LLM
from sal.models.reward_models import load_prm
from sal.search.best_of_n import best_of_n
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

class LocalLlamaAdapter(ProviderAdapter):
    def init_client(self):
        # Build a Config object from model_config.kwargs, override approach and n
        config_kwargs = dict(self.model_config.kwargs)
        config_kwargs.setdefault('approach', 'best_of_n')
        config_kwargs['n'] = 1
        config = Config(**config_kwargs)
        llm = LLM(model=config.model_path,
                  gpu_memory_utilization=getattr(config, 'gpu_memory_utilization', 0.5),
                  enable_prefix_caching=True,
                  seed=getattr(config, 'seed', 42))
        prm_config = self.model_config.kwargs.get('prm_config', {})
        prm = load_prm(Config(**prm_config))
        self._config = config
        self._llm = llm
        self._prm = prm
        return (llm, prm)

    def make_prediction(self, prompt: str, task_id: str, test_id: str, pair_index: int) -> Attempt:
        start_time = datetime.now(timezone.utc)
        x = {"problem": [prompt]}
        logger.info(f"LocalLlamaAdapter prompt: {prompt}")
        # result = best_of_n(x, self._config, self._llm, self._prm)  # For future use
        # Prepare input for LLM
        sampling_params = self._llm.sampling_params_class(
            temperature=getattr(self._config, 'temperature', 0.0),
            max_tokens=getattr(self._config, 'max_tokens', 512),
            top_p=getattr(self._config, 'top_p', 1.0),
            n=1
        )
        responses = self._llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
        # Assume responses is a list of objects with an 'outputs' attribute
        raw_response = None
        answer = ""
        for r in responses:
            for output in r.outputs:
                raw_response = output.text
                answer = raw_response
                break
            if answer:
                break
        logger.info(f"LocalLlamaAdapter raw response: {raw_response}")
        # For now, use dummy values for token/cost accounting
        prompt_tokens = 0
        completion_tokens_count = 0
        total_tokens = prompt_tokens + completion_tokens_count
        reasoning_tokens = 0
        input_choices = [
            Choice(index=0, message=Message(role="user", content=prompt))
        ]
        response_choices = [
            Choice(index=1, message=Message(role="assistant", content=answer))
        ]
        all_choices = input_choices + response_choices
        input_cost_per_token = getattr(self.model_config.pricing, 'input', 0) / 1_000_000
        output_cost_per_token = getattr(self.model_config.pricing, 'output', 0) / 1_000_000
        prompt_cost = prompt_tokens * input_cost_per_token
        completion_cost = completion_tokens_count * output_cost_per_token
        reasoning_cost = reasoning_tokens * output_cost_per_token
        end_time = datetime.now(timezone.utc)
        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens_count,
                total_tokens=total_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=reasoning_tokens,
                    accepted_prediction_tokens=completion_tokens_count,
                    rejected_prediction_tokens=0
                )
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                reasoning_cost=reasoning_cost,
                total_cost=prompt_cost + completion_cost + reasoning_cost
            ),
            task_id=task_id,
            pair_index=pair_index,
            test_id=test_id
        )
        return Attempt(answer=answer, metadata=metadata)

    def extract_json_from_response(self, input_response: str):
        # JSON schema for a list of lists of integers
        json_schema = {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"}
            }
        }
        # Prompt for extraction
        prompt = f"Extract the test output as a JSON array of arrays of integers from the following response. Only output the JSON, nothing else.\n\nResponse:\n{input_response}"
        conv = [
            {"role": "system", "content": getattr(self._config, 'system_prompt', "You are a helpful assistant.")},
            {"role": "user", "content": prompt}
        ]
        tokenizer = self._llm.get_tokenizer()
        if getattr(self._config, 'custom_chat_template', None) is not None:
            tokenizer.chat_template = self._config.custom_chat_template
        templated_conv = tokenizer.apply_chat_template([conv], tokenize=False, add_generation_prompt=True)
        sampling_params = self._llm.sampling_params_class(
            temperature=0.0,  # Greedy decoding for extraction
            max_tokens=512,
            top_p=1.0,
            n=1
        )
        responses = self._llm.generate(
            templated_conv,
            sampling_params=sampling_params,
            use_tqdm=False,
            guided_options_request={"guided_json": json_schema}
        )
        # Extract the JSON from the first response
        for r in responses:
            for output in r.outputs:
                try:
                    return output.text if isinstance(output.text, list) else json.loads(output.text)
                except Exception:
                    continue
        return None 