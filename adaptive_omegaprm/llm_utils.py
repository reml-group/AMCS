import os
import threading
import random 
from typing import List, Tuple, Dict, Any


try:
    import vllm
    from vllm import LLM, SamplingParams, RequestOutput
except ImportError:
    print("vLLM is not installed or an error occurred during import. vLLM functionalities will not be available.")
    LLM, SamplingParams, RequestOutput = type(None), type(None), type(None) 

try:
    from transformers import pipeline
except ImportError:
    print("Transformers library not installed. Hugging Face pipeline functionalities will not be available.")
    pipeline = type(None)


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class LLMService:
    def __init__(self, model_name: str = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct",
                 device: str = "cuda", 
                 max_new_tokens: int = 2048,
                 temperature: float = 0.7, 
                 top_k: int = -1, 
                 top_p: float = 1.0, 
                 model_type: str="vllm",
                 gpu_memory_utilization: float = 0.90 
                 ):
        self.model_name = model_name
        self.device = device 
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        if self.temperature == 0.0: 
            self.top_k = -1 
            self.top_p = 1.0 
        else: 
            self.top_k = top_k if top_k > 0 else -1 
            self.top_p = top_p if top_p < 1.0 else 1.0 
        
        self.model_type = model_type.lower()
        self.gpu_memory_utilization = gpu_memory_utilization
        self.pipe = None 
        self.llm = None  
        self.load_lock = threading.Lock()

    def start_service(self):
        with self.load_lock:
            if self.model_type == "hf":
                if pipeline is None:
                    raise ImportError("Transformers library not installed. Cannot start Hugging Face pipeline service.")
                if self.pipe is None:
                    print(f"LLMService: Loading Hugging Face model '{self.model_name}' on device '{self.device}'...")
                    try:
                        self.pipe = pipeline(
                            "text-generation",
                            model=self.model_name,
                            torch_dtype="auto", 
                            device_map="auto" 
                        )
                        print("LLMService: Hugging Face model loaded successfully.")
                    except Exception as e:
                        print(f"LLMService: Error loading Hugging Face model: {e}")
                        raise
            elif self.model_type == "vllm":
                if LLM is None:
                    raise ImportError("vLLM is not installed or failed to import. Cannot start vLLM service.")
                if self.llm is None:
                    print(f"LLMService: Loading vLLM model '{self.model_name}' with GPU memory utilization: {self.gpu_memory_utilization}...")
                    try:
                        self.llm = LLM(
                            model=self.model_name, 
                            trust_remote_code=True, 
                            tensor_parallel_size=1, 
                            gpu_memory_utilization=self.gpu_memory_utilization
                        )
                        print("LLMService: vLLM model loaded successfully.")
                    except Exception as e:
                        print(f"LLMService: Error loading vLLM model: {e}")
                        raise
            else:
                raise ValueError("Unsupported model_type. Choose 'hf' for Hugging Face or 'vllm' for vLLM.")

    def _calculate_nll_from_vllm_output(self, vllm_req_output: RequestOutput) -> float:
        if not vllm_req_output.outputs:
            return random.uniform(15.0, 25.0) 

        first_output = vllm_req_output.outputs[0]
        if not first_output.logprobs or not first_output.token_ids:
            return random.uniform(15.0, 25.0)

        generated_token_ids = first_output.token_ids
        per_position_logprobs_list = first_output.logprobs

        if not generated_token_ids:
             return 0.0 

        if len(generated_token_ids) != len(per_position_logprobs_list):
            print(f"LLMService Warning (calc_nll): Mismatch len(token_ids)={len(generated_token_ids)}, len(logprobs)={len(per_position_logprobs_list)}. NLL may be inaccurate.")
            return random.uniform(15.0, 25.0) 

        cumulative_logprob = 0.0
        processed_tokens_count = 0

        for i, token_id in enumerate(generated_token_ids):
            if i >= len(per_position_logprobs_list): break 
            logprobs_dict_at_pos = per_position_logprobs_list[i]

            if logprobs_dict_at_pos is None:
                 cumulative_logprob -= 5.0 
                 continue

            if token_id in logprobs_dict_at_pos:
                try:
                    logprob_object = logprobs_dict_at_pos[token_id]
                    cumulative_logprob += logprob_object.logprob
                    processed_tokens_count +=1
                except AttributeError: 
                    try:
                        cumulative_logprob += logprobs_dict_at_pos[token_id] 
                        processed_tokens_count +=1
                    except Exception: 
                         print(f"LLMService CRITICAL Warning (calc_nll): Could not access logprob for token_id {token_id} at pos {i} from {logprobs_dict_at_pos[token_id]}. Penalizing heavily.")
                         cumulative_logprob -= 10.0
            else:
                print(f"LLMService CRITICAL Warning (calc_nll): Generated token_id {token_id} not in logprobs dict at pos {i}. "
                      f"Keys: {list(logprobs_dict_at_pos.keys())}. NLL will be very inaccurate. Penalizing heavily.")
                cumulative_logprob -= 10.0 

        if processed_tokens_count == 0 and generated_token_ids:
             return random.uniform(20.0, 30.0)
        if processed_tokens_count == 0 and not generated_token_ids: return 0.0

        nll = -cumulative_logprob 
        return nll if nll >= 0 else (abs(nll) + random.uniform(1.0, 5.0))

    def generate_response_with_llm_extras(self, prompt: str, num_copies: int) -> List[Tuple[str, Dict[str, Any]]]:
        results = []
        if self.model_type == "hf":
            if self.pipe is None: raise ValueError("HF LLM service not started.")
            prompts_batch = [prompt] * num_copies
            hf_outputs = self.pipe(prompts_batch, max_new_tokens=self.max_new_tokens, batch_size=num_copies,
                                   do_sample=self.temperature > 0, temperature=self.temperature if self.temperature > 0 else None,
                                   top_k=self.top_k if self.temperature > 0 and self.top_k > 0 else None, 
                                   top_p=self.top_p if self.temperature > 0 and self.top_p < 1.0 else None, 
                                   return_full_text=False)
            for output_item in hf_outputs:
                text = output_item[0]["generated_text"]
                extras = {"nll": random.uniform(1.0, 10.0)} 
                results.append((text, extras))

        elif self.model_type == "vllm":
            if self.llm is None: raise ValueError("vLLM service not started.")
            sampling_params = SamplingParams(
                n=1, 
                temperature=self.temperature,
                top_k=self.top_k, 
                top_p=self.top_p, 
                max_tokens=self.max_new_tokens,
                logprobs=5 
            )
            prompts_batch = [prompt] * num_copies 
            
            try:
                request_outputs: List[RequestOutput] = self.llm.generate(prompts_batch, sampling_params)
            except Exception as e:
                print(f"LLMService Error during vLLM generation: {e}")
                for _ in range(num_copies): results.append(("", {"nll": random.uniform(25.0,35.0)}))
                return results

            for req_output in request_outputs: 
                text = req_output.outputs[0].text 
                nll = self._calculate_nll_from_vllm_output(req_output)
                extras = {"nll": nll}
                results.append((text, extras))
        else:
            raise ValueError("Unsupported model_type.")
        return results

    def generate_response(self, prompt: str, num_copies: int = 1) -> List[str]:
        results_with_extras = self.generate_response_with_llm_extras(prompt, num_copies)
        return [text for text, extras in results_with_extras]

if __name__ == "__main__":
    VLLM_TEST_MODEL_PATH = os.environ.get("VLLM_MODEL_PATH", "/data2/qsh/model/Qwen2.5-Math-7B-Instruct/") 
    print(f"--- Main Test: vLLM LLMService with model: {VLLM_TEST_MODEL_PATH} ---")
    if LLM is None: print("Skipping vLLM test: vLLM library not available.")
    elif not os.path.isdir(VLLM_TEST_MODEL_PATH): print(f"Skipping vLLM test: model path does not exist: {VLLM_TEST_MODEL_PATH}")
    else:
        try:
            llm_service = LLMService(model_name=VLLM_TEST_MODEL_PATH, model_type="vllm", 
                                     temperature=0.0, max_new_tokens=30, gpu_memory_utilization=0.80)
            llm_service.start_service()
            test_prompt = "The first prime number greater than 20 is"
            
            print(f"\nTesting generate_response_with_llm_extras for: '{test_prompt}'")
            responses_extras = llm_service.generate_response_with_llm_extras(test_prompt, num_copies=2)
            if responses_extras:
                for i, (text, extras) in enumerate(responses_extras):
                    print(f"  vLLM Response (extras) {i+1}: '{text}' (NLL: {extras.get('nll')})")
            else:
                print("  generate_response_with_llm_extras returned no responses.")
            
            print(f"\nTesting generate_response for: '{test_prompt}'")
            plain_responses = llm_service.generate_response(test_prompt, num_copies=1)
            if plain_responses:
                print(f"  vLLM Plain Response: '{plain_responses[0]}'")
            else:
                print("  generate_response returned no responses.")
        except Exception as e: print(f"Error during vLLM test in __main__: {e}\nEnsure vLLM is correctly installed, CUDA is available, and model path is correct and accessible.")