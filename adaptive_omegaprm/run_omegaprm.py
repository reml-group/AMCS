import argparse
import json
import logging
import os
import re 
import time
from typing import Dict, List, Any
from tqdm import tqdm

from omegaprm import LanguageModel, OmegaPRM

DS_NAME = "math-problems"
logger: logging.Logger


def setup_logging(log_file_prefix: str):
    log_dir = os.path.dirname(log_file_prefix)
    if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_file_prefix}.log"
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler); handler.close()
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(process)d][%(name)s][%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, mode='a'), logging.StreamHandler()])
    return logging.getLogger(__name__)


def load_questions(filepath: str) -> List[Dict[str, str]]:
    questions = []
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            content = f.read()
            try: questions = json.loads(content)
            except json.JSONDecodeError:
                f.seek(0);
                for line_num, line in enumerate(f):
                    try: questions.append(json.loads(line))
                    except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON line {line_num+1} in {filepath}: {line.strip()}")
        if not isinstance(questions, list): logger.error(f"Not a list from {filepath}."); return []
        return questions
    except Exception as e: logger.error(f"Error loading {filepath}: {e}"); return []


def should_process_question(q_data: Dict[str, str], llm: LanguageModel, num_rollouts: int, logger_obj: logging.Logger) -> bool:
    prompt, answer = q_data.get("problem"), q_data.get("final_answer")
    if not prompt or not answer: logger_obj.warning("Filter: Skip missing problem/answer."); return False
    correct, incorrect = False, False
    try:
        rollouts = llm.generate_rollout(prompt, num_rollouts)
        if not rollouts: logger_obj.info(f"Filter: Skip (no rollouts) for: {prompt[:70]}..."); return False
        for ans_text in rollouts:
            if llm.evaluate_correctness(prompt + "\n\n" + ans_text, answer): correct = True
            else: incorrect = True
            if correct and incorrect: logger_obj.info(f"Filter: PASSED (mixed) for: {prompt[:70]}..."); return True
        status = "all_correct" if correct else ("all_incorrect" if incorrect else "no_clear_result")
        logger_obj.info(f"Filter: SKIPPED ({status}) for: {prompt[:70]}..."); return False
    except Exception as e: logger_obj.error(f"Filter error for '{prompt[:70]}...': {e}", exc_info=False); return False


def process_question(omega_prm_instance: OmegaPRM, q_data: Dict[str, str], logger_obj: logging.Logger) -> Dict[str, Any]:
    logger_obj.info(f"Processing question via OmegaPRM: {q_data.get('problem','NO_PROBLEM_TEXT')[:100]}...")

    reasoning_tree_output, run_statistics = omega_prm_instance.run(q_data["problem"], q_data["final_answer"])
    
    final_data = {
        "question": q_data["problem"],
        "final_answer": q_data["final_answer"],
        "reasoning_steps": reasoning_tree_output,
        "run_statistics": run_statistics
    }
    return final_data


def save_question_data(data: Dict, q_id: Any, path: str, logger_obj: logging.Logger):
    data["question_id"] = q_id
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding='utf-8') as fd:
        fd.write(json.dumps(data) + "\n")
    logger_obj.info(f"Saved data for question_id {q_id} to {path}")


def main(args):
    global logger; logger = setup_logging(args.log_file_prefix)
    logger.info(f"Run Config: Output to: {args.output_file_path}")
    logger.info(f"Run Config: Model: {args.model_name} ({args.model_type}) on {args.device}, GPU Mem Util: {args.gpu_memory_utilization}")
    logger.info(f"Run Config: Questions from: {args.question_file}")
    if args.max_questions_to_process > 0: logger.info(f"Run Config: Max questions per file: {args.max_questions_to_process}")

    all_loaded_questions = load_questions(args.question_file)
    if not all_loaded_questions: logger.error(f"No questions loaded from {args.question_file}. Exiting."); return
    questions_to_attempt = all_loaded_questions[:args.max_questions_to_process] if args.max_questions_to_process > 0 else all_loaded_questions
    logger.info(f"Loaded {len(all_loaded_questions)} questions. Will attempt {len(questions_to_attempt)} based on --max_questions_to_process.")

    llm = LanguageModel(
        model_name=args.model_name, device=args.device, model_type=args.model_type,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, gpu_memory_utilization=args.gpu_memory_utilization
    )
    feature_names_list = [n.strip() for n in args.feature_names_mc.split(',') if n.strip()] if args.feature_names_mc else ['nll', 'log_length']
    logger.info(f"Run Config: Adaptive MC Features: {feature_names_list}")
    logger.info(f"Run Config: Epsilon Node Confidence Stop: {args.epsilon_node_confidence_stop}")
    logger.info(f"Run Config: Dynamic k_step_amc - Min={args.min_dynamic_k_step}, Max={args.max_dynamic_k_step}, ScaleFactor={args.hw_to_k_step_scaling_factor}")
    logger.info(f"Run Config: NLL Filter Threshold={args.nll_filter_threshold}, Garbled Filter Level={args.garbled_text_filter_level}")


    omega_prm_instance = OmegaPRM(
        LM=llm, logger=logger, c_puct=args.c_puct, alpha=args.alpha, beta=args.beta,
        L_q_len_penalty=args.L_q_len_penalty, k_fixed_rollouts=args.k_fixed_rollouts,
        N_max_search=args.max_search_count, max_rollout_budget=args.max_rollout_budget,
        save_data_tree=args.save_data_tree,
        k0_adaptive_mc=args.k0_adaptive_mc,
        min_dynamic_k_step_amc=args.min_dynamic_k_step,
        max_dynamic_k_step_amc=args.max_dynamic_k_step,
        hw_to_k_step_scaling_factor_amc=args.hw_to_k_step_scaling_factor,
        k_min_cluster_mc=args.k_min_cluster_mc,
        epsilon_cluster_mc=args.epsilon_cluster_mc, k_max_node_mc=args.k_max_node_mc,
        num_clusters_k_mc=args.num_clusters_k_mc, feature_names_mc=feature_names_list,
        epsilon_node_confidence_stop=args.epsilon_node_confidence_stop,
        nll_filter_threshold=args.nll_filter_threshold,
        garbled_text_filter_level=args.garbled_text_filter_level
    )

    final_questions_for_processing = []
    if args.use_filter:
        logger.info(f"Filtering {len(questions_to_attempt)} questions...")
        for q_item_filter in tqdm(questions_to_attempt, desc=f"Filtering {os.path.basename(args.question_file)}", unit="q_filter"):
            if "problem" not in q_item_filter or "final_answer" not in q_item_filter: logger.warning(f"Skipping filter: malformed: {str(q_item_filter)[:100]}"); continue
            if should_process_question(q_item_filter, llm, args.num_filter_rollouts, logger): final_questions_for_processing.append(q_item_filter)
        logger.info(f"After filtering: {len(final_questions_for_processing)} questions selected.")
    else: final_questions_for_processing = questions_to_attempt; logger.info("Filtering disabled.")
    if not final_questions_for_processing: logger.info("No questions to process. Exiting."); return

    question_file_shortname = os.path.basename(args.question_file)
    
    for idx, q_data_item in enumerate(tqdm(final_questions_for_processing, desc=f"MainProcessing {question_file_shortname}", unit="q_proc")):
        if not isinstance(q_data_item, dict) or "problem" not in q_data_item or "final_answer" not in q_data_item: logger.warning(f"Skipping malformed item: {str(q_data_item)[:100]}"); continue
        
        original_q_id = q_data_item.get("id", q_data_item.get("question_id", f"auto_idx_{idx}"))
        
        numeric_id = None
        if isinstance(original_q_id, int):
            numeric_id = original_q_id
        elif isinstance(original_q_id, str):
            numbers = re.findall(r'\d+', original_q_id)
            if numbers:
                numeric_id = int(numbers[-1])
        
        if numeric_id is None:
            numeric_id = idx

        try:
            start_time = time.time()
            result_data_item = process_question(omega_prm_instance, q_data_item, logger)
            duration_s = time.time() - start_time
            
            if "run_statistics" in result_data_item:
                result_data_item["run_statistics"]["wall_clock_time_s"] = round(duration_s, 2)
            
            save_question_data(result_data_item, numeric_id, args.output_file_path, logger)
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR processing q_id {numeric_id} (Original: {original_q_id}): {e}", exc_info=True)
            error_placeholder = {"question_id": numeric_id, "question": q_data_item.get('problem'), "error_message": str(e), "reasoning_steps": None}
            save_question_data(error_placeholder, numeric_id, args.output_file_path, logger)
            
    logger.info(f"Finished {args.question_file}. Total items attempted: {len(final_questions_for_processing)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OmegaPRM with Clustered-Adaptive MC and detailed logging.")
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--log_file_prefix", type=str, default="logs/omegaprm_default_run")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_type", type=str, default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7); parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1.0); parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--c_puct", type=float, default=0.125); parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.9); parser.add_argument("--L_q_len_penalty", type=int, default=500)
    parser.add_argument("--k_fixed_rollouts", type=int, default=16)
    parser.add_argument("--max_search_count", type=int, default=100)
    parser.add_argument("--max_rollout_budget", type=int, default=2000)
    parser.add_argument("--save_data_tree", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--k0_adaptive_mc", type=int, default=8)
    parser.add_argument("--min_dynamic_k_step", type=int, default=1, help="Min k_step for dynamic adaptive MC.")
    parser.add_argument("--max_dynamic_k_step", type=int, default=8, help="Max k_step for dynamic adaptive MC.")
    parser.add_argument("--hw_to_k_step_scaling_factor", type=float, default=16.0, help="Scaling factor: WilsonHW * factor = approx k_step.")
    parser.add_argument("--k_min_cluster_mc", type=int, default=3)
    parser.add_argument("--epsilon_cluster_mc", type=float, default=0.2)
    parser.add_argument("--k_max_node_mc", type=int, default=100)
    parser.add_argument("--num_clusters_k_mc", type=int, default=3)
    parser.add_argument("--feature_names_mc", type=str, default="nll,log_length")
    parser.add_argument("--epsilon_node_confidence_stop", type=float, default=0.1)
    parser.add_argument("--nll_filter_threshold", type=float, default=100.0, help="Avg NLL threshold to filter catastrophic rollouts.")
    parser.add_argument("--garbled_text_filter_level", type=int, default=3, help="Max count of Unicode replacement chars before filtering.")
    parser.add_argument("--use_filter", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--num_filter_rollouts", type=int, default=32)
    parser.add_argument("--max_questions_to_process", type=int, default=-1)
    
    args = parser.parse_args()

    main(args)
