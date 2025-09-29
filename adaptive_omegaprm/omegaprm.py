import heapq
import math
import random
import re
import json
from typing import List, Tuple, Dict, Any, Optional
import itertools
import numpy as np
import logging

from llm_utils import LLMService
from grader import math_equal


def _extract_last_boxed_content(text: str) -> Optional[str]:
    idx_start_box = text.rfind('\\boxed{')
    if idx_start_box == -1: return None
    start_pos = idx_start_box + len('\\boxed{')
    brace_level = 1
    for i in range(start_pos, len(text)):
        char = text[i]
        if char == '{': brace_level += 1
        elif char == '}': brace_level -= 1
        if brace_level == 0: return text[start_pos:i]
    return None

def check_correctness(generated_response: str, expected_answer: str) -> bool:
    extracted_answer = _extract_last_boxed_content(generated_response)
    if extracted_answer is None: return False
    extracted_answer = extracted_answer.strip()
    logging.getLogger(__name__).debug(f"Checking correctness: prediction={repr(extracted_answer)}, reference={repr(expected_answer)}")
    return math_equal(prediction=extracted_answer, reference=expected_answer)


def separate_steps(steps_text: str) -> List[str]:
    delimiter = "\n\n"
    return [step.strip() for step in steps_text.split(delimiter) if step.strip()]


class LanguageModel:
    def __init__(self, model_name="/root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct",
                 device="cuda", max_new_tokens=512, temperature=0.7,
                 top_k=-1, top_p=1.0, model_type="vllm",
                 gpu_memory_utilization: float = 0.90):
        self.llm_service = LLMService(
            model_name=model_name, device=device, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p, model_type=model_type,
            gpu_memory_utilization=gpu_memory_utilization
        )
        self.default_prompt_template = (
            "Please complete the answer for the question based on the given steps without generating existing steps again, "
            "and separate your following steps using \\n\\n.\n\nQuestion and current steps:\n{state_prefix}"
        )

    def _get_prompt(self, state_prefix: str) -> str:
        return self.default_prompt_template.format(state_prefix=state_prefix)
    
    def generate_rollout_with_features(self, state_prefix: str, num_copies: int) -> List[Tuple[str, Dict[str, Any]]]:
        self.llm_service.start_service()
        return self.llm_service.generate_response_with_llm_extras(self._get_prompt(state_prefix), num_copies)

    def evaluate_correctness(self, response_path_text: str, expected_answer: str) -> bool:
        return check_correctness(response_path_text, expected_answer)


class AdaptiveMCCluster:
    def __init__(self, cluster_id: int):
        self.id = cluster_id; self.rollout_indices: List[int] = []
        self.centroid: Optional[np.ndarray] = None
        self.n_samples, self.n_successes = 0, 0
        self.wilson_ci_lower, self.wilson_ci_upper, self.wilson_half_width = 0.0, 1.0, 1.0
        self.is_frozen = False
    def update_stats(self, n_successes: int, n_samples: int, conf_level_z: float = 1.96):
        self.n_samples, self.n_successes = n_samples, n_successes
        if self.n_samples > 0:
            p_hat, z_sq, n_val = self.n_successes / self.n_samples, conf_level_z**2, self.n_samples
            denominator, center_adj = 1 + z_sq / n_val, p_hat + z_sq / (2 * n_val)
            term_sqrt = max(0, (p_hat * (1 - p_hat) / n_val) + (z_sq / (4 * n_val**2)))
            hw = (conf_level_z * math.sqrt(term_sqrt)) / denominator
            self.wilson_ci_lower, self.wilson_ci_upper = max(0.0, (center_adj / denominator) - hw), min(1.0, (center_adj / denominator) + hw)
            if self.wilson_ci_lower > self.wilson_ci_upper: self.wilson_ci_lower = self.wilson_ci_upper = p_hat
            self.wilson_half_width = (self.wilson_ci_upper - self.wilson_ci_lower) / 2
        else: self.wilson_ci_lower, self.wilson_ci_upper, self.wilson_half_width = 0.0, 1.0, 1.0
    def check_frozen(self, k_min_cluster: int, epsilon_cluster: float):
        if self.n_samples >= k_min_cluster and self.wilson_half_width <= epsilon_cluster: self.is_frozen = True


class State:
    def __init__(self, solution_prefix: str, parent: Optional['State'] = None):
        self.solution_prefix = solution_prefix
        self.parent = parent
        self.N, self.total_rollouts_for_mc, self.correct_rollouts_for_mc = 0, 0, 0
        self.MC: Optional[float] = None
        self.incorrect_rollouts: List[str] = []
        self.children: List['State'] = []
        self.clusters: List[AdaptiveMCCluster] = []
        self.all_mc_rollouts_details: List[Dict[str, Any]] = []
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    def add_incorrect_continuation(self, continuation_text: str):
        if continuation_text not in self.incorrect_rollouts:
            self.incorrect_rollouts.append(continuation_text)

    def get_full_text_prefix(self) -> str:
        return self.solution_prefix

    def get_new_step_text(self) -> str:
        if self.parent:
            parent_prefix = self.parent.get_full_text_prefix()
            if self.solution_prefix.startswith(parent_prefix):
                new_text = self.solution_prefix[len(parent_prefix):]
                return new_text.lstrip('\n').strip()
        return self.solution_prefix

    def get_text_with_labels_for_tree_output(self) -> Dict[str, Any]:
        output_data = {
            "text": self.get_new_step_text(), 
            "mc_value": self.MC,
            "children": [child.get_text_with_labels_for_tree_output() for child in self.children],
            "incorrect_continuations": self.incorrect_rollouts
        }
        output_data["N_visits"] = self.N
        output_data["total_mc_rollouts_for_node"] = self.total_rollouts_for_mc
        return output_data


class SearchTree:
    def __init__(self): self.root: Optional[State] = None; self.nodes: List[State] = []
    def add_state(self, state: State):
        if state not in self.nodes: self.nodes.append(state)


class CandidatePool:
    def __init__(self):
        self.heap: List[Tuple[float, int]] = []; self.entry_finder: Dict[int, Tuple[float, int]] = {}
        self.counter = itertools.count(); self.id_to_rollout: Dict[int, Tuple[State, str]] = {}
        self.latest_id_per_rollout: Dict[Tuple[int, str], int] = {}
    def add_or_update(self, state: State, rollout: str, priority: float):
        state_id = id(state); rollout_key = (state_id, rollout)
        if rollout_key in self.latest_id_per_rollout:
            old_id = self.latest_id_per_rollout[rollout_key]
            if old_id in self.entry_finder: del self.entry_finder[old_id]
        uid = next(self.counter)
        self.latest_id_per_rollout[rollout_key] = uid
        heapq.heappush(self.heap, (-priority, uid)); self.entry_finder[uid] = (-priority, uid)
        self.id_to_rollout[uid] = (state, rollout)
    def pop(self) -> Tuple[Optional[State], Optional[str]]:
        while self.heap:
            neg_p, uid = heapq.heappop(self.heap)
            if uid in self.entry_finder:
                state, rollout = self.id_to_rollout.pop(uid); del self.entry_finder[uid]
                state_id = id(state); r_key = (state_id, rollout)
                if self.latest_id_per_rollout.get(r_key) == uid: del self.latest_id_per_rollout[r_key]
                return state, rollout
        return None, None
    def is_empty(self) -> bool: return not self.entry_finder


class OmegaPRM:
    def __init__(self, LM: LanguageModel, logger: logging.Logger, c_puct: float, alpha: float, beta: float,
                 L_q_len_penalty: int, k_fixed_rollouts: int, N_max_search: int,
                 max_rollout_budget: int, save_data_tree: bool,
                 k0_adaptive_mc: int, min_dynamic_k_step_amc: int, max_dynamic_k_step_amc: int,
                 hw_to_k_step_scaling_factor_amc: float, k_min_cluster_mc: int, epsilon_cluster_mc: float,
                 k_max_node_mc: int, num_clusters_k_mc: int, feature_names_mc: Optional[List[str]],
                 epsilon_node_confidence_stop: float = 0.1, nll_filter_threshold: float = 100.0,
                 garbled_text_filter_level: int = 3):
        self.LM, self.logger = LM, logger
        self.expected_answer: Optional[str] = None
        self.c_puct, self.alpha, self.beta = c_puct, alpha, beta
        self.L_q_len_penalty = L_q_len_penalty; self.k_fixed_rollouts = k_fixed_rollouts
        self.N_max_search, self.max_rollout_budget = N_max_search, max_rollout_budget
        self.save_data_tree = save_data_tree
        self.T, self.C = SearchTree(), CandidatePool()
        self.current_search_iteration, self.total_llm_calls_for_mc_estimation = 0, 0
        self.k0_amc = k0_adaptive_mc
        self.min_dynamic_k_step_amc = max(1, min_dynamic_k_step_amc)
        self.max_dynamic_k_step_amc = max(self.min_dynamic_k_step_amc, max_dynamic_k_step_amc)
        self.hw_to_k_step_scaling_factor_amc = hw_to_k_step_scaling_factor_amc
        self.k_min_cluster_amc, self.epsilon_cluster_amc = k_min_cluster_mc, epsilon_cluster_mc
        self.k_max_node_amc, self.num_clusters_amc = k_max_node_mc, num_clusters_k_mc
        self.feature_names_amc = feature_names_mc if feature_names_mc else ['nll', 'log_length']
        self.epsilon_node_amc_stop = epsilon_node_confidence_stop
        self.nll_filter_threshold = nll_filter_threshold
        self.garbled_text_filter_level = garbled_text_filter_level
        self.run_stats: Dict[str, Any] = {}

    def reset(self):
        self.logger.info("OmegaPRM: Resetting state.")
        self.expected_answer = None; self.T, self.C = SearchTree(), CandidatePool()
        self.current_search_iteration, self.total_llm_calls_for_mc_estimation = 0, 0
        self.run_stats = {"nodes_sampled": 0, "early_exit_nodes": 0, "samples_per_node_list": [], "cluster_stats_per_node": []}

    def run(self, question: str, answer: str) -> Tuple[Any, Dict]:
        self.reset()
        self.logger.info(f"OmegaPRM: Run start. Q: '{question[:100]}...'")
        self.expected_answer = answer
        initial_state = State(solution_prefix=question)
        self.T.root, self.T.nodes = initial_state, [initial_state]
        
        self.monte_carlo_estimation_adaptive(initial_state)
        
        while (self.current_search_iteration < self.N_max_search and
               self.total_llm_calls_for_mc_estimation < self.max_rollout_budget and
               not self.C.is_empty()):
            self.current_search_iteration += 1
            self.logger.info(f"--- MCTS Iter: {self.current_search_iteration}/{self.N_max_search} ---")
            parent_state, incorrect_rollout = self.selection_phase()
            if parent_state is None: break
            
            self.expansion_phase_binary_search(parent_state, incorrect_rollout)
            self.maintenance_phase(parent_state)
        
        self.logger.info(f"OmegaPRM: Run finished. Total MC LLM calls: {self.total_llm_calls_for_mc_estimation}")
        tree_output = self.collect_tree_structure()
        self.run_stats["total_llm_calls"] = self.total_llm_calls_for_mc_estimation
        return tree_output, self.run_stats


    def _process_mc_results(self, state: State):
        if state.MC is None:
            self.logger.warning(f"ProcessResults: State has no MC value!"); return
        
        correct_rollouts = [d['text'] for d in state.all_mc_rollouts_details if d['is_correct']]
        incorrect_details = [d for d in state.all_mc_rollouts_details if not d['is_correct']]

        self.add_incorrect_rollouts_to_candidate_pool(state, incorrect_details)

        if state.MC == 1.0:
            seen_chains = set()
            for r_text in correct_rollouts:
                if r_text not in seen_chains:
                    self.add_correct_rollout_to_tree(state, r_text)
                    seen_chains.add(r_text)

    def add_correct_rollout_to_tree(self, parent_state: State, rollout_text: str):
        new_solution_prefix = parent_state.get_full_text_prefix() + "\n\n" + rollout_text
        if any(c.solution_prefix == new_solution_prefix for c in parent_state.children):
            return
        
        new_state = State(solution_prefix=new_solution_prefix, parent=parent_state)
        new_state.MC = 1.0
        self.T.add_state(new_state)
        parent_state.children.append(new_state)

    def expansion_phase_binary_search(self, parent_state: State, incorrect_rollout: str):
        self.logger.info(f"BinarySearchPhase: Exploring incorrect rollout: '{incorrect_rollout[:50].replace(chr(10),' ')}...'")
        steps = separate_steps(incorrect_rollout)
        if not steps: return
        self.binary_search_recursive(parent_state, steps, 0, len(steps) - 1)

    def binary_search_recursive(self, s_ast: State, steps: List[str], left: int, right: int):
        if left > right: return

        mid = (left + right) // 2
        prefix_steps = steps[left : mid + 1]
        prefix_full_text = s_ast.get_full_text_prefix() + "\n\n" + "\n\n".join(prefix_steps)
        s_new = next((child for child in self.T.nodes if child.solution_prefix == prefix_full_text.strip()), None)

        if not s_new:

            s_new = State(solution_prefix=prefix_full_text.strip(), parent=s_ast)
            self.T.add_state(s_new)
            s_ast.children.append(s_new)
            self.monte_carlo_estimation_adaptive(s_new)
        
        if s_new.MC == 0:
            self.binary_search_recursive(s_ast, steps, left, mid - 1)
        else:
            self.binary_search_recursive(s_new, steps, mid + 1, right)

    def monte_carlo_estimation_adaptive(self, state: State):
        state_prefix_snip = state.get_full_text_prefix()[:70].replace("\n", " ") + "..."
        self.logger.info(f"AMC Enter: State='{state_prefix_snip}', N_before={state.N}")
        self.run_stats["nodes_sampled"] += 1
        state.N += 1
        
        if state.MC is not None:
             self.logger.info(f"AMC Skip: State already has MC value {state.MC:.4f}")
             self._process_mc_results(state)
             return

        state.clusters = [AdaptiveMCCluster(i) for i in range(self.num_clusters_amc)]
        state.all_mc_rollouts_details = []
        node_samps, node_succs = 0, 0

        num_k0_sample = min(self.k0_amc, self.max_rollout_budget - self.total_llm_calls_for_mc_estimation, self.k_max_node_amc - node_samps)
        if num_k0_sample > 0:
            k0_conts = self.LM.generate_rollout_with_features(state.get_full_text_prefix(), num_k0_sample)
            self.total_llm_calls_for_mc_estimation += len(k0_conts)
            for i, (text, extras) in enumerate(k0_conts):
                full_path_text = state.get_full_text_prefix() + "\n\n" + text
                is_corr = self.LM.evaluate_correctness(full_path_text, self.expected_answer)
                feats = self._extract_raw_features(text, extras)
                self.logger.info(f"AMC k0 Rollout {i+1}: Correct={is_corr}, AvgNLL={feats.get('nll',-1):.2f}, Text='{text[:30].replace(chr(10),' ')}...'")
                state.all_mc_rollouts_details.append({'text': text, 'is_correct': is_corr, 'raw_features': feats, 'llm_extras': extras, 'features_std': None, 'cluster_id': -1})
                node_samps += 1
                if is_corr: node_succs += 1
        
        if not state.all_mc_rollouts_details:
            self.logger.warning(f"AMC: No rollouts after k0 for '{state_prefix_snip}'"); state.MC = 0.0
            self._process_mc_results(state)
            return

        state.total_rollouts_for_mc, state.correct_rollouts_for_mc = node_samps, node_succs
        state.MC = node_succs / node_samps if node_samps > 0 else 0.0

        if node_samps > 0 and (state.MC == 1.0 or state.MC == 0.0):
            self.run_stats["early_exit_nodes"] += 1
            self.logger.info(f"AMC k0 Shortcut: All rollouts {'CORRECT' if state.MC==1.0 else 'INCORRECT'}. Finalizing node.")
            self._process_mc_results(state)
            return

        raw_f_matrix = np.array([[d['raw_features'][name] for name in self.feature_names_amc] for d in state.all_mc_rollouts_details])
        std_f_matrix, state.feature_means, state.feature_stds = self._standardize_features(raw_f_matrix)
        for i, d in enumerate(state.all_mc_rollouts_details): d['features_std'] = std_f_matrix[i]
        
        valid_f_vecs = [d['features_std'] for d in state.all_mc_rollouts_details]
        if valid_f_vecs and self.num_clusters_amc > 0:
            labels, centroids = self._simple_kmeans(valid_f_vecs, self.num_clusters_amc)
            for i, d in enumerate(state.all_mc_rollouts_details): d['cluster_id'] = labels[i]
            for c_idx, c_obj in enumerate(state.clusters):
                c_obj.rollout_indices = [i for i,d in enumerate(state.all_mc_rollouts_details) if d['cluster_id'] == c_idx]
                samps, succs = len(c_obj.rollout_indices), sum(1 for r_idx in c_obj.rollout_indices if state.all_mc_rollouts_details[r_idx]['is_correct'])
                c_obj.update_stats(succs, samps)
                if c_idx < len(centroids): c_obj.centroid = centroids[c_idx]
                c_obj.check_frozen(self.k_min_cluster_amc, self.epsilon_cluster_amc)
        
        while node_samps < self.k_max_node_amc:
            node_overall_hw = self._get_node_wilson_half_width(node_succs, node_samps)
            if node_overall_hw <= self.epsilon_node_amc_stop:
                self.logger.info(f"AMC AdLoop: Node confidence met. HW={node_overall_hw:.4f}"); break
            
            active_cs = [c for c in state.clusters if c.centroid is not None and not c.is_frozen and c.n_samples > 0]
            if not active_cs: self.logger.info("AMC AdLoop: All active clusters frozen/empty."); break
            
            active_cs.sort(key=lambda c: c.wilson_half_width, reverse=True); target_c = active_cs[0]
            
            raw_dynamic_step_calc = math.ceil(target_c.wilson_half_width * self.hw_to_k_step_scaling_factor_amc)
            k_step_dynamic_val = int(max(self.min_dynamic_k_step_amc, min(self.max_dynamic_k_step_amc, raw_dynamic_step_calc)))
            
            num_adapt = min(k_step_dynamic_val, self.k_max_node_amc - node_samps, self.max_rollout_budget - self.total_llm_calls_for_mc_estimation)
            if num_adapt <= 0: break
            
            adaptive_conts = self.LM.generate_rollout_with_features(state.get_full_text_prefix(), num_adapt)
            self.total_llm_calls_for_mc_estimation += len(adaptive_conts)

            for text, extras in adaptive_conts:
                full_path_text = state.get_full_text_prefix() + "\n\n" + text
                is_corr = self.LM.evaluate_correctness(full_path_text, self.expected_answer)
                feats = self._extract_raw_features(text, extras)
                std_f_arr, _, _ = self._standardize_features(np.array([[feats[n] for n in self.feature_names_amc]]), state.feature_means, state.feature_stds)
                std_f_vec = std_f_arr[0]
                
                min_d = float('inf'); best_c = target_c
                for c_iter in state.clusters:
                    if c_iter.centroid is not None:
                        d = np.linalg.norm(std_f_vec - c_iter.centroid)
                        if d < min_d: min_d = d; best_c = c_iter
                
                new_idx = len(state.all_mc_rollouts_details)
                state.all_mc_rollouts_details.append({'text': text, 'is_correct': is_corr, 'raw_features': feats, 'llm_extras': extras, 'features_std': std_f_vec, 'cluster_id': best_c.id})
                node_samps += 1
                if is_corr: node_succs += 1
                
                best_c.rollout_indices.append(new_idx)
                samps_c, succs_c = len(best_c.rollout_indices), sum(1 for r_idx in best_c.rollout_indices if state.all_mc_rollouts_details[r_idx]['is_correct'])
                best_c.update_stats(succs_c, samps_c)
                best_c.check_frozen(self.k_min_cluster_amc, self.epsilon_cluster_amc)
                
            state.MC = node_succs / node_samps if node_samps > 0 else 0.0

        state.total_rollouts_for_mc, state.correct_rollouts_for_mc = node_samps, node_succs
        self.run_stats["samples_per_node_list"].append(state.total_rollouts_for_mc)
        self._process_mc_results(state)

    def add_incorrect_rollouts_to_candidate_pool(self, parent_state: State, incorrect_details: List[Dict]):
        num_added_to_C = 0
        for detail in incorrect_details:
            r_text = detail['text']
            is_filtered_out = False
            
            replacement_char_count = r_text.count("\ufffd")
            if replacement_char_count > self.garbled_text_filter_level:
                is_filtered_out = True
                self.logger.warning(f"Filter: Garbled text. Skipping. Count={replacement_char_count}")
            
            avg_nll = detail.get('llm_extras', {}).get('nll', 0.0)
            if not is_filtered_out and avg_nll > self.nll_filter_threshold:
                is_filtered_out = True
                self.logger.warning(f"Filter: High NLL. Skipping. AvgNLL={avg_nll:.2f}")

            if not is_filtered_out:
                parent_state.add_incorrect_continuation(r_text)
                priority = self.compute_selection_score(parent_state, r_text)
                self.C.add_or_update(parent_state, r_text, priority)
                num_added_to_C += 1
        self.logger.info(f"Added {num_added_to_C} incorrect rollouts to candidate pool C.")
    
    def compute_Q(self, state: State, cont_text: str) -> float:
        mc = state.MC if state.MC is not None else 0.0
        wc = len(cont_text.split())
        return (self.alpha**(1-mc)) * (self.beta**(wc/self.L_q_len_penalty))

    def compute_U(self, state: State) -> float:
        N_total = sum(s_node.N for s_node in self.T.nodes if s_node.N > 0)
        return self.c_puct * (math.sqrt(N_total if N_total > 0 else 1.0)) / (1 + state.N)

    def compute_selection_score(self, state: State, cont_text: str) -> float:
        time=self.current_search_iteration+1
        weight=(math.e)**(-time/20)
        return (1-weight)*self.compute_Q(state, cont_text) + weight*self.compute_U(state)

    def selection_phase(self) -> Tuple[Optional[State], Optional[str]]:
        self.logger.info(f"SelectionPhase: Candidate Pool C size: {len(self.C.entry_finder)}")
        if self.C.is_empty(): self.logger.info("SelectionPhase: Candidate Pool is empty."); return None, None
        selected_parent_state, selected_continuation = self.C.pop()
        return selected_parent_state, selected_continuation

    def maintenance_phase(self, state_processed: State):
        if state_processed:
            for cont_text in list(state_processed.incorrect_rollouts):
                new_priority = self.compute_selection_score(state_processed, cont_text)
                self.C.add_or_update(state_processed, cont_text, new_priority)

    def collect_tree_structure(self) -> Dict[str, Any]:
        self.logger.info("Collecting full tree structure.")
        return self.T.root.get_text_with_labels_for_tree_output() if self.T.root else {}
        
    def _extract_raw_features(self, cont_text: str, llm_extras: Dict[str, Any]) -> Dict[str, float]:
        feats = {}
        num_words = len(cont_text.split())
        if num_words == 0: num_words = 1
        if 'nll' in self.feature_names_amc:
            total_nll = llm_extras.get('nll', 50.0 * num_words)
            feats['nll'] = total_nll / num_words
        if 'log_length' in self.feature_names_amc:
            feats['log_length'] = math.log(num_words + 1e-6)
        return feats
        
    def _standardize_features(self, raw_f_matrix: np.ndarray, means: Optional[np.ndarray]=None, stds: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if raw_f_matrix.size == 0: return raw_f_matrix, np.array([]), np.array([])
        curr_f_matrix = np.atleast_2d(raw_f_matrix)
        if means is None: means = np.mean(curr_f_matrix, axis=0)
        if stds is None:
            stds = np.std(curr_f_matrix, axis=0)
            stds[stds == 0] = 1.0
        return (curr_f_matrix - means) / stds, means, stds

    def _simple_kmeans(self, data_pts_std: List[np.ndarray], num_clust: int, max_iters: int = 20) -> Tuple[List[int], List[np.ndarray]]:
        if not data_pts_std or num_clust <= 0: return [], []
        num_samps = len(data_pts_std)
        if num_samps < num_clust: num_clust = num_samps
        if num_clust == 0: return [], []
        
        data_np = np.array(data_pts_std)
        initial_indices = random.sample(range(num_samps), num_clust)
        centroids = [data_np[i] for i in initial_indices]
        labels = np.zeros(num_samps, dtype=int)
        
        for _ in range(max_iters):
            changed = False
            for i, point in enumerate(data_np):
                distances = [np.linalg.norm(point - c) for c in centroids]
                new_label = np.argmin(distances)
                if labels[i] != new_label: changed = True
                labels[i] = new_label
            
            new_centroids_list = [[] for _ in range(num_clust)]
            for i, label_val in enumerate(labels): new_centroids_list[label_val].append(data_np[i])
            
            for i in range(num_clust):
                if new_centroids_list[i]:
                    centroids[i] = np.mean(new_centroids_list[i], axis=0)
            
            if not changed: break
        return labels.tolist(), centroids

    def _get_node_wilson_half_width(self, successes: int, samples: int, conf_level_z: float = 1.96) -> float:
        if samples == 0: return 1.0
        p_hat = successes / samples; z_sq = conf_level_z**2; n_val = samples
        denominator = 1 + z_sq / n_val; center_adj = p_hat + z_sq / (2 * n_val)
        term_sqrt_val = max(0, (p_hat * (1 - p_hat) / n_val) + (z_sq / (4 * n_val**2)))
        lower = max(0.0, (center_adj / denominator) - (conf_level_z * math.sqrt(term_sqrt_val)) / denominator)
        upper = min(1.0, (center_adj / denominator) + (conf_level_z * math.sqrt(term_sqrt_val)) / denominator)
        if lower > upper: lower = upper = p_hat
        return (upper - lower) / 2