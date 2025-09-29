import torch
import torch.nn.functional as F 

@torch.inference_mode()
def _qwen_math_infer_fn(input_str: str, model, tokenizer, device):
    GOOD_TOKEN = '+'
    BAD_TOKEN = '-'
    STEP_TAG = '\n\n\n\n\n'

    candidate_tokens = tokenizer.encode(f" {GOOD_TOKEN} {BAD_TOKEN}") 
    step_tag_id = torch.tensor([tokenizer.encode(f" {STEP_TAG}")], device=device)
    input_id = torch.tensor(
        [tokenizer.encode(input_str)], device=device)
    logits = model(input_id).logits[:,:,candidate_tokens]

    scores = logits.softmax(dim=-1)[:,:,0]
    mask = input_id == step_tag_id
    step_scores = scores[mask]
    return step_scores


@torch.inference_mode()
def _math_shepherd_infer_fn(input_str: str, model, tokenizer, device):
    GOOD_TOKEN = '+'
    BAD_TOKEN = '-'
    STEP_TAG = 'ки'
    candidate_tokens = tokenizer.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:]
    step_tag_id = tokenizer.encode(f"{STEP_TAG}")[-1] 

    input_id = torch.tensor(
        [tokenizer.encode(input_str)], device=device)
    logits = model(input_id).logits[:,:,candidate_tokens]
    scores = logits.softmax(dim=-1)[:,:,0] 
    step_scores = scores[input_id == step_tag_id]
    return step_scores
 
@torch.inference_mode()
def _qwen2_5_prm_infer_fn(input_str: str, model, tokenizer, device):
    step_sep_token = "<extra_0>"
    input_ids = tokenizer(input_str, return_tensors="pt").input_ids.to(device)
    step_sep_id = tokenizer.convert_tokens_to_ids(step_sep_token)
    if input_ids.shape[0] != 1:
        raise ValueError("This function is designed for a batch size of 1.")

    logits = model(input_ids).logits
    mask = (input_ids == step_sep_id)
    probabilities = F.softmax(logits, dim=-1)
    sample_probs = probabilities.squeeze(0)
    sample_mask = mask.squeeze(0)
    step_token_distributions = sample_probs[sample_mask] 
    step_scores = step_token_distributions[step_token_distributions > 0].view(-1, 2)[:, 1]

    return step_scores