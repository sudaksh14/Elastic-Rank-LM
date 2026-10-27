import torch
import torch.nn as nn
import json
import math
from types import MethodType
from transformers import AutoTokenizer, AutoModel, AutoConfig
from fvcore.nn import FlopCountAnalysis

from lowrank.models.layers import DynamicLowRankLinear, decompose_and_replace


class SequenceClassificationWrapper(nn.Module):
    """
    A wrapper to place a classification head on top of a base transformer model.
    This provides explicit control and avoids AutoClass compatibility issues.
    """
    def __init__(self, base_model, num_labels, pad_token_id):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        
        # The classification head
        self.score = nn.Linear(base_model.config.hidden_size, num_labels, bias=False)

    def forward(self, input_ids, attention_mask=None):
        # 1. Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False, # We only need the last hidden state
        )
        
        # 2. Get the last hidden state
        last_hidden_state = outputs.last_hidden_state

        # 3. Find the representation of the last non-padding token
        # This is the standard method for getting a sequence representation from causal LMs
        if self.pad_token_id is not None:
            sequence_lengths = (torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1).to(
                last_hidden_state.device
            )
            batch_size = input_ids.shape[0]
            pooled_hidden_state = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            pooled_hidden_state = last_hidden_state[:, -1] # Just take the last token
            
        # 4. Pass through the classification head
        logits = self.score(pooled_hidden_state)
        
        return logits


# ==============================================================================
# ========= CORE ANALYSIS LOGIC ================================================
# ==============================================================================

def dynamic_low_rank_linear_handler(module: DynamicLowRankLinear, inputs: tuple) -> float:
    in_features = module.A.weight.shape[1]
    out_features = module.B.weight.shape[0]
    active_rank = module.active_rank
    macs_A = in_features * active_rank
    macs_B = active_rank * out_features
    total_macs_per_token = macs_A + macs_B
    return total_macs_per_token

def get_target_module(model, model_id):
    if 'gemma' in model_id.lower() or 'qwen' in model_id.lower():
        return model.base_model # We target the base_model inside our wrapper
    elif 'gpt-neo' in model_id.lower():
        # GPT-Neo's main block is called 'transformer'
        return model.base_model.transformer
    elif 'pythia' in model_id.lower():
        # Pythia's main block is called 'gpt_neox'
        return model.base_model.gpt_neox
    else:
        raise ValueError(f"Unknown model architecture for {model_id}")

def analyze_model_flops(model_id: str, max_rank: int, seq_len: int = 256):
    print(f"\n--- Analyzing {model_id} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    num_labels = 3 # For MNLI

    # --- Setup Tokenizer and Config ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Standard practice

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.use_cache = False # Prevents JIT tracer error

    # --- 1. Baseline FLOPs Calculation ---
    print("Calculating baseline FLOPs...")
    base_transformer_model = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True)
    base_model = SequenceClassificationWrapper(base_transformer_model, num_labels, tokenizer.pad_token_id).to(device)
    
    dummy_input = (torch.zeros((1, seq_len), dtype=torch.long, device=device),)
    flop_analyzer_base = FlopCountAnalysis(base_model, dummy_input)
    base_flops = flop_analyzer_base.total()
    results["base_model_flops_g"] = base_flops / 1e9
    print(f"Base FLOPs: {base_flops / 1e9:.2f} GFLOPs")
    
    # --- 2. Low-Rank FLOPs Calculation ---
    print("Calculating low-rank FLOPs...")
    low_rank_transformer_model = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True)
    low_rank_model = SequenceClassificationWrapper(low_rank_transformer_model, num_labels, tokenizer.pad_token_id).to(device)

    target_module = get_target_module(low_rank_model, model_id)
    decompose_and_replace(target_module, max_rank=max_rank, target_substring="mlp")

    def _set_rank_all_method(self, rank):
        for m in self.modules():
            if isinstance(m, DynamicLowRankLinear):
                m.set_rank(rank)
    low_rank_model._set_rank_all = MethodType(_set_rank_all_method, low_rank_model)

    flop_analyzer_low_rank = FlopCountAnalysis(low_rank_model, dummy_input)
    flop_analyzer_low_rank.set_op_handle(DynamicLowRankLinear, dynamic_low_rank_linear_handler)

    ranks_to_eval = [max_rank // (2**k) for k in range(int(math.log2(max_rank)) + 1) if max_rank // (2**k) > 0]
    rank_analysis = {}
    
    for rank in sorted(ranks_to_eval, reverse=True):
        low_rank_model._set_rank_all(rank)
        current_flops = flop_analyzer_low_rank.total()
        percentage_change = ((current_flops - base_flops) / base_flops) * 100
        
        rank_analysis[rank] = {
            "flops_g": current_flops / 1e9,
            "percentage_change": round(percentage_change, 2)
        }
        print(f"  Rank {rank:>4}: {current_flops / 1e9:>6.2f} GFLOPs ({percentage_change:+.2f}%)")
        
    results["low_rank_analysis"] = rank_analysis
    
    del base_model, low_rank_model, base_transformer_model, low_rank_transformer_model
    torch.cuda.empty_cache()
    
    return results

if __name__ == '__main__':
    models_to_analyze = {
        'gemma-2b': 'google/gemma-2b',
        'qwen2-0.5b': 'Qwen/Qwen2-0.5B',
        'gpt-neo-2.7b': 'EleutherAI/gpt-neo-2.7B',
        'pythia-2.8b': 'EleutherAI/pythia-2.8b',
    }
    
    MAX_RANK = 1024
    MAX_SEQ_LENGTH = 256
    
    all_results = {}
    for name, model_id in models_to_analyze.items():
        all_results[name] = analyze_model_flops(
            model_id=model_id,
            max_rank=MAX_RANK,
            seq_len=MAX_SEQ_LENGTH
        )
        
    output_filename = "flops_analysis_results.json"
    with open(output_filename, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n✅ Analysis complete. Results saved to {output_filename}")