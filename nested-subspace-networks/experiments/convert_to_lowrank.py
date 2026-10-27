import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import os
import shutil

# ==============================================================================
# 1. DYNAMIC LOW-RANK MODULE
# ==============================================================================

class DynamicLowRankLinear(nn.Module):
    """
    A dynamically rank-adjustable low-rank linear layer.
    """
    def __init__(self, in_features, out_features, max_rank, bias=True):
        super().__init__()
        self.max_rank = max_rank
        self.in_features = in_features
        self.out_features = out_features
        self.active_rank = max_rank 

        self.A = nn.Linear(in_features, max_rank, bias=False) 
        self.B = nn.Linear(max_rank, out_features, bias=bias)

    def set_rank(self, rank):
        self.active_rank = min(max(1, rank), self.max_rank)

    def forward(self, x):
        r = self.active_rank
        A_weight_slice = self.A.weight[:r, :]
        B_weight_slice = self.B.weight[:, :r]
        h = nn.functional.linear(x, A_weight_slice)
        return nn.functional.linear(h, B_weight_slice, self.B.bias)

    def __repr__(self):
        return (f"DynamicLowRankLinear(in={self.in_features}, out={self.out_features}, "
                f"max_rank={self.max_rank}, active_rank={self.active_rank})")


# ==============================================================================
# 2. REPLACEMENT FUNCTION
# ==============================================================================

def replace_linear_with_lowrank(model: nn.Module, max_rank: int, target_substring: str, initialize_weights: bool = True):
    """
    Iteratively finds and replaces nn.Linear layers in the model.
    If `initialize_weights` is True, it performs SVD to initialize the new layers.
    If False, it only creates the structure, assuming weights will be loaded.
    """
    for full_name, module in list(model.named_modules()):
        if target_substring in full_name and isinstance(module, nn.Linear):
            name_parts = full_name.split('.')
            parent_name = '.'.join(name_parts[:-1])
            local_name = name_parts[-1]
            parent_module = model.get_submodule(parent_name)

            in_features = module.in_features
            out_features = module.out_features
            
            new_layer = DynamicLowRankLinear(
                in_features, out_features, max_rank, bias=(module.bias is not None)
            )

            if initialize_weights:
                original_weight = module.weight.detach().clone()
                
                # Perform SVD to initialize the new layer with pre-trained knowledge
                U, S, Vh = torch.linalg.svd(original_weight.float(), full_matrices=False)
                
                # Truncate to the desired rank
                r = min(max_rank, S.size(0)) 
                U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
                
                # Decompose into two matrices A and B
                S_r_sqrt = torch.diag(torch.sqrt(S_r))
                B_init = U_r @ S_r_sqrt
                A_init = S_r_sqrt @ Vh_r

                # Copy the initialized weights into the new layer
                # Note: B_init is [out, r] and A_init is [r, in]. This is correct.
                new_layer.B.weight.data.copy_(B_init.to(original_weight.dtype))
                new_layer.A.weight.data.copy_(A_init.to(original_weight.dtype))
                
                if module.bias is not None:
                    new_layer.B.bias.data.copy_(module.bias.detach().clone())

                print(f"Replaced layer: {module} with dynamic low-rank linear")
            setattr(parent_module, local_name, new_layer)
    return model


# ==============================================================================
# 3. CUSTOM MODEL CLASS
# This class ensures the model is built correctly during `from_pretrained`.
# ==============================================================================
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM

class CustomGemmaForCausalLM(GemmaForCausalLM):
    """
    A custom Gemma model class that automatically applies low-rank factorization
    to its MLP layers upon initialization, based on the rank stored in its config.
    """
    def __init__(self, config):
        # First, initialize the standard Gemma model
        super().__init__(config)
        
        # Now, check the config for our custom rank attribute.
        if hasattr(self.config, "low_rank_alpha"):
            max_rank = self.config.low_rank_alpha
            print(f"Detected low_rank_alpha in config. Applying low-rank structure with rank={max_rank}...")
            # IMPORTANT: We set initialize_weights=False here because we are only
            # building the model structure. The actual weights will be loaded
            # from the checkpoint file immediately after this.
            replace_linear_with_lowrank(self, max_rank=max_rank, target_substring='mlp', initialize_weights=False)


# ==============================================================================
# 4. MAIN SCRIPT FOR CONVERSION
# ==============================================================================

def main(args):
    """
    Main function to load a model, convert it, and save it with the
    correct custom configuration for later loading.
    """
    print(f"Loading original model: {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print("\nStarting conversion to low-rank and initializing with SVD...")
    # Here, we want to perform the SVD, so we call with initialize_weights=True (the default)
    model = replace_linear_with_lowrank(model, args.max_rank, args.target_substring, initialize_weights=True)
    
    # --- Configure the model for successful loading ---
    # 1. Store the rank in the config. This is the crucial step.
    model.config.low_rank_alpha = args.max_rank
    
    # 2. Map the AutoModel class to our new custom class.
    model.config.auto_map = {
        "AutoModelForCausalLM": "convert_to_lowrank.CustomGemmaForCausalLM",
    }

    print(f"\nConversion complete. Saving converted model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the model (which now has the updated config) and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Copy the script itself into the output directory
    shutil.copyfile(__file__, os.path.join(args.output_dir, "convert_to_lowrank.py"))
    
    print("\nModel saved successfully.")
    print(f"To load, use: model = AutoModelForCausalLM.from_pretrained('{args.output_dir}', trust_remote_code=True)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a Hugging Face model's linear layers to dynamic low-rank layers.")
    parser.add_argument('--model_id', type=str, default='google/gemma-2b',
                        help='The Hugging Face model ID to load.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The directory to save the converted model.')
    parser.add_argument('--target_substring', type=str, default='mlp',
                        help="Substring to identify target layers for replacement (e.g., 'mlp', 'self_attn').")
    parser.add_argument('--max_rank', type=int, default=64,
                        help='The maximum rank for the low-rank decomposition.')
    
    args = parser.parse_args()
    main(args)
