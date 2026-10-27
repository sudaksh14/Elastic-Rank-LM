import torch
from torch import nn
from transformers import GemmaPreTrainedModel, GemmaModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import math

from lowrank.models.layers import DynamicLowRankLinear, replace_linear_with_lowrank, decompose_and_replace


class CustomGemmaForSequenceClassification(GemmaPreTrainedModel):
    def __init__(self, config, train_ranks=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # This creates the base GemmaModel with standard nn.Linear layers
        self.model = GemmaModel(config) 
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        # --- Low-Rank Specific Additions ---
        self.max_rank = 1024 # Or get from config
        
        # Define the ranks you want to train
        if train_ranks is None:
            self.train_ranks = [self.max_rank // (2**k) for k in range(int(math.log2(self.max_rank)) + 1)]
        else:
            self.train_ranks = train_ranks
        
        # Create learnable uncertainty parameters for each rank
        self.log_vars_ce = nn.ParameterDict({
            str(r): nn.Parameter(torch.zeros(())) for r in self.train_ranks
        })

        # Initialize weights and apply final processing
        self.post_init()

    def _set_rank_all(self, rank):
        """Sets the active rank for all DynamicLowRankLinear layers in the network."""
        for m in self.modules():
            if isinstance(m, DynamicLowRankLinear):
                m.set_rank(rank)

    @classmethod
    def from_pretrained_with_lowrank(
        cls,
        model_name_or_path: str,
        max_rank: int,
        train_ranks: list = None,
        **kwargs,
    ):

        model = cls.from_pretrained(model_name_or_path, **kwargs)
        decompose_and_replace(model.model, max_rank=max_rank, target_substring='mlp')

        # Attach the custom attributes
        model.max_rank = max_rank
        if train_ranks is None:
            model.train_ranks = [
                max_rank // (2**k) for k in range(int(math.log2(max_rank)) + 1)
            ]
        else:
            model.train_ranks = train_ranks
            
        model.log_vars_ce = nn.ParameterDict({
            str(r): nn.Parameter(torch.zeros(())) for r in model.train_ranks
        }).to(model.device)

        print("Successfully created low-rank model from pre-trained weights.")
        return model
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        # The code expects the logits for the last token of the sequence for classification
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            # Find the index of the last non-padding token
            sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(logits.device)
            # Gather the logits at the last token position
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        else:
            pooled_logits = logits[:, -1, :] # Fallback for inputs_embeds

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
