import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig


from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .configs import EConfig

# from suffix_decoding_simulator_v2 import *



class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path,use_fast=False)
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.ea_layer = Model(config,bias=bias,total_tokens=total_token,depth=depth,top_k=top_k,threshold=threshold)

        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()
        self.dumped_tree_mask=False

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            total_token=59,
            depth=5,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type=='Qwen2ForCausalLM':
            base_model=KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )



        if total_token==-1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans=[40,48,50,56,60]
            x=[1,1.05,1.07,1.1,1.13]
            times=[]

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token=cans[times.index(min(times))]
            model.ea_layer.total_tokens=total_token-1




        return model

    # Calls forward on the base model
    # returns output of the model (except lm head) and hidden states (i.e. outputs[0]). If output_orig=true, also output the output of the lm head
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            # Calling self.base_model.model() instead of self.base_model() to be able not to run the lm head
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states


    def populate_best_suffix_tree_candidates(self, prompt_tree, input_ids):
        max_prefix_length = min(len(input_ids), self.max_depth)
        candidate = ([], [])
        prefix_length = 0
        best_score = 0
        for length in range(1, max_prefix_length + 1):
            token_ids, parents, score = prompt_tree.find_best_path_or_tree(
                input_ids[len(input_ids) - length :],
                self.matching_strategy,
                self.max_spec_factor,
            )
            if score > best_score:
                candidate = (token_ids, parents)
                prefix_length = length
                best_score = score
        for length in range(1, max_prefix_length + 1):
            token_ids, parents, score = self.suffix_tree.find_best_path_or_tree(
                input_ids[len(input_ids) - length :],
                self.matching_strategy,
                self.max_spec_factor,
            )
            if score > best_score:
                candidate = (token_ids, parents)
                prefix_length = length
                best_score = score
        return candidate, prefix_length
    
    def generate_suffix_decoding_mask_and_positions(self, candidate):
        token_ids, parents = candidate
        tree_mask = torch.zeros((self.max_sd_tree_size +1, self.max_sd_tree_size +1)) # +1 for the root
        tree_position_ids = torch.zeros(self.max_sd_tree_size, dtype=torch.long)
        for i, parent in enumerate(parents):
            assert parent < i
            idx_to_copy_from = 0 if i == 0 else parent + 1
            assert idx_to_copy_from < i+1
            # Set causal relationship to all tokens with causal relationship to the parent
            tree_mask[i+1, :] = tree_mask[idx_to_copy_from, :]
            # Set causal relationship to self
            tree_mask[i+1, i+1] = 1
            # set position ids: position of current token = position of the parent + 1
            tree_position_ids[i] = tree_position_ids[parent] + 1
        return tree_mask, tree_position_ids
    
    def verify_and_commit(self, outputs):
        pass

    @torch.no_grad()
    def suffix_dec_generate(self,
                            input_ids,
                            temperature=0.0,
                            top_p=0.0,
                            top_k=0.0,
                            max_new_tokens=512,
                            max_length=2048,
                            log=False,
                            is_llama3=False):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        assert temperature == 0.0, "Suffix decoding does not support temperature"
        assert top_p == 0.0, "Suffix decoding does not support top_p"
        assert top_k == 0.0, "Suffix decoding does not support top_k"
        input_ids = input_ids.clone()
        # Initialize prompt tree
        prompt_tree = SuffixTree([input_ids], max_depth=self.max_depth)
        # Initialize the KV cache
        self.ea_layer.reset_kv()
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # prefilling
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        
        new_tokens = 0
        for new_token_idx in range(max_length):
            # generate predictions
            candidate, prefix_length = self.populate_best_suffix_tree_candidates(prompt_tree, input_ids)
            # Build tree mask, position ids
            tree_mask, tree_position_ids = self.generate_suffix_decoding_mask_and_positions(candidate)
            self.base_model.model.tree_mask = tree_mask
            position_ids = tree_position_ids + input_ids.shape[1]
            # Verification LLM
            outputs = self.base_model(input_ids, past_key_values=past_key_values, position_ids=position_ids, use_cache=False)
            # Figure out which tokens were accepted
            accepted_tokens, input_ids = self.verify_and_commit(outputs, input_ids)
            # Update KV cache to store committed tokens (accepted tokens + bonus)
            self.update_kv_cache(accepted_tokens, input_ids)
            print("new input_ids: ", input_ids)

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_tokens > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        print(f"Calling eagenerate with params: temperature={temperature}, top_p={top_p}, top_k={top_k}, max_new_tokens={max_new_tokens}, max_length={max_length}, log={log}, is_llama3={is_llama3}")
        max_length=max_length-self.ea_layer.total_tokens-10
        print(f"self.ea_layer.total_tokens={self.ea_layer.total_tokens}, max_length={max_length}")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        print("padding: ", padding)

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        print("input_len: ", input_len)
        # print("past_key_values: ", past_key_values)
        print("Calling initalize tree...")
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        # Dump tree mask to file
        if not self.dumped_tree_mask:
            torch.save(tree_mask, "tree_mask.pt")
            self.dumped_tree_mask = True


        print("draft_tokens: ", draft_tokens)
        print("retrieve_indices: ", retrieve_indices)
        print("tree_mask: ", tree_mask)
        print("tree_position_ids: ", tree_position_ids)
        print("logits: ", logits)

        new_token = 0

        # Setting max_length to 2 for testing
        max_length = 2
        for idx in range(max_length):
            print("\tEntering loop with idx: ", idx)
            #with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            #retrieve_indices=tree_buffers["retrieve_indices"]
            #logits = logits[0, retrieve_indices]
            draft_tokens=torch.cat((draft_tokens,padding),dim=1)
            candidates=draft_tokens[0,retrieve_indices]
            print("candidates (draft tokens after rearranging by branch): ", candidates)
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            print("Accept lenght:", accept_length)
            print("Best candidate: ", best_candidate)
            #with Timer("update_inference_inputs"):
            print("calling update_inference_inputs...")
            input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )
            print("new input_ids: ", input_ids)

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx


    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # prefilling
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0

        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                # extract last token for incremental decoding
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            # append last token
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token+=1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    # for webui
    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length=max_length-self.ea_layer.total_tokens-10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0

        for idx in range(max_length):
            #with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            #retrieve_indices=tree_buffers["retrieve_indices"]
            #logits = logits[0, retrieve_indices]
            draft_tokens=torch.cat((draft_tokens,padding),dim=1)
            candidates=draft_tokens[0,retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            #with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    # for webui
    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0


        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids



            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break



