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

from suffix_decoding_simulator_v2 import *



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
            ea_layer_state_dict,
            suffix_tree_trace_filepath,
            suffix_tree_partition_name,
            suffix_tree_matching_strategy,
            suffix_tree_max_spec_factor,
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

        # Suffix Tree initialization
        self.suffix_tree = None
        self.suffix_tree_max_depth = 64
        self.suffix_tree_trace_filepath = suffix_tree_trace_filepath
        self.suffix_tree_partition_name = suffix_tree_partition_name
        self.suffix_tree_matching_strategy = suffix_tree_matching_strategy
        self.suffix_tree_max_spec_factor = suffix_tree_max_spec_factor
        assert((self.suffix_tree_trace_filepath is None) == (self.suffix_tree_partition_name is None))
        if not self.suffix_tree_trace_filepath is None:
            self.init_suffix_tree(self.suffix_tree_trace_filepath, self.suffix_tree_partition_name)
        


    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer
    
    def init_suffix_tree(self, trace_filepath: str, partition_name: str):
        if self.suffix_tree is not None:
            raise ValueError("Suffix tree is already initialized.")
        if self.suffix_tree_max_depth <= 0:
            raise ValueError("Suffix tree max depth must be greater than 0.")
        if not os.path.exists(trace_filepath):
            raise ValueError(f"Trace file {trace_filepath} does not exist.")
        trace = load_trace(trace_filepath)
        training_data = None
        for partition in trace.partitions:
            if partition.partition_name == partition_name:
                training_data = [self.tokenizer.encode(training_entry.response, add_special_tokens=False) for training_entry in partition.training_entries]
                break
        if training_data is None:
            raise ValueError(f"Partition {partition_name} not found in trace.")
        
        print(f"Constructing suffix tree for partition {partition_name} with {len(training_data)} training entries.")
        start_time = time.time()
        self.suffix_tree = SuffixTree(
            prompt_response_pairs=training_data,
            max_depth=self.suffix_tree_max_depth,
        )
        end_time = time.time()
        print(f"Suffix tree construction completed in {end_time - start_time:.2f} seconds.")

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
            suffix_tree_trace_filepath=None,
            suffix_tree_partition_name=None,
            suffix_tree_matching_strategy=MatchingStrategy.DYNAMIC_TOKEN_TREE,
            suffix_tree_max_spec_factor=4.0,
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
            ea_layer_state_dict,
            suffix_tree_trace_filepath,
            suffix_tree_partition_name,
            suffix_tree_matching_strategy,
            suffix_tree_max_spec_factor,
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
        max_prefix_length = min(len(input_ids), self.suffix_tree_max_depth)
        candidate = ([], [])
        prefix_length = 0
        best_score = 0
        for length in range(1, max_prefix_length + 1):
            token_ids, parents, score = prompt_tree.find_best_path_or_tree(
                input_ids[len(input_ids) - length :],
                self.suffix_tree_matching_strategy,
                self.suffix_tree_max_spec_factor,
                self.ea_layer.total_tokens-1,
            )
            if score > best_score:
                candidate = (token_ids, parents)
                prefix_length = length
                best_score = score
        for length in range(1, max_prefix_length + 1):
            token_ids, parents, score = self.suffix_tree.find_best_path_or_tree(
                input_ids[len(input_ids) - length :],
                self.suffix_tree_matching_strategy,
                self.suffix_tree_max_spec_factor,
                self.ea_layer.total_tokens-1,
            )
            if score > best_score:
                candidate = (token_ids, parents)
                prefix_length = length
                best_score = score
        return candidate[0], candidate[1], prefix_length
    
    def generate_suffix_decoding_mask_and_positions(self, draft_token_ids, draft_token_parents, input_ids):
        assert(len(draft_token_ids) == len(draft_token_parents))
        assert(input_ids is not None)
        assert(input_ids.shape[0] == 1)
        assert(input_ids.shape[1] >= 1)
        
        draft_tokens, tree_mask, tree_position_ids = input_ids[:,-1:], None, None
        if len(draft_token_parents) > 0:
            tree_mask = torch.zeros((self.ea_layer.total_tokens, self.ea_layer.total_tokens))
            tree_position_ids = torch.zeros(len(draft_token_parents), dtype=torch.long)
            for i, parent in enumerate(draft_token_parents):
                assert parent < i
                idx_to_copy_from = 0 if i == 0 else parent + 1
                assert idx_to_copy_from < i+1
                # Set causal relationship to all tokens with causal relationship to the parent
                tree_mask[i+1, :] = tree_mask[idx_to_copy_from, :]
                # Set causal relationship to self
                tree_mask[i+1, i+1] = 1
                # set position ids: position of current token = position of the parent + 1
                tree_position_ids[i] = tree_position_ids[parent] + 1
            tree_position_ids = tree_position_ids + input_ids.shape[1]-1
        return draft_tokens, tree_mask, tree_position_ids

    def verify_and_commit(self, draft_tokens, draft_token_parents, logits, past_key_values, past_key_values_data, current_length_data):
        accepted_length = 0
        # Root is always accepted
        accepted_tokens = draft_tokens[:,0:1]
        # remove root from draft tokens
        draft_tokens = draft_tokens[:,1:]
        correct_tokens = torch.argmax(logits, dim=-1)
        print("draft tokens.shape: ", draft_tokens.shape)
        print("draft tokens: ", draft_tokens)
        print("correct tokens.shape: ", correct_tokens.shape)
        print("correct tokens: ", correct_tokens)
        return accepted_tokens, accepted_length       

    @torch.no_grad()
    def suffix_decoding_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=3500,
            max_length=8200,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length=max_length-self.ea_layer.total_tokens-10

        assert temperature == 0.0, "Suffix decoding does not support temperature"
        assert top_p == 0.0, "Suffix decoding does not support top_p"
        assert top_k == 0.0, "Suffix decoding does not support top_k"

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize prompt tree
        prompt_tree = SuffixTree([input_ids.squeeze().tolist()], max_depth=self.suffix_tree_max_depth)


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
        # Prefilling
        outputs, logits, hidden_states = self(
            input_ids, 
            past_key_values=past_key_values, 
            output_orig=True,
        )
        bonus_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # Append to input_ids
        input_ids = torch.cat([input_ids, bonus_token.to(input_ids.device)], dim=-1)

        new_token = 1


        # speculated_tokens_per_step= []
        # accepted_tokens_per_step= []
        # generated_tokens_per_step= []
        # speculation_times = [spec_time]

        print("Prefilled ", input_len, "tokens")
        print("Input ids shape: ", input_ids.shape)
        print("Input ids: ", input_ids)
        print("Bonus token shape: ", bonus_token.shape)
        print("Bonus token: ", bonus_token)
       
        # print("retrieve_indices.shape", retrieve_indices.shape)
        # print("retrieve_indices: ", retrieve_indices)
        
        print("logits.shape: ", logits.shape)
        print("logits", logits)
        # print("past_key_values_data.shape: ", past_key_values_data.shape)
        # print("len(past_key_values_data)", len(past_key_values_data))
        # print("past_key_values_data: ", past_key_values_data)
        # print("current_length_data.shape: ", current_length_data.shape)
        # print("current_length_data: ", current_length_data)


        for idx in range(max_length):
            print("\nStep ", idx)

            # generate predictions
            draft_tokens_ids, draft_token_parents, prefix_length = self.populate_best_suffix_tree_candidates(prompt_tree, input_ids)
            num_speculated_tokens = len(draft_tokens_ids)

            # print("Draft tokens shape: ", draft_tokens.shape)
            print("Draft token ids: ", draft_tokens_ids)
            print("Draft tokens parents: ", draft_token_parents)
            print("Prefix length: ", prefix_length)
            # print("tree_mask.shape: ", tree_mask.shape)
            # print("tree_mask: ", tree_mask)

            # Build tree mask, position ids
            draft_tokens, tree_mask, tree_position_ids = self.generate_suffix_decoding_mask_and_positions(draft_tokens_ids, draft_token_parents, input_ids)
            print("Draft tokens: ", draft_tokens)
            print("Tree mask: ", tree_mask)
            print("Tree position ids: ", tree_position_ids)
            self.base_model.model.tree_mask = tree_mask

            assert (tree_mask is None) == (tree_position_ids is None)
            if tree_mask is not None:
                assert num_speculated_tokens == len(tree_position_ids.squeeze())

            
            # Verification LLM
            outputs, logits, hidden_states = self(
                draft_tokens, 
                past_key_values=past_key_values, 
                position_ids=tree_position_ids, 
                output_orig=True,
            )
            # Figure out which tokens were accepted, and update the kv cache. Root is always accepted
            accepted_tokens, accepted_length = self.verify_and_commit(draft_tokens, draft_token_parents, logits, past_key_values, past_key_values_data, current_length_data)
            print("Accepted tokens.shape: ", accepted_tokens.shape)
            print("Accepted tokens: ", accepted_tokens)
            # Append tokens to the input_ids
            input_ids = torch.cat([input_ids, accepted_tokens.to(input_ids.device)], dim=-1)
            # Update KV cache to store committed tokens (accepted tokens + bonus)
            
            print("new input_ids: ", input_ids)
            
            
            # num_accepted_tokens = accept_length.item()
            # num_generated_tokens = num_accepted_tokens + 1 # bonus token
            # speculated_tokens_per_step.append(num_speculated_tokens)
            # accepted_tokens_per_step.append(num_accepted_tokens)
            # generated_tokens_per_step.append(num_generated_tokens)
            # speculation_times.append(spec_time)

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
            assert False
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx, speculated_tokens_per_step, accepted_tokens_per_step, generated_tokens_per_step, speculation_times

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=3500,
            max_length=8200,
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
        print("Input ids BEFORE entering initialize_tree: ", input_ids)
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token, spec_time = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        print("Input ids AFTER entering initialize_tree: ", input_ids)
        new_token = 0


        speculated_tokens_per_step= []
        accepted_tokens_per_step= []
        generated_tokens_per_step= []
        speculation_times = [spec_time]

        print("Prefilled ", input_len, "tokens")
        print("Input ids shape: ", input_ids.shape)
        print("Input ids: ", input_ids)
        print("Draft tokens shape: ", draft_tokens.shape)
        print("Draft tokens: ", draft_tokens)
        print("retrieve_indices.shape", retrieve_indices.shape)
        print("retrieve_indices: ", retrieve_indices)
        print("tree_mask.shape: ", tree_mask.shape)
        print("tree_mask: ", tree_mask)
        print("logits.shape: ", logits.shape)
        print("logits", logits)
        # print("past_key_values_data.shape: ", past_key_values_data.shape)
        print("len(past_key_values_data)", len(past_key_values_data))
        print("past_key_values_data: ", past_key_values_data)
        print("current_length_data.shape: ", current_length_data.shape)
        print("current_length_data: ", current_length_data)


        for idx in range(max_length):
            #with Timer("all"):
            self.base_model.model.tree_mask = tree_mask
            print("Step ", idx)

            num_speculated_tokens = len(draft_tokens.squeeze())
            assert num_speculated_tokens == len(tree_position_ids.squeeze())

            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            print("verification...")
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
            
            print("draft tokens after applying padding: ", draft_tokens.shape)
            print(draft_tokens)
            print("candidates after applying retrieve_indices: ", candidates.shape)
            print(candidates)

            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            print("best candidate.shape: ", best_candidate.shape)
            print("best candidate: ", best_candidate)
            print("accept length.shape: ", accept_length.shape)
            print("accept length: ", accept_length)
            
            
            num_accepted_tokens = accept_length.item()
            num_generated_tokens = num_accepted_tokens + 1 # bonus token
            speculated_tokens_per_step.append(num_speculated_tokens)
            accepted_tokens_per_step.append(num_accepted_tokens)
            generated_tokens_per_step.append(num_generated_tokens)

            # print(accept_length)
            #with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token, spec_time = update_inference_inputs(
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
            speculation_times.append(spec_time)

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
            assert False
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx, speculated_tokens_per_step, accepted_tokens_per_step, generated_tokens_per_step, speculation_times


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



