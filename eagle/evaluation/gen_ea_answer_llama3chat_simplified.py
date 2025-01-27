"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from accelerate.utils import set_seed
set_seed(0)

import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

from ..model.ea_model import EaModel
from ..model.kv_cache import initialize_past_key_values
from ..model.utils import *

import sys
from dataclasses import asdict, dataclass, field
sys.path.append('/usr/suffix-tree-decoding/simulators')
from suffix_decoding_simulator_v2 import (
    TraceEntry,
    TracePartition,
    TraceMetadata,
    Trace,
    load_trace
)

@dataclass
class TraceEntryEA:
    prompt: str
    response: str
    ea_response: str
    prompt_length: int
    response_length: int
    ea_response_length: int
    num_decoding_steps: int = 0
    decoding_time: float = 0.0

def save_ea_trace(trace: Trace, output_path: str):
    # Convert the Trace instance to a dictionary
    trace_dict = asdict(trace)
    
    # Save the dictionary as a JSON file
    with open(output_path, 'w') as f:
        json.dump(trace_dict, f, indent=2)
    
    print(f"Trace saved to {output_path}")


@torch.inference_mode()
def run_eval(
        base_model_path,
        ea_model_path,
        trace_path,
        partitions,
        max_requests_per_partition,
        output_file,
        csv_output_file,
        max_spec_tokens,
        max_depth,
        top_k,
):
    trace = load_trace(trace_path)
    print("Trace metadata:")
    print(trace.metadata)
    # Filter partitions from the trace
    partition_names = partitions.split(",")
    trace.partitions = [trace.partitions[i] for i in range(len(trace.partitions)) if trace.partitions[i].partition_name in partition_names]

    model = EaModel.from_pretrained(
        base_model_path = base_model_path,
        ea_model_path = ea_model_path,
        total_token = max_spec_tokens,
        depth = max_depth,
        top_k = top_k,
        torch_dtype = torch.float16,
        low_cpu_mem_usage = True,
        device_map = "auto"
    )

    tokenizer = model.get_tokenizer()

    logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    warmup_entry = trace.partitions[0].eval_entries[0]
    print("Performing warmup with entry:", warmup_entry)

    # warmup
    for run_idx in range(3):
        torch.manual_seed(0)
        
        input_ids = tokenizer([warmup_entry.prompt], add_special_tokens=False,).input_ids

        # try:
        torch.cuda.synchronize()
        start_time = time.time()

        output_ids, new_tokens, num_decoding_steps = model.eagenerate(
            torch.as_tensor(input_ids).cuda(),
            temperature=0.0,
            log=True,
            is_llama3=True,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        output_ids = output_ids[0][len(input_ids[0]):]
        # be consistent with the template's stop_token_ids
        stop_token_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        if stop_token_ids:
            stop_token_ids_index = [
                i
                for i, id in enumerate(output_ids)
                if id in stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        # stop_str = "</s>"
        # if stop_str and output.find(stop_str) > 0:
        #     output = output[: output.find(stop_str)]
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()

        print(f"Warmup run {run_idx} took {total_time:.2f} seconds to generate {len(output_ids)} tokens (original: {warmup_entry.response_length}) with {num_decoding_steps} decoding steps.")
        
    print('Warmup done')

    assert (len(trace.partitions) == len(partition_names))
    for partition in trace.partitions:
        assert (partition.partition_name in partition_names)
        print("Processing partition:", partition.partition_name)

        new_entries = []
        for eval_entry_idx, entry in tqdm(enumerate(partition.eval_entries)):
            if eval_entry_idx >= max_requests_per_partition:
                new_entries.append(TraceEntryEA(
                    prompt=entry.prompt,
                    response=entry.response,
                    ea_response="",
                    prompt_length=entry.prompt_length,
                    response_length=entry.response_length,
                    ea_response_length=0,
                    num_decoding_steps=0,
                    decoding_time=0.0
                ))
                continue
            
            torch.manual_seed(0)
            input_ids = tokenizer([entry.prompt], add_special_tokens=False, ).input_ids

            # try:
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, num_new_tokens, decoding_steps = model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=0.0,
                log=True,
                is_llama3=True,
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            
            output_ids = output_ids[0][len(input_ids[0]):]
            output = tokenizer.decode(output_ids)

            print((int(num_new_tokens), len(output_ids), entry.response_length))

            new_entries.append(TraceEntryEA(
                prompt=entry.prompt,
                response=entry.response,
                ea_response=output,
                prompt_length=entry.prompt_length,
                response_length=entry.response_length,
                ea_response_length=len(output_ids),
                num_decoding_steps=decoding_steps,
                decoding_time=total_time
            ))

        partition.eval_entries = new_entries
    save_ea_trace(trace, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="yuhuili/EAGLE-LLaMA3-Instruct-70B"
    )
    parser.add_argument("--base-model-path", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument(
        "--trace-path",
        type=str,
        required=True,
        help="Path to the trace file in JSON format",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        default="all",
        help="Comma-separated list of partitions to run on.",
    )
    parser.add_argument(
        "--max-requests-per-partition",
        type=int,
        default=1_000_000,
        help="Max number of requests to run on each partition.",
    )
    parser.add_argument("--output-file", type=str, required=True, help="The output answer file.")
    parser.add_argument("--csv-output-file", type=str, required=True, help="The metrics output file.")
    parser.add_argument(
        "--max-spec-tokens",
        type=int,
        default=60,
        help="The maximum number of tokens to speculate.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="The maximum depth of the speculation tree.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()
    print("Loading trace from ", args.trace_path)
    # trace = load_trace(args.trace_path)
    # print("Trace metadata:")
    print(f"Output to {args.output_file}, with metrics in {args.csv_output_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.trace_path,
        args.partitions,
        args.max_requests_per_partition,
        args.output_file,
        args.csv_output_file,
        args.max_spec_tokens,
        args.max_depth,
        args.top_k
    )