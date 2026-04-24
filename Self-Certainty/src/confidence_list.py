import argparse
import json
import os
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tqdm
import torch.nn.functional as F
import pandas as pd
import numpy as np

# Mistral-native models require Mistral3ForConditionalGeneration instead of
# AutoModelForCausalLM. They use tekken.json tokenizer and a multimodal
# architecture wrapper even for text-only variants.
MISTRAL_NATIVE_MODELS = ["ministral", "pixtral", "mistral-large"]


def _is_mistral_native(model_dir: str) -> bool:
    """Check if a model needs Mistral3ForConditionalGeneration."""
    return any(m in model_dir.lower() for m in MISTRAL_NATIVE_MODELS)


def _ensure_ministral3_registered():
    """Register 'ministral3' as alias for 'mistral3' in transformers CONFIG_MAPPING.

    Ministral models (e.g., Ministral-3-14B) use model_type='ministral3' in
    their config.json, but some transformers versions only register 'mistral3'.
    Without this alias, AutoConfig.from_pretrained() raises KeyError: 'ministral3'.
    """
    try:
        from transformers import CONFIG_MAPPING
        if "ministral3" not in CONFIG_MAPPING:
            if "mistral3" in CONFIG_MAPPING:
                from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
                CONFIG_MAPPING.register("ministral3", Mistral3Config)
                print("  Registered 'ministral3' as alias for 'mistral3' in CONFIG_MAPPING", flush=True)
            else:
                print("  Warning: 'mistral3' not in CONFIG_MAPPING either, "
                      "transformers may be too old for Mistral3 support", flush=True)
    except Exception as e:
        print(f"  Warning: Could not register ministral3 alias: {e}", flush=True)

def confidence_logprob_sum(logprob_sum: torch.Tensor, attention_mask: torch.Tensor, V: int):
    """
    Calculate the confidence of the logprob_sum.
    logprob_sum: torch.Tensor, shape (batch_size, seq_length) or (seq_length)
    attention_mask: torch.Tensor, shape (batch_size, seq_length) or (seq_length)
    V: int, the vocab size
    """
    logprob_sum = logprob_sum.contiguous()
    attention_mask = attention_mask.contiguous()
    V_tensor = torch.tensor(V, dtype=logprob_sum.dtype, device=logprob_sum.device)
    conf = -1/V * logprob_sum - torch.log(V_tensor)
    valid_conf = conf * attention_mask
    batch_confidence_list = (valid_conf.sum(dim=-1) / attention_mask.sum(dim=-1)).tolist()
    return batch_confidence_list


def _load_examples(filepath: str, output_field_name: str = "output"):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":                                   # existing path
        with open(filepath, "r") as f:
            return json.load(f)

    elif ext == ".parquet":                              # new path
        df = pd.read_parquet(filepath)                   # uses pyarrow by default
        # If the column `output` is a JSON-encoded string instead of a Python list,
        # decode it so the rest of the pipeline stays unchanged.
        if df[output_field_name].dtype == object and isinstance(df.iloc[0][output_field_name], str):
            df[output_field_name] = df[output_field_name].apply(json.loads)

        return df.to_dict(orient="records")              # == list[dict] like your JSON file

    else:
        raise ValueError(f"Unsupported input format: {ext}")

@torch.no_grad()
def confidence_with_file(filepath, output_file=None, batch_size=4, model_dir=None,
                         input_field_name="model_input", output_field_name="output",
                         device_map_strategy=None):
    data = _load_examples(filepath, output_field_name)

    if model_dir is None:
        model_dir = data[0]["generator"]

    print("Loading model:", model_dir, flush=True)
    torch.set_grad_enabled(False)

    is_mistral = _is_mistral_native(model_dir)

    if is_mistral:
        # Mistral-native models (Ministral, Pixtral, mistral-large) use
        # Mistral3ForConditionalGeneration — a multimodal architecture wrapper.
        # AutoModelForCausalLM cannot load them.
        _ensure_ministral3_registered()
        from transformers import Mistral3ForConditionalGeneration
        print(f"  Detected Mistral-native model, using Mistral3ForConditionalGeneration", flush=True)

        load_kwargs = dict(
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        # FP8 quantized models: dequantize to bf16 for stable forward pass
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            qcfg = getattr(cfg, 'quantization_config', None)
            if qcfg and (isinstance(qcfg, dict) and qcfg.get('quant_method') == 'fp8'
                         or hasattr(qcfg, 'quant_method') and qcfg.quant_method == 'fp8'):
                from transformers import FineGrainedFP8Config
                load_kwargs['quantization_config'] = FineGrainedFP8Config(dequantize=True)
                print(f"  FP8 model detected, dequantizing to bf16", flush=True)
            else:
                load_kwargs['torch_dtype'] = torch.float16
        except Exception:
            load_kwargs['torch_dtype'] = torch.float16

        llm = Mistral3ForConditionalGeneration.from_pretrained(model_dir, **load_kwargs)
        input_device = next(llm.parameters()).device
        if hasattr(llm, 'hf_device_map'):
            devices_used = sorted(set(str(v) for v in llm.hf_device_map.values()))
            print(f"  Model distributed across devices: {devices_used}", flush=True)

    elif device_map_strategy == "auto":
        # Multi-GPU: spread model layers across all visible GPUs via pipeline parallelism.
        # Used for models too large to fit on a single GPU (e.g., 120B in fp16 ≈ 240GB).
        print(f"  Loading model with device_map='auto' (multi-GPU)...", flush=True)
        llm = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        input_device = next(llm.parameters()).device
        if hasattr(llm, 'hf_device_map'):
            devices_used = sorted(set(str(v) for v in llm.hf_device_map.values()))
            print(f"  Model distributed across devices: {devices_used}", flush=True)
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            if mem > 0:
                print(f"    GPU {i}: {mem:.1f}GB / {total:.1f}GB", flush=True)
    else:
        # Single-GPU: load entire model to one device.
        # Use CUDA if available; load directly to GPU to avoid CPU RAM copy.
        # The chained `.from_pretrained(...).to(device)` pattern loads the full
        # model into CPU RAM first, causing OOM when 8 workers run simultaneously
        # (e.g., 8 × 64GB = 512GB CPU RAM for 32B models).
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Loading model weights to {device} (low_cpu_mem_usage=True)...", flush=True)
        llm = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map={"": device} if device.type == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        input_device = device
        print(f"  Model loaded. GPU memory: {torch.cuda.memory_allocated(device)/1024**3:.1f}GB / "
              f"{torch.cuda.get_device_properties(device).total_memory/1024**3:.1f}GB", flush=True)

    # Resolve vocab_size and hidden_size — multimodal wrappers store them in text_config
    if hasattr(llm.config, 'text_config'):
        _vocab_size = llm.config.text_config.vocab_size
        _hidden_size = llm.config.text_config.hidden_size
    else:
        _vocab_size = llm.config.vocab_size
        _hidden_size = llm.config.hidden_size

    tokenizer = None
    # Mistral models' tokenizer_config.json references 'TokenizersBackend' (from
    # mistral_common), which AutoTokenizer cannot resolve. Additionally,
    # PreTrainedTokenizerFast.from_pretrained() fails because tokenizer_config.json
    # has list-format fields incompatible with HF's parser.
    # Solution: load tokenizer.json directly via tokenizers.Tokenizer, bypassing
    # tokenizer_config.json entirely. This mirrors vLLM's tokenizer_mode="mistral".
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, padding=True, trust_remote_code=True)
    except Exception as tok_err:
        if is_mistral:
            print(f"  AutoTokenizer failed ({tok_err}), trying direct tokenizer.json load...", flush=True)
            # Load tokenizer.json directly, bypassing tokenizer_config.json
            try:
                from tokenizers import Tokenizer as HFTokenizer
                from transformers import PreTrainedTokenizerFast

                # Resolve tokenizer.json path (local dir or HuggingFace Hub)
                tokenizer_json_path = os.path.join(model_dir, "tokenizer.json") if os.path.isdir(model_dir) else None
                if tokenizer_json_path is None or not os.path.exists(tokenizer_json_path):
                    from huggingface_hub import hf_hub_download
                    tokenizer_json_path = hf_hub_download(repo_id=model_dir, filename="tokenizer.json")

                raw_tokenizer = HFTokenizer.from_file(tokenizer_json_path)
                tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
                print(f"  Loaded tokenizer from tokenizer.json directly (type: {type(tokenizer).__name__})", flush=True)
            except Exception as e2:
                raise RuntimeError(
                    f"Cannot load tokenizer for Mistral model {model_dir}. "
                    f"AutoTokenizer error: {tok_err}. "
                    f"Direct tokenizer.json error: {e2}"
                ) from e2
        else:
            raise

    # Add padding token if it is not already present.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        llm.config.pad_token_id = tokenizer.pad_token_id
        llm.resize_token_embeddings(len(tokenizer))
        llm.embed_tokens = torch.nn.Embedding(
            _vocab_size, _hidden_size, padding_idx=llm.config.pad_token_id
        ).to(input_device)
        print("Added padding token to tokenizer")
        
    tokenizer.padding_side = "right"
    
    llm.eval()
    print("Loaded model and tokenizer.", flush=True)

    # Determine the output file path
    if output_file is None:
        output_file = os.path.splitext(filepath)[0] + f"-confidence-list.json"

    # Use JSONL as intermediate progress format to avoid O(N²) re-serialization
    # and unbounded memory growth. Each processed item is appended as one line.
    # Final JSON array is assembled only after all items are done.
    progress_file = output_file + '.progress.jsonl'

    # Determine resume offset from existing progress or output file
    processed_items = 0
    if os.path.exists(progress_file):
        # Resume from JSONL progress file
        with open(progress_file, 'r') as f:
            for line in f:
                if line.strip():
                    processed_items += 1
        print(f"Resuming from JSONL progress: {processed_items} items already done.", flush=True)
    elif os.path.exists(output_file):
        # Backward compat: resume from old JSON array output
        try:
            with open(output_file, "r") as f_out:
                old_data = json.load(f_out)
                processed_items = len(old_data)
                print(f"Found existing JSON output with {processed_items} items.", flush=True)
                # Convert to JSONL progress file so we don't re-load it again
                with open(progress_file, 'w') as pf:
                    for item in old_data:
                        pf.write(json.dumps(item, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o) + '\n')
                del old_data  # free memory immediately
                print(f"Converted to JSONL progress format.", flush=True)
        except json.JSONDecodeError:
            print("Output file is corrupted. Starting fresh.", flush=True)

    total_items = len(data)
    print(f"Total items to process: {total_items}. Already processed: {processed_items}.", flush=True)
    
    for index in tqdm.tqdm(range(processed_items, total_items), desc="Processing inputs"):
        item = data[index]
        new_item = {k: v for k, v in item.items()}

        # Encode the input prompt.
        input_encoded = tokenizer(
            item[input_field_name],
            return_tensors="pt",
            padding=False,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = input_encoded['input_ids'].reshape(-1)
        input_attention_mask = input_encoded['attention_mask'].reshape(-1)
        input_length = input_attention_mask.sum().item()  # Actual token count of the prompt

        # Retrieve the top N outputs.
        outputs = item[output_field_name]
        print(f"\n  [{index+1}/{total_items}] input_tokens={input_length}, "
              f"num_outputs={len(outputs)}, "
              f"output_chars=[{min(len(o) for o in outputs)}-{max(len(o) for o in outputs)}]",
              flush=True)

        # Classify outputs based on their raw text length (before tokenization).
        # Conditions ordered from largest to smallest to avoid short-circuit bug.
        groups = {
            "small": {"outputs": [], "indices": []},
            "medium": {"outputs": [], "indices": []},
            "large": {"outputs": [], "indices": []}
        }
        for idx, text in enumerate(outputs):
            if len(text) > 6 * 1024:
                groups["large"]["outputs"].append(text)
                groups["large"]["indices"].append(idx)
            elif len(text) > 3 * 1024:
                groups["medium"]["outputs"].append(text)
                groups["medium"]["indices"].append(idx)
            else:
                groups["small"]["outputs"].append(text)
                groups["small"]["indices"].append(idx)
        
        # Prepare a list for the final confidence scores (in the original order).
        final_confidences = [None] * len(outputs)
        
        # Define batch sizes for each group.
        group_batch_sizes = {
            "small": batch_size,
            "medium": max(1, batch_size // 2),
            "large": max(1, batch_size // 4)
        }
        
        # Process each group separately.
        for group_name in ["small", "medium", "large"]:
            group_texts = groups[group_name]["outputs"]
            group_indices = groups[group_name]["indices"]
            if not group_texts:
                continue

            current_batch_size = group_batch_sizes[group_name]
            print(f"    group={group_name}: {len(group_texts)} outputs, batch_size={current_batch_size}",
                  flush=True)

            # Tokenize the outputs in this group.
            group_tokenized = tokenizer(
                group_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
            group_outputs_ids = group_tokenized['input_ids']            # (n, seq_length_out)
            group_outputs_attention_mask = group_tokenized['attention_mask']  # (n, seq_length_out)
            print(f"      tokenized: output_seq_len={group_outputs_ids.shape[1]}, "
                  f"full_seq_len={input_length + group_outputs_ids.shape[1]}", flush=True)
            
            # Build full sequences by concatenating the prompt and each output.
            full_ids_list = []
            full_attention_mask_list = []
            for i in range(group_outputs_ids.size(0)):
                combined_ids = torch.cat((input_ids, group_outputs_ids[i]), dim=0)
                combined_attention_mask = torch.cat((input_attention_mask, group_outputs_attention_mask[i]), dim=0)
                full_ids_list.append(combined_ids)
                full_attention_mask_list.append(combined_attention_mask)
            full_ids = torch.stack(full_ids_list)            # shape: (n, total_seq_length)
            full_attention_mask = torch.stack(full_attention_mask_list)  # shape: (n, total_seq_length)
            
            # Process logits in batches to avoid CUDA OOM.
            # On OOM, automatically halve batch size and retry. If batch_size=1
            # still OOMs, assign -inf confidence to that sample and continue.
            group_confidences = []
            ptr = 0  # pointer into the group's full_ids
            while ptr < full_ids.shape[0]:
                effective_bs = min(current_batch_size, full_ids.shape[0] - ptr)
                computed = False
                while effective_bs >= 1:
                    torch.cuda.empty_cache()
                    end_idx = ptr + effective_bs
                    batch_ids = full_ids[ptr:end_idx].to(input_device)
                    batch_attention_mask = full_attention_mask[ptr:end_idx].to(input_device)
                    try:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            batch_logprob_sum = llm(batch_ids, attention_mask=batch_attention_mask).logits
                            batch_logprob_sum = batch_logprob_sum[:, input_length:, :]
                            batch_logprob_sum = F.log_softmax(batch_logprob_sum, dim=-1)
                            batch_logprob_sum = batch_logprob_sum.sum(dim=-1).to('cpu').to(torch.float32)

                        # Success — compute confidence for this batch
                        batch_output_attention_mask = group_outputs_attention_mask[ptr:end_idx]
                        batch_confidence_list = confidence_logprob_sum(
                            batch_logprob_sum, batch_output_attention_mask, _vocab_size
                        )
                        group_confidences.extend(batch_confidence_list)
                        ptr = end_idx
                        computed = True
                        break  # done with this chunk, advance ptr
                    except RuntimeError as e:
                        if "out of memory" not in str(e).lower():
                            raise
                        del batch_ids, batch_attention_mask
                        torch.cuda.empty_cache()
                        if effective_bs > 1:
                            effective_bs = max(1, effective_bs // 2)
                            print(f"    OOM: reducing batch size to {effective_bs} and retrying...")
                        else:
                            break  # bs=1 still OOMs, exit inner loop

                if not computed:
                    # batch_size=1 still OOMs — skip this one sample
                    print(f"    OOM at batch_size=1 (seq_len={full_ids.shape[1]}), "
                          f"assigning -inf confidence.")
                    group_confidences.append(float('-inf'))
                    ptr += 1
            
            # Place the computed confidences back in the correct (original) positions.
            for i, orig_idx in enumerate(group_indices):
                final_confidences[orig_idx] = group_confidences[i]
        
        if any(conf is None for conf in final_confidences):
            print(f"Warning: Some confidences were not computed for item at index {index}.")
        
        print("all_confidences:", final_confidences, flush=True)

        # Save the confidence list with the current item.
        new_item["confidence_list"] = final_confidences
        new_item["processed_index"] = index

        # Append to JSONL progress file (O(1) per item, no re-serialization).
        try:
            with open(progress_file, 'a') as pf:
                pf.write(json.dumps(
                    new_item,
                    default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o
                ) + '\n')
        except Exception as e:
            print(f"Error writing to progress file: {e}", flush=True)
            print("Exiting to prevent data loss.", flush=True)
            break

        print(f"Processed {index + 1}/{total_items}", flush=True)
        torch.cuda.empty_cache()

    # Assemble final JSON array from JSONL progress file.
    if os.path.exists(progress_file):
        print(f"\nAssembling final output from JSONL progress...", flush=True)
        all_results = []
        with open(progress_file, 'r') as pf:
            for line in pf:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))
        try:
            with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(output_file)) as tmp_file:
                json.dump(all_results, tmp_file, indent=4,
                          default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
                temp_name = tmp_file.name
            os.replace(temp_name, output_file)
            os.remove(progress_file)
            print(f"Final output: {output_file} ({len(all_results)} items)", flush=True)
        except Exception as e:
            print(f"Error assembling final output: {e}", flush=True)
            print(f"JSONL progress preserved at: {progress_file}", flush=True)

# Example usage:
# python3 script.py --input_file /path/to/input.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute confidence scores for model outputs using DataParallel for faster inference."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to the output file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing.")
    parser.add_argument("--model_name", type=str, default=None, help="Path to the model directory or huggingface model name.")
    parser.add_argument("--input_field_name", type=str, default="model_input", help="Field name for the input text in the input file.")
    parser.add_argument("--output_field_name", type=str, default="output", help="Field name for the output text in the input file.")
    parser.add_argument("--device_map", type=str, default=None, help="Device map strategy. Use 'auto' for multi-GPU pipeline parallelism (required for models >80B).")
    args = parser.parse_args()

    confidence_with_file(args.input_file, args.output_file, args.batch_size, args.model_name, args.input_field_name, args.output_field_name, args.device_map)
