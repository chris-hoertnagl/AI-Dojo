# coding=utf-8
# Copyright 2024 Sourab Mangrulkar. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import gc
import os
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from peft import LoraConfig, replace_lora_weights_loftq
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
current_mse = float("inf")


def create_datasets(tokenizer, data_args):
    dataset = load_dataset('json', data_files=[data_args.data_file_path])
    # dataset = Dataset.from_dict(dataset['train'][:40])
    dataset = dataset['train'] 
    dataset
    
    def to_chat_template(sample):
        sample['text'] = tokenizer.apply_chat_template(sample['messages'], tokenize=False)
        return sample
    messages_dataset = dataset.map(to_chat_template, remove_columns=dataset.features)
    messages_dataset

    return messages_dataset, None


def create_and_prepare_model(args, data_args, training_args):
    bnb_config = None
    quant_storage_stype = None
    load_in_8bit = args.use_8bit_qunatization
    load_in_4bit = args.use_4bit_quantization

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_stype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_stype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

    torch_dtype = quant_storage_stype if quant_storage_stype and quant_storage_stype.is_floating_point else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        torch_dtype=torch_dtype,
    )

    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer

def get_mae(x, y):
    return (x - y).abs().mean()


def get_mse(x, y):
    return torch.pow(x - y, 2).mean()


def error_report(x, y):
    mae = get_mae(x, y)
    mse = get_mse(x, y)
    print(
        f"Mean absolute error: {mae:>8.5f}\n"
        f"Mean squared error:  {mse:>8.5f}"
    )


def loftq_init(model, tokenizer, train_dataset, max_seq_length, args):
    if args.use_loftq_callback:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=compute_dtype)
        base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        random_input_ids = torch.randint(0, len(train_dataset), size=(1,)).numpy().tolist()
        random_inputs = [train_dataset[i]['content'] for i in random_input_ids]
        random_inputs = tokenizer(random_inputs, return_tensors="pt", padding=True, truncation="max_length", max_length=max_seq_length)
        logits_base = base_model(**random_inputs).logits
        del base_model
        gc.collect()
        
        def loftq_callback(model, module_name):
            """Callable to replace weights with LoFTQ if the mse is lower than the current best one."""
            global current_mse
            logits = model(**random_inputs).logits
            mse = get_mse(logits_base, logits)
            if mse < current_mse:
                current_mse = mse
                print(f"MSE improved for module {module_name}")
                return True
            print(f"MSE did not improve for module {module_name}")
            return False
        
        replace_lora_weights_loftq(model, callback=loftq_callback)
        logits_loftq_callback = model(**random_inputs).logits
        error_report(logits_base, logits_loftq_callback)
    else:
        replace_lora_weights_loftq(model)
    

def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class