import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("./logs/train.log"), logging.StreamHandler()],
)


#########################################################################################################
#######################################     Training Code    ############################################
#########################################################################################################
def load_model(id: str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        id,
        load_in_8bit=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(id)
    return model, tokenizer


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def preprocess_model(model, config):
    # Parameter freezing
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    model.embed_out = CastOutputToFloat(model.embed_out)

    # LORA
    config = LoraConfig(
        r=config["LORA_R"],
        lora_alpha=config["LORA_ALPHA"],
        lora_dropout=config["LORA_DROPOUT"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Add Low Rank Adapters + freezing
    model = get_peft_model(model, config)
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logging.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    return model


def generate_prompt(data_point):
    return f"""{data_point["instruction"]}

{data_point["output"]}"""


def load_data():
    data = load_dataset("json", data_files="./data/chris_train.json")

    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=256,
            padding="max_length",
        )
    )
    return data


def train(model, tokenizer, data, config):
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=config["PER_DEVICE_TRAIN_BATCH_SIZE"],
            gradient_accumulation_steps=config["GRADIENT_ACCUMULATION_STEPS"],
            warmup_steps=config["WARMUP_STEPS"],
            max_steps=config["MAX_STEPS"],
            learning_rate=config["LEARNING_RATE"],
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.train()
    return model


#########################################################################################################
#######################################   Execute training   ############################################
#########################################################################################################
if __name__ == '__main__':
    logging.info("Start loading model")
    model, tokenizer = load_model("databricks/dolly-v2-3b")
    logging.info("Model loaded")

    with open("./config.json", "r") as f:
        config = json.load(f)

    logging.info("Start model preprocessing")
    model = preprocess_model(model, config)
    logging.info("Model preprocessing finished")

    logging.info("Start data loading")
    data = load_data()
    logging.info("Data loaded")

    logging.info("Start Training")
    model = train(model, tokenizer, data, config)
    logging.info("Training finished")

    logging.info("Saving model locally")
    model.save_pretrained("/src/best_training/dolly-v2-3b-chris")

    logging.info("Saving model to hub")
    model.push_to_hub("chrishoertnagl/dolly-v2-3b-chris")
