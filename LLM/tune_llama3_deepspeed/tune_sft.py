from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import bitsandbytes as bnb
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

DTYPE = "bfloat16"

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_model_tokenizer(model_path: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=DTYPE,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config, 
    )
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=find_all_linear_names(model),
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer




if __name__ == "__main__":
    # TODO fix these 2:
    MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    OUTPUT_PATH = "/llm/model/test"
    DATA_PATH = "/llm/data/train.jsonl"
    TEXT_COL = "text"
    dataset = load_dataset('json', data_files=[DATA_PATH])['train']
    config = {
        "epochs": 1,
        "lr": 0.00001,
        "max_length": 128,
        "batch_size": 1,
        "accumulate_grad_batches": 4,
        "precision": "16-mixed",
        "gradient_clip_val": 1.0,
    }
    model, tokenizer = get_model_tokenizer(MODEL_PATH)

    def make_prompt(sample):
        sample[TEXT_COL] = tokenizer.apply_chat_template(conversation=sample['messages'], tokenize=False)
        return sample
    train_dataset = dataset.map(make_prompt)
    print(train_dataset)
    print(train_dataset["text"][0])

    training_arguments = TrainingArguments(
        output_dir=OUTPUT_PATH,                    # directory to save and repository id
        num_train_epochs=config["epochs"],                       # number of training epochs
        per_device_train_batch_size=config["batch_size"],            # batch size per device during training
        gradient_accumulation_steps=config["accumulate_grad_batches"],            # number of steps before performing a backward/update pass
        gradient_checkpointing=True,              # use gradient checkpointing to save memory
        logging_steps=1,                         
        learning_rate=2.0e-04,
        bf16=True
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=train_dataset,
        dataset_text_field=TEXT_COL,
        max_seq_length=config["max_length"],
    )
    trainer.train()


