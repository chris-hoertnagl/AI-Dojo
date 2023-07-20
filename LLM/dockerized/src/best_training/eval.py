import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./logs/eval.log"),
        logging.StreamHandler()
    ]
)

#########################################################################################################
#######################################    Inference Code    ############################################
#########################################################################################################

def generate_text(prompt: str, model: AutoModelForCausalLM, tokenizer:AutoTokenizer) -> str:
    batch = tokenizer(prompt, return_tensors='pt')

    with torch.cuda.amp.autocast():
      output_tokens = model.generate(**batch, max_new_tokens=50)

    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# generate_text = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         use_cache=True,
#         device_map="auto",
#         max_length=64,
#         do_sample=True,
#         top_k=10,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,
# )

# def generate_text(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     input_ids = inputs["input_ids"].cuda()
#     generation_output = model.generate(
#         input_ids=input_ids,
#         generation_config=GenerationConfig(temperature=1.0, top_p=0.95, num_beams=4),
#         return_dict_in_generate=True,
#         output_scores=True,
#         max_new_tokens=64
#     )
#     output = []
#     for seq in generation_output.sequences:
#         output.append(tokenizer.decode(seq))
#     return output


#########################################################################################################
#######################################   Execute inference  ############################################
#########################################################################################################
logging.info('Loading model')
peft_model_id = "/src/best_training/dolly-v2-3b-chris"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

logging.info('Starting text generation\n\n')
logging.info(generate_text("What is your name?", model, tokenizer))

logging.info(generate_text("Tell me your name!", model, tokenizer))
logging.info('Text generated')