import torch
from torch.utils.data import DataLoader
import lightning as L
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig


# Sharing Datasets Across Process Boundaries
# The LightningDataModule class provides an organized way to decouple data loading from training logic, 
# with prepare_data() being used for downloading and pre-processing the dataset on a single process, 
# and setup() loading the pre-processed data for each process individually:
class DataModule(L.LightningDataModule):
    def __init__(self, data_path: str, processor_path: str, max_length: str, batch_size: int, num_workers: int = 4):
        self.data_path = data_path
        self.processor_path = processor_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # can be used to e.g. download data from cloud
        pass
        
    def setup(self):
        self.dataset = load_dataset('json', data_files=[self.data_path])['train']
        self.tokenizer = AutoTokenizer.from_pretrained(self.processor_path)
        self.tokenizer.padding_side = "right" # during training, one always uses padding on the right


    def collate_fn(self, examples):
        texts = []

        for example in examples:
            text_prompt = self.tokenizer.apply_chat_template(example)
            texts.append(text_prompt)

        batch = self.tokenizer(text=texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        return input_ids, attention_mask, labels

    def train_loader(self):
        return DataLoader(self.dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)
    

class LlamaModelPLModule(L.LightningModule):
    def __init__(self, model_path: str, learning_rate:float):
        super().__init__()
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
    
    def setup(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype="float16",
            quantization_config=self.bnb_config, 
            attn_implementation="flash_attention_2"
        )

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, labels = batch

        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                          )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss


    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        return optimizer


if __name__ == "__main__":
    MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    DATA_PATH = "./data/train.jsonl"
    config = {
        "epochs": 1,
        "lr": 1e-4,
        "max_length": 128,
        "batch_size": 2,
        "accumulate_grad_batches": 4,
        "precision": "16-mixed",
        "gradient_clip_val": 1.0,
    }
    model_module = LlamaModelPLModule(MODEL_ID, learning_rate=config["lr"])
    data_module = DataModule(DATA_PATH, processor_path=MODEL_ID, max_length=config["max_length"], batch_size=config["batch_size"])
    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        max_epochs=config["epochs"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        gradient_clip_val=config["gradient_clip_val"],
        precision=config["precision"]
    )
    trainer.fit(model_module, data_module)
