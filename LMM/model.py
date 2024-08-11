from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
from abc import abstractmethod, ABC
from pydantic.dataclasses import dataclass
from typing import Union
from PIL import ImageFile

@dataclass
class ModelConfig:
    # model_id: str
    troch_dtype: str
    bnb_config: Union[BitsAndBytesConfig, dict]
    max_new_tokens: int

    def __post_init__(self):
        if isinstance(self.bnb_config, dict):
            self.bnb_config = BitsAndBytesConfig(**self.bnb_config)
        else:
            raise ValueError("Invalid bnb_config type")

class Model(ABC):
    
    @abstractmethod
    def chat(self, messages: list, image: bytes) -> str:
        pass

# # Open the image file
# image = Image.open("path/to/image.jpg")

# # Check the image file type
# if image.mode == "RGB":
#     print("The image is in RGB mode.")
# else:
#     print("The image is not in RGB mode.")

class LlavaNext(Model):
    MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
    
    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(LlavaNext.MODEL_ID)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            LlavaNext.MODEL_ID,
            torch_dtype=self.config.troch_dtype,
            quantization_config=self.config.bnb_config,
        )
    
    def to_llava_messages(messages, has_image: bool = False) -> list:
        if has_image:
            llava_messages = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
            # Add image token to last message in converstaion
            llava_messages[-1]["content"].append({"type": "image"})
        else:
            llava_messages = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        return llava_messages

    def chat(self, messages: list, image: ImageFile = None) -> str:
        if image is None:
            text_prompt = self.processor.apply_chat_template(LlavaNext.to_llava_messages(messages), add_generation_prompt=True)
            inputs = self.processor(text=text_prompt, return_tensors="pt").to("cuda")
        else:
            text_prompt = self.processor.apply_chat_template(LlavaNext.to_llava_messages(messages, True), add_generation_prompt=True)
            inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt").to("cuda")
        # Generate token IDs
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        # Decode back into text
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts[0].split("assistant\n\n\n")[-1]