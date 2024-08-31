from huggingface_hub import snapshot_download

model_id = "llava-hf/llama3-llava-next-8b-hf"
snapshot_download(model_id)