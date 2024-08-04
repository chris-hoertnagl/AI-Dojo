from huggingface_hub import snapshot_download

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
snapshot_download(model_id)