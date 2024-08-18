from huggingface_hub import snapshot_download

# it's possible to download the entire dataset
snapshot_download(
    repo_id="McGill-NLP/WebLINX-full", repo_type="dataset"
)