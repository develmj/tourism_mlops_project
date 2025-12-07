from huggingface_hub import upload_folder

SPACE_ID = "mjiyer/tourism-wellness-space"

upload_folder(
    folder_path=".",
    repo_id=SPACE_ID,
    repo_type="space"
)

print("Deployment folder uploaded to Hugging Face Space.")

