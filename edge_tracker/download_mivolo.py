from huggingface_hub import snapshot_download

MIVOLO_REVISION = "53393526c220e34cdd7b722b36d22b6f9e5f4241"


def main():
    print("Downloading the complete MiVOLO V2 Transformers model...")
    mivolo_path = snapshot_download(
        repo_id="iitolstykh/mivolo_v2",
        revision=MIVOLO_REVISION,
        allow_patterns=[
            "config.json",
            "configuration_mivolo.py",
            "mivolo_image_processor.py",
            "model.safetensors",
            "modeling_mivolo.py",
            "preprocessor_config.json",
        ],
    )
    print(f"Saved to: {mivolo_path}")


if __name__ == "__main__":
    main()
