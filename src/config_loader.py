# utils/config_loader.py
import os, yaml

def load_and_set_env(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    aws = config.get("aws", {})
    llm = config.get("llm", {})

    # Set AWS keys
    os.environ["AWS_ACCESS_KEY_ID"] = aws.get("access_key_id", "")
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws.get("secret_access_key", "")
    os.environ["AWS_DEFAULT_REGION"] = aws.get("region", "us-east-1")

    # Set LLM provider
    if llm.get("provider") == "bedrock":
        os.environ["BEDROCK_MODEL_ID"] = llm.get("model", "")
        os.environ["LITELLM_PROVIDER"] = "bedrock"
        print(f"✅ Using Bedrock model: {os.environ['BEDROCK_MODEL_ID']}")
    else:
        os.environ["OPENAI_API_KEY"] = llm.get("api_key", "")
        os.environ["MODEL_NAME"] = llm.get("model", "gpt-4o-mini")
        os.environ["LITELLM_PROVIDER"] = "openai"
        print(f"✅ Using OpenAI model: {os.environ['MODEL_NAME']}")

    return config
