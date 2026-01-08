from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
	if not os.path.exists(path):
		return {}
	with open(path, "r", encoding="utf-8") as f:
		content = f.read()
		# Normalize tabs (common in editors) to spaces so PyYAML can parse
		if "\t" in content:
			content = content.replace("\t", "  ")
		return yaml.safe_load(content) or {}


def apply_env_from_config(cfg: Dict[str, Any]) -> None:
	aws = cfg.get("aws", {})
	if aws:
		os.environ.setdefault("AWS_REGION", str(aws.get("region", "us-east-1")))
		if aws.get("access_key_id"):
			os.environ["AWS_ACCESS_KEY_ID"] = str(aws["access_key_id"])
		if aws.get("secret_access_key"):
			os.environ["AWS_SECRET_ACCESS_KEY"] = str(aws["secret_access_key"])
		if aws.get("session_token"):
			os.environ["AWS_SESSION_TOKEN"] = str(aws["session_token"])  # optional

	llm = cfg.get("llm", {})
	if llm:
		# Prefer explicit provider settings
		provider = str(llm.get("provider", "bedrock"))
		model = str(llm.get("model", "anthropic.claude-3-5-sonnet-20240620-v1:0"))
		os.environ.setdefault("MODEL_NAME", model)
	if provider == "openai" and llm.get("api_key"):
		os.environ["OPENAI_API_KEY"] = str(llm["api_key"]) 
	elif provider == "groq" and llm.get("api_key"):
		os.environ["GROQ_API_KEY"] = str(llm["api_key"]) 
	elif provider == "bedrock":
		# Set BEDROCK_MODEL_ID explicitly (don't use setdefault)
		os.environ["BEDROCK_MODEL_ID"] = str(model)
		# Also set MODEL_NAME for backward compatibility
		os.environ["MODEL_NAME"] = str(model)
		print(f"✅ [DEBUG] Configured Bedrock model: {model}")
		# Optional: Bedrock inference profile if the model requires it
		if llm.get("inference_profile_id"):
			os.environ["BEDROCK_INFERENCE_PROFILE_ID"] = str(llm["inference_profile_id"]).strip()
		if llm.get("inference_profile_arn"):
			os.environ["BEDROCK_INFERENCE_PROFILE_ARN"] = str(llm["inference_profile_arn"]).strip()
	if provider == "huggingface":
		if llm.get("api_key"):
			os.environ["HUGGINGFACEHUB_API_TOKEN"] = str(llm["api_key"]).strip()
		os.environ.setdefault("HF_MODEL_ID", model)

	temp = cfg.get("generation", {}).get("temperature")
	if temp is not None:
		os.environ["MODEL_TEMPERATURE"] = str(temp)

	# OCR tool paths (Poppler + Tesseract)
	ocr = cfg.get("ocr", {})
	# if ocr:
	# 	# poppler_path = ocr.get("poppler_path")
		# tesseract_cmd = ocr.get("tesseract_cmd")
		# if poppler_path:
		# 	os.environ["POPPLER_PATH"] = str(poppler_path)
		# if tesseract_cmd:
		# 	os.environ["TESSERACT_CMD"] = str(tesseract_cmd)


__all__ = ["load_config", "apply_env_from_config"]


