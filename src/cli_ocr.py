from __future__ import annotations

import sys

from .config import load_config, apply_env_from_config
from .pdf_ingestion import save_full_text_to_dir_ocr
from .extraction import _get_llm
from .agent_builder import build_tasks_with_llm, run_task


def main() -> int:
	if len(sys.argv) < 2:
		print("Usage: python -m src.cli_ocr <pdf_path> [lang]")
		return 1
	pdf_path = sys.argv[1]
	lang = sys.argv[2] if len(sys.argv) > 2 else "eng"
	cfg = load_config()
	apply_env_from_config(cfg)
	out = save_full_text_to_dir_ocr(pdf_path, out_dir="Ext_text", dpi=300, lang=lang)
	print(out)

	# Optional: run ingest agent to produce structured JSON per new schema
	try:
		llm = _get_llm()
		tasks = build_tasks_with_llm(llm)
		container = {
			"source_pdf": pdf_path,
			"text_pages": [open(out, "r", encoding="utf-8").read()],
			"tables": [],
		}
		result = run_task(tasks["ingest_task"], container)
		print("\nINGEST_JSON:\n" + str(result))
	except Exception:
		pass
	try:
		with open(out, "r", encoding="utf-8") as f:
			preview = f.read(800)
		print("\nPREVIEW:\n" + preview)
	except Exception:
		pass
	return 0


if __name__ == "__main__":
	sys.exit(main())


