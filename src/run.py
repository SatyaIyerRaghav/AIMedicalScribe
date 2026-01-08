# src/run.py
import os
# Load config FIRST before any other imports
from src.config import load_config, apply_env_from_config

# Load and apply config from config.yaml
config = load_config("config.yaml")
apply_env_from_config(config)

# Now import the app (which will import agents with proper config)
from src.graph import app

from src.logger_setup import init_logging

init_logging()

def run_case(app, audio_file):
    """
    Run the LangGraph pipeline with a given audio file.
    """
    print(f"\n Processing audio file: {audio_file}\n")

    # Convert to absolute path (important for Whisper)
    audio_path = os.path.abspath(audio_file)
    print(f" [DEBUG] Absolute path: {audio_path}")

    # Build initial state that LangGraph will carry through all nodes
    state = {
        "audio_file": audio_path
    }
    # Config dictionary for LangGraph runtime
    config = {
        "configurable": {
            "thread_id": "case_001",
            "audio_file": audio_path  
        }
    }

    print(" [DEBUG] Invoking LangGraph pipeline...")
    result = app.invoke(state, config=config)

    if result is None:
        print("⚠️ [DEBUG] app.invoke() returned None — using fallback state.")
        result = state

    print(" [DEBUG] LangGraph pipeline completed.\n")
    return result


def main():
    import sys

    if len(sys.argv) < 2:
        print("Please provide an audio file path as argument.")
        return

    audio_file = sys.argv[1]
    print(f"[DEBUG] Command-line argument received: {audio_file}")

    final_state = run_case(app, audio_file)
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
