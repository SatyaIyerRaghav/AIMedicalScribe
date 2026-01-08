import os
import subprocess

def transcribe_audio(audio_path: str) -> str:
    """
    Local Whisper transcription helper.
    Uses 'whisper' CLI if available, otherwise returns dummy text.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        print(f"🎙️ Running Whisper on: {audio_path}")
        # This assumes Whisper CLI is installed (`pip install openai-whisper`)
        result = subprocess.run(
            ["whisper", audio_path, "--model", "base", "--language", "en", "--output_format", "txt"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print("⚠️ Whisper failed, returning fallback transcript.")
            print(result.stderr)
            return "(transcription unavailable)"

        # Try to locate output .txt file
        txt_path = os.path.splitext(audio_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
            return transcript

        print("⚠️ Whisper did not produce a text file, returning empty transcript.")
        return "(no transcript generated)"

    except Exception as e:
        print(f"❌ Whisper error: {e}")
        return "(error during transcription)"
