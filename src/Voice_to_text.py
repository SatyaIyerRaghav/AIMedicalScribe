import os
import whisper

def voice_to_text_task(state, config):
    print("🔍 [DEBUG] Entered voice_to_text_task")

    # ✅ Check both state and config for audio file
    audio_file = (
        state.get("audio_file")
        or config.get("configurable", {}).get("audio_file")
    )
    print(f"audio file details {audio_file}")

    if not audio_file or not os.path.exists(audio_file):
        print("⚠️ [DEBUG] No audio file found in config or state!")
        state["transcript"] = None
        return state

    print(f"🎙️ [DEBUG] Transcribing audio file: {audio_file}")
    model = whisper.load_model("small")
    result = model.transcribe(audio_file)

    transcript = result["text"]
    print("✅ [DEBUG] Transcription complete.")
    state["transcript"] = transcript
    return state
