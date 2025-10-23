"""
Audio Pipeline for Lab 3: process an audio file, perform STT, multi-factor confidence scoring,
PII redaction, summarization, TTS generation, and audit logging.
"""

import os
import re
import json
import datetime
from typing import Tuple, List

from dotenv import load_dotenv
import soundfile as sf
import numpy as np
from google.cloud import speech, texttospeech
import spacy

# Load spaCy model globally for reuse
nlp = spacy.load("en_core_web_sm")

def load_env(env_path: str = ".env") -> None:
    """Load environment variables and authenticate Google Cloud."""
    load_dotenv(env_path)
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds or not os.path.exists(creds):
        raise EnvironmentError("Missing or invalid GOOGLE_APPLICATION_CREDENTIALS in .env")
    print("[INFO] Environment loaded successfully.")


def preprocess_audio(input_path: str, output_path: str) -> str:
    import librosa
    
    y, sr = librosa.load(input_path, sr=None)
    
    sf.write(output_path, y, sr)
    
    return output_path


def transcribe_audio(audio_path: str) -> Tuple[str, List]:
    """Transcribe audio with Google Speech-to-Text."""
    client = speech.SpeechClient()

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        enable_word_confidence=True,
        enable_automatic_punctuation=True,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    transcript = ""
    words = []
    api_conf = 0.0

    for result in response.results:
        transcript += result.alternatives[0].transcript + " "
        api_conf += result.alternatives[0].confidence
        for w in result.alternatives[0].words:
            words.append({"word": w.word, "confidence": w.confidence})

    api_conf /= len(response.results) if response.results else 1
    print("[INFO] Transcription complete.")
    return transcript.strip(), words


def calculate_snr(audio_path: str) -> float:
    y, sr = sf.read(audio_path)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    threshold = 0.01 * np.max(np.abs(y))
    noise = y[np.abs(y) < threshold]
    signal = y[np.abs(y) >= threshold]

    if len(noise) == 0:
        noise_power = 1e-9
    else:
        noise_power = np.mean(noise ** 2)

    signal_power = np.mean(signal ** 2)

    snr = 10 * np.log10(signal_power / (noise_power + 1e-9))

    return float(np.clip(snr, 0, 50))


def calculate_perplexity(word_confidences: List[float]) -> float:
    """Compute inverse of average confidence as a pseudo-perplexity."""
    if not word_confidences:
        return float("inf")
    avg_conf = sum(word_confidences) / len(word_confidences)
    return round(1.0 / max(avg_conf, 1e-5), 3)


def multi_factor_confidence(api_confidence: float, snr: float, perplexity: float) -> Tuple[float, str]:
    """Weighted average of multiple confidence indicators."""
    snr_norm = min(snr / 40.0, 1.0)
    perp_norm = 1 - min(perplexity / 10.0, 1.0)
    combined = round((0.6 * api_confidence + 0.25 * snr_norm + 0.15 * perp_norm), 3)

    if combined > 0.85:
        label = "HIGH"
    elif combined > 0.6:
        label = "MEDIUM"
    else:
        label = "LOW"

    return combined, label


def redact_pii(text: str) -> Tuple[str, List[dict]]:
    """Remove credit card numbers and named entities."""
    redactions = []
    redacted = text

    # Regex for credit card pattern
    pattern = re.compile(r"(\b\d{4}[\s,-]?){3}\d{4}\b")
    matches = pattern.findall(text)
    for m in matches:
        redacted = re.sub(pattern, "[REDACTED_CARD]", redacted)
        redactions.append({"type": "CREDIT_CARD", "value": m.strip()})

    # NER for names
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"]:
            redacted = redacted.replace(ent.text, "[REDACTED_" + ent.label_ + "]")
            redactions.append({"type": ent.label_, "value": ent.text})

    return redacted, redactions


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Naive summarizer using first N sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:max_sentences]).strip()


def synthesize_speech(text: str, output_path: str, voice_name: str = "en-US-Neural2-A") -> str:
    """Generate MP3 summary via Google Text-to-Speech."""
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    print(f"[INFO] Summary audio saved to {output_path}")
    return output_path


def write_audit_log(log_data: dict, log_path: str) -> None:
    """Append structured JSON to log file."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")
    print(f"[INFO] Log entry written to {log_path}")


def main():
    load_env()
    input_file = os.getenv("INPUT_AUDIO", "test_audio.mp3")

    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        return

    processed = preprocess_audio(input_file, "processed_audio.wav")
    transcript, words = transcribe_audio(processed)
    word_confidences = [w.get("confidence", 0.0) for w in words]

    snr_val = calculate_snr(processed)
    perplexity_val = calculate_perplexity(word_confidences)
    api_conf = sum(word_confidences) / len(word_confidences) if word_confidences else 0.0

    combined_score, level = multi_factor_confidence(api_conf, snr_val, perplexity_val)
    redacted_text, redactions = redact_pii(transcript)
    summary = summarize_text(redacted_text, max_sentences=10)
    synthesize_speech(summary, "output_summary.mp3")

    # Write outputs
    with open("output_transcript.txt", "w", encoding="utf-8") as f:
        f.write(redacted_text)

    log_data = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input_file": input_file,
        "api_confidence": api_conf,
        "snr": snr_val,
        "perplexity": perplexity_val,
        "combined_score": combined_score,
        "confidence_level": level,
        "redactions": redactions,
    }
    write_audit_log(log_data, "audit.log")

    print("[SUCCESS] Pipeline completed successfully!")


if __name__ == "__main__":
    main()
