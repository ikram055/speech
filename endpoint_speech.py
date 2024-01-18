from fastapi import FastAPI, File, UploadFile
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import whisper

app = FastAPI()


class SpeechProcessingModel:
    def __init__(self, model_name):
        self.model = whisper.load_model(model_name)

    def load_audio(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        return audio, mel

    def detect_language(self, mel):
        _, probs = self.model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        return detected_language

    def decode_audio(self, audio, mel):
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(self.model, mel, options)
        return result.text

    def translate(self, text, target):
        translated = GoogleTranslator(source="auto", target=target).translate(text)
        return translated

    def text_to_speech(self, text, language="en", output_path="output.mp3"):
        tts = gTTS(text=text, lang=language, slow=False, tld="us")
        tts.save(output_path)
        return output_path  # Return the path to the generated audio file


model_instance = SpeechProcessingModel("base")


@app.post("/process_audio")
async def process_audio(
    audio_file: UploadFile = File(...), target_language: str = "en"
):
    # Save the uploaded audio file
    with open("uploaded_audio.wav", "wb") as audio:
        audio.write(audio_file.file.read())

    # Load and process the audio
    loaded_audio, mel = model_instance.load_audio("uploaded_audio.wav")
    detected_language = model_instance.detect_language(mel=mel)
    recognized_text = model_instance.decode_audio(loaded_audio, mel=mel)
    translated_text = model_instance.translate(recognized_text, target=target_language)

    # Generate text-to-speech and get the path to the generated audio file
    tts_output_path = model_instance.text_to_speech(
        translated_text, language=target_language
    )

    return {
        "detected_language": detected_language,
        "recognized_text": recognized_text,
        "translated_text": translated_text,
        "tts_output_path": tts_output_path,  # Include the path to the generated audio file in the response
    }
