"""
Voice Interface for RAG Chatbot
Handles Speech-to-Text (STT) and Text-to-Speech (TTS)
"""

import os
import io
import logging
from typing import Optional
from pathlib import Path
import tempfile

# Google Cloud Speech-to-Text & Text-to-Speech
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as texttospeech

# Audio processing
from pydub import AudioSegment
import wave

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceInterface:
    """Handles voice input/output using Google Cloud APIs"""

    def __init__(self):
        """Initialize Google Cloud Speech and TTS clients"""
        try:
            # Initialize clients
            self.speech_client = speech.SpeechClient()
            self.tts_client = texttospeech.TextToSpeechClient()

            logger.info("✅ Voice interface initialized successfully")

        except Exception as e:
            logger.error(f"❌ Error initializing voice interface: {e}")
            raise

    def transcribe_audio(self, audio_path: str, language_code: str = "en-US") -> Optional[str]:
        """
        Transcribe audio file to text using Google Speech-to-Text

        Args:
            audio_file_path: Path to audio file (WAV, MP3, etc.)
            language_code: Language code (default: en-US)

        Returns:
            Transcribed text or None if failed
        """
        try:
            import subprocess
            import os

            # Convert WebM to WAV using ffmpeg
            wav_path = audio_path.replace('.webm', '.wav')

            # Convert to LINEAR16 WAV format (what Google expects)
            result = subprocess.run([
                'ffmpeg', '-i', audio_path,
                '-acodec', 'pcm_s16le',  # LINEAR16 encoding
                '-ac', '1',               # Mono
                '-ar', '16000',           # 16kHz sample rate
                '-y',                     # Overwrite output file
                wav_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            logger.info(f"FFmpeg stderr: {result.stderr.decode()}")
            logger.info(f"Converted {audio_path} to {wav_path}")

            # Check if WAV file has actual audio content
            file_size = os.path.getsize(wav_path)
            logger.info(f"WAV file size: {file_size} bytes")

            if file_size < 1000:  # Less than 1KB is likely silence
                logger.warning(
                    "WAV file too small - likely silence or no audio")
                return ""

            # Read the converted WAV file
            with open(wav_path, 'rb') as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
                enable_automatic_punctuation=True,
                model="default",
                audio_channel_count=1,
                enable_word_time_offsets=False,
            )

            logger.info("Sending audio to Google Speech-to-Text...")
            response = self.speech_client.recognize(config=config, audio=audio)

            if not response.results:
                logger.error(f"❌ Google returned empty results: {response}")
                # Try alternative config for very short audio
                config.model = "default"
                response = self.speech_client.recognize(
                    config=config, audio=audio)

                if not response.results:
                    logger.error("Still no results with alternative model")
                    return ""

            transcript = " ".join([
                result.alternatives[0].transcript
                for result in response.results
            ])

            # Clean up temporary WAV file
            try:
                os.remove(wav_path)
            except:
                pass

            logger.info(f"✅ Transcription: {transcript}")
            return transcript

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
            return ""
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def synthesize_speech(
        self,
        text: str,
        output_path: str,
        language_code: str = "en-US",
        voice_name: str = "en-US-Neural2-J",  # Male voice
        speaking_rate: float = 1.0
    ) -> bool:
        """
        Convert text to speech using Google Text-to-Speech

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            language_code: Language code
            voice_name: Voice to use (Neural2 voices are best)
            speaking_rate: Speed of speech (0.25 to 4.0)

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🔊 Synthesizing speech: {text[:50]}...")

            # Set up synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Configure voice
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )

            # Configure audio
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate
            )

            # Perform synthesis
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # Save audio to file
            with open(output_path, "wb") as out:
                out.write(response.audio_content)

            logger.info(f"✅ Speech saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Speech synthesis error: {e}")
            return False

    def _convert_to_wav(self, audio_file_path: str) -> str:
        """
        Convert audio file to WAV format if needed

        Args:
            audio_file_path: Path to audio file

        Returns:
            Path to WAV file
        """
        file_ext = Path(audio_file_path).suffix.lower()

        # If already WAV, check format
        if file_ext == ".wav":
            return self._ensure_wav_format(audio_file_path)

        # Convert to WAV
        try:
            audio = AudioSegment.from_file(audio_file_path)

            # Convert to mono, 16kHz, 16-bit
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            audio = audio.set_sample_width(2)  # 16-bit

            # Save as WAV
            wav_path = str(Path(audio_file_path).with_suffix('.wav'))
            audio.export(wav_path, format="wav")

            logger.info(f"✅ Converted audio to WAV: {wav_path}")
            return wav_path

        except Exception as e:
            logger.error(f"❌ Audio conversion error: {e}")
            return audio_file_path

    def _ensure_wav_format(self, wav_path: str) -> str:
        """
        Ensure WAV file is in correct format (mono, 16kHz, 16-bit)

        Args:
            wav_path: Path to WAV file

        Returns:
            Path to properly formatted WAV file
        """
        try:
            # Check current format
            with wave.open(wav_path, 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()

            # If format is correct, return as is
            if channels == 1 and sample_rate == 16000 and sample_width == 2:
                return wav_path

            # Otherwise, convert
            audio = AudioSegment.from_wav(wav_path)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            audio = audio.set_sample_width(2)

            new_path = str(Path(wav_path).with_stem(
                Path(wav_path).stem + "_converted"))
            audio.export(new_path, format="wav")

            return new_path

        except Exception as e:
            logger.error(f"❌ WAV format check error: {e}")
            return wav_path


# Available voices for different use cases
VOICE_OPTIONS = {
    "male_professional": "en-US-Neural2-J",
    "female_professional": "en-US-Neural2-F",
    "male_casual": "en-US-Neural2-D",
    "female_casual": "en-US-Neural2-C",
    "male_warm": "en-US-Neural2-A",
    "female_warm": "en-US-Neural2-E"
}


if __name__ == "__main__":
    # Test voice interface
    logger.info("Testing Voice Interface...")

    try:
        voice = VoiceInterface()

        # Test TTS
        test_text = "Hello! I am your company information assistant. How can I help you today?"
        output_file = "test_voice.mp3"

        if voice.synthesize_speech(test_text, output_file):
            logger.info(f"✅ Test TTS successful: {output_file}")

        # Note: STT test requires an actual audio file
        logger.info("✅ Voice interface test complete!")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
