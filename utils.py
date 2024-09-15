import sys
from pathlib import Path
import wave
from pydub import AudioSegment
import loguru


def is_audio_file(file: Path):
    return file.suffix.lower() in [".wav", ".mp3", ".ogg"]


def get_audio_duration_ms(file_path):
    try:
        with wave.open(str(file_path), "r") as wav_file:
            return wav_file.getnframes() / wav_file.getframerate() * 1000
    except wave.Error as e:
        audio = AudioSegment.from_file(file_path)
        return len(audio)
    except Exception as e:
        raise e


logger = loguru.logger
logger.remove()

log_format = (
    "<g>{time:MM-DD HH:mm:ss}</g> |<lvl>{level:^8}</lvl>| {file}:{line} | {message}"
)
logger.add(sys.stdout, format=log_format)
