import os
import base64
import argparse
import mimetypes
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI


CHUNK_DURATION_SECONDS = 300


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe an MP3/WAV file with Mimo ASR.")
    parser.add_argument("--input", required=True, help="Path to the MP3 or WAV file to transcribe.")
    parser.add_argument(
        "--output",
        help="Path for the transcription text file (defaults to the input filename with a .txt suffix).",
    )
    return parser.parse_args()


def split_audio(audio_file, output_dir):
    """Re-encode the input and split it into API-sized MP3 segments."""
    chunk_pattern = Path(output_dir) / "chunk_%03d.mp3"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_file),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
        "-f",
        "segment",
        "-segment_time",
        str(CHUNK_DURATION_SECONDS),
        "-reset_timestamps",
        "1",
        str(chunk_pattern),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is not installed or is not available on PATH") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg failed: {detail}") from error

    chunks = sorted(
        Path(output_dir).glob("chunk_*.mp3"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not chunks:
        raise RuntimeError("ffmpeg completed without producing any audio chunks")
    return chunks


def transcribe_audio(client, audio_file):
    with open(audio_file, "rb") as input_file:
        audio_bytes = input_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    mime_type = mimetypes.guess_type(audio_file)[0] or "audio/mpeg"

    completion = client.chat.completions.create(
        model="mimo-v2.5-asr",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:{mime_type};base64,{audio_base64}"
                        },
                    }
                ],
            }
        ],
        extra_body={"asr_options": {"language": "auto"}},
        stream=True,
    )

    text_parts = []
    for chunk in completion:
        content = []
        for choice in chunk.choices:
            if choice.delta.content:
                content.append(choice.delta.content)
        chunk_text = "".join(content)
        print(f"Chunk text: {chunk_text}")
        text_parts.append(chunk_text)
    return "".join(text_parts)


def main():
    args = parse_args()
    audio_file = Path(args.input)
    audio_text_file = Path(args.output or audio_file.with_suffix(".txt"))

    if not audio_file.is_file():
        raise SystemExit(f"Input audio file does not exist: {audio_file}")

    if audio_text_file.is_file():
        print(f"{audio_text_file} is exists!")
        raise SystemExit(-1)

    client = OpenAI(
        api_key=os.environ.get("MIMO_API_KEY"),
        base_url="https://api.xiaomimimo.com/v1",
    )

    with tempfile.TemporaryDirectory(prefix="mimo_asr_") as temp_dir:
        audio_chunks = split_audio(audio_file, temp_dir)
        transcriptions = []
        for index, audio_chunk in enumerate(audio_chunks, start=1):
            print(f"Transcribing audio segment {index}/{len(audio_chunks)}: {audio_chunk.name}")
            transcriptions.append(transcribe_audio(client, audio_chunk))

    # 在多个切片文本之间添加换行符，防止大片文本连在一起
    combined_text = "\n".join(text.strip() for text in transcriptions if text.strip())
    with open(audio_text_file, "w", encoding="utf-8") as output_file:
        output_file.write(combined_text)
    print(f"Transcription saved to {audio_text_file}")


if __name__ == "__main__":
    main()
