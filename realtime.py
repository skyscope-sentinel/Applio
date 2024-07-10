import os
import sys
import pyaudio
import torch
import numpy as np
import argparse

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.configs.config import Config

config = Config()

from rvc.infer.infer_realtime import VoiceConverter
from rvc.lib.utils import load_embedding


def load_hubert(embedder_model, embedder_model_custom=None):
    """
    Loads the HuBERT model for speaker embedding extraction.

    Args:
        embedder_model: Path to the pre-trained embedder model.
        embedder_model_custom: Path to a custom embedder model (if any).
    """
    models, _, _ = load_embedding(embedder_model, embedder_model_custom)
    hubert_model = models[0].to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    return hubert_model.eval()


def load_model(weight_root):
    return (
        torch.load(weight_root, map_location="cpu")
        if os.path.isfile(weight_root)
        else None
    )


def realtime(
    input_device_index,
    output_device_index,
    buffer_size,
    quality,
    f0_up_key,
    protect,
    hop_length,
    f0_method,
    model_path,
):
    model_data = load_model(model_path)
    hubert_model = load_hubert("contentvec")

    tgt_sr = model_data["config"][-1]

    repeat = 3 if config.is_half else 1
    repeat *= quality

    pa = pyaudio.PyAudio()
    print(
        "input_device: %s"
        % (
            pa.get_device_info_by_index(input_device_index)
            if input_device_index is not None
            else "Default"
        )
    )
    print(
        "output_device: %s"
        % (
            pa.get_device_info_by_index(output_device_index)
            if output_device_index is not None
            else "Default"
        )
    )

    input_frame_rate = 16000
    frames_per_buffer = input_frame_rate * buffer_size // 1000

    input_stream = pa.open(
        rate=input_frame_rate,
        channels=1,
        format=pyaudio.paFloat32,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=frames_per_buffer,
    )
    output_stream = pa.open(
        rate=tgt_sr,
        channels=1,
        format=pyaudio.paFloat32,
        output=True,
        output_device_index=output_device_index,
    )
    input_stream.start_stream()

    try:
        while input_stream.is_active():
            audio_input = np.frombuffer(
                input_stream.read(frames_per_buffer), dtype=np.float32
            )
            print(
                "audio_input: %s, %s, %s, %s"
                % (
                    audio_input.shape,
                    audio_input.dtype,
                    np.min(audio_input).item(),
                    np.max(audio_input).item(),
                )
            )
            infer_pipeline = VoiceConverter(
                repeat, hubert_model, model_data
            ).infer_pipeline
            audio_output = infer_pipeline(
                audio_input,
                f0_up_key,
                f0_method,
                protect,
                hop_length,
            )
            audio_output = audio_output.astype(np.float32, order="C") / 32768.0

            print(
                "audio_output: %s, %s, %s, %s"
                % (
                    audio_output.shape,
                    audio_output.dtype,
                    np.min(audio_output).item() if audio_output.size > 0 else None,
                    np.max(audio_output).item() if audio_output.size > 0 else None,
                )
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if np.isnan(audio_output).any():
                continue

            output_stream.write(audio_output.tobytes())

    finally:
        output_stream.close()
        input_stream.close()
        pa.terminate()


def list_audio_devices():
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        print(pa.get_device_info_by_index(i))
    pa.terminate()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the realtime.py script with specific parameters."
    )
    parser.add_argument("--list-audio-devices", action="store_true")
    parser.add_argument("-i", "--input-device-index", type=int, default=None)
    parser.add_argument("-o", "--output-device-index", type=int, default=None)
    parser.add_argument(
        "--buffer_size", type=int, default=2000, help="buffering size in ms"
    )
    parser.add_argument(
        "--quality", type=int, default=1, help="quality of the audio output"
    )
    parser.add_argument(
        "--f0_up_key",
        type=str,
        help="Value for f0_up_key",
        choices=[str(i) for i in range(-24, 25)],
        default="0",
    )
    parser.add_argument(
        "--rms_mix_rate",
        type=str,
        help="Value for rms_mix_rate",
        choices=[str(i / 10) for i in range(11)],
        default="1",
    )
    parser.add_argument(
        "--protect",
        type=str,
        help="Value for protect",
        choices=[str(i / 10) for i in range(6)],
        default="0.33",
    )
    parser.add_argument(
        "--hop_length",
        type=str,
        help="Value for hop_length",
        choices=[str(i) for i in range(1, 513)],
        default="128",
    )
    parser.add_argument(
        "--f0_method",
        type=str,
        help="Value for f0_method",
        choices=[
            "pm",
            "harvest",
            "dio",
            "crepe",
            "crepe-tiny",
            "rmvpe",
            "fcpe",
            "hybrid[crepe+rmvpe]",
            "hybrid[crepe+fcpe]",
            "hybrid[rmvpe+fcpe]",
            "hybrid[crepe+rmvpe+fcpe]",
        ],
        default="rmvpe",
    )
    parser.add_argument("--pth_path", type=str, help="Path to the .pth file")

    return parser.parse_args()


def main():
    if len(sys.argv) == 1:
        print("Please run the script with '-h' for more information.")
        sys.exit(1)

    args = parse_arguments()
    if args.list_audio_devices:
        list_audio_devices()
    else:
        try:
            realtime(
                int(args.input_device_index),
                int(args.output_device_index),
                int(args.buffer_size),
                int(args.quality),
                str(args.f0_up_key),
                str(args.protect),
                str(args.hop_length),
                str(args.f0_method),
                str(args.pth_path),
            )
        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
