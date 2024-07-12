import os
import sys
import pyaudio
import torch
import numpy as np
import argparse
import time

pa = pyaudio.PyAudio()

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

    infer_pipeline = VoiceConverter(repeat, hubert_model, model_data).infer_pipeline

    start_time = time.time()
    last_time = start_time
    total_frames = 0
    try:
        while input_stream.is_active():
            audio_input = np.frombuffer(
                input_stream.read(frames_per_buffer), dtype=np.float32
            )
            total_frames += frames_per_buffer

            # Measure inference time in milliseconds
            current_time = time.time()
            elapsed_time_ms = (current_time - last_time) * 1000
            last_time = current_time
            inference_ms = elapsed_time_ms
            print(f"Inference Time: {inference_ms:.2f} ms")

            audio_output = infer_pipeline(
                audio_input,
                f0_up_key,
                f0_method,
                protect,
                hop_length,
            )

            audio_output = audio_output.astype(np.float32, order="C") / 32768.0

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if np.isnan(audio_output).any():
                continue

            output_stream.write(audio_output.tobytes())

    finally:
        output_stream.close()
        input_stream.close()
        pa.terminate()

        # Calculate average inference time in milliseconds
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        avg_inference_ms = total_time_ms / total_frames
        print(f"Average Inference Time: {avg_inference_ms:.2f} ms")


def list_audio_devices(sample_rate=16000, channels=1):
    pa = pyaudio.PyAudio()
    print(
        f"Checking devices for compatibility with sample rate: {sample_rate} Hz and channels: {channels}"
    )

    compatible_devices = []
    for i in range(pa.get_device_count()):
        device_info = pa.get_device_info_by_index(i)
        is_input = device_info["maxInputChannels"] >= channels
        is_output = (
            device_info["maxOutputChannels"] >= channels
            and device_info["maxOutputChannels"] > 0
        )

        try:
            if is_input:
                pa.is_format_supported(
                    sample_rate,
                    input_device=device_info["index"],
                    input_channels=channels,
                    input_format=pyaudio.paFloat32,
                )
                compatible_devices.append((i, device_info["name"], "input"))

            if is_output:
                pa.is_format_supported(
                    sample_rate,
                    output_device=device_info["index"],
                    output_channels=channels,
                    output_format=pyaudio.paFloat32,
                )
                compatible_devices.append((i, device_info["name"], "output"))

        except ValueError:
            continue

    if not compatible_devices:
        print("No compatible devices found.")
    else:
        for device in compatible_devices:
            print(f"Index: {device[0]}, Name: {device[1]}, Type: {device[2]}")

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
            "rmvpe",
            "fcpe",
        ],
        default="fcpe",
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