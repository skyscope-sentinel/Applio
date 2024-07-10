import os, sys
import gradio as gr
import subprocess, psutil

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

python = sys.executable

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
model_root_relative = os.path.relpath(model_root, now_dir)

names = [
    os.path.join(root, file)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for file in files
    if (
        file.endswith((".pth", ".onnx"))
        and not (file.startswith("G_") or file.startswith("D_"))
    )
]


def change_choices():
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if (
            file.endswith((".pth", ".onnx"))
            and not (file.startswith("G_") or file.startswith("D_"))
        )
    ]
    return ({"choices": sorted(names), "__type__": "update"},)


def stop_realtime():
    global process
    if process:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
        process.wait()
        print("Realtime process and all its children have been terminated.")


def start_realtime(
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
    global process
    command_realtime = [
        python,
        "realtime.py",
        "-i",
        str(input_device_index),
        "-o",
        str(output_device_index),
        "--buffer_size",
        str(buffer_size),
        "--quality",
        str(quality),
        "--f0_up_key",
        str(f0_up_key),
        "--protect",
        str(protect),
        "--hop_length",
        str(hop_length),
        "--f0_method",
        str(f0_method),
        "--pth_path",
        str(model_path),
    ]
    process = subprocess.Popen(command_realtime, shell=True)


# Single inference tab
def realtime_tab():
    default_weight = names[0] if names else None
    with gr.Tab(label=i18n("Realtime")):
        model_path = gr.Dropdown(
            label=i18n("Voice Model"),
            info=i18n("Select the voice model to use for the conversion."),
            choices=sorted(names, key=lambda path: os.path.getsize(path)),
            interactive=True,
            value=default_weight,
            allow_custom_value=True,
        )
        refresh_button = gr.Button(i18n("Refresh"))
        with gr.Row():
            with gr.Column():
                input_device_index = gr.Textbox(
                    label=i18n("Input Device Index"),
                    info=i18n("The index of the input device to use."),
                    value=None,
                    interactive=True,
                )
                output_device_index = gr.Textbox(
                    label=i18n("Output Device Index"),
                    info=i18n("The index of the output device to use."),
                    value=None,
                    interactive=True,
                )
            with gr.Column():
                buffer_size = gr.Slider(
                    minimum=0,
                    maximum=10000,
                    step=100,
                    label=i18n("Buffer Size"),
                    info=i18n(
                        "Buffering size in milliseconds. The higher the value, the more latency but less chance of audio dropouts."
                    ),
                    value=2000,
                    interactive=True,
                )
                quality = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    label=i18n("Quality"),
                    info=i18n(
                        "Quality of the audio output. The higher the value, the better the quality but the more CPU usage."
                    ),
                    value=1,
                    interactive=True,
                )
        with gr.Accordion(i18n("Advanced Settings"), open=False):
            with gr.Column():
                pitch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label=i18n("Pitch"),
                    info=i18n(
                        "Set the pitch of the audio, the higher the value, the higher the pitch."
                    ),
                    value=0,
                    interactive=True,
                )
                protect = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n("Protect Voiceless Consonants"),
                    info=i18n(
                        "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                    ),
                    value=0.5,
                    interactive=True,
                )
                hop_length = gr.Slider(
                    minimum=1,
                    maximum=512,
                    step=1,
                    label=i18n("Hop Length"),
                    info=i18n(
                        "Denotes the duration it takes for the system to transition to a significant pitch change. Smaller hop lengths require more time for inference but tend to yield higher pitch accuracy."
                    ),
                    visible=False,
                    value=128,
                    interactive=True,
                )
                f0_method = gr.Radio(
                    label=i18n("Pitch extraction algorithm"),
                    info=i18n(
                        "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                    ),
                    choices=[
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                        "hybrid[rmvpe+fcpe]",
                    ],
                    value="rmvpe",
                    interactive=True,
                )

        start_realtime_button = gr.Button(i18n("Start Realtime"))
        stop_realtime_button = gr.Button(i18n("Stop Realtime"))

    refresh_button.click(
        fn=change_choices,
        inputs=[],
        outputs=[
            model_path,
        ],
    )

    start_realtime_button.click(
        fn=start_realtime,
        inputs=[
            input_device_index,
            output_device_index,
            buffer_size,
            quality,
            pitch,
            protect,
            hop_length,
            f0_method,
            model_path,
        ],
        outputs=[],
    )

    stop_realtime_button.click(
        fn=stop_realtime,
        inputs=[],
        outputs=[],
    )
