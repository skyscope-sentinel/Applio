import os
import torch
import numpy as np
from rvc.infer.pipeline_realtime import Pipeline as VC
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.configs.config import Config

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class VoiceConverter:
    """
    A class for performing voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self, x_pad, hubert_model, cpt):
        """
        Initializes the VoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load RVC configuration
        self.hubert_model = hubert_model  # HuBERT model for speaker embedding extraction
        self.tgt_sr = None  # Target sampling rate for the output audio
        self.net_g = None  # Generator network for voice conversion
        self.vc = None  # Voice conversion pipeline instance
        self.cpt = cpt  # Checkpoint for loading model weights
        self.version = None  # Model version
        self.n_spk = None  # Number of speakers in the model
        self.use_f0 = None  # Whether the model uses F0
        self.x_pad = x_pad  # Padding value for the input audio

    def voice_conversion(
        self,
        sid=0,
        audio=None,
        f0_up_key=None,
        f0_method=None,
        protect=None,
        hop_length=None,
    ):
        """
        Performs voice conversion on the input audio using the loaded model and settings.

        Args:
            sid: Speaker ID for the target voice.
            input_audio_path: Path to the input audio file.
            f0_up_key: Pitch shift value in semitones.
            f0_file: Path to an external F0 file for pitch guidance.
            f0_method: F0 estimation method to use.
            file_index: Path to the FAISS index for speaker embedding retrieval.
            index_rate: Weighting factor for speaker embedding retrieval.
            resample_sr: Target sampling rate for resampling.
            rms_mix_rate: Mixing ratio for adjusting RMS levels.
            protect: Protection level for preserving the original pitch.
            hop_length: Hop length for F0 estimation.
            output_path: Path for saving the converted audio.
            split_audio: Whether to split the audio into segments for processing.
            f0_autotune: Whether to apply autotune to the F0 contour.
            filter_radius: Radius for median filtering of the F0 contour.
            embedder_model: Path to the embedder model.
            embedder_model_custom: Path to a custom embedder model.
        """
        f0_up_key = int(f0_up_key)
        try:
            audio_max = np.abs(audio).max() / 0.95

            if audio_max > 1:
                audio /= audio_max

            audio_opt = self.vc.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                f0_up_key,
                f0_method,
                protect,
                hop_length,
                self.version,
            )
            return audio_opt

        except Exception as error:
            print(error)

    def setup_network(self):
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
        self.use_f0 = self.cpt.get("f0", 1)

        self.version = self.cpt.get("version", "v1")
        self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
        self.net_g = Synthesizer(
            *self.cpt["config"],
            use_f0=self.use_f0,
            text_enc_hidden_dim=self.text_enc_hidden_dim,
            is_half=self.config.is_half,
        )
        del self.net_g.enc_q
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        self.net_g = (
            self.net_g.half() if self.config.is_half else self.net_g.float()
            )

    def setup_vc_instance(self):
        self.vc = VC(self.tgt_sr, self.x_pad)
        self.n_spk = self.cpt["config"][-3]

    def infer_pipeline(
        self,
        audio,
        f0_up_key,
        f0_method,
        protect,
        hop_length,
    ):
        """
        Main inference pipeline for voice conversion.

        Args:
            f0_up_key: Pitch shift value.
            filter_radius: Filter radius for F0 smoothing.
            index_rate: Speaker embedding retrieval rate.
            rms_mix_rate: RMS mixing ratio.
            protect: Pitch protection level.
            hop_length: Hop length for F0 estimation.
            f0_method: F0 estimation method.
            audio_input_path: Input audio file path.
            audio_output_path: Output audio file path.
            model_path: Model weight file path.
            index_path: FAISS index file path.
            split_audio: Whether to split audio.
            f0_autotune: Whether to apply autotune.
            clean_audio: Whether to apply noise reduction.
            clean_strength: Noise reduction strength.
            export_format: Output audio format.
            embedder_model: Embedder model path.
            embedder_model_custom: Custom embedder model path.
            upscale_audio: Whether to upscale audio.
        """
        self.setup_network()
        self.setup_vc_instance()
        try:
            audio_opt = self.voice_conversion(
                sid=0,
                audio=audio,
                f0_up_key=f0_up_key,
                f0_method=f0_method,
                protect=float(protect),
                hop_length=hop_length,
            )
            return audio_opt

        except Exception as error:
            print(f"Voice conversion failed: {error}")