import os
import gc
import re
import sys
import torch
import torch.nn.functional as F
import parselmouth
import torchcrepe
import pyworld
import librosa
import numpy as np
from scipy import signal
from functools import lru_cache
from torch import Tensor

now_dir = os.getcwd()
sys.path.append(now_dir)
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.lib.predictors.FCPE import FCPEF0Predictor

from rvc.configs.config import Config


# Constants for high-pass filter
FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48  # Hz
SAMPLE_RATE = 16000  # Hz
bh, ah = signal.butter(
    N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE
)

class Pipeline:
    """
    The main pipeline class for performing voice conversion, including preprocessing, F0 estimation,
    voice conversion using a model, and post-processing.
    """

    def __init__(self, tgt_sr, x_pad):
        """
        Initializes the Pipeline class with target sampling rate and configuration parameters.

        Args:
            tgt_sr: The target sampling rate for the output audio.
            config: A configuration object containing various parameters for the pipeline.
        """
        config = Config()
        self.x_pad = x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.is_half = config.is_half
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = config.device
        self.ref_freqs = [
            65.41,
            82.41,
            110.00,
            146.83,
            196.00,
            246.94,
            329.63,
            440.00,
            587.33,
            783.99,
            1046.50,
        ]

    def get_f0_crepe(
        self,
        x,
        f0_min,
        f0_max,
        p_len,
        hop_length,
        model="full",
    ):
        """
        Estimates the fundamental frequency (F0) of a given audio signal using the Crepe model.

        Args:
            x: The input audio signal as a NumPy array.
            f0_min: Minimum F0 value to consider.
            f0_max: Maximum F0 value to consider.
            p_len: Desired length of the F0 output.
            hop_length: Hop length for the Crepe model.
            model: Crepe model size to use ("full" or "tiny").
        """
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        audio = torch.from_numpy(x).to(self.device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        pitch: Tensor = torchcrepe.predict(
            audio,
            self.sample_rate,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=hop_length * 2,
            device=self.device,
            pad=True,
        )
        p_len = p_len or x.shape[0] // hop_length
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0 = np.nan_to_num(target)
        return f0

    def get_f0_hybrid(
        self,
        methods_str,
        x,
        f0_min,
        f0_max,
        p_len,
        hop_length,
    ):
        """
        Estimates the fundamental frequency (F0) using a hybrid approach combining multiple methods.

        Args:
            methods_str: A string specifying the methods to combine (e.g., "hybrid[crepe+rmvpe]").
            x: The input audio signal as a NumPy array.
            f0_min: Minimum F0 value to consider.
            f0_max: Maximum F0 value to consider.
            p_len: Desired length of the F0 output.
            hop_length: Hop length for F0 estimation methods.
        """
        methods_str = re.search("hybrid\[(.+)\]", methods_str)
        if methods_str:
            methods = [method.strip() for method in methods_str.group(1).split("+")]
        f0_computation_stack = []
        print(f"Calculating f0 pitch estimations for methods {str(methods)}")
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        for method in methods:
            f0 = None
            if method == "crepe":
                f0 = self.get_f0_crepe_computation(
                    x, f0_min, f0_max, p_len, int(hop_length)
                )
            elif method == "rmvpe":
                self.model_rmvpe = RMVPE0Predictor(
                    os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
                    is_half=self.is_half,
                    device=self.device,
                )
                f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
                f0 = f0[1:]
            elif method == "fcpe":
                self.model_fcpe = FCPEF0Predictor(
                    os.path.join("rvc", "models", "predictors", "fcpe.pt"),
                    f0_min=int(f0_min),
                    f0_max=int(f0_max),
                    dtype=torch.float32,
                    device=self.device,
                    sampling_rate=self.sample_rate,
                    threshold=0.03,
                )
                f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
                del self.model_fcpe
                gc.collect()
            f0_computation_stack.append(f0)

        f0_computation_stack = [fc for fc in f0_computation_stack if fc is not None]
        f0_median_hybrid = None
        if len(f0_computation_stack) == 1:
            f0_median_hybrid = f0_computation_stack[0]
        else:
            f0_median_hybrid = np.nanmedian(f0_computation_stack, axis=0)
        return f0_median_hybrid

    def get_f0(
        self,
        x,
        p_len,
        f0_up_key,
        f0_method,
        hop_length,
        inp_f0=None,
    ):
        """
        Estimates the fundamental frequency (F0) of a given audio signal using various methods.

        Args:
            input_audio_path: Path to the input audio file.
            x: The input audio signal as a NumPy array.
            p_len: Desired length of the F0 output.
            f0_up_key: Key to adjust the pitch of the F0 contour.
            f0_method: Method to use for F0 estimation (e.g., "pm", "harvest", "crepe").
            filter_radius: Radius for median filtering the F0 contour.
            hop_length: Hop length for F0 estimation methods.
            inp_f0: Optional input F0 contour to use instead of estimating.
        """
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sample_rate)
                .to_pitch_ac(
                    time_step=self.time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=self.f0_min,
                    pitch_ceiling=self.f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.sample_rate,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=10,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sample_rate)
            f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe":
            f0 = self.get_f0_crepe(x, self.f0_min, self.f0_max, p_len, int(hop_length))
        elif f0_method == "crepe-tiny":
            f0 = self.get_f0_crepe(
                x, self.f0_min, self.f0_max, p_len, int(hop_length), "tiny"
            )
        elif f0_method == "rmvpe":
            self.model_rmvpe = RMVPE0Predictor(
                os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
                is_half=self.is_half,
                device=self.device,
            )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        elif f0_method == "fcpe":
            self.model_fcpe = FCPEF0Predictor(
                os.path.join("rvc", "models", "predictors", "fcpe.pt"),
                f0_min=int(self.f0_min),
                f0_max=int(self.f0_max),
                dtype=torch.float32,
                device=self.device,
                sampling_rate=self.sample_rate,
                threshold=0.03,
            )
            f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
            del self.model_fcpe
            gc.collect()
        elif "hybrid" in f0_method:
            f0 = self.get_f0_hybrid(
                f0_method,
                x,
                self.f0_min,
                self.f0_max,
                p_len,
                hop_length,
            )

        f0 *= pow(2, f0_up_key / 12)
        tf0 = self.sample_rate // self.window
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int)

        return f0_coarse, f0bak

    def voice_conversion(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        version,
        protect,
    ):
        """
        Performs voice conversion on a given audio segment.

        Args:
            model: The feature extractor model.
            net_g: The generative model for synthesizing speech.
            sid: Speaker ID for the target voice.
            audio0: The input audio segment.
            pitch: Quantized F0 contour for pitch guidance.
            pitchf: Original F0 contour for pitch guidance.
            index: FAISS index for speaker embedding retrieval.
            big_npy: Speaker embeddings stored in a NumPy array.
            index_rate: Blending rate for speaker embedding retrieval.
            version: Model version ("v1" or "v2").
            protect: Protection level for preserving the original pitch.
        """
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5 and pitch != None and pitchf != None:
            feats0 = feats.clone()
        if (
            isinstance(index, type(None)) == False
            and isinstance(big_npy, type(None)) == False
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * feats

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch != None and pitchf != None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch != None and pitchf != None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch != None and pitchf != None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch != None and pitchf != None:
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        f0_up_key,
        f0_method,
        protect,
        hop_length,
        version,
    ):
        """
        The main pipeline function for performing voice conversion.

        Args:
            model: The feature extractor model.
            net_g: The generative model for synthesizing speech.
            sid: Speaker ID for the target voice.
            audio: The input audio signal.
            input_audio_path: Path to the input audio file.
            f0_up_key: Key to adjust the pitch of the F0 contour.
            f0_method: Method to use for F0 estimation.
            resample_sr: Resampling rate for the output audio.
            rms_mix_rate: Blending rate for adjusting the RMS level of the output audio.
            version: Model version.
            protect: Protection level for preserving the original pitch.
            hop_length: Hop length for F0 estimation methods.
            f0_file: Path to a file containing an F0 contour to use.
        """
        index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        pitch, pitchf = self.get_f0(
            audio_pad,
            p_len,
            f0_up_key,
            f0_method,
            hop_length,
            inp_f0,
        )
        pitch = pitch[:p_len]
        pitchf = pitchf[:p_len]
        if self.device == "mps":
            pitchf = pitchf.astype(np.float32)
        pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
        pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        for t in opt_ts:
            t = t // self.window * self.window
            audio_opt.append(
                self.voice_conversion(
                    model,
                    net_g,
                    sid,
                    audio_pad[s : t + self.t_pad2 + self.window],
                    pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                    pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                    index,
                    big_npy,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
            s = t
        audio_opt.append(
            self.voice_conversion(
                model,
                net_g,
                sid,
                audio_pad[t:],
                pitch[:, t // self.window :] if t is not None else pitch,
                pitchf[:, t // self.window :] if t is not None else pitchf,
                index,
                big_npy,
                version,
                protect,
            )[self.t_pad_tgt : -self.t_pad_tgt]
        )
        audio_opt = np.concatenate(audio_opt)
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
