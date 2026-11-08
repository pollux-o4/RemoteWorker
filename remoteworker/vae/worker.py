"""Peer execution: reuse native decoders; persist only lossless CPU results."""

from __future__ import annotations

from pathlib import Path

from .. import _native
from .codec import dumps
from .schema import DECODE_DEFAULTS, decode_fields, video_fields, video_options


def load_request_latent(filename):
    _native.vae_upload_relative(filename)
    import folder_paths

    from .codec import MAX_BYTES

    return _native.vae_load_upload(filename, folder_paths.get_temp_directory(), MAX_BYTES)


def check_output_budget(vae, samples, kind):
    from .codec import MAX_BYTES

    return _native.vae_output_budget(vae, samples, kind, MAX_BYTES)


def decode(
    vae,
    samples,
    kind,
    tiled,
    tile_size,
    overlap,
    temporal_size,
    temporal_overlap,
    *,
    transfer=True,
):
    from .codec import MAX_BYTES

    return _native.vae_decode(
        vae,
        samples,
        kind,
        tiled,
        tile_size,
        overlap,
        temporal_size,
        temporal_overlap,
        transfer=transfer,
        max_bytes=MAX_BYTES,
    )


class RemoteVAEDecodeWorker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "filename": ("STRING",),
                "kind": (["image", "audio"],),
                **decode_fields(),
            }
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "run"
    CATEGORY = "RemoteWorker/internal"
    DEV_ONLY = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(
        self,
        vae,
        filename,
        kind,
        tiled,
        tile_size,
        overlap,
        temporal_size,
        temporal_overlap,
    ):
        samples = load_request_latent(filename)
        return (
            decode(
                vae,
                samples,
                kind,
                tiled,
                tile_size,
                overlap,
                temporal_size,
                temporal_overlap,
            ),
        )


class SaveRemoteVAEOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",), "filename_prefix": ("STRING",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "RemoteWorker/internal"
    DEV_ONLY = True

    def save(self, value, filename_prefix):
        import folder_paths

        data = dumps(value)
        full, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        path = Path(full) / f"{name}_{counter:05}_.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        locator = f"{subfolder}/{path.name}" if subfolder else path.name
        return {"ui": {"vae_output": [locator]}, "result": (locator,)}


class RemoteVAEWarmupWorker:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "kind": (["image", "audio"],)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "RemoteWorker/internal"
    DEV_ONLY = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, vae, kind):
        import torch

        # H3 video uses 24xTxHxW, audio uses 32xstereo xT. For other
        # families derive the rank/channels from the loaded native VAE.
        channels, dims = vae.latent_channels, vae.latent_dim
        shape = tuple(_native.vae_warmup_shape(channels, dims, getattr(vae, "output_channels", None), kind))
        output = decode(vae, {"samples": torch.zeros(shape)}, kind, **DECODE_DEFAULTS)
        del output
        return ("ready",)


def _load_vae(name):
    from nodes import VAELoader

    return VAELoader().load_vae(name)[0]


def _encode_wav(audio, waveform, path):
    import av

    channels = waveform.shape[1]
    with av.open(str(path), "w", format="wav") as container:
        stream = container.add_stream("pcm_f32le", rate=audio["sample_rate"])
        frame = av.AudioFrame.from_ndarray(
            waveform[0].float().numpy(),
            format="fltp",
            layout="stereo" if channels == 2 else "mono",
        )
        frame.sample_rate = audio["sample_rate"]
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def _encode_mp4(frames, audio, path, options):
    from fractions import Fraction

    from comfy_api.latest import InputImpl, Types

    video = InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, audio=audio, frame_rate=Fraction(str(options["fps"]))))
    video.save_to(
        str(path),
        format=Types.VideoContainer.MP4,
        codec=Types.VideoCodec.H264,
        crf=options["crf"],
    )


def encode_video(samples, path, options):
    return _native.vae_encode_video(samples, path, options, _load_vae, decode, _encode_wav, _encode_mp4)


class SharedVAEVideoWorker:
    CATEGORY = "RemoteWorker/internal"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    DEV_ONLY = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "request_id": ("STRING",),
                "vae_name": ("STRING",),
                **video_fields(),
                "audio_vae_name": ("STRING", {"default": ""}),
            },
            "optional": {"kind": (["image", "audio"],)},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, request_id, **options):
        from .shared_store import publish

        receipt = publish(request_id, video_options(**options), encode_video)
        return {"ui": {"shared_video": [receipt]}, "result": (receipt["path"],)}


NODE_CLASS_MAPPINGS = {
    cls.__name__: cls
    for cls in (
        RemoteVAEDecodeWorker,
        SaveRemoteVAEOutput,
        RemoteVAEWarmupWorker,
        SharedVAEVideoWorker,
    )
}
NODE_DISPLAY_NAME_MAPPINGS = {name: name for name in NODE_CLASS_MAPPINGS}
