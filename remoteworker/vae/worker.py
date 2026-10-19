"""Peer execution: reuse native decoders; persist only lossless CPU results."""
from __future__ import annotations

from pathlib import Path
import re

from .codec import dumps, loads, validate_latent
from .schema import DECODE_DEFAULTS, decode_fields, video_fields, video_options


def load_request_latent(filename):
    if not re.fullmatch(r"remote-conditioning/[A-Za-z0-9_-]{1,64}/samples\.pt \[temp\]", filename):
        raise ValueError("Remote VAE latent must belong to a request-scoped temp upload")
    import folder_paths
    root = Path(folder_paths.get_temp_directory()).resolve()
    path = (root / filename.removesuffix(" [temp]")).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Remote VAE latent path escapes temp storage")
    with path.open("rb") as stream:
        from .codec import MAX_BYTES
        return validate_latent(loads(stream.read(MAX_BYTES + 1)))


def check_output_budget(vae, samples, kind):
    from math import prod
    from .codec import MAX_BYTES
    latent = samples["samples"]
    if getattr(latent, "is_nested", False):
        latent = latent.unbind()[0 if kind == "image" else -1]
    shape = list(latent.shape)
    ratios = vae.upscale_ratio
    if kind == "audio":
        length = shape[-1] * ratios if isinstance(ratios, int) else shape[-1]
        elements = shape[0] * getattr(vae, "output_channels", 2) * length
    else:
        ratios = ratios if isinstance(ratios, tuple) else (ratios,) * (len(shape) - 2)
        axes = shape[2:]
        pixels = [ratio(axis) if callable(ratio) else ratio * axis for ratio, axis in zip(ratios, axes)]
        elements = shape[0] * getattr(vae, "output_channels", 3) * prod(pixels)
    if elements * 4 > MAX_BYTES - 1024 * 1024:
        raise ValueError("Decoded output exceeds the Remote VAE transfer budget; reduce the batch or clip length")


def decode(vae, samples, kind, tiled, tile_size, overlap, temporal_size, temporal_overlap, *, transfer=True):
    channels = getattr(vae, "output_channels", None)
    if kind == "audio" and channels is not None and channels >= 3:
        raise ValueError("선택한 모델은 이미지·영상 VAE입니다. 오디오 VAE를 선택하세요.")
    if kind == "image" and channels == 2:
        raise ValueError("선택한 모델은 오디오 VAE입니다. 오디오 디코딩 또는 오디오 대상을 선택하세요.")
    if transfer:
        check_output_budget(vae, samples, kind)
    if kind == "audio":
        from comfy_extras.nodes_audio import vae_decode_audio
        return vae_decode_audio(vae, samples)
    from nodes import VAEDecode, VAEDecodeTiled
    if tiled:
        return VAEDecodeTiled().decode(vae, samples, tile_size, overlap, temporal_size, temporal_overlap)[0]
    return VAEDecode().decode(vae, samples)[0]


class RemoteVAEDecodeWorker:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",), "filename": ("STRING",), "kind": (["image", "audio"],),
            **decode_fields(),
        }}

    RETURN_TYPES = ("*",)
    FUNCTION = "run"
    CATEGORY = "RemoteWorker/internal"
    DEV_ONLY = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, vae, filename, kind, tiled, tile_size, overlap, temporal_size, temporal_overlap):
        samples = load_request_latent(filename)
        return (decode(vae, samples, kind, tiled, tile_size, overlap, temporal_size, temporal_overlap),)


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
        full, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
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
        if kind == "audio" and channels == 32 and dims == 2:
            shape = (1, channels, 2, 8)
        elif kind == "image" and dims == 3:
            shape = (1, channels, 2, 2, 2)
        elif kind == "image" and dims == 2 and getattr(vae, "output_channels", 3) != 2:
            shape = (1, channels, 8, 8)
        else:
            raise ValueError("Dummy warmup is unsupported for this VAE family")
        output = decode(vae, {"samples": torch.zeros(shape)}, kind, **DECODE_DEFAULTS)
        del output
        return ("ready",)


def encode_video(samples, path, options):
    """Native Comfy adapter at the shared store's encoder seam."""
    from fractions import Fraction
    from nodes import VAELoader
    from comfy_api.latest import InputImpl, Types

    vae = VAELoader().load_vae(options['vae_name'])[0]
    frames = decode(vae, samples, 'image', **(DECODE_DEFAULTS | {'tiled': options['tiled']}), transfer=False)
    audio = samples.get('remote_vae_audio')
    if options['audio_vae_name']:
        vae = VAELoader().load_vae(options['audio_vae_name'])[0]
        audio = decode(vae, samples, 'audio', **DECODE_DEFAULTS, transfer=False)
    video = InputImpl.VideoFromComponents(Types.VideoComponents(
        images=frames, audio=audio, frame_rate=Fraction(str(options['fps']))))
    video.save_to(str(path), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264, crf=options['crf'])
    return dict(frames=int(frames.shape[0]), width=int(frames.shape[2]),
                height=int(frames.shape[1]), audio=audio is not None)


class SharedVAEVideoWorker:
    CATEGORY = 'RemoteWorker/internal'
    RETURN_TYPES = ('STRING',)
    FUNCTION = 'run'
    OUTPUT_NODE = True
    DEV_ONLY = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'request_id': ('STRING',), 'vae_name': ('STRING',), **video_fields()}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def run(self, request_id, **options):
        from .shared_store import publish
        receipt = publish(request_id, video_options(**options), encode_video)
        return {'ui': {'shared_video': [receipt]}, 'result': (receipt['path'],)}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    RemoteVAEDecodeWorker, SaveRemoteVAEOutput, RemoteVAEWarmupWorker, SharedVAEVideoWorker)}
NODE_DISPLAY_NAME_MAPPINGS = {name: name for name in NODE_CLASS_MAPPINGS}
