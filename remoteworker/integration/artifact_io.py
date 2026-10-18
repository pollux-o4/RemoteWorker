"""conditioning_io — CONDITIONING 을 디스크로 저장하고 다른 런타임에서 로드하는 ComfyUI 노드.

목적: 생성 측 런타임이 텍스트 인코더로 만든 CONDITIONING을 CPU `.cond` 파일로 보관하고,
소비 측 런타임은 텍스트 인코더를 다시 적재하지 않고 그 파일을 쓴다.

UI: LoadConditioning 노드는 LoadImage 처럼 업로드 버튼·드래그-드롭·붙여넣기를 지원한다.
  파일을 노드에 떨구면 /upload/image 로 서버 input/ 에 올라가고 드롭다운에 바로 잡힌다.

호환: 생성·소비 측은 같은 텍스트 인코더·ComfyUI conditioning 계약을 사용해야 한다.
`.cond` v1에는 호환성 메타데이터가 없으므로, 이 확인은 현재 작업자가 수행한다.
"""
import os

import torch

try:
    import folder_paths
except Exception:                       # ComfyUI 밖 import 대비(검증용)
    folder_paths = None


def _to_cpu(obj):
    """conditioning 안의 모든 텐서를 detach + CPU 로 (기기 독립 직렬화)."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _list_input_conds():
    if folder_paths is None:
        return []
    d = folder_paths.get_input_directory()
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".cond"))


def _resolve(filename):
    """input/ → output/ → 그대로 순으로 파일을 찾는다."""
    cands = []
    if folder_paths is not None:
        cands.append(os.path.join(folder_paths.get_input_directory(), filename))
        cands.append(os.path.join(folder_paths.get_output_directory(), filename))
    cands.append(filename)
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


class SaveConditioning:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory() if folder_paths else "."

    @classmethod
    def INPUT_TYPES(cls):
        # filename_prefix: SaveImage/SaveLatent 와 동일 — "sub/name" 이면 sub 는 하위폴더, name 은 접두어.
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "filename_prefix": ("STRING", {"default": "conditioning/cond"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("artifact_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/io"

    def save(self, conditioning, filename_prefix):
        # get_save_image_path 가 하위폴더 생성 + 기존 파일 스캔해 다음 카운터를 잡아줌 → _00001_ 로 쌓임(덮어쓰기 X).
        full, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        os.makedirs(full, exist_ok=True)
        file = f"{name}_{counter:05}_.cond"
        path = os.path.join(full, file)
        torch.save(_to_cpu(conditioning), path)
        print(f"[SaveConditioning] saved -> {path}")
        display_name = subfolder.replace("\\", "/") + "/" + file if subfolder else file
        return {"ui": {"text": [display_name]}, "result": (display_name,)}


class SaveRemoteConditioning:
    """Persist one trusted remote-conditioning artifact without shadowing Comfy's built-in node.

    Recent ComfyUI releases register their own ``SaveConditioning`` node.  The
    remote protocol therefore uses this deliberately distinct node id and UI
    field, so a peer cannot silently serialize a different artifact format.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory() if folder_paths else "."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "filename_prefix": ("STRING", {"default": "remote-conditioning/cond"}),
            },
            "optional": {
                # Explicit ordering makes VRAM cleanup safe after every Qwen
                # positive/negative/latent artifact has reached disk.
                "dependency": ("*",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("artifact_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/io"

    def save(self, conditioning, filename_prefix, dependency=None):
        full, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        os.makedirs(full, exist_ok=True)
        file = f"{name}_{counter:05}_.cond"
        path = os.path.join(full, file)
        torch.save(_to_cpu(conditioning), path)
        print(f"[SaveRemoteConditioning] saved -> {path}")
        display_name = subfolder.replace("\\", "/") + "/" + file if subfolder else file
        return {"ui": {"conditioning": [display_name]}, "result": (display_name,)}


class SaveRemoteLatent:
    """Persist a LATENT through ``torch.save`` for cross-runtime remote return.

    MiniMax H3 returns a PyTorch NestedTensor.  ComfyUI's built-in SaveLatent
    writes safetensors and therefore requires ``contiguous()``, which that type
    deliberately does not expose.  The remote adapter needs a lossless CPU
    transport artifact instead of a portable safetensors checkpoint.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory() if folder_paths else "."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "remote-conditioning/latent"}),
            },
            "optional": {
                # Allows remote workflows to serialize CONDITIONING first, then
                # LATENT, then run a cleanup node after both artifacts exist.
                "dependency": ("*",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("artifact_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/io"

    def save(self, samples, filename_prefix, dependency=None):
        if not isinstance(samples, dict) or "samples" not in samples:
            raise ValueError("Save Remote Latent requires a LATENT dictionary with samples")
        full, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        os.makedirs(full, exist_ok=True)
        file = f"{name}_{counter:05}_.latent"
        path = os.path.join(full, file)
        torch.save(_to_cpu(samples), path)
        print(f"[SaveRemoteLatent] saved -> {path}")
        display_name = subfolder.replace("\\", "/") + "/" + file if subfolder else file
        return {"ui": {"latents": [display_name]}, "result": (display_name,)}


class SaveRemoteText:
    """Write one UTF-8 STRING artifact for a request-scoped remote response."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory() if folder_paths else "."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True}),
            "filename_prefix": ("STRING", {"default": "remote-conditioning/prompt"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/io"

    def save(self, text, filename_prefix):
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise ValueError("Save Remote Text requires a non-empty UTF-8 text value")
        full, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        os.makedirs(full, exist_ok=True)
        file = f"{name}_{counter:05}_.txt"
        path = os.path.join(full, file)
        with open(path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        print(f"[SaveRemoteText] saved -> {path}")
        display_name = subfolder.replace("\\", "/") + "/" + file if subfolder else file
        return {"ui": {"text": [display_name]}}


class LoadConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        # input/ 의 .cond 목록을 드롭다운으로. web/uploadcond.js 가 업로드 버튼·드래그·붙여넣기를 붙인다.
        return {"required": {
            "filename": (_list_input_conds(), {"cond_upload": True}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load"
    CATEGORY = "conditioning/io"

    @classmethod
    def IS_CHANGED(cls, filename):
        p = _resolve(filename)
        return os.path.getmtime(p) if p else float("nan")     # 파일 바뀌면 캐시 무효화

    @classmethod
    def VALIDATE_INPUTS(cls, filename):
        # 업로드 직후 등 object_info 캐시보다 최신 파일도 통과시킨다(콤보 고정목록 검증 우회).
        return True if _resolve(filename) else f"conditioning 파일 없음: {filename}"

    def load(self, filename):
        path = _resolve(filename)
        if path is None:
            raise FileNotFoundError(
                f"[LoadConditioning] '{filename}' 못 찾음 — input/ 에 올리거나 드롭해줘")
        # weights_only=False: conditioning 은 state_dict 가 아니라 list/dict 구조라 필요(torch 2.6+ 기본 True 면 실패)
        cond = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[LoadConditioning] loaded <- {path}")
        return (cond,)


NODE_CLASS_MAPPINGS = {
    "SaveConditioning": SaveConditioning,
    "SaveRemoteConditioning": SaveRemoteConditioning,
    "SaveRemoteLatent": SaveRemoteLatent,
    "SaveRemoteText": SaveRemoteText,
    "LoadConditioning": LoadConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveConditioning": "Save Conditioning (.cond)",
    "SaveRemoteConditioning": "Save Remote Conditioning (.cond)",
    "SaveRemoteLatent": "Save Remote Latent (.latent)",
    "SaveRemoteText": "Save Remote Text (.txt)",
    "LoadConditioning": "Load Conditioning (.cond)",
}
