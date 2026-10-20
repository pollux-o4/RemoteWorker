"""Public shared-latent nodes and peer execution facade."""
from __future__ import annotations

from ..core.execution import completed_prompt
from ..core.artifacts import successful_outputs
from ..core.protocol import build_release_cleanup_node
from ..defaults import DEFAULT_SERVER_URL
from . import shared_store as store
from .schema import LATENT_TYPE, VIDEO_FPS, VIDEO_CRF, video_inputs, video_options, validate_model_choice
from .service import peer_api, preflight
from . import warmup as warmup_tasks


class SaveSharedVAELatent:
    DESCRIPTION = '공유 폴더에 latent를 원자적으로 저장합니다. 영상 저장·경로 확인 성공 후에만 임시 latent를 삭제합니다. 실패한 작업 ID로 재시도할 수 있습니다.'
    CATEGORY = 'RemoteWorker/VAE'
    RETURN_TYPES = (LATENT_TYPE, 'STRING')
    RETURN_NAMES = ('latent', 'request_id')
    FUNCTION = 'run'
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'samples': ('LATENT',)}, 'optional': {'audio': ('AUDIO',)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def run(self, samples, audio=None):
        request_id = store.save_latent(samples, audio)
        return {'ui': {'text': [request_id]}, 'result': (request_id, request_id)}


class LoadSharedVAELatent:
    DESCRIPTION = '실패 후 보존된 작업 ID를 입력해 공유 latent를 다시 사용합니다. 완료 후 삭제된 latent는 읽을 수 없습니다.'
    CATEGORY = 'RemoteWorker/VAE'
    RETURN_TYPES = (LATENT_TYPE, 'LATENT')
    RETURN_NAMES = ('latent', 'samples')
    FUNCTION = 'run'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'request_id': ('STRING', {'default': ''})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def run(self, request_id):
        # LATENT output deliberately requires the input file; completed jobs can
        # be retried directly with RemoteVAEVideo's request_id override instead.
        return request_id, store.load_latent(request_id)


class RemoteVAEVideo:
    DESCRIPTION = '공유 latent를 처리 서버에서 디코딩하고 MP4(H.264)로 저장합니다. 영상 텐서를 전송하지 않고 경로만 반환합니다. 결과 확인 후 latent 삭제; 실패 시 보존합니다.'
    CATEGORY = 'RemoteWorker/VAE'
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('video_path',)
    OUTPUT_NODE = True
    FUNCTION = 'run'

    @classmethod
    def INPUT_TYPES(cls):
        return video_inputs()

    @classmethod
    def VALIDATE_INPUTS(cls, vae_name, audio_vae_name=""):
        return validate_model_choice(vae_name)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def run(self, vae_name, fps=VIDEO_FPS, audio_vae_name='', crf=VIDEO_CRF, tiled=False,
            server_url=DEFAULT_SERVER_URL, timeout_seconds=600, memory_policy='release_after_result',
            latent=None, request_id='', warmup=None, kind='image'):
        if latent and request_id and latent != request_id:
            raise ValueError('연결된 latent와 재시도 작업 ID가 다릅니다.')
        request_id = latent or request_id
        store.request_path('temp', request_id, 'latent.pt')
        options = video_options(vae_name, audio_vae_name, fps, crf, tiled, kind)
        path = execute_shared(peer_api(server_url, timeout_seconds), request_id, options,
            server_url=server_url, memory_policy=memory_policy, warmup=warmup)
        return {'ui': {'text': [path]}, 'result': (path,)}


def execute_shared(api, request_id, options, *, server_url, memory_policy, warmup=None):
    try:
        preflight(api, options['vae_name'], 'SharedVAEVideoWorker', memory_policy)
        if options['audio_vae_name']:
            preflight(api, options['audio_vae_name'], 'SharedVAEVideoWorker')
        warmup_tasks.finish(api, warmup, vae_name=options['vae_name'], kind=options.get('kind', 'image'), server_url=server_url)
        graph = {'save': {'class_type': 'SharedVAEVideoWorker', 'inputs': {'request_id': request_id, **options}}}
        if memory_policy == 'release_after_result':
            graph['cleanup'] = build_release_cleanup_node(['save', 0])
        with completed_prompt(api, graph) as (prompt_id, history):
            outputs = successful_outputs(history, prompt_id)
            receipts = outputs.get('save', {}).get('shared_video')
            if not isinstance(receipts, list) or len(receipts) != 1:
                raise ValueError('Shared VAE video receipt is missing')
            path = store.acknowledge(request_id, receipts[0], options)
        return path
    except Exception as exc:
        raise RuntimeError(f'공유 VAE 작업 실패 · 재시도 ID: {request_id} · 임시 latent 보존: {exc}') from exc


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    SaveSharedVAELatent, LoadSharedVAELatent, RemoteVAEVideo)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'SaveSharedVAELatent': 'Save Shared VAE Latent',
    'LoadSharedVAELatent': 'Load Shared VAE Latent',
    'RemoteVAEVideo': 'Remote VAE Video (Shared File)',
}
