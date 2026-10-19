"""Isolated H3 queue lookahead and remote decode acceptance run."""
import json
import os
from pathlib import Path
import time
import urllib.request
from urllib.error import HTTPError

CALLER=os.environ.get('REMOTEWORKER_E2E_CALLER','http://127.0.0.1:18288')
PEER=os.environ.get('REMOTEWORKER_E2E_PEER','http://127.0.0.1:18289')
ROOT=Path(os.environ.get('REMOTEWORKER_E2E_ROOT','./remote-preparation-amd'))
REPORTS=Path(os.environ.get('REMOTEWORKER_E2E_REPORTS',str(ROOT)))
REPORTS.mkdir(parents=True,exist_ok=True)


def request(url, path, payload=None):
    data=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request(url+path,data=data,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=20) as response:
            body=response.read()
            return json.loads(body) if body else {}
    except HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def workflow(prompt, name):
    return {
      'clip': {'class_type':'RemoteMiniMaxH3ReferenceEncode','inputs':{
        'prompt':prompt,'clip_name':'nvmes/minimaxH3INT4Convrot_qwen3vl32bInt4.safetensors',
        'clip_type':'minimax','device':'default','vae_name':'nvmes/minimax_h3_video_vae_fp16.safetensors',
        'audio_vae_name':'nvmes/minimax_h3_audio_vae_fp32.safetensors','width':32,'height':32,
        'length':5,'ref_image_size':'match','server_url':PEER,'timeout_seconds':600,
        'peer_memory_policy':'release_after_result','cache_policy':'reuse_same_inputs'}},
      'model':{'class_type':'UNETLoader','inputs':{'unet_name':'nvmes/minimaxH3INT8INT4_ref2va_flREF2VAPruned.safetensors','weight_dtype':'default'}},
      'shift':{'class_type':'MiniMaxH3SigmaShift','inputs':{'model':['model',0],'shift_video':12,'shift_audio':3}},
      'guider':{'class_type':'BasicGuider','inputs':{'model':['shift',0],'conditioning':['clip',0]}},
      'noise':{'class_type':'RandomNoise','inputs':{'noise_seed':42}},
      'sampler':{'class_type':'KSamplerSelect','inputs':{'sampler_name':'euler'}},
      'sigma':{'class_type':'BasicScheduler','inputs':{'model':['shift',0],'scheduler':'simple','steps':1,'denoise':1}},
      'sample':{'class_type':'SamplerCustomAdvanced','inputs':{'noise':['noise',0],'guider':['guider',0],
        'sampler':['sampler',0],'sigmas':['sigma',0],'latent_image':['clip',1]}},
      'video_decode':{'class_type':'RemoteVAEDecode','inputs':{'samples':['sample',0],
        'vae_name':'nvmes/minimax_h3_video_vae_fp16.safetensors','server_url':PEER,'timeout_seconds':600,'memory_policy':'release_after_result','tiled':False,'tile_size':512,'overlap':64,'temporal_size':64,'temporal_overlap':8}},
      'audio_decode':{'class_type':'RemoteVAEDecodeAudio','inputs':{'samples':['sample',0],
        'vae_name':'nvmes/minimax_h3_audio_vae_fp32.safetensors','server_url':PEER,'timeout_seconds':600,'memory_policy':'release_after_result'}},
      'frames':{'class_type':'ImageFromBatch','inputs':{'image':['video_decode',0],'batch_index':0,'length':3}},
      'video':{'class_type':'CreateVideo','inputs':{'images':['frames',0],'audio':['audio_decode',0],'fps':30,'bit_depth':'auto','color_space':'sRGB','codec':'h264'}},
      'save':{'class_type':'SaveVideo','inputs':{'video':['video',0],'filename_prefix':'preparation/'+name,'format':'auto','format.codec':'auto'}},
    }



def validate_acceptance(ids, completed, status):
    assert len(completed) == len(ids)
    assert all(job['status']['status_str'] == 'success' for job in completed.values())
    events = status['events']
    for prompt_id in ids[1:]:
        ready = [e for e in events if e['state'] == 'conditioning_ready' and e.get('prompt_id') == prompt_id]
        assert ready, 'next conditioning was not prepared: ' + prompt_id
        started = next(data['timestamp']/1000 for kind, data in completed[prompt_id]['status']['messages']
                       if kind == 'execution_start')
        assert ready[0]['at'] <= started, 'conditioning was not ahead of queue execution'
    assert any(e['state'] == 'vae_ready' for e in events), 'no automatic VAE warmup'
    for job in completed.values():
        for item in job['outputs']['save']['images']:
            assert item['metadata']['frame_count'] == 3
            assert item['metadata']['duration'] == .1


def main():
    for url in (CALLER, PEER):
        for attempt in range(60):
            try:
                request(url, '/system_stats')
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError('development server unavailable: '+url)
    ids=[]
    for i,prompt in enumerate(['A red ball rolls on a table.','A blue ball rolls on a table.','A blue ball rolls on a table.']):
        graph=workflow(prompt, f'case-{i+1}')
        reply=request(CALLER,'/prompt',{'prompt':graph,'client_id':'preparation-acceptance'})
        ids.append(reply['prompt_id']);print('queued',i+1,ids[-1],flush=True)
        (ROOT/f'case-{i+1}.api.json').write_text(json.dumps(graph,indent=2))
    completed={}
    end=time.monotonic()+1800
    while time.monotonic()<end:
        for prompt_id in ids:
            if prompt_id in completed:continue
            history=request(CALLER,'/history/'+prompt_id).get(prompt_id)
            if history:
                completed[prompt_id]=history
                print('finished',prompt_id,history['status']['status_str'],flush=True)
                if history['status']['status_str']!='success':print(json.dumps(history['status']),flush=True)
        if len(completed)==len(ids):break
        time.sleep(2)
    status=request(CALLER,'/remoteworker/preparation')
    (ROOT/'acceptance.json').write_text(json.dumps({'jobs':completed,'preparation':status},indent=2))
    validate_acceptance(ids, completed, status)
    print('all jobs succeeded',json.dumps(status),flush=True)

if __name__=='__main__':main()
