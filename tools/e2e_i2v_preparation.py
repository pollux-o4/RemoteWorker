"""Real linked I2V + always_refresh queue acceptance on isolated GPU servers."""
import json
import time
from pathlib import Path
from e2e_preparation import workflow, request, CALLER, PEER, ROOT, REPORTS


def create_fixtures():
    from PIL import Image, ImageDraw
    directory=ROOT/'input'
    directory.mkdir(parents=True,exist_ok=True)
    for name,x in [('first',8),('last',20)]:
        image=Image.new('RGB',(32,32),'#dadada')
        ImageDraw.Draw(image).ellipse((x-4,12,x+4,20),fill='#cc3333')
        image.save(directory/f'remoteworker-e2e-{name}.png')


def graph(label, steps=1):
    g=workflow('A '+label+' geometric ball rolls on a plain table.','i2v-'+label)
    g['clip']={'class_type':'RemoteMiniMaxH3ImageToVideo','inputs':{
        'prompt':'A '+label+' geometric ball rolls on a plain table.',
        'clip_name':'nvmes/minimaxH3INT4Convrot_qwen3vl32bInt4.safetensors','clip_type':'minimax','device':'default',
        'vae_name':'nvmes/minimax_h3_video_vae_fp16.safetensors','width':['size',0],'height':['size',1],
        'length':['length',1],'first_frame':['first',0],'last_frame':['last',0],
        'server_url':PEER,'timeout_seconds':600,'peer_memory_policy':'keep','cache_policy':'always_refresh'}}
    g.update({
      'first':{'class_type':'LoadImage','inputs':{'image':'remoteworker-e2e-first.png'}},
      'last':{'class_type':'LoadImage','inputs':{'image':'remoteworker-e2e-last.png'}},
      'scale':{'class_type':'ImageScaleToTotalPixels','inputs':{'image':['last',0],'upscale_method':'nearest-exact','megapixels':.01,'resolution_steps':32}},
      'size':{'class_type':'GetImageSize','inputs':{'image':['scale',0]}},
      'seconds':{'class_type':'PrimitiveFloat','inputs':{'value':.1}},
      'length':{'class_type':'ComfyMathExpression','inputs':{'expression':'max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17','values.a':['seconds',0]}},
    })
    g['sigma']['inputs']['steps']=steps
    return g


def wait_jobs(ids, destination):
    until=time.monotonic()+900
    while time.monotonic()<until:
        try:
            jobs={p:request(CALLER,'/history/'+p).get(p) for p in ids}
        except OSError:
            time.sleep(1);continue
        if all(jobs.values()):
            data={'jobs':jobs,'preparation':request(CALLER,'/remoteworker/preparation')}
            (REPORTS/destination).write_text(json.dumps(data,indent=2))
            return data
        time.sleep(1)
    raise TimeoutError('I2V acceptance timed out')


def validate_short_video(job, output_root):
    import av
    item=job['outputs']['save']['images'][0]
    with av.open(str(Path(output_root)/item['subfolder']/item['filename'])) as clip:
        # Container timestamps include the audio stream; video packets can round by 1 ms.
        duration=clip.duration/av.time_base
        frames=list(clip.decode(video=0))
    assert len(frames)==3
    assert abs(duration-.1)<.002, duration


if __name__=='__main__':
    create_fixtures()
    for url in (CALLER,PEER):
        for _ in range(60):
            try:
                request(url,'/system_stats');break
            except OSError:time.sleep(1)
        else:raise RuntimeError('server unavailable: '+url)
    a=request(CALLER,'/prompt',{'prompt':graph('red',256)})['prompt_id']
    time.sleep(5)
    b=request(CALLER,'/prompt',{'prompt':graph('blue')})['prompt_id']
    print('A then B after five seconds',a,b,flush=True)
    data=wait_jobs([a,b],'i2v-acceptance.json')
    assert all(v['status']['status_str']=='success' for v in data['jobs'].values()), {k:v['status']['status_str'] for k,v in data['jobs'].items()}
    events=data['preparation']['events']
    ready=next(e for e in events if e['state']=='conditioning_ready' and e.get('prompt_id')==b)
    used=next(e for e in events if e['state']=='conditioning_consumed' and e.get('prompt_id')==b)
    started=next(v['timestamp']/1000 for k,v in data['jobs'][b]['status']['messages'] if k=='execution_start')
    assert ready['at']<started<=used['at']
    assert data['preparation']['handoff_bytes']==0
    for job in data['jobs'].values():
        validate_short_video(job, ROOT/'output')
    print('PASS: linked I2V, refresh, ahead-of-execution, one-shot consumption, zero retained handoff',flush=True)
