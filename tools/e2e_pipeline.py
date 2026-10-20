"""Run against the isolated CPU server with the opt-in pipeline_nodes fixture."""
import json,urllib.request,time
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--server',default='http://127.0.0.1:18294')
parser.add_argument('--log',default='/tmp/pipeline-dev-events.jsonl')
args=parser.parse_args()
base=args.server.rstrip('/')
def req(path,data=None):
 request=urllib.request.Request(base+path,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(request,timeout=10) as response:
   body=response.read();return json.loads(body) if body else None
 except urllib.error.HTTPError as e:raise RuntimeError(e.read().decode()[:1800]) from e
print('pipeline:',req('/remoteworker/preparation').get('pipeline'))
import uuid
run_token=uuid.uuid4().hex
start=time.monotonic();ids=[]
for tag in ('A','B','C'):
 graph={'s':{'class_type':'PipelineSampler','inputs':{'tag':tag,'delay':0.6,'model':['m',0]}},'d':{'class_type':'RemoteVAEDecode','inputs':{'samples':['s',0],'vae_name':'test.vae','server_url':'http://127.0.0.1:18299','timeout_seconds':30,'memory_policy':'keep','output_mode':'tensor','tiled':False,'tile_size':512,'overlap':64,'temporal_size':64,'temporal_overlap':8}},'p':{'class_type':'PipelineSink','inputs':{'image':['d',0],'tag':tag}}}
 graph['m']={'class_type':'PipelineModel','inputs':{'strength':0.5 if tag=='C' else 1.0,'token':run_token}}
 ids.append(req('/prompt',{'prompt':graph,'client_id':'pipeline-test'})['prompt_id'])
end=time.monotonic()+15
while time.monotonic()<end:
 histories=[req('/history/'+pid) for pid in ids]
 if all(histories):break
 time.sleep(.1)
for pid,h,tag in zip(ids,histories,('A','B','C')):
 result=h.get(pid,{})
 assert result.get('status',{}).get('status_str')=='success',result
 assert result['outputs']['p']['tag']==[tag],result
print('PASS native queue histories and downstream previews retain original prompt IDs',ids)
events=[json.loads(l) for l in Path(args.log).read_text().splitlines()]
events=[e for e in events if e['at']>=start]
print([(e['event'],round(e['at']-start,3),e['prompt']) for e in events])
a=next(e for e in events if e['event']=='sample-end-A')
b=next(e for e in events if e['event']=='sample-start-B')
decodes=[e for e in events if e['event']=='decode-end']
assert decodes and b['at']<decodes[0]['at'], 'B sampling waited for A decode'
print('PASS next sampling begins before previous decode finishes')

loads=[e['event'] for e in events if e['event'].startswith('model-load-')]
assert sorted(loads)==['model-load-0.5','model-load-1.0'],loads
print('PASS native model signature reused across two executors; changed settings invalidate reuse')
print(req('/remoteworker/preparation')['pipeline'])

for endpoint in ('interrupt','jobs'):
 ids=[]
 for tag in ('A','B'):
  graph={'s':{'class_type':'PipelineSampler','inputs':{'tag':tag,'delay':0.6}},'d':{'class_type':'RemoteVAEDecode','inputs':{'samples':['s',0],'vae_name':'test.vae','server_url':'http://127.0.0.1:18299','timeout_seconds':30,'memory_policy':'keep','output_mode':'tensor','tiled':False,'tile_size':512,'overlap':64,'temporal_size':64,'temporal_overlap':8}},'p':{'class_type':'PipelineSink','inputs':{'image':['d',0],'tag':tag}}}
  ids.append(req('/prompt',{'prompt':graph,'client_id':'pipeline-test'})['prompt_id'])
 end=time.monotonic()+10
 while time.monotonic()<end:
  s=req('/remoteworker/preparation')['pipeline']
  if s.get('local_prompt_id')==ids[1] and any(j['prompt_id']==ids[0] and j['phase']=='remote' for j in s['jobs']):break
  time.sleep(.05)
 else:raise RuntimeError('No overlap window')
 req('/interrupt',{'prompt_id':ids[0]}) if endpoint=='interrupt' else req('/api/jobs/'+ids[0]+'/cancel',{})
 while time.monotonic()<end:
  h=[req('/history/'+pid).get(pid) for pid in ids]
  if all(h):break
  time.sleep(.1)
 assert h[0]['status']['status_str']=='error',h[0]
 assert any(m[0]=='execution_interrupted' for m in h[0]['status']['messages']),h[0]
 assert h[1]['status']['status_str']=='success',h[1]
 assert h[1]['outputs']['p']['tag']==['B']
 print('PASS targeted '+endpoint+' cancellation: waiting A interrupted; sampling B preview succeeds',flush=True)

# The native preview must reach its original websocket client during B's long local node.
import asyncio
import aiohttp
async def check_early_preview():
 client_id = 'early-preview-' + uuid.uuid4().hex
 session = aiohttp.ClientSession()
 ws = await session.ws_connect(base.replace('http', 'ws', 1) + '/ws?clientId=' + client_id)
 preview_ids = []
 for tag, delay in [('early-A', 0.1), ('long-B', 4.0)]:
  graph = {
   's': {'class_type': 'PipelineSampler', 'inputs': {'tag': tag, 'delay': delay}},
   'd': {'class_type': 'RemoteVAEDecode', 'inputs': {'samples': ['s', 0], 'vae_name': 'test.vae',
          'server_url': 'http://127.0.0.1:18299', 'timeout_seconds': 30, 'memory_policy': 'keep',
          'output_mode': 'tensor', 'tiled': False, 'tile_size': 512, 'overlap': 64,
          'temporal_size': 64, 'temporal_overlap': 8}},
   'p': {'class_type': 'PreviewImage', 'inputs': {'images': ['d', 0]}}}
  preview_ids.append(req('/prompt', {'prompt': graph, 'client_id': client_id})['prompt_id'])
 try:
  end = time.monotonic() + 10
  while time.monotonic() < end:
   message = await ws.receive(timeout=10)
   message = message.data
   if not isinstance(message, str): continue
   message = json.loads(message)
   data = message.get('data', {})
   if message['type'] == 'executed' and data.get('node') == 'p' and data.get('prompt_id') == preview_ids[0]:
    early_image = data['output']['images'][0]
    from urllib.parse import urlencode
    async with session.get(base + '/view?' + urlencode({k: early_image[k] for k in ('filename', 'subfolder', 'type')})) as image:
     assert image.status == 200
     assert (await image.read()).startswith(b'\x89PNG'), 'preview file unavailable'
    status = req('/remoteworker/preparation')['pipeline']
    assert status['local_prompt_id'] == preview_ids[1], status
    assert not req('/history/' + preview_ids[1]), 'B completed before A preview'
    assert any(j['prompt_id'] == preview_ids[0] and j.get('preview_nodes') == ['p'] for j in status['jobs']), status
    break
  else: raise AssertionError('native preview never arrived during next sampling')
 finally:
  await ws.close()
  await session.close()
 while time.monotonic() < end:
  histories = [req('/history/' + pid).get(pid) for pid in preview_ids]
  if all(histories): break
  time.sleep(.1)
 for history in histories:
  assert history['status']['status_str'] == 'success', history
 assert all(histories[0]['outputs']['p']['images'][0][key] == early_image[key]
            for key in ('filename', 'subfolder', 'type')), 'preview was encoded twice'
 print('PASS native PreviewImage websocket arrives during B sampling; same temp file retained in A history', flush=True)
asyncio.run(check_early_preview())
