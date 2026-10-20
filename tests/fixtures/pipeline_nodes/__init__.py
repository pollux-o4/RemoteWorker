"""CPU-only integration fixture; never register in production."""
import os
if os.environ.get("REMOTEWORKER_PIPELINE_FIXTURE") != "1":
    raise RuntimeError("Pipeline fixtures require REMOTEWORKER_PIPELINE_FIXTURE=1")
import sys,time,json,threading
import torch
from comfy_execution.progress import get_progress_state
log_lock=threading.Lock()
def record(event):
 with log_lock:
  from comfy_execution.utils import get_executing_context
  context=get_executing_context()
  if event.startswith(('sample','preview','model')):assert get_progress_state().prompt_id==context.prompt_id
  with open(os.environ.get('REMOTEWORKER_PIPELINE_TEST_LOG','/tmp/pipeline-dev-events.jsonl'),'a') as f:f.write(json.dumps(dict(event=event,at=time.monotonic(),prompt=context.prompt_id))+'\n')
class PipelineSampler:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'tag':('STRING',),'delay':('FLOAT',{'default':0.5})},'optional':{'model':('MODEL',)}}
 RETURN_TYPES=('LATENT',);FUNCTION='run';CATEGORY='test'
 @classmethod
 def IS_CHANGED(cls,**kw):return float('nan')
 def run(self,tag,delay,model=None):
  service=next(m for n,m in sys.modules.copy().items() if n.endswith('.remoteworker.vae.service'))
  service.execute_decode=decode
  record('sample-start-'+tag);time.sleep(delay);record('sample-end-'+tag);return ({'samples':torch.zeros(1,4,2,2)},)
class PipelineSink:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'image':('IMAGE',),'tag':('STRING',)}}
 RETURN_TYPES=();FUNCTION='run';OUTPUT_NODE=True;CATEGORY='test'
 def run(self,image,tag):
  record('preview-'+tag);return {'ui':{'tag':[tag],'shape':[list(image.shape)]},'result':()}
def decode(api,**kw):
 record('decode-start')
 cancellation=next(m for n,m in sys.modules.copy().items() if n.endswith('.remoteworker.core.cancellation'))
 for _ in range(30):
  time.sleep(0.05);cancellation.check_cancelled()
 record('decode-end');return torch.ones(1,16,16,3)
NODE_CLASS_MAPPINGS={'PipelineSampler':PipelineSampler,'PipelineSink':PipelineSink}
NODE_DISPLAY_NAME_MAPPINGS={key:key for key in NODE_CLASS_MAPPINGS}

class FakeModel:pass
class PipelineModel:
 @classmethod
 def INPUT_TYPES(cls):return {'required':{'strength':('FLOAT',{'default':1.0})},'optional':{'token':('STRING',{'default':''})}}
 RETURN_TYPES=('MODEL',);FUNCTION='run';CATEGORY='test'
 def run(self,strength,token=""):record('model-load-'+str(strength));return (FakeModel(),)
NODE_CLASS_MAPPINGS['PipelineModel']=PipelineModel
NODE_DISPLAY_NAME_MAPPINGS['PipelineModel']='PipelineModel'
