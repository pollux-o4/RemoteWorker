"""Cancellation and foreground priority with linked, always-refresh I2V."""
import json
import time
from e2e_i2v_preparation import graph,request,wait_jobs,create_fixtures,CALLER,PEER,ROOT


def empty_queue():
    end=time.monotonic()+300
    while time.monotonic()<end:
        if not any(request(CALLER,'/queue').values()):return
        time.sleep(1)
    raise TimeoutError('caller busy')


def pair(label,steps):
    a=request(CALLER,'/prompt',{'prompt':graph(label,steps)})['prompt_id']
    b=request(CALLER,'/prompt',{'prompt':graph(label+'-next')})['prompt_id']
    return a,b


if __name__=='__main__':
    create_fixtures()
    empty_queue()
    a,b=pair('cancel-pending',64)
    until=time.monotonic()+300
    while time.monotonic()<until:
        events=request(CALLER,'/remoteworker/preparation')['events']
        if any(e['state']=='conditioning' and e.get('prompt_id')==b for e in events):
            request(CALLER,'/queue',{'delete':[b]});break
        time.sleep(.1)
    else:raise AssertionError('prefetch did not start')
    d=wait_jobs([a],'i2v-pending-cancel.json')
    assert d['jobs'][a]['status']['status_str']=='success'
    assert any(e['state']=='cancelled' and e.get('prompt_id')==b for e in d['preparation']['events'])
    assert d['preparation']['handoff_bytes']==0
    assert not request(CALLER,'/history/'+b)
    print('PASS pending cancellation: only B preparation cancelled; A video success; handoff bytes zero',flush=True)
    empty_queue()
    a,b=pair('foreground-priority',16)
    d=wait_jobs([a,b],'i2v-foreground-priority.json')
    assert all(j['status']['status_str']=='success' for j in d['jobs'].values())
    assert any(e['state']=='conditioning' and e.get('prompt_id')==b for e in d['preparation']['events'])
    assert any(e['state']=='cancelled' and e.get('prompt_id')==b for e in d['preparation']['events'])
    assert not any(e['state']=='conditioning_consumed' and e.get('prompt_id')==b for e in d['preparation']['events'])
    assert d['preparation']['handoff_bytes']==0
    print('PASS foreground priority: incomplete B prefetch cancelled for A decode; B runs normally; both videos success',flush=True)
