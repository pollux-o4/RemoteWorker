"""Cancel lookahead while preserving the currently sampling development job."""
import json
import time
from e2e_preparation import request, workflow, CALLER, PEER, ROOT

for attempt in range(60):
    try:
        request(CALLER, '/system_stats')
        break
    except OSError:
        time.sleep(1)

active = workflow('A green ball rests on a table.', 'cancel-foreground')
active['sigma']['inputs']['steps'] = 32
current = request(CALLER, '/prompt', {'prompt': active})['prompt_id']
following = workflow('A yellow ball rolls across a table.', 'cancelled-lookahead')
pending = request(CALLER, '/prompt', {'prompt': following})['prompt_id']
print('queued', current, pending, flush=True)
end = time.monotonic() + 600
cancelled = False
while time.monotonic() < end:
    status = request(CALLER, '/remoteworker/preparation')
    event = status.get('active') or {}
    if event.get('state') == 'conditioning' and event.get('prompt_id') == pending and not cancelled:
        request(CALLER, '/queue', {'delete': [pending]})
        cancelled = True
        print('cancelled queued lookahead', flush=True)
    history = request(CALLER, '/history/' + current).get(current)
    if history:
        status = request(CALLER, '/remoteworker/preparation')
        result = {'foreground': history, 'cancelled_prompt': pending, 'requested': cancelled, 'preparation': status}
        (ROOT/'cancellation.json').write_text(json.dumps(result, indent=2))
        assert cancelled, 'did not observe a real lookahead window'
        assert history['status']['status_str'] == 'success', history['status']
        assert any(e['state'] == 'cancelled' for e in status['events']), status
        assert not request(CALLER, '/queue')['queue_pending']
        print('foreground success; speculative request cancelled', flush=True)
        break
    time.sleep(.2)
else:
    raise TimeoutError('development cancellation acceptance timed out')
