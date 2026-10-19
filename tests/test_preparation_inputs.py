import unittest
from types import SimpleNamespace
from remoteworker.preparation.inputs import Resolver, UnsupportedInput
from remoteworker.preparation.handoff import Handoffs, signature

class InputPreparationTests(unittest.TestCase):
    def test_resolves_shared_upstream_once_and_refresh_is_not_a_skip(self):
        calls=[]
        class Primitive:
            FUNCTION='execute'
            def execute(self,value):calls.append(value);return (value,)
        g={'p':{'class_type':'PrimitiveFloat','inputs':{'value':5}},
           'clip':{'class_type':'RemoteMiniMaxH3ImageToVideo','inputs':{'width':['p',0],'height':['p',0],'cache_policy':'always_refresh'}}}
        values=Resolver(g,{'PrimitiveFloat':Primitive}).inputs('clip')
        self.assertEqual(values,{'width':5,'height':5,'cache_policy':'always_refresh'})
        self.assertEqual(calls,[5])
    def test_arbitrary_model_or_llm_upstream_is_not_executed(self):
        g={'x':{'class_type':'UNETLoader','inputs':{}},'clip':{'inputs':{'first_frame':['x',0]}}}
        with self.assertRaises(UnsupportedInput):Resolver(g,{}).inputs('clip')
    def test_cancelled_handoff_is_purged_and_refresh_consumed_exactly_once(self):
        h=Handoffs(max_bytes=100)
        h.put(('B','clip'),'sig',b'bytes');self.assertEqual(h.take(('B','clip'),'sig'),b'bytes')
        self.assertIsNone(h.take(('B','clip'),'sig'))
        h.put(('B','clip'),'sig',b'bytes');h.retain({'A'})
        self.assertEqual(h.bytes_used,0)
    def test_mismatched_input_discards_staged_result(self):
        h=Handoffs();h.put(('B','clip'),'old',b'bytes')
        self.assertIsNone(h.take(('B','clip'),'new'));self.assertEqual(h.bytes_used,0)
    def test_request_signature_preserves_refresh_and_media_identity(self):
        self.assertNotEqual(signature({'cache_policy':'always_refresh','width':32}),signature({'cache_policy':'always_refresh','width':64}))

    def test_managed_foreground_executes_without_staged_result(self):
        from remoteworker.preparation.integration import managed
        class Example:
            @managed
            def run(self,prompt,server_url='http://peer'):
                return prompt
        self.assertEqual(Example().run('B'),'B')

    def test_ancestor_change_or_priority_change_cancels_speculation(self):
        from remoteworker.preparation.coordinator import Coordinator
        g={'p':{'class_type':'PrimitiveFloat','inputs':{'value':5}}}
        pending=[(1,'B',g)]
        server=SimpleNamespace(prompt_queue=SimpleNamespace(get_current_queue=lambda: ([(0,'A',{})],pending)))
        c=Coordinator(server,{},None,sampling=lambda *_:True)
        self.assertFalse(c.preparation_cancelled('A','B',{'p':{'class_type':'PrimitiveFloat','inputs':{'value':5}}}))
        g['p']['inputs']['value']=6
        self.assertTrue(c.preparation_cancelled('A','B',{'p':{'class_type':'PrimitiveFloat','inputs':{'value':5}}}))
        pending.insert(0,(-1,'C',{}))
        self.assertTrue(c.preparation_cancelled('A','B',g))

    def test_foreground_vae_waits_for_admitted_next_clip(self):
        from remoteworker.preparation.coordinator import Coordinator
        server=SimpleNamespace(prompt_queue=SimpleNamespace(get_current_queue=lambda: ([(0,'A',{})],[(1,'B',{})])))
        c=Coordinator(server,{},None,sampling=lambda *_:False)
        self.assertFalse(c.preparation_cancelled('A','B',{}))

    def test_memory_pressure_clears_disposable_cache_not_handoff(self):
        from unittest.mock import patch
        from remoteworker.preparation.memory import PressureRelief
        h=Handoffs();h.put(('B','clip'),'sig',b'bytes')
        calls=[];relief=PressureRelief(lambda:calls.append('clear'))
        with patch('remoteworker.preparation.memory.snapshot',return_value={'available':1}),patch('remoteworker.preparation.memory.time.monotonic',return_value=100):
            relief.tick();relief.tick()
        self.assertEqual(calls,['clear']);self.assertEqual(h.take(('B','clip'),'sig'),b'bytes')

    def test_image_size_does_not_access_shared_native_hidden_state(self):
        class Loader:
            FUNCTION='run'
            def run(self):return (SimpleNamespace(shape=(1,64,96,3)),)
        g={'image':{'class_type':'LoadImage','inputs':{}},'size':{'class_type':'GetImageSize','inputs':{'image':['image',0]}},'clip':{'inputs':{'width':['size',0],'height':['size',1]}}}
        self.assertEqual(Resolver(g,{'LoadImage':Loader,'GetImageSize':object}).inputs('clip'),{'width':96,'height':64})

    def test_resolves_vae_selection_without_evaluating_sampler(self):
        g={'decode':{'class_type':'RemoteVAEDecode','inputs':{'samples':['sample',0],'vae_name_override':['name',0]}},
           'sample':{'class_type':'SamplerCustomAdvanced','inputs':{}},
           'name':{'class_type':'PrimitiveString','inputs':{'value':'chosen.vae'}}}
        class Primitive:
            FUNCTION='run'
            def run(self,value):return (value,)
        self.assertEqual(Resolver(g,{'PrimitiveString':Primitive}).inputs('decode',exclude=('samples',)),{'vae_name_override':'chosen.vae'})

    def test_genrel_draft_input_is_cpu_safe_and_preserves_text(self):
        class RequestInput:
            FUNCTION = 'run'
            def run(self, name, text): return (text,)
        graph = {'draft': {'class_type': 'RequestConditionInput', 'inputs': {'name': 'prompt', 'text': '최종 Draft\nline 2'}},
                 'clip': {'inputs': {'prompt': ['draft', 0]}}}
        self.assertEqual(Resolver(graph, {'RequestConditionInput': RequestInput}).inputs('clip'), {'prompt': '최종 Draft\nline 2'})
