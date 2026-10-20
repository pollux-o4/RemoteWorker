"""Real CPU mux: publication, retry and cleanup use actual MP4/WAV files."""
import os
from array import array
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from remoteworker.vae import shared_store as store
from remoteworker.vae import mux
from web_harness import run_web


class SharedMuxTests(unittest.TestCase):
    def setUp(self):
        root = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, REMOTEWORKER_SHARED_VAE_DIR=root))
        self.sources = []
        for identity, filename, source in [('a'*32, 'video.mp4', 'color=c=blue:s=64x64:r=24:d=0.125'),
                                           ('b'*32, 'audio.wav', 'sine=frequency=440:sample_rate=32000:duration=0.125')]:
            path = store.request_path('outputs', identity, filename)
            path.parent.mkdir(parents=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', source, '-y', str(path)], check=True)
            store.write_json(path.with_name('receipt.json'), dict(request_id=identity, path=str(path),
                bytes=path.stat().st_size, options={'kind': 'audio'} if filename.endswith('wav') else {}))
            self.sources.append(path)

    def test_mux_publishes_av_then_consumes_sources_and_replays(self):
        def packets(path):
            return subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_packets','-show_data_hash','sha256','-show_entries','packet=data_hash',
                '-of','csv=p=0',str(path)])
        original = packets(self.sources[0])
        result = mux.combine(*map(str, self.sources))
        info = mux.probe(Path(result['path']))
        self.assertEqual({s['codec_type'] for s in info['streams']}, {'video', 'audio'})
        waveform = array('f', subprocess.check_output(['ffmpeg','-v','error','-i',result['path'],
            '-map','0:a:0','-ac','1','-f','f32le','pipe:1']))
        self.assertGreater(max(map(abs, waveform)), 0.01)
        self.assertTrue(all(not path.exists() for path in self.sources))
        stamp = Path(result['path']).stat().st_mtime_ns
        self.assertEqual(mux.combine(*map(str, self.sources)), result)
        self.assertEqual(Path(result['path']).stat().st_mtime_ns, stamp)
        # Upstream cached decode receipts remain replayable after consumption.
        self.assertIsNotNone(store.read_receipt('a'*32))
        self.assertIsNotNone(store.read_receipt('b'*32))
        self.assertEqual(packets(result['path']), original)

    def test_mux_failure_preserves_sources_and_removes_partial_file(self):
        with patch.object(mux, 'encode', side_effect=RuntimeError('mux failed')):
            with self.assertRaisesRegex(RuntimeError, 'mux failed'):
                mux.combine(*map(str, self.sources))
        self.assertTrue(all(path.is_file() for path in self.sources))
        self.assertEqual(len(list(store.shared_root().glob('outputs/*/*.mp4'))), 1)

    def test_rejects_foreign_path_before_encoding(self):
        with self.assertRaises(ValueError):
            mux.combine('/tmp/foreign.mp4', str(self.sources[1]))
        self.assertTrue(all(path.exists() for path in self.sources))

    def test_keep_sources_and_video_only_are_explicit_supported_modes(self):
        result = mux.combine(str(self.sources[0]), cleanup=False)
        self.assertFalse(result['audio'])
        self.assertTrue(all(path.is_file() for path in self.sources))

    def test_cancel_before_mux_preserves_both_sources(self):
        def cancel():
            raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            mux.combine(*map(str, self.sources), interrupt=cancel)
        self.assertTrue(all(path.is_file() for path in self.sources))

    def test_cancel_terminates_the_active_encoder_process(self):
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('cancelled')
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            mux.run([sys.executable, '-c', 'import time; time.sleep(10)'], cancel)
        self.assertLess(time.monotonic() - started, 2)

    def test_migration_preserves_save_video_branch_and_uses_file_outputs(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {convertSharedVideo} from './custom/web/shared_combine.js';
            const linked=[];
            const source=(type)=>({type,widgets:[{name:'output_mode',value:'shared'},{name:'fps',value:24}],
                connect:(out,target,slot)=>linked.push([type,out,target.type,slot])});
            const video=source('RemoteVAEDecode'), audio=source('RemoteVAEDecodeAudio');
            const save={type:'SaveVideo'};
            const graph={links:{1:{origin_id:1},2:{origin_id:2},3:{target_id:4,target_slot:0}},
                getNodeById:id=>({1:video,2:audio,4:save})[id],
                add:node=>node.graph=graph,remove:node=>node.removed=true};
            globalThis.LiteGraph={createNode:type=>({type,connect:(out,target,slot)=>linked.push([type,out,target.type,slot])})};
            const old={type:'CreateVideo',graph,pos:[10,20],inputs:[{name:'images',link:1},{name:'audio',link:2}],
                outputs:[{links:[3]}],widgets:[{name:'fps',value:30}]};
            const node=convertSharedVideo(old);
            assert.equal(old.removed,true);
            assert.deepEqual(node.pos,[10,20]);
            assert.deepEqual(linked,[['RemoteVAEDecode',1,'RemoteVAECombineMedia',0],
                ['RemoteVAEDecodeAudio',1,'RemoteVAECombineMedia',1],['RemoteVAECombineMedia',0,'SaveVideo',0]]);
            assert.equal(video.widgets[1].value,30);
        ''')
