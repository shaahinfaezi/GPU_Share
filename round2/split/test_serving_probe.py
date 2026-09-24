"""Offline serving-profile and probe regressions; never contacts Kubernetes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


renderer = load('serving_renderer', '50-render-split.py')
probe = load('serving_probe', '56-serving-probe.py')


class ServingTests(unittest.TestCase):
    def test_candidate_is_explicitly_two_workers_in_every_arm(self):
        for role in ('tp', 'pp'):
            for arm in ('dedicated', 'hami', 'hami-mps'):
                doc = yaml.safe_load(renderer.render(role, arm, '100', 'lowmem-serve'))
                pod = doc['spec']['template']
                main = pod['containers'][0]
                args = main['args']
                def val(flag):
                    self.assertEqual(args.count(flag), 1)
                    return args[args.index(flag) + 1]
                tp, pp = int(val('--tensor-parallel-size')), int(val('--pipeline-parallel-size'))
                self.assertEqual(tp * pp, 2)
                self.assertEqual((tp, pp), (2, 1) if role == 'tp' else (1, 2))
                self.assertEqual(val('--max-model-len'), '8192')
                self.assertEqual(val('--model'), '/mnt/models')
                self.assertEqual(val('--max-num-batched-tokens'), '1024')
                self.assertIn('--enable-chunked-prefill', args)
                self.assertIn('--enforce-eager', args)
                self.assertNotIn('--disable-frontend-multiprocessing', args)
                self.assertIn('vllm.entrypoints.openai.api_server', main['command'])
                self.assertEqual(main['resources']['limits']['memory'], '6Gi')
                for section in ('limits', 'requests'):
                    gpu = main['resources'][section]
                    self.assertEqual(gpu['nvidia.com/gpu'], '2')
                    self.assertEqual(gpu['nvidia.com/gpumem'], '4096')
                    self.assertEqual(gpu.get('nvidia.com/gpucores'), None if arm == 'dedicated' else '50')
                env = {x['name']: x['value'] for x in main['env']}
                self.assertEqual('CUDA_MPS_PIPE_DIRECTORY' in env, arm == 'hami-mps')
                self.assertNotIn('VLLM_ENABLE_V1_MULTIPROCESSING', env)
                if arm == 'hami-mps':
                    self.assertEqual(pod['volumes'][0]['hostPath']['path'], '/tmp/nvidia-mps')

    def test_probe_traffic_is_identical_bounded_and_labelled_synthetic(self):
        a, b = probe.make_trace('tp'), probe.make_trace('tp')
        self.assertEqual(a, b)
        self.assertEqual(a['params']['source'], 'synthetic')
        live = [r for r in a['requests'] if not r['warmup']]
        self.assertGreater(len(live), 0)
        self.assertTrue(all(0 <= r['t'] < 60 and r['max_tokens'] == 64 for r in live))
        self.assertNotEqual(a['trace_id'], probe.make_trace('pp')['trace_id'])

    def test_pss_matches_uid_in_both_cgroup_formats_and_excludes_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for pid, group in ((10, 'podabc-def'), (11, 'podabc_def'), (12, 'podother')):
                p = root / str(pid)
                p.mkdir()
                (p / 'cgroup').write_text('0::/kubepods/' + group)
                (p / 'comm').write_text('worker')
                (p / 'smaps_rollup').write_text('Pss: 2048 kB\nRss: 4096 kB\n')
            result = probe.pod_pss('abc-def', root)
            self.assertEqual(result['pss_mib'], 4)
            self.assertEqual(len(result['processes']), 2)
            self.assertTrue(result['complete'])
            self.assertFalse(probe.pod_pss('absent', root)['complete'])

    def test_memory_guard_fails_closed(self):
        monitor = probe.Monitor(Path('.'))
        with patch.object(probe.H, 'host_mem_mib', side_effect=[{'available': 1600}, {'available': 1500}]):
            monitor.sample()
            self.assertFalse(monitor.abort.is_set())
            monitor.sample()
            self.assertTrue(monitor.abort.is_set())
            self.assertEqual(monitor.minimum, 1500)
        missing = probe.Monitor(Path('.'))
        with patch.object(probe.H, 'host_mem_mib', return_value={}):
            missing.sample()
        self.assertTrue(missing.abort.is_set())

    def test_cleanup_refuses_unowned_object(self):
        other = {'metadata': {'labels': {'r2-probe': 'someone-else'}}}
        with patch.object(probe, 'kube', return_value=json.dumps(other)) as kube:
            with self.assertRaisesRegex(RuntimeError, 'not owned'):
                probe.cleanup('r2-tp', 'mine', 'tp')
            self.assertEqual(kube.call_count, 1)

    def test_failure_saves_evidence_and_cleans_only_current_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(probe, 'stable_idle', return_value=(8000, [8000] * 5)), \
                 patch.object(probe.Monitor, 'run'), \
                 patch.object(probe, 'kube', side_effect=RuntimeError('create failed')), \
                 patch.object(probe, 'cleanup') as clean, \
                 patch.object(probe.H, 'pods_for', return_value=[]), \
                 patch.object(probe.H, 'kubectl', return_value=(0, '')):
                with self.assertRaisesRegex(RuntimeError, 'create failed'):
                    probe.run_profile(renderer, 'tp', 'lowmem-serve', Path(tmp), probe.make_trace('tp'))
                self.assertEqual(clean.call_args.args[0], 'r2-tp')
                self.assertEqual(clean.call_args.args[2], 'tp')
                result = json.loads((Path(tmp) / 'lowmem-serve' / 'probe.json').read_text())
                self.assertFalse(result['ok'])
                self.assertFalse(result['calibration'])
                self.assertFalse((Path(tmp) / 'calibration.json').exists())


if __name__ == '__main__':
    unittest.main()
