"""Offline regressions for split admission; no Kubernetes or GPU calls."""
import unittest
import json
import csv
import tempfile
import importlib.util
from pathlib import Path
from split_checks import memory_budget, same_two_devices, remap_trace, replay_quality


class AdmissionTests(unittest.TestCase):
    def test_eager_profile_preserves_split_and_quota_across_arms(self):
        spec = importlib.util.spec_from_file_location(
            'split_renderer', Path(__file__).with_name('50-render-split.py'))
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        for role in ('tp', 'pp'):
            for arm in ('dedicated', 'hami', 'hami-mps'):
                original = renderer.render(role, arm, '100', 'original')
                eager = renderer.render(role, arm, '100', 'lowmem-eager')
                self.assertEqual(eager.replace(' --enforce-eager', ''), original)
                self.assertIn('--%s-parallel-size 2' % ('tensor' if role == 'tp' else 'pipeline'), eager)
                self.assertEqual(eager.count('nvidia.com/gpumem: "4096"'), 2)

    def test_peak_host_delta_prevents_double_counting_shared_rss(self):
        cal = {'solo': {r: {'vllm_rss_mib': 6600, 'peak_host_delta_mib': 2900}
                        for r in ('tp', 'pp')}}
        self.assertTrue(memory_budget(cal, 7609)['ok'])
        self.assertFalse(memory_budget(cal, 6500)['ok'])

    def test_renamed_trace_has_consistent_metadata_and_id(self):
        doc = {'requests': [{'service': 'llm2a'}, {'service': 'llm2b'}], 'trace_id': 'stale'}
        remap_trace(doc)
        self.assertEqual(doc['services'], ['pp', 'tp'])
        self.assertNotEqual(doc['trace_id'], 'stale')
        digest = doc['trace_id']
        self.assertEqual(remap_trace(doc)['trace_id'], digest)

    def test_replay_success_exit_does_not_hide_errors_missing_rows_or_delay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'trace.json').write_text(json.dumps({'requests': [
                {'service': 'tp', 'warmup': False}, {'service': 'pp', 'warmup': False}]}))
            rows = [dict(service=r, warmup=False, ok=True, t_sched=0,
                         t_sent=0.01, error='') for r in ('tp', 'pp')]

            def check(current):
                with (path / 'replay.requests.csv').open('w', newline='') as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader(); w.writerows(current)
                return replay_quality(path, ['tp', 'pp'])['ok']

            self.assertTrue(check(rows))
            self.assertFalse(check(rows[:1]))
            rows[1]['ok'] = False
            rows[1]['error'] = 'Too many open files'
            self.assertFalse(check(rows))
            rows[1]['ok'] = True
            rows[1]['t_sent'] = 2
            self.assertFalse(check(rows))

    def test_historical_solo_success_is_not_shared_admission(self):
        cal = {'solo': {'tp': {'vllm_rss_mib': 6575},
                        'pp': {'vllm_rss_mib': 6162}}}
        self.assertFalse(memory_budget(cal, 9000)['ok'])
        self.assertTrue(memory_budget(cal, 14000)['ok'])

    def test_unknown_rss_does_not_silently_use_single_rank_default(self):
        with self.assertRaises(ValueError):
            memory_budget({'solo': {'tp': {}, 'pp': {}}}, 16000)

    def test_one_tenant_on_both_cards_is_not_shared_placement(self):
        self.assertFalse(same_two_devices({'tp': [], 'pp': ['GPU-a', 'GPU-b']}))
        self.assertFalse(same_two_devices({'tp': ['GPU-a', 'GPU-b'],
                                           'pp': ['GPU-b', 'GPU-c']}))
        self.assertTrue(same_two_devices({'tp': ['GPU-a', 'GPU-b'],
                                          'pp': ['GPU-b', 'GPU-a']}))


if __name__ == '__main__':
    unittest.main()
