"""Offline checks for the one-attempt Qwen 7B configuration."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('qwen_attempt_test', HERE / '59-qwen7b-attempt.py')
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)


class QwenAttemptTests(unittest.TestCase):
    def test_rendered_settings_are_bounded_and_preserve_four_rank_shared_design(self):
        for arm in ('dedicated', 'hami', 'hami-mps'):
            for role in Q.series.ROLES:
                with patch.object(Q.series, 'tiny', Q.model_adapter()):
                    doc = yaml.safe_load(Q.series.render(role, arm, 'owned'))
                self.assertEqual(doc['spec']['model'], {
                    'name': 'Qwen/Qwen2.5-7B-Instruct-AWQ', 'uri': 'pvc://r2-models-pvc/models/llm'})
                pod = doc['spec']['template']
                main = pod['containers'][0]
                args = main['args']
                expected = {'--tensor-parallel-size': '2', '--pipeline-parallel-size': '1',
                            '--max-num-seqs': '2', '--max-model-len': '1024',
                            '--max-num-batched-tokens': '256', '--kv-cache-memory-bytes': '268435456',
                            '--safetensors-load-strategy': 'lazy', '--quantization': 'awq_marlin',
                            '--dtype': 'half'}
                for flag, value in expected.items():
                    self.assertEqual(args.count(flag), 1)
                    self.assertEqual(args[args.index(flag) + 1], value)
                self.assertIn('--no-enable-prefix-caching', args)
                self.assertIn('--enforce-eager', args)
                self.assertNotIn('--disable-frontend-multiprocessing', args)
                self.assertNotIn('--max-parallel-loading-workers', args)
                self.assertNotIn('--cpu-offload-gb', args)
                self.assertNotIn('--swap-space', args)
                for section in ('requests', 'limits'):
                    self.assertEqual(main['resources'][section]['nvidia.com/gpu'], '2')
                    self.assertEqual(main['resources'][section]['nvidia.com/gpumem'], '5632')
                env = {x['name']: x['value'] for x in main['env']}
                self.assertEqual(env['VLLM_WORKER_MULTIPROC_METHOD'], 'fork')
                self.assertEqual('CUDA_MPS_PIPE_DIRECTORY' in env, arm == 'hami-mps')
                self.assertIn('--kv-cache-memory-bytes 268435456', env['VLLM_ADDITIONAL_ARGS'])

    def test_mismatched_or_unquantized_model_is_refused(self):
        config = dict(model_type='qwen2', hidden_size=3584, num_hidden_layers=28,
                      num_attention_heads=28, num_key_value_heads=4,
                      quantization_config={'quant_method': 'awq', 'bits': 4})
        Q.validate_config(config)
        with self.assertRaisesRegex(RuntimeError, 'AWQ checkpoint'):
            Q.validate_config(dict(config, quantization_config={}))
        with self.assertRaisesRegex(RuntimeError, 'architecture'):
            Q.validate_config(dict(config, hidden_size=896))

    def test_chat_trace_carries_new_model_and_context_and_keeps_solo_subset(self):
        with patch.object(Q.series, 'tiny', Q.model_adapter()):
            trace = Q.series.make_trace(.5, 300, 64, 20260924)
            self.assertEqual(trace['params']['model'], Q.MODEL)
            self.assertEqual(trace['params']['max_model_len'], 1024)
            self.assertTrue(all(r['kind'] == 'chat' for r in trace['requests']))
            self.assertEqual(Q.series.selected_trace(trace, ['tp-a'])['requests'],
                             [r for r in trace['requests'] if r['service'] == 'tp-a'])


if __name__ == '__main__':
    unittest.main()
