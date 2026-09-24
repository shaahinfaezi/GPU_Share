"""Offline regressions for TP2+TP2 manifests, replay protocol and admission."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import yaml

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load('tiny_series_test', HERE / '58-tiny-series.py')
R = load('tiny_replay_test', HERE.parent / 'workload/replay.py')


class TinyTests(unittest.TestCase):
    def test_real_http_completion_replay_writes_successful_request_csv(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append((self.path, payload))
                if self.path != '/v1/completions' or 'prompt' not in payload:
                    self.send_error(400)
                    return
                if payload.get('stream'):
                    body = ('data: ' + json.dumps({'choices': [{'text': 'hello'}]}) + '\n\n' +
                            'data: ' + json.dumps({'choices': [], 'usage': {
                                'prompt_tokens': 4, 'completion_tokens': 1}}) + '\n\n' +
                            'data: [DONE]\n\n').encode()
                else:
                    body = json.dumps({'choices': [{'text': 'hello'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Content-Type', 'text/event-stream' if payload.get('stream') else 'application/json')
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows = [dict(service='tp-a', kind='completion', warmup=False, t=0,
                             prompt='A short prompt', max_tokens=1, ignore_eos=True)]
                trace = dict(requests=rows, params={'duration': 1}, trace_id=S.digest(rows))
                (root / 'trace.json').write_text(json.dumps(trace))
                argv = ['replay.py', '--trace', str(root / 'trace.json'), '--arm', 'dedicated',
                        '--out-dir', str(root), '--endpoint', 'tp-a=http://127.0.0.1:%d' % server.server_port,
                        '--model', 'tp-a=pythia', '--max-inflight', '1']
                with patch.object(R.sys, 'argv', argv):
                    R.main()
                summary = json.loads((root / 'replay.summary.json').read_text())
                self.assertEqual(summary['requests_ok'], 1)
                self.assertEqual(summary['aggregate']['output_tokens'], 1)
                self.assertTrue(S.replay_quality(root, ['tp-a'])['ok'])
                self.assertTrue(all(path == '/v1/completions' and 'messages' not in payload
                                    for path, payload in captured))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_both_tenants_are_tp2_same_model_quota_and_correct_mps_mounts(self):
        for arm in ('dedicated', 'hami', 'hami-mps'):
            for role in S.ROLES:
                doc = yaml.safe_load(S.render(role, arm, 'owned-token'))
                self.assertEqual(doc['metadata']['name'], 'r2-' + role)
                self.assertEqual(doc['metadata']['labels']['r2-probe'], 'owned-token')
                self.assertEqual(doc['spec']['model']['name'], 'EleutherAI/pythia-70m')
                self.assertEqual(doc['spec']['model']['uri'],
                                 'pvc://r2-models-pvc/models/pythia-70m-split-probe')
                pod = doc['spec']['template']
                self.assertEqual(pod['nodeSelector']['kubernetes.io/hostname'], 'mr-gholami')
                main = pod['containers'][0]
                args = main['args']
                for flag, value in (('--tensor-parallel-size', '2'), ('--pipeline-parallel-size', '1'),
                                    ('--max-model-len', '2048'), ('--max-num-seqs', '8')):
                    self.assertEqual(args.count(flag), 1)
                    self.assertEqual(args[args.index(flag) + 1], value)
                self.assertEqual(main['resources']['limits']['nvidia.com/gpu'], '2')
                self.assertEqual(main['resources']['limits']['nvidia.com/gpumem'], '4096')
                self.assertEqual(main['resources']['limits'].get('nvidia.com/gpucores'),
                                 None if arm == 'dedicated' else '50')
                env = {item['name']: item['value'] for item in main['env']}
                self.assertEqual(env['VLLM_WORKER_MULTIPROC_METHOD'], 'fork')
                self.assertEqual('CUDA_MPS_PIPE_DIRECTORY' in env, arm == 'hami-mps')
                if arm == 'hami-mps':
                    self.assertEqual(env['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'], '100')
                    self.assertEqual(pod['volumes'][0]['hostPath']['path'], '/tmp/nvidia-mps')

    def test_solo_requests_are_exact_subsets_of_shared_trace(self):
        trace = S.make_trace(2, 300, 128, 20260924)
        self.assertEqual(trace, S.make_trace(2, 300, 128, 20260924))
        for role in S.ROLES:
            subset = S.selected_trace(trace, [role])
            self.assertEqual(subset['requests'], [r for r in trace['requests'] if r['service'] == role])
            live = [r for r in subset['requests'] if not r['warmup']]
            self.assertTrue(all(r['kind'] == 'completion' and r['ignore_eos'] and r['max_tokens'] == 128 for r in live))
            self.assertTrue(all(0 <= r['t'] < 300 for r in live))
        self.assertNotEqual(trace['trace_id'], S.make_trace(3, 300, 128, 20260924)['trace_id'])
        self.assertEqual(S.selected_trace(trace, S.ROLES)['trace_id'], trace['trace_id'])

    def test_gate_uses_both_full_solo_measurements_without_lowering_reserve(self):
        solos = {r: {'peak_host_delta_mib': 3949} for r in S.ROLES}
        gate = S.admission(solos, 7505)
        self.assertFalse(gate['ok'])
        self.assertEqual(gate['required_mib'], 8922)
        self.assertTrue(S.admission({r: {'peak_host_delta_mib': 3000} for r in S.ROLES}, 7505)['ok'])
        with self.assertRaises(ValueError):
            S.admission({r: {'peak_host_delta_mib': float('nan')} for r in S.ROLES}, 7505)

    def test_completion_resolver_does_not_probe_chat(self):
        response = SimpleNamespace(status_code=200)
        with patch.object(R.SESSION, 'post', return_value=response) as post:
            url, status = R.resolve_chat('http://local', 'pythia', 30, False, 'completion')
            self.assertEqual((url, status), ('http://local/v1/completions', 200))
            self.assertIn('prompt', post.call_args.kwargs['json'])
            self.assertNotIn('messages', post.call_args.kwargs['json'])

    def test_chat_and_completion_streams_keep_usage_and_count_tokens_not_chunks(self):
        for kind in ('chat', 'completion'):
            chunks = [{'choices': [({'text': 'hello'} if kind == 'completion' else {'delta': {'content': 'hello'}})]},
                      {'choices': [({'text': ' world'} if kind == 'completion' else {'delta': {'content': ' world'}})]},
                      {'choices': [], 'usage': {'prompt_tokens': 12, 'completion_tokens': 7}}]
            response = MagicMock()
            response.__enter__.return_value = response
            response.status_code, response.ok = 200, True
            response.iter_lines.return_value = ['data: ' + json.dumps(x) for x in chunks] + ['data: [DONE]']
            with patch.object(R.SESSION, 'post', return_value=response) as post:
                record = R.do_chat('http://local', 'pythia', {'prompt': 'test', 'max_tokens': 8,
                                   'ignore_eos': True}, 30, False, 0, 0, kind)
            self.assertTrue(record['ok'])
            self.assertEqual(record['output_tokens'], 7)
            self.assertEqual(record['chunks'], 2)
            self.assertEqual(len(record['itl']), 1)
            self.assertEqual('prompt' in post.call_args.kwargs['json'], kind == 'completion')
            self.assertEqual('messages' in post.call_args.kwargs['json'], kind == 'chat')

    def test_intervals_preserve_failures_and_empty_latency_bins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = dict(params={'duration': 10}, requests=[
                dict(service='tp-a', warmup=False, t=1), dict(service='tp-a', warmup=False, t=6)])
            (root / 'trace.json').write_text(json.dumps(trace))
            with (root / 'replay.requests.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['service', 'warmup', 't_sent', 't_last', 'ok', 'output_tokens'])
                writer.writeheader()
                writer.writerow(dict(service='tp-a', warmup=False, t_sent=1, t_last=7, ok=True, output_tokens=10))
                writer.writerow(dict(service='tp-a', warmup=False, t_sent=6, t_last=8, ok=False, output_tokens=0))
            S.write_intervals(root, ['tp-a'])
            with (root / 'intervals.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]['scheduled_arrivals'], '1')
            self.assertEqual(rows[0]['latency_p95_s'], '')
            self.assertEqual(rows[1]['completed_failed'], '1')
            self.assertEqual(float(rows[1]['latency_p99_s']), 6)
            self.assertEqual(float(rows[1]['output_tokens_per_s']), 2)

    def test_cleanup_attempts_both_roles_but_never_deletes_unowned_controller(self):
        def fake(args, **kwargs):
            if args[0] == 'get':
                token = 'other' if args[2] == 'r2-tp-a' else 'ours'
                return json.dumps({'metadata': {'labels': {'r2-probe': token}}})
            return ''
        with patch.object(S.P, 'kube', side_effect=fake) as kube:
            with self.assertRaisesRegex(RuntimeError, 'unowned'):
                S.cleanup_roles(S.ROLES, 'ours')
            deleted = [c.args[0][2] for c in kube.call_args_list if c.args[0][0] == 'delete']
            self.assertEqual(deleted, ['r2-tp-b'])

    def test_memory_refusal_writes_failed_result_without_creating_tenants(self):
        args = SimpleNamespace(rate_per_tenant=2, seconds=300, hold_seconds=0)
        solos = {r: {'peak_host_delta_mib': 3949} for r in S.ROLES}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(S, 'get_pods', return_value=[]), \
             patch.object(S.H, 'wait_gpu_idle', return_value=True), \
             patch.object(S.P, 'stable_idle', return_value=(7505, [7505] * 5)), \
             patch.object(S.P, 'kube') as kube:
            with self.assertRaisesRegex(RuntimeError, 'Pair needs'):
                S.run_arm(Path(tmp), 'hami', 'hami', S.ROLES, S.make_trace(2, 300, 128, 1), args, solos)
            kube.assert_not_called()
            result = json.loads((Path(tmp) / 'hami/arm.json').read_text())
            self.assertFalse(result['ok'])
            self.assertEqual(json.loads((Path(tmp) / 'hami/admission.json').read_text())['required_mib'], 8922)

    def test_mps_clients_must_belong_to_both_workers_of_each_pod(self):
        pss = {role: {'processes': [{'pid': index * 10 + n, 'name': 'VLLM::Worker_TP'} for n in (1, 2)]}
               for index, role in enumerate(S.ROLES)}
        with tempfile.TemporaryDirectory() as tmp, patch.object(S.H, 'assert_mps_matches_arm',
             return_value=(True, {'mps_client_pids': [1, 2, 11]})):
            with self.assertRaisesRegex(RuntimeError, 'two GPU workers'):
                S.mps_check('hami-mps', pss, Path(tmp), 'test')


if __name__ == '__main__':
    unittest.main()
