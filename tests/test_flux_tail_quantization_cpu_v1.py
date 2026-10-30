import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from lowlight_restoration.scalar_residual import quantized_scalar_residual
from smoke import replay_flux_tail_quantization_cpu_v1 as replay


class QuantizationReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_rational_all_65536_pairs_and_ties_even(self):
        values = torch.arange(256, dtype=torch.uint8)
        left = values[:, None].expand(256, 256)
        right = values[None, :].expand(256, 256)
        exact, ties = replay.rational_blend(left, right)
        expected = torch.tensor([[round((7*i+3*j)/10) for j in range(256)]
                                 for i in range(256)], dtype=torch.uint8)
        self.assertTrue(torch.equal(exact, expected))
        self.assertTrue(bool((exact[ties].int() % 2 == 0).all()))

    def test_synthetic_does_not_access_assets_or_cuda(self):
        with patch.object(replay, 'load_inputs', side_effect=AssertionError('no assets')):
            result = replay.synthetic_audit()
        self.assertEqual(result['uint8_pairs'], 65536)
        self.assertEqual(result['disagreements_off_exact_ties'], 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_difference_signed_no_uint8_overflow(self):
        left = torch.tensor([0, 255, 10], dtype=torch.uint8).reshape(1, 3, 1, 1)
        right = torch.tensor([255, 0, 10], dtype=torch.uint8).reshape_as(left)
        result = replay.difference(left, right)
        self.assertEqual(result['max_abs'], 255)
        self.assertEqual(result['signed_mean'], 0)
        self.assertEqual(result['different_elements'], 2)
        self.assertEqual(result['different_rgb_pixels'], 1)
        with self.assertRaises(RuntimeError):
            replay.difference(left.float()*float('nan'), right.float())

    @staticmethod
    def tensors():
        obs = torch.arange(3*512*512, dtype=torch.int32).remainder(256).to(torch.uint8).reshape(1, 3, 512, 512)
        normalized = obs.float()/127.5-1
        raw = (normalized*.6 + .017).clamp(-1, 1)
        raw_u8, raw_norm = quantized_scalar_residual(normalized, raw, 1.)
        proposal, _ = quantized_scalar_residual(normalized, raw_norm, .3)
        return dict(observation_normalized=normalized, raw_clamped=raw, raw_uint8=raw_u8,
                    proposal_uint8=proposal), dict(observation=obs, proposal=proposal.clone())

    def test_exact_replay_reports_zero_and_does_not_mutate(self):
        trace, cache = self.tensors()
        before = {k: v.clone() for k, v in trace.items()}
        result = replay.replay(trace, cache)
        self.assertEqual(result['first_quantization_cpu_vs_saved']['different_elements'], 0)
        self.assertEqual(result['variants']['cpu_full_replay']['vs_saved_gpu']['different_elements'], 0)
        for key in trace:
            self.assertTrue(torch.equal(before[key], trace[key]))

    def test_geometry_dtype_observation_and_nonfinite_guards(self):
        trace, cache = self.tensors()
        for name, value in [('raw_clamped', trace['raw_clamped'].double()),
                            ('raw_clamped', trace['raw_clamped'][:, :, :8]),
                            ('raw_clamped', trace['raw_clamped']*float('nan')),
                            ('observation_normalized', torch.zeros_like(trace['raw_clamped']))]:
            with self.assertRaises(RuntimeError):
                replay.replay(trace | {name: value}, cache)

    def test_source_guard_rejects_nontraining_or_wrong_index(self):
        identity = dict(index=0, role='train', row=dict(split='train', category='05Noise'))
        replay.validate_identity(identity)
        for update in [dict(index=1), dict(role='evaluation'), dict(row=dict(split='validation', category='05Noise'))]:
            with self.assertRaises(RuntimeError):
                replay.validate_identity(identity | update)

    def test_path_and_content_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError):
                replay.inside(root, '../escape')
            (root/'outside').symlink_to('/tmp')
            with self.assertRaises(ValueError):
                replay.inside(root, 'outside/escape')
            (root/'file').write_text('original')
            with self.assertRaises(RuntimeError):
                replay.bound(root, 'file', '0'*64)

    def test_report_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            filename = Path(d)/'report.json'
            replay.write_report(filename, dict(value=1))
            with self.assertRaises(FileExistsError):
                replay.write_report(filename, dict(value=2))
            self.assertEqual(json.loads(filename.read_text()), dict(value=1))

    def test_missing_remote_inputs_fail_without_claiming_a_report(self):
        with tempfile.TemporaryDirectory() as d, patch('sys.argv', ['replay', '--root', d]), \
                patch.object(replay, 'synthetic_audit', side_effect=AssertionError('must fail first')):
            with self.assertRaisesRegex(FileNotFoundError, 'not mounted here'):
                replay.main()
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_bound_loading_decodes_only_six_allowed_tensors(self):
        from safetensors import safe_open as original_open
        trace, cache = self.tensors()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, diag = root/replay.PARENT, root/replay.DIAGNOSTIC
            parent.mkdir(parents=True)
            diag.mkdir(parents=True)
            scalar = root/replay.SCALAR_SOURCE
            scalar.parent.mkdir(parents=True)
            scalar.write_bytes((replay.ROOT/replay.SCALAR_SOURCE).read_bytes())
            identity = dict(index=0, role='train', row=dict(split='train', category='05Noise'))
            source = {replay.SCALAR_SOURCE: replay.sha(scalar)}
            parent_receipt = dict(selected=[identity], config_sha256='c'*64, source_sha256=source)
            replay.write_report(parent/'preflight.json', parent_receipt)
            save_file({k: v[0] for k, v in cache.items()} | {'target': torch.ones(1)},
                      str(parent/'cache_000.safetensors'))
            cache_sha = replay.sha(parent/'cache_000.safetensors')
            replay.write_report(parent/'cache_000.json', dict(identity=identity,
                                config_sha256='c'*64, source_sha256=source,
                                file='cache_000.safetensors', sha256=cache_sha))
            replay.write_report(diag/'preflight.json', dict(identity=identity, parent_run=replay.PARENT,
                                parent_preflight_sha256=replay.sha(parent/'preflight.json'),
                                cache_metadata_sha256=replay.sha(parent/'cache_000.json'),
                                assets={replay.PARENT+'/cache_000.safetensors': cache_sha}, source_sha256=source))
            save_file(trace | {'unused': torch.ones(1)}, str(diag/'gpu0_repeat0.safetensors'))
            trace_sha = replay.sha(diag/'gpu0_repeat0.safetensors')
            replay.write_report(diag/'gpu0_repeat0.json', dict(file='gpu0_repeat0.safetensors', sha256=trace_sha,
                                stage_sha256={k: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                                              for k, v in trace.items()}))
            replay.write_report(diag/'result.json', dict(preflight_sha256=replay.sha(diag/'preflight.json'),
                                artifact_sha256={n: replay.sha(diag/n) for n in
                                                 ('gpu0_repeat0.safetensors', 'gpu0_repeat0.json')}))
            decoded = []

            class TrackedOpen:
                def __init__(self, *args, **kwargs):
                    self.handle = original_open(*args, **kwargs)

                def __enter__(self):
                    self.handle.__enter__()
                    return self

                def get_tensor(self, name):
                    decoded.append(name)
                    if name in ('target', 'unused'):
                        raise AssertionError('disallowed tensor decoded')
                    return self.handle.get_tensor(name)

                def __exit__(self, *args):
                    return self.handle.__exit__(*args)

            with patch.object(replay, 'RESULT_SHA', replay.sha(diag/'result.json')), \
                    patch.object(replay, 'TRACE_SHA', trace_sha), patch('safetensors.safe_open', TrackedOpen):
                loaded_trace, loaded_cache, evidence = replay.load_inputs(root)
                self.assertEqual(decoded, list(replay.TRACE_KEYS)+['observation', 'proposal'])
                self.assertEqual(evidence['cache_sha256'], cache_sha)
                self.assertTrue(torch.equal(loaded_trace['raw_clamped'], trace['raw_clamped']))
                self.assertTrue(torch.equal(loaded_cache['proposal'], cache['proposal']))


if __name__ == '__main__':
    unittest.main()
