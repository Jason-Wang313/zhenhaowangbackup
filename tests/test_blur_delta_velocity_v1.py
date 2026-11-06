import tempfile
import unittest
from pathlib import Path
import torch
from lowlight_restoration import blur_delta_velocity_v1 as core
from lowlight_restoration.lora import inject_lora
from smoke import blur_delta_state_v1 as state


def tiny_model():
    torch.manual_seed(12)
    model = torch.nn.Sequential(torch.nn.Linear(3, 3))
    inject_lora(model, ['0'], 2, 2.)
    return model


class DeltaTests(unittest.TestCase):
    def test_identity_preserves_all_uint8_levels(self):
        lq = torch.arange(256).float().view(1, 1, 16, 16)/127.5-1
        decoded = torch.randn_like(lq)*4
        torch.testing.assert_close(core.corrected_rgb(lq, decoded, decoded), lq, rtol=0, atol=0)
        self.assertTrue(torch.equal(core.uint8_output(core.corrected_rgb(lq, decoded, decoded)),
                                    torch.arange(256).byte().view_as(lq)))

    def test_difference_sign_and_nonzero_gradient_at_identity(self):
        z = torch.full((1, 1, 2, 2), .1)
        reference = torch.full_like(z, .2)
        velocity = reference.clone().requires_grad_()
        corrected = core.corrected_latent(z, velocity, reference)
        output = core.corrected_rgb(z, corrected*2, z*2)
        output.sum().backward()
        torch.testing.assert_close(velocity.grad, torch.full_like(z, -2))
        torch.testing.assert_close(core.corrected_latent(z, reference+.03, reference), z-.03)

    def test_reference_graph_rejected(self):
        x = torch.ones(1, 1, 2, 2)
        with self.assertRaises(ValueError):
            core.corrected_latent(x, x, x.clone().requires_grad_())
        with self.assertRaises(ValueError):
            core.corrected_rgb(x, x, x.clone().requires_grad_())

    def test_native_cfg_and_token_layout(self):
        z = torch.arange(12).float().reshape(1, 3, 2, 2)
        def pack(value):
            return value.flatten(2).transpose(1, 2), torch.zeros(1, 4, 4, dtype=torch.long)
        class Recorder(torch.nn.Module):
            def forward(self, **args):
                self.args = args
                out = args['x'].clone()
                out[0] = 0
                return out
        model = Recorder()
        prompt = {'cfg_context': torch.zeros(2, 1, 1), 'cfg_ids': torch.zeros(2, 1, 4)}
        result = core.native_velocity(model, z, prompt, pack)
        torch.testing.assert_close(result, z, rtol=0, atol=0)
        self.assertEqual(model.args['x'].shape, (2, 8, 3))
        self.assertTrue(torch.equal(model.args['x_ids'][:, 4:, 0], torch.full((2, 4), 10)))
        self.assertTrue(torch.equal(model.args['timesteps'], torch.ones(2)))

    def test_epoch_coverage_and_no_rank_duplicates(self):
        epochs = []
        for epoch in range(2):
            values = []
            for step in range(epoch*16, (epoch+1)*16):
                a, b = [core.sample_indices(step, rank) for rank in range(2)]
                self.assertFalse(set(a) & set(b))
                values.extend(sum(([x, y] for x, y in zip(a, b)), []))
            self.assertEqual(sorted(values), list(range(128)))
            epochs.append(values)
        self.assertNotEqual(*epochs)
        self.assertEqual(core.sample_indices(17, 1), core.sample_indices(17, 1))

    def test_batch_mean_not_double_divided(self):
        x = torch.arange(8).float()
        parameter = torch.tensor(.2, requires_grad=True)
        ((parameter*x-1).square().mean()).backward()
        expected = parameter.grad.clone()
        observed = 0.
        for rank in range(2):
            p = torch.tensor(.2, requires_grad=True)
            for i in range(rank, 8, 2):
                ((p*x[i]-1).square()/8).backward()
            observed = observed + p.grad
        torch.testing.assert_close(observed, expected)

    def test_save_restore_exact_next_adam_step_and_rng(self):
        model = tiny_model()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        x = torch.randn(8, 3)
        def update(m, o):
            o.zero_grad()
            m(x).square().mean().backward()
            o.step()
        update(model, optimizer)
        with tempfile.TemporaryDirectory() as folder:
            state.save(folder, model, optimizer, step=0, world=2, rank_rng=[state.rng_state('cpu')]*2,
                       receipt_sha='binding', history=[])
            # Step0 above intentionally has Adam step1: must be rejected.
            with self.assertRaises(ValueError):
                state.load(folder, 'binding', 2)
        model = tiny_model()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        with tempfile.TemporaryDirectory() as folder:
            state.save(folder, model, optimizer, step=0, world=2, rank_rng=[state.rng_state('cpu')]*2,
                       receipt_sha='binding', history=[])
            update(model, optimizer)
            state.save(folder, model, optimizer, step=1, world=2, rank_rng=[state.rng_state('cpu')]*2,
                       receipt_sha='binding', history=[{'step': 1}])
            saved = state.load(folder, 'binding', 2)
            expected_rng = torch.rand(3)
            update(model, optimizer)
            other = tiny_model()
            o = torch.optim.AdamW([p for p in other.parameters() if p.requires_grad], lr=.001)
            state.restore(other, o, saved, 0, 'cpu')
            torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
            update(other, o)
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            with self.assertRaises(ValueError): state.load(folder, 'binding', 4)
            with self.assertRaises(ValueError): state.load(folder, 'other-binding', 2)
            record = state.read(Path(folder)/'steps/step_000001.json')
            with (Path(folder)/record['file']).open('ab') as handle: handle.write(b'corrupt')
            with self.assertRaises(ValueError): state.load(folder, 'binding', 2)


if __name__ == '__main__':
    unittest.main()
