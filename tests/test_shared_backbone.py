"""Shared single/multislice operators with the correction replaced by identity.

This isolates the common two-layer backbone. Active adaptive_depth_gate corrections
are covered separately by their rule/model tests, not disabled in production.
"""
import unittest
from unittest.mock import patch

import torch

from seareg.loss import execution_plan, prepare_ego_l2_coefficients
from seareg.model import LightGCN, batch_edge_weights, compute_loss
from seareg.multislice.grouped_loss import bpr_only, group_regularization
from seareg.multislice.model import AdaptiveDepthGate


def fixture():
    torch.manual_seed(739)
    single = LightGCN(4, 5, 0, 4)
    multi = AdaptiveDepthGate(
        4, 5, 4, torch.tensor([2., 3., 1., 2.]),
        torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
        torch.tensor([0, 0, 1, 1]),
    )
    with torch.no_grad():
        single.embedding_spot.mul_(10.)
        single.embedding_peak.mul_(10.)
        multi.embedding_spot.copy_(single.embedding_spot)
        multi.embedding_peak.copy_(single.embedding_peak)
    rows = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3])
    cols = torch.tensor([0, 2, 0, 1, 4, 3, 1, 3]) + 4
    indices = torch.stack((torch.cat((rows, cols)), torch.cat((cols, rows))))
    values = torch.tensor([.4, .6, .2, .3, .5, .7, .8, .2] * 2)
    graph = torch.sparse_coo_tensor(indices, values, (9, 9)).coalesce()
    spatial = torch.sparse_coo_tensor(
        torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        torch.tensor([.5, .5, .75, .75]), (4, 4)).coalesce()
    return single, multi, graph, spatial


class SharedBackboneTests(unittest.TestCase):
    def test_seeded_initialization_and_trainable_parameters_are_shared(self):
        torch.manual_seed(739)
        single = LightGCN(4, 5, 0, 4)
        torch.manual_seed(739)
        multi = AdaptiveDepthGate(
            4, 5, 4, torch.tensor([2., 3., 1., 2.]),
            torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
            torch.tensor([0, 0, 1, 1]))
        expected = dict(single.named_parameters())
        actual = dict(multi.named_parameters())
        self.assertEqual(set(actual), {"embedding_spot", "embedding_peak"})
        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_identity_gate_preserves_shared_two_layer_forward_and_gradients(self):
        for use_spatial in (False, True):
            with self.subTest(use_spatial=use_spatial):
                single, multi, graph, spatial = fixture()
                self.assertIsInstance(multi, LightGCN)
                self.assertEqual(
                    [(n, tuple(p.shape)) for n, p in single.named_parameters()],
                    [(n, tuple(p.shape)) for n, p in multi.named_parameters()])
                with patch.object(multi, "_correct_first_layer_peak_message",
                                  side_effect=lambda message, _: message) as correction:
                    left = single.propagate(graph, None, spatial, use_spatial, .37, 0.)
                    right = multi.propagate(graph, None, spatial, use_spatial, .37, 0.)
                    self.assertEqual(correction.call_count, int(use_spatial))
                    self.assertIsNone(left[2])
                    self.assertIsNone(right[2])
                    for actual, expected in zip(right[:2], left[:2]):
                        self.assertTrue(torch.equal(actual, expected))
                    # A nonzero objective involving both exported node types.
                    left_loss = left[0].square().sum() + .37 * left[1].sin().sum()
                    right_loss = right[0].square().sum() + .37 * right[1].sin().sum()
                    left_loss.backward()
                    right_loss.backward()
                for (name, expected), (_, actual) in zip(single.named_parameters(), multi.named_parameters()):
                    self.assertGreater(float(expected.grad.abs().max()), 0., name)
                    self.assertTrue(torch.equal(actual.grad, expected.grad), name)

    def test_shared_bpr_ego_l2_clip_and_adam(self):
        single, multi, graph, spatial = fixture()
        users = torch.tensor([0, 0, 2, 2, 3, 3])
        positive = torch.tensor([0, 0, 3, 3, 1, 1])
        negative = torch.tensor([2, 4, 0, 4, 0, 2])
        weights = batch_edge_weights(users, single.num_spot, dtype=single.embedding_spot.dtype)
        coefficients = prepare_ego_l2_coefficients(
            users, positive, negative, weights, n_spots=4, n_peaks=5, assume_uniform=True)
        plan = execution_plan(
            len(users), 4, 4, backend="torch", workspace_mb=1.,
            group_size=2, device="cpu", dtype=torch.float32)
        left_optimizer = torch.optim.Adam(single.parameters(), lr=.001, weight_decay=0.)
        right_optimizer = torch.optim.Adam(multi.parameters(), lr=.001, weight_decay=0.)
        with patch.object(multi, "_correct_first_layer_peak_message", side_effect=lambda message, _: message):
            left, parts = compute_loss(
                single, users, positive, negative, graph, spatial=spatial,
                r_spatial=.37, decay=.001, group_size=2, loss_backend="torch",
                weighted_ego_l2=True, ego_coefficients=coefficients, loss_plan=plan)
            bpr = bpr_only(multi, users, positive, negative, graph, spatial, True, .37, plan, 2)
            reg, diagnostics = group_regularization(multi, [coefficients])
            self.assertTrue(torch.equal(bpr, parts["bpr"]))
            self.assertTrue(torch.equal(reg, parts["reg"]))
            self.assertTrue(torch.equal(diagnostics[0], reg.detach()))
            right = bpr + .001 * reg
            self.assertTrue(torch.equal(right, left))
            left.backward()
            right.backward()
        for p, q in zip(single.parameters(), multi.parameters()):
            self.assertTrue(torch.equal(p.grad, q.grad))
        left_norm = torch.nn.utils.clip_grad_norm_(single.parameters(), .5)
        right_norm = torch.nn.utils.clip_grad_norm_(multi.parameters(), .5)
        self.assertGreater(float(left_norm), .5)  # Exercise an actual clipping operation.
        self.assertTrue(torch.equal(left_norm, right_norm))
        for p, q in zip(single.parameters(), multi.parameters()):
            self.assertTrue(torch.equal(p.grad, q.grad))
        left_optimizer.step()
        right_optimizer.step()
        for p, q in zip(single.parameters(), multi.parameters()):
            self.assertTrue(torch.equal(p, q))
            a, b = left_optimizer.state[p], right_optimizer.state[q]
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]), key)


if __name__ == "__main__":
    unittest.main()
