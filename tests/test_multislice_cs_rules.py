import unittest
import importlib.util
import torch


class MicroRulesTests(unittest.TestCase):
    def test_matched_budget_is_independent_per_slice(self):
        spec = importlib.util.find_spec('seareg.multislice.microgain')
        self.assertIsNotNone(spec, 'microgain implementation is required')
        from seareg.multislice.microgain import allocate_matched03
        cap = torch.tensor([.5, .8, .3, .9, .6])
        q = torch.tensor([1., 0., .5, 0., 0.])
        norm = torch.tensor([1., 2., 3., 4., 0.])
        valid = torch.ones(5, dtype=torch.bool)
        codes = torch.tensor([0, 0, 1, 1, 1])
        out = allocate_matched03(cap, q, norm, valid, codes)
        for code in (0, 1):
            m = codes == code
            self.assertTrue(torch.allclose((out['matched03'][m] * norm[m]).sum(), (out['local03'][m] * norm[m]).sum(), atol=1e-8))
        self.assertGreater(out['matched03'][1].item(), 0)  # valid Q=0 shares its slice budget
        self.assertEqual(out['matched03'][4].item(), 0)  # zero displacement is ineligible
        self.assertTrue(((out['alpha'] >= 0) & (out['alpha'] <= .03)).all())
        self.assertTrue((out['matched03'] <= cap).all())

    def test_zero_quality_slice_has_no_allocation(self):
        from seareg.multislice.microgain import allocate_matched03
        x = torch.ones(3)
        out = allocate_matched03(x, torch.zeros(3), x, torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.long))
        self.assertTrue(torch.equal(out['matched03'], torch.zeros(3)))
        self.assertEqual(out['alpha'].item(), 0)

    def test_cardinal_coverage_and_cross_slice_rejection(self):
        from seareg.multislice.microgain import local_quality
        xy = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        h = torch.tensor([[0., 1.], [2., 0.], [3., 0.], [4., 0.], [5., 0.]])
        idx = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]])
        graph = torch.sparse_coo_tensor(idx, torch.ones(4), (5, 5)).coalesce()
        codes = torch.zeros(5, dtype=torch.long)
        result = local_quality(h, xy, graph, h + torch.tensor([3.5, -1.]), torch.ones(5, dtype=torch.bool), codes)
        self.assertTrue(result['valid'][0])
        self.assertTrue(torch.allclose(result['alignment'][0], torch.tensor(1.)))
        self.assertTrue((result['quality'][1:] == 0).all())
        changed = codes.clone(); changed[1] = 1
        with self.assertRaisesRegex(ValueError, 'cross-slice'):
            local_quality(h, xy, graph, h, torch.ones(5, dtype=torch.bool), changed)
        xy[1, 0] = 2
        result = local_quality(h, xy, graph, h + 1, torch.ones(5, dtype=torch.bool), codes)
        self.assertFalse(result['valid'][0])

    def test_frozen_local_quality_and_allocation_match_exactly(self):
        import importlib.util
        from pathlib import Path
        from seareg.multislice import microgain
        frozen = Path('/mnt/disk1/xmy/seareg/workspaces/b1_multislice_20261025_v1/runs/gate_cs_local_microgain_7118_r100n5_lab66_20261031_v1/models/cs_matched03/code/micro_rules.py')
        if not frozen.exists():
            self.skipTest('frozen experiment source is unavailable')
        spec = importlib.util.spec_from_file_location('frozen_micro_rules', frozen)
        oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
        torch.manual_seed(84)
        xy = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.]] * 2)
        idx = torch.tensor([[0]*4 + [5]*4, [1, 2, 3, 4, 6, 7, 8, 9]])
        graph = torch.sparse_coo_tensor(idx, torch.rand(8), (10, 10)).coalesce()
        h = torch.randn(10, 4); clean = torch.randn(10, 4)
        valid = torch.ones(10, dtype=torch.bool); codes = torch.tensor([0]*5 + [1]*5)
        actual = microgain.local_quality(h, xy, graph, clean, valid, codes)
        expected = oracle.local_quality(h, xy, graph, clean, valid, codes)
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)
        cap = torch.rand(10); norm = (clean-h).norm(dim=1)
        actual = microgain.allocate_matched03(cap, actual['quality'], norm, actual['valid'], codes)
        expected = oracle.allocate(cap, expected['quality'], norm, expected['valid'], codes)
        for key in actual:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def test_spatial_coverage_reports_only_geometric_support(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('seareg.multislice.geometry'))
        from seareg.multislice.geometry import spatial_coverage_report
        xy = torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.], [-1.,1.], [-1.,0.], [-1.,-1.], [0.,-1.], [1.,-1.]])
        idx = torch.tensor([[0]*8, list(range(1,9))])
        graph = torch.sparse_coo_tensor(idx, torch.ones(8), (9,9)).coalesce()
        report = spatial_coverage_report(xy, graph, torch.zeros(9, dtype=torch.long))
        self.assertEqual(report['total_spots'], 9)
        self.assertEqual(report['per_slice'][0]['eight_sector_covered'], 1)
        self.assertEqual(report['per_slice'][0]['cardinal_covered'], 1)
        codes = torch.zeros(9, dtype=torch.long); codes[1] = 1
        with self.assertRaisesRegex(ValueError, 'cross-slice'):
            spatial_coverage_report(xy, graph, codes)


    def test_cs_factor_outputs_match_frozen_source(self):
        import importlib.util
        from pathlib import Path
        from seareg.multislice.geometry import compute_selective_geometry
        from seareg.multislice.protection import compute_protection
        from seareg.multislice.coherence import compute_cs_gain
        base = Path('/mnt/disk1/xmy/seareg/workspaces/b1_multislice_20261025_v1/runs/gate_cs_local_microgain_7118_r100n5_lab66_20261031_v1/models/cs_matched03/code')
        if not base.exists(): self.skipTest('frozen experiment source unavailable')
        def load(name):
            spec = importlib.util.spec_from_file_location('frozen_' + name, base / (name + '.py'))
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
        oldgeo = load('selective_geometry'); oldpro = load('reliability_factors')
        oldgain = load('aligned_factors'); oldelig = load('eligibility_factors')
        torch.manual_seed(654)
        xy = torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.], [-1.,1.], [-1.,0.], [-1.,-1.], [0.,-1.], [1.,-1.]])
        idx = torch.tensor([[0]*8, list(range(1,9))]); s = torch.sparse_coo_tensor(idx, torch.rand(8), (9,9)).coalesce()
        h = torch.randn(9, 4); g = torch.rand(9); codes = torch.zeros(9, dtype=torch.long)
        geo = compute_selective_geometry(h, xy, s, g); reference = oldgeo.compute_selective_geometry(h, xy, s, g)
        for name in geo:
            self.assertTrue(torch.equal(geo[name], reference[name]), name)
        p = compute_protection(h, s, g, geo, codes); refp = oldpro.compute_protection(h, s, g, reference, codes, 'cross_boundary_risk_selective')
        for name in ('effective_P', 'risk', 'displacement_norm'):
            self.assertTrue(torch.equal(p[name], refp[name]), name)
        gain = compute_cs_gain(h, s, g, p['effective_P'], geo['valid_coverage'], codes)
        refg = oldgain.compute_gain(h, s, g, refp['effective_P'], reference['valid_coverage'], codes)
        refe = oldelig.compute_eligibility(h, s, g, refg['local_delta'], refp['risk'], refg['eligible'], codes)
        for name, ref in [('local_delta', refg['local_delta']), ('consensus', refe['consensus']), ('effective_delta', refe['coherence_gain']), ('displacement_norm', refg['displacement_norm'])]:
            self.assertTrue(torch.equal(gain[name], ref), name)
