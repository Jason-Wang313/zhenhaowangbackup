import unittest
from tests import test_blur_delta_official_v1 as original_tests
from smoke.evaluate_foundir_blur_fullgrid_official_v1 import assigned,metric_summary


class ExposureOfficialTests(unittest.TestCase):
    def test_four_ranks_cover_exact150(self):
        rows=original_tests.OfficialTests().rows();groups=[assigned(rows,r) for r in range(4)]
        self.assertEqual([len(g) for g in groups],[38,38,37,37])
        self.assertEqual(len({i['row']['record_id'] for g in groups for i in g}),150)
        with self.assertRaises(ValueError):assigned(rows[:-1]+rows[:1],0)
        with self.assertRaises(ValueError):assigned(rows,4)

    def test_same_complete_final_candidate_must_meet_both_gates(self):
        fixture=original_tests.OfficialTests()
        self.assertTrue(metric_summary(fixture.records())['official_Blur_pass'])
        self.assertFalse(metric_summary(fixture.records(psnr=25.8))['official_Blur_pass'])
        self.assertFalse(metric_summary(fixture.records(lpips=.181))['official_Blur_pass'])
        with self.assertRaises(ValueError):metric_summary(fixture.records()[:-1])


if __name__=='__main__':unittest.main()
