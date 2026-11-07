import copy
import unittest
from smoke import blur_multiview_data_v1 as data
from smoke.official_vlm_router_v4_contract import tiles


class ViewTests(unittest.TestCase):
    def rows(self):
        return [{'role':'train', 'crop_xy':[17,23], 'row':{
            'record_id':f'train_{i}', 'split':'train','category':'01Blur',
            'width':'1920', 'height':'1080', 'gt_sha256':'unused'}} for i in range(16)]

    def test_four_distinct_deployment_views_and_no_old_key_collisions(self):
        rows = self.rows(); result = data.views(rows)
        self.assertEqual(len(result),64)
        self.assertEqual(len(set(data.key(i) for i in result)),64)
        old = {data.base.parent.key(i) for i in rows}
        self.assertFalse(old & {data.key(i) for i in result})
        for index, row in enumerate(rows):
            chosen = result[index*4:index*4+4]
            grid = {(t['x'],t['y']) for t in tiles(row['row'])}
            self.assertEqual(len({tuple(i['crop_xy']) for i in chosen}),4)
            self.assertTrue(all(tuple(i['crop_xy']) in grid for i in chosen))

    def test_deterministic_selection_independent_of_GT_and_old_crop(self):
        rows=self.rows(); expected=data.views(rows); altered=copy.deepcopy(rows)
        for row in altered:
            row['row']['gt_sha256']='other';row['crop_xy']=[0,0]
        actual=data.views(altered)
        self.assertEqual([i['crop_xy'] for i in expected],[i['crop_xy'] for i in actual])
        self.assertEqual(expected,data.views(rows))
        self.assertEqual(rows,self.rows())

    def test_nontraining_rows_rejected(self):
        rows=self.rows(); rows[0]['row']['split']='test'
        with self.assertRaises((ValueError,RuntimeError)): data.views(rows)


if __name__=='__main__': unittest.main()
