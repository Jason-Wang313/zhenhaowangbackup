import copy
import unittest
from smoke import blur_fullgrid_data_v1 as data
from smoke.official_vlm_router_v4_contract import tiles


class ViewTests(unittest.TestCase):
    def rows(self):
        return [{'role':'train', 'crop_xy':[17,23], 'row':{
            'record_id':f'train_{i}', 'split':'train','category':'01Blur',
            'width':'1920', 'height':'1080', 'gt_sha256':'unused'}} for i in range(16)]

    def test_all15_distinct_deployment_views_and_no_old_key_collisions(self):
        rows = self.rows(); result = data.views(rows)
        self.assertEqual(len(result),240)
        self.assertEqual(len(set(data.key(i) for i in result)),240)
        old = {data.base.parent.key(i) for i in rows}
        self.assertFalse(old & {data.key(i) for i in result})
        for index, row in enumerate(rows):
            chosen = result[index*15:index*15+15]
            grid = {(t['x'],t['y']) for t in tiles(row['row'])}
            self.assertEqual(len({tuple(i['crop_xy']) for i in chosen}),15)
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


    def test_reuse_by_source_and_coordinates_not_old_view_number(self):
        from smoke import blur_multiview_data_v1 as old
        rows=self.rows(); new=data.views(rows); prior=old.views(rows)
        reused=data.reuse_map(prior,new)
        self.assertEqual(len(reused),64)
        for item in new:
            if data.key(item) in reused:
                p=reused[data.key(item)]
                self.assertEqual(item['row'],p['row']);self.assertEqual(item['crop_xy'],p['crop_xy'])
        prior[0]['row']=dict(prior[0]['row'],gt_sha256='changed')
        with self.assertRaises((ValueError,RuntimeError)):data.reuse_map(prior,new)


if __name__=='__main__': unittest.main()
