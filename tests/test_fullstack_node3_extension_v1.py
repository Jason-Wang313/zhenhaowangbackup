import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from smoke import foundir_fullstack_executor_3x2_v2 as legacy
from smoke import foundir_fullstack_node3_extension_v1 as new

class Node3Tests(unittest.TestCase):
    def test_node3_only_scope(self):
        new.validate_extension_scope({'nodes':[3],'sources':{new.SELF:'digest'}})
        for nodes in ([0,1,2,3],[0,3],[3,3]):
            with self.assertRaises(ValueError):new.validate_extension_scope({'nodes':nodes,'sources':{new.SELF:'digest'}})
        with self.assertRaises(ValueError):new.validate_extension_scope({'nodes':[3],'sources':{}})

    def test_legacy_skips_new_node_job_before_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'job.json').write_text(json.dumps({'nodes':[3]}))
            with patch.object(legacy,'validate') as validate,patch.object(legacy.subprocess,'Popen') as popen:
                for node in (0,1,2):legacy.run_job(root,node)
                validate.assert_not_called();popen.assert_not_called()
            self.assertEqual([p.name for p in root.iterdir()],['job.json'])

    def test_import_does_not_mutate_live_worker_configuration(self):
        self.assertEqual(legacy.NODES,3);self.assertNotIn(3,legacy.NODE_MAP)
        self.assertEqual(new.NODE['ip'],'10.244.90.116')

if __name__=='__main__':unittest.main()
