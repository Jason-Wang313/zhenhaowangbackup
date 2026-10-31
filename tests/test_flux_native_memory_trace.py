"""Allocation-summary accounting uses only standard Python, no CUDA."""
import unittest
from smoke.profile_foundir_flux_native_energy_memory_v1 import summarize_memory_snapshot


class TraceTests(unittest.TestCase):
    def test_live_stack_aggregation_excludes_freed_and_keeps_pending_separate(self):
        frame=[dict(filename='attention.py',line=12,name='native_vjp')]
        snapshot={'segments':[{'blocks':[
            dict(state='active_allocated',size=100,frames=frame),
            dict(state='active_allocated',size=200,frames=frame),
            dict(state='active_awaiting_free',size=40,frames=frame),
            dict(state='inactive',size=900,frames=frame),
        ]}],'device_traces':[[dict(action='alloc',size=100),dict(action='oom',size=80,device_free=50,frames=frame)]]}
        result=summarize_memory_snapshot(snapshot)
        self.assertEqual(result['live_bytes_by_state'],dict(active_allocated=300,active_awaiting_free=40))
        self.assertEqual(result['largest_live_stacks'][0]['blocks'],2)
        self.assertEqual(result['largest_live_stacks'][0]['bytes'],300)
        self.assertEqual(result['oom_events'][0]['size'],80)
        self.assertEqual(result['capture_scope'],'post_exception_or_completion_not_peak')

    def test_untraced_preexisting_allocations_are_retained(self):
        result=summarize_memory_snapshot({'segments':[{'blocks':[dict(state='active_allocated',size=7)]}]})
        self.assertEqual(result['live_bytes_by_state']['active_allocated'],7)
        self.assertEqual(result['largest_live_stacks'][0]['frames'],[])
        self.assertEqual(result['oom_events'],[])


if __name__=='__main__':unittest.main()
