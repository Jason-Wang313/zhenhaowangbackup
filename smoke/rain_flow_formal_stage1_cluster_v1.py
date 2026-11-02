"""Reuse four-node failure-aware barriers under the isolated Rain train-fit root."""
import argparse
import time
from smoke import rain_flow_formal_stage1_contract_v1 as p
from smoke.existing_flux_replay_cluster_v1 import publish,check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["publish","wait"])
    parser.add_argument("event",choices=["boot","prepared","ready","done","verified","failed"])
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--launch-id",required=True)
    parser.add_argument("--node-rank",type=int,choices=range(4),required=True)
    parser.add_argument("--timeout",type=float,default=3600)
    args=parser.parse_args();output=p.run_path(p.ROOT,args.run_id)
    if args.mode=="publish":
        publish(output,args.launch_id,args.node_rank,args.event)
    else:
        deadline=time.monotonic()+args.timeout
        while not check(output,args.launch_id,args.event):
            if time.monotonic()>=deadline:raise TimeoutError("waiting for four nodes: "+args.event)
            time.sleep(2)


if __name__=="__main__":main()
