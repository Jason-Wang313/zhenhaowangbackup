"""Shared-filesystem launch barriers, independent of CUDA. Fresh launch ID per attempt."""
import argparse
import os
from pathlib import Path
import socket
import time

from smoke import existing_flux_replay_contract_v1 as c


def directory(output, launch):
    c.run_path(output, launch)  # Validate identifier, not a runtime output path.
    return output / "launches" / launch


def publish(output, launch, node, event):
    if node not in range(4):
        raise ValueError("invalid node rank")
    if event in ("prepared", "verified") and node != 0:
        raise ValueError("only node0 may publish common completion")
    folder = directory(output, launch)
    folder.mkdir(parents=True, exist_ok=True)
    value = {"launch": launch, "node": node, "event": event}
    if event == "boot":
        # Never reuse an attempt ID, even when the run itself is being resumed.
        with (folder / f"claim_node{node}").open("x") as f:
            f.write(f"{socket.gethostname()} {os.getpid()}\n")
        value["hostname"] = socket.gethostname()
        value["master_addr"] = os.environ["REPLAY_MASTER_ADDR"]
        value["master_port"] = os.environ["REPLAY_MASTER_PORT"]
    elif event in ("prepared", "ready", "done", "verified"):
        value["receipt_sha256"] = c.sha(output / "receipt.json")
        if event in ("ready", "done"):
            value["runtime_sha256"] = c.sha(output / f"runtime_node{node}.json")
        if event == "verified":
            value["result_sha256"] = c.sha(output / "result.json")
    c.write_new(folder / f"{event}_node{node}.json", value)


def check(output, launch, event):
    folder = directory(output, launch)
    if list(folder.glob("failed_node*.json")):
        raise RuntimeError("a node failed; inspect per-node log; resume with a new launch ID")
    nodes = [0] if event in ("prepared", "verified") else range(4)
    records = []
    for node in nodes:
        p = folder / f"{event}_node{node}.json"
        if not p.exists():
            return False
        value = c.read(p)
        if any(value.get(k) != v for k, v in dict(launch=launch, node=node, event=event).items()):
            raise ValueError("stale/mismatched cluster marker")
        if event != "boot":
            if value["receipt_sha256"] != c.sha(output / "receipt.json"):
                raise ValueError("cluster receipt changed")
            if event in ("ready", "done") and value["runtime_sha256"] != c.sha(output / f"runtime_node{node}.json"):
                raise ValueError("node runtime changed")
            if event == "verified" and value["result_sha256"] != c.sha(output / "result.json"):
                raise ValueError("result changed")
        records.append(value)
    if event == "boot":
        if len({v["hostname"] for v in records}) != 4:
            raise ValueError("four distinct nodes required")
        if len({(v["master_addr"], v["master_port"]) for v in records}) != 1:
            raise ValueError("nodes disagree on rendezvous address/port")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["publish", "wait"])
    p.add_argument("event", choices=["boot", "prepared", "ready", "done", "verified", "failed"])
    p.add_argument("--run-id", required=True)
    p.add_argument("--launch-id", required=True)
    p.add_argument("--node-rank", type=int, choices=range(4), required=True)
    p.add_argument("--timeout", type=float, default=7200)
    args = p.parse_args()
    output = c.run_path(c.ROOT, args.run_id)
    if args.mode == "publish":
        publish(output, args.launch_id, args.node_rank, args.event)
    else:
        end = time.monotonic() + args.timeout
        while not check(output, args.launch_id, args.event):
            if time.monotonic() >= end:
                raise TimeoutError(f"waiting for {args.event}; check all four nodes/shared filesystem")
            time.sleep(2)


if __name__ == "__main__":
    main()
