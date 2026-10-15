#!/usr/bin/env python3
"""Minimal cross-node TCP listener/client probe with no torch dependency."""

from __future__ import annotations

import argparse
import socket


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    server = subparsers.add_parser("server")
    server.add_argument("--port", type=int, required=True)
    server.add_argument("--clients", type=int, required=True)
    client = subparsers.add_parser("client")
    client.add_argument("--host", required=True)
    client.add_argument("--port", type=int, required=True)
    client.add_argument("--name", required=True)
    client.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    if args.mode == "server":
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", args.port))
            listener.listen(args.clients)
            print(f"CROSS_NODE_LISTENER_READY port={args.port}", flush=True)
            for _ in range(args.clients):
                connection, address = listener.accept()
                with connection:
                    name = connection.recv(128).decode("utf-8", errors="replace")
                print(f"CROSS_NODE_CONNECTED name={name} address={address[0]}:{address[1]}", flush=True)
        print("CROSS_NODE_SERVER_PASS", flush=True)
    else:
        with socket.create_connection((args.host, args.port), timeout=args.timeout) as connection:
            connection.sendall(args.name.encode("utf-8"))
        print(f"CROSS_NODE_CLIENT_PASS name={args.name} server={args.host}:{args.port}", flush=True)


if __name__ == "__main__":
    main()
