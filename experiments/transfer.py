"""aioquic bulk transfer: the SERVER sends, the CLIENT downloads.

The server is the sender, so the congestion controller under test is set on the
server (--cc). Shaping in testbed/netns.sh is applied on the server->client
direction, matching this.

    python experiments/transfer.py server --cc cubic --log logs/x.csv --once
    python experiments/transfer.py client --host 10.0.0.2 --size-mb 10

Server log (CSV, sampled every --sample-ms): t, cwnd, bytes_in_flight,
rtt_latest, rtt_smoothed, rtt_min, sent_bytes, lost_bytes, lost_packets.
The same sampler works for every controller (reno, cubic, reno_copy, and the
learned one later), so logs are directly comparable.
A summary JSON is written next to the CSV (<log>.json).
"""
import argparse
import asyncio
import contextlib
import csv
import datetime
import json
import os
import ssl
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import aioquic.quic.recovery  # noqa: E402,F401  (registers reno, cubic)
import controllers.fixed_cwnd  # noqa: E402,F401  (registers fixed4/8/16/40)
import controllers.learned  # noqa: E402,F401  (registers learned, learned_<model>)
import controllers.reno_copy  # noqa: E402,F401  (registers reno_copy)
from aioquic.asyncio import QuicConnectionProtocol, serve  # noqa: E402
from aioquic.quic.configuration import QuicConfiguration  # noqa: E402
from aioquic.quic.connection import QuicConnection  # noqa: E402
from aioquic.quic.events import ConnectionTerminated, StreamDataReceived  # noqa: E402

CERT = os.path.join(ROOT, "testbed", "cert.pem")
KEY = os.path.join(ROOT, "testbed", "key.pem")
BIG_WINDOW = 256 * 1024 * 1024  # keep flow control from limiting the sender

DONE = None  # asyncio.Event, set when the (single) connection ends


def ensure_cert():
    if os.path.exists(CERT) and os.path.exists(KEY):
        return
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    with open(KEY, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()))
    with open(CERT, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


# ----------------------------------------------------------------- server
class SenderProtocol(QuicConnectionProtocol):
    log_path = None
    sample_s = 0.01

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._req = b""
        self._rows = []
        self._sampler = None
        self._t0 = None
        self._t0_epoch = None
        self.sent_bytes = 0
        self.lost_bytes = 0
        self.lost_packets = 0

    def _wrap_cc(self):
        cc = self._cc = self._quic._loss._cc
        orig_lost, orig_sent = cc.on_packets_lost, cc.on_packet_sent

        def lost(*, now, packets):
            packets = list(packets)
            self.lost_packets += len(packets)
            self.lost_bytes += sum(p.sent_bytes for p in packets)
            return orig_lost(now=now, packets=packets)

        def sent(*, packet):
            self.sent_bytes += packet.sent_bytes
            return orig_sent(packet=packet)

        cc.on_packets_lost, cc.on_packet_sent = lost, sent

    async def _sample(self):
        loss = self._quic._loss
        while True:
            self._rows.append((
                round(time.monotonic() - self._t0, 4),
                loss.congestion_window, loss.bytes_in_flight,
                loss._rtt_latest, loss._rtt_smoothed,
                loss._rtt_min if loss._rtt_min != float("inf") else 0.0,
                self.sent_bytes, self.lost_bytes, self.lost_packets))
            await asyncio.sleep(self.sample_s)

    def quic_event_received(self, event):
        if isinstance(event, StreamDataReceived):
            self._req += event.data
            if event.end_stream:
                n = int(self._req.decode().split()[1])
                self._wrap_cc()
                self._t0 = time.monotonic()
                self._t0_epoch = time.time()
                self._sampler = asyncio.ensure_future(self._sample())
                self._quic.send_stream_data(event.stream_id, b"x" * n, end_stream=True)
                self.transmit()
        elif isinstance(event, ConnectionTerminated):
            if self._sampler:
                self._sampler.cancel()
            self._flush()
            if DONE is not None:
                DONE.set()

    def _flush(self):
        if not self.log_path or not self._rows:
            return
        with open(self.log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "cwnd", "bytes_in_flight", "rtt_latest", "rtt_smoothed",
                        "rtt_min", "sent_bytes", "lost_bytes", "lost_packets"])
            w.writerows(self._rows)
        with open(self.log_path + ".json", "w") as f:
            json.dump({"t0_epoch": self._t0_epoch, "sent_bytes": self.sent_bytes,
                       "lost_bytes": self.lost_bytes,
                       "lost_packets": self.lost_packets}, f)
        # controllers that make per-interval decisions (the learned one) log them too
        rows = getattr(getattr(self, "_cc", None), "decision_rows", None)
        if rows:
            with open(self.log_path + ".decisions.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(self._cc.decision_header)
                w.writerows(rows)


async def run_server(args):
    global DONE
    DONE = asyncio.Event()
    ensure_cert()
    cfg = QuicConfiguration(is_client=False, congestion_control_algorithm=args.cc,
                            max_data=BIG_WINDOW, max_stream_data=BIG_WINDOW)
    cfg.load_cert_chain(CERT, KEY)
    SenderProtocol.log_path = args.log
    SenderProtocol.sample_s = args.sample_ms / 1000
    await serve(args.host, args.port, configuration=cfg, create_protocol=SenderProtocol)
    print("listening", flush=True)
    if args.once:
        await asyncio.wait_for(DONE.wait(), timeout=args.timeout + 30)
    else:
        await asyncio.Event().wait()


# ----------------------------------------------------------------- client
class ReceiverProtocol(QuicConnectionProtocol):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.done = asyncio.get_event_loop().create_future()
        self.received = 0

    def quic_event_received(self, event):
        if isinstance(event, StreamDataReceived):
            self.received += len(event.data)
            if event.end_stream and not self.done.done():
                self.done.set_result(time.monotonic())


@contextlib.asynccontextmanager
async def connect_v4(host, port, cfg):
    """Plain IPv4 client socket. aioquic's own connect() opens a dual-stack IPv6
    socket, which fails on machines where IPv6 is disabled; the lab is IPv4 only."""
    loop = asyncio.get_running_loop()
    cfg.server_name = cfg.server_name or host
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: ReceiverProtocol(QuicConnection(configuration=cfg)),
        local_addr=("0.0.0.0", 0))
    try:
        protocol.connect((host, port))
        await protocol.wait_connected()
        yield protocol
    finally:
        protocol.close()
        await protocol.wait_closed()
        transport.close()


async def run_client(args):
    n = int(args.size_mb * 1024 * 1024)
    cfg = QuicConfiguration(is_client=True, verify_mode=ssl.CERT_NONE,
                            max_data=BIG_WINDOW, max_stream_data=BIG_WINDOW)
    async with connect_v4(args.host, args.port, cfg) as client:
        sid = client._quic.get_next_available_stream_id()
        t0 = time.monotonic()
        client._quic.send_stream_data(sid, f"GET {n}".encode(), end_stream=True)
        client.transmit()
        t1 = await asyncio.wait_for(client.done, timeout=args.timeout)
        secs = t1 - t0
        print(json.dumps({"bytes": client.received, "seconds": round(secs, 4),
                          "goodput_mbps": round(client.received * 8 / secs / 1e6, 3)}),
              flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=4433)
    s.add_argument("--cc", default="reno", help="reno | cubic | reno_copy | ...")
    s.add_argument("--log", default=None)
    s.add_argument("--sample-ms", type=float, default=10)
    s.add_argument("--once", action="store_true", help="exit after one connection")
    s.add_argument("--timeout", type=float, default=300)
    c = sub.add_parser("client")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=4433)
    c.add_argument("--size-mb", type=float, default=10)
    c.add_argument("--timeout", type=float, default=300)
    args = p.parse_args()
    asyncio.run(run_server(args) if args.mode == "server" else run_client(args))


if __name__ == "__main__":
    main()
