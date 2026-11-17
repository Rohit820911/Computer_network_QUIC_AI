"""Phase 3 plug-in test: a Reno copy registered as "reno_copy".

Importing this module registers the controller with aioquic. Select it with
QuicConfiguration(congestion_control_algorithm="reno_copy").

It behaves exactly like built-in Reno but records one row per call to
on_rtt_measurement (cwnd, RTT, bytes in flight, cumulative acked/lost bytes),
which is the logging format all later controllers will share.
"""
import csv
import time
from collections.abc import Iterable

from aioquic.quic.congestion.base import register_congestion_control
from aioquic.quic.congestion.reno import RenoCongestionControl
from aioquic.quic.packet_builder import QuicSentPacket

LOG_PATH = None  # set to a CSV path before the connection starts


class RenoCopy(RenoCongestionControl):
    def __init__(self, *, max_datagram_size: int) -> None:
        super().__init__(max_datagram_size=max_datagram_size)
        self._t0 = time.monotonic()
        self._acked = 0
        self._lost = 0
        self._rows = []

    def on_packet_acked(self, *, now: float, packet: QuicSentPacket) -> None:
        self._acked += packet.sent_bytes
        super().on_packet_acked(now=now, packet=packet)

    def on_packets_lost(self, *, now: float, packets: Iterable[QuicSentPacket]) -> None:
        packets = list(packets)
        self._lost += sum(p.sent_bytes for p in packets)
        super().on_packets_lost(now=now, packets=packets)

    def on_rtt_measurement(self, *, now: float, rtt: float) -> None:
        super().on_rtt_measurement(now=now, rtt=rtt)
        self._rows.append(
            (time.monotonic() - self._t0, self.congestion_window, rtt,
             self.bytes_in_flight, self._acked, self._lost)
        )

    def flush(self, path=None) -> None:
        path = path or LOG_PATH
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "cwnd", "rtt", "bytes_in_flight", "acked_bytes", "lost_bytes"])
            w.writerows(self._rows)


register_congestion_control("reno_copy", RenoCopy)
