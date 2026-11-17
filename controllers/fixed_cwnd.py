"""Fixed-window controllers: the window never changes, whatever happens.

Registered names: fixed4, fixed8, fixed16, fixed40. Importing this module registers them.

Two uses:
1. Phase 3 proof of control. If the window really is under our control, throughput on a
   link that is not the bottleneck follows cwnd / RTT (double the window, double the
   rate). tests/check_control.py checks this.
2. A real-lab reference matching the simulator's "fixed40" baseline (40 packets is about
   one BDP at 10 Mbit/s and 40 ms). Not a deployable controller: it is tuned for S1.
"""
from collections.abc import Iterable

from aioquic.quic.congestion.base import QuicCongestionControl, register_congestion_control
from aioquic.quic.packet_builder import QuicSentPacket


class FixedWindow(QuicCongestionControl):
    PACKETS = 40

    def __init__(self, *, max_datagram_size: int) -> None:
        super().__init__(max_datagram_size=max_datagram_size)
        self.congestion_window = self.PACKETS * max_datagram_size

    def on_packet_acked(self, *, now: float, packet: QuicSentPacket) -> None:
        self.bytes_in_flight -= packet.sent_bytes

    def on_packet_sent(self, *, packet: QuicSentPacket) -> None:
        self.bytes_in_flight += packet.sent_bytes

    def on_packets_expired(self, *, packets: Iterable[QuicSentPacket]) -> None:
        for p in packets:
            self.bytes_in_flight -= p.sent_bytes

    def on_packets_lost(self, *, now: float, packets: Iterable[QuicSentPacket]) -> None:
        for p in packets:
            self.bytes_in_flight -= p.sent_bytes

    def on_persistent_congestion(self) -> None:
        pass

    def on_rtt_measurement(self, *, now: float, rtt: float) -> None:
        pass


def _make(n):
    return type(f"Fixed{n}", (FixedWindow,), {"PACKETS": n})


for _n in (4, 8, 16, 40):
    register_congestion_control(f"fixed{_n}", _make(_n))
