"""
PYNQ bring-up on the KR260: load the overlay, sanity-check against the golden model, read cycle counts.
Copy deploy/overlay/ and deploy/testvec/ next to deploy/board/ on the board, then:
    sudo -E python3 driver.py [--n 2000]
Leaves the bitstream loaded, so bench.c can run afterwards.
"""
import argparse
import os
import sys

# sudo strips the environment; restore what Kria-PYNQ's /etc/profile.d/pynq_venv.sh sets up.
os.environ.setdefault("XILINX_XRT", "/usr")
os.environ.setdefault("BOARD", "KR260")
# venv bin first (as `activate` does): PYNQ shells out to xclbinutil, and sudo's secure_path would
# otherwise pick /usr/bin/xclbinutil, an XRT wrapper that fails silently on this image.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

import numpy as np  # noqa: E402
from pynq import MMIO, Overlay  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GPIO_GO, GPIO_STATUS = 0x0, 0x8
QNN_IN = [0x10 + 8 * k for k in range(8)]
QNN_OUT = [0x50, 0x60, 0x70]
CORE_CLK_HZ = 199_998_000


def pack(xq):
    u = xq.astype(np.uint16).astype(np.uint32)
    u = np.append(u, np.uint32(0))                    # pad to 16 elements
    return [int(u[2 * k] | (u[2 * k + 1] << 16)) for k in range(8)]


def unpack(words):
    raw = [(words[j // 2] >> (16 * (j % 2))) & 0xFFFF for j in range(5)]
    return np.array(raw, np.uint16).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="events to check (Python is slow; bench.c does all)")
    args = ap.parse_args()

    ol = Overlay(os.path.join(HERE, "..", "overlay", "qnn.bit"))
    print("IPs:", sorted(ol.ip_dict))
    gpio_ip, core_ip = ol.ip_dict["axi_gpio_0"], ol.ip_dict["qnn_axi_0"]
    gpio = MMIO(gpio_ip["phys_addr"], gpio_ip["addr_range"])
    core = MMIO(core_ip["phys_addr"], core_ip["addr_range"])
    print(f"gpio @ {gpio_ip['phys_addr']:#x}, qnn_axi @ {core_ip['phys_addr']:#x}")

    tv = os.path.join(HERE, "..", "testvec")
    xq = np.fromfile(os.path.join(tv, "x_q.bin"), np.int16).reshape(-1, 15)
    gold = np.fromfile(os.path.join(tv, "golden_q.bin"), np.int16).reshape(-1, 5)
    y = np.fromfile(os.path.join(tv, "y.bin"), np.uint8)

    gpio.write(GPIO_GO, 0)
    n, bad, cycles, pred = min(args.n, len(y)), 0, [], []
    for e in range(n):
        for addr, w in zip(QNN_IN, pack(xq[e])):
            core.write(addr, w)
        gpio.write(GPIO_GO, 1)
        while not (s := gpio.read(GPIO_STATUS)) & 0x8000_0000:
            pass
        out = unpack([core.read(a) for a in QNN_OUT])
        gpio.write(GPIO_GO, 0)
        cycles.append(s & 0x7FFF_FFFF)
        bad += int((out != gold[e]).any())
        pred.append(int(out.argmax()))
    cyc = np.array(cycles)
    print(f"{n} events: {bad} events differ from golden -> {'FAIL' if bad else 'PASS'}")
    print(f"accuracy on these events: {(np.array(pred) == y[:n]).mean():.4f} "
          f"(golden: {(gold[:n].argmax(1) == y[:n]).mean():.4f})")
    print(f"on-chip cycles: min {cyc.min()} max {cyc.max()} -> {cyc.min() * 1e9 / CORE_CLK_HZ:.1f} ns")


if __name__ == "__main__":
    main()
