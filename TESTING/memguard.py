#!/usr/bin/env python3
"""Run a command under a hard resident-memory ceiling.

    python TESTING/memguard.py --cap-gb 20 [--interval-ms 200] -- <command...>

The child is started in its own process group. Every interval we sum the
resident size of every live process under it; if that total crosses --cap-gb we
SIGKILL the whole group and exit 137. This is a *hard* ceiling: it does not rely
on the child cooperating or on any allocator limit.

Why libproc and not ps/psutil: in this sandbox /bin/ps is denied by the seatbelt
("Operation not permitted") and psutil is not installed in the benchmark env.
proc_pid_rusage() from libSystem needs no extra privilege and returns each
process's resident size directly.

Peak is tracked here as the max of the summed tree, and independently by
/usr/bin/time -l wrapped around this script (which reports the max RSS of any
single process in the subtree). Both are printed so a run can be cross-checked.
"""
import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time

# --- libproc bindings -------------------------------------------------------
_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)

# libproc.h: #define RUSAGE_INFO_V0 1
RUSAGE_INFO_V0 = 1

# proc_pid_rusage() writes a whole rusage_info struct (the kernel knows it is
# larger than v0), so the destination must be a generously sized raw buffer --
# defining only the fields we want and passing that struct overflows into
# adjacent memory. resident_size sits at byte offset 64 in the v0 layout:
# uuid[16] then user_time, system_time, pkg_idle_wkups, interrupt_wkups,
# pageins, wired_size (six uint64s = 48 bytes), then resident_size.
_RUSAGE_BUF = 2048
_RESIDENT_OFFSET = 64

_libc.proc_pid_rusage.restype = ctypes.c_int
_libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


def _rss_bytes(pid):
    """Resident size of pid in bytes, or None if it is gone/unreadable."""
    buf = ctypes.create_string_buffer(_RUSAGE_BUF)
    rc = _libc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(RUSAGE_INFO_V0),
                               buf)
    if rc != 0:
        return None
    return int.from_bytes(buf.raw[_RESIDENT_OFFSET:_RESIDENT_OFFSET + 8], "little")


_libc.proc_listchildpids.restype = ctypes.c_int
_libc.proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def _children(pid):
    """Direct children of pid (empty when libproc cannot answer)."""
    buf = (ctypes.c_int * 4096)()
    n = _libc.proc_listchildpids(ctypes.c_int(pid), buf, ctypes.sizeof(buf))
    if n <= 0:
        return []
    count = min(n, len(buf)) // ctypes.sizeof(ctypes.c_int)
    return [int(buf[i]) for i in range(count) if buf[i] > 0]


def _tree_rss(pid, acc):
    """Sum RSS over pid and its descendants (best effort)."""
    total = 0
    stack, seen = [pid], set()
    while stack:
        p = stack.pop()
        if p in seen or p <= 0:
            continue
        seen.add(p)
        r = _rss_bytes(p)
        if r:
            total += r
        stack.extend(_children(p))
    acc.append(total)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-gb", type=float, default=20.0)
    ap.add_argument("--interval-ms", type=int, default=200)
    ap.add_argument("--label", default="")
    # everything after -- is the command; argparse REMAINDER keeps it verbatim
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given (use: memguard.py --cap-gb N -- cmd args...)")

    cap_bytes = int(args.cap_gb * 1024 ** 3)
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, start_new_session=True)
    pgid = os.getpgid(proc.pid)

    peak_sum = 0
    killed = False
    try:
        while True:
            rc = proc.poll()
            total = _tree_rss(proc.pid, [])
            peak_sum = max(peak_sum, total)
            if rc is not None:
                break
            if cap_bytes and total > cap_bytes:
                killed = True
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                break
            time.sleep(max(args.interval_ms, 20) / 1000.0)
    except KeyboardInterrupt:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise

    wall = time.perf_counter() - t0
    code = proc.returncode if proc.returncode is not None else 137
    out = {
        "memguard_label": args.label,
        "cap_gb": args.cap_gb,
        "peak_tree_rss_gb": round(peak_sum / 1e9, 3),
        "killed": killed,
        "exit_code": code if not killed else 137,
        "wall_s": round(wall, 3),
    }
    print("MEMGUARD_JSON " + json.dumps(out), flush=True)
    if killed:
        print(f"[memguard] KILLED {args.label or cmd[0]} at "
              f"{peak_sum / 1e9:.2f} GB (cap {args.cap_gb:g} GB)", flush=True)
    sys.exit(137 if killed else (code or 0))


if __name__ == "__main__":
    main()
