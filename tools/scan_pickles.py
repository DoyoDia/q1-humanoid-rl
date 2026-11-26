"""Check model weights from an untrusted source before anything unpickles them.

    python tools/scan_pickles.py FILE [FILE ...]

Pickle can call any importable function while loading, so a .ckpt / .pth / .pkl from a
third-party mirror can run code the moment torch.load() opens it. This walks the pickle
opcodes without executing them and lists every global the file references (module.name).
Torch zip checkpoints are opened and each pickle inside is scanned. A file passes when every
global is on the allowlist below (tensor rebuilding, storages, containers, numpy arrays, and
a few config/metadata types); anything that can run code -- os, subprocess, builtins.eval,
exec, getattr chains, etc. -- fails it. Exit code 1 if any file fails.
"""

from __future__ import annotations

import pickletools
import re
import sys
import zipfile
from pathlib import Path

ALLOW = [
    r"torch\._utils\._rebuild_(tensor|tensor_v2|parameter|parameter_with_state|device_tensor_from_numpy|qtensor|sparse_tensor|nested_tensor)",
    r"torch\.[A-Za-z]*Storage", r"torch\.storage\._load_from_bytes", r"torch\.(device|Size|float32|float16|bfloat16|int64|int32|bool|dtype|_tensor\._rebuild_from_type_v2|Tensor)",
    r"torch\.nn\.parameter\.Parameter", r"torch\.serialization\._get_layout",
    r"collections\.(OrderedDict|defaultdict)",
    r"numpy\.core\.multiarray\.(_reconstruct|scalar)", r"numpy\._core\.multiarray\.(_reconstruct|scalar)",
    r"numpy\.(ndarray|dtype)", r"numpy\.dtypes\.[A-Za-z0-9]+DType",
    r"_codecs\.encode",
    r"(builtins|__builtin__)\.(set|frozenset|slice|complex|bytearray|tuple|list|dict|int|float|str|bool)",
    r"argparse\.Namespace",
    r"omegaconf\.[A-Za-z_.]+", r"typing\.Any",
    r"pathlib\.(PosixPath|WindowsPath|Path)",
]
DANGER = re.compile(r"^(os|posix|nt|subprocess|sys|shutil|socket|importlib|runpy|pty|ctypes|builtins\.(eval|exec|compile|getattr|setattr|__import__|open|input|globals|locals|vars|apply))\b")


def globals_in(data: bytes) -> set[str]:
    """Globals of every pickle at the head of `data`. Legacy (non-zip) torch files are a
    run of pickles -- magic number, protocol, sys info, the object, storage keys -- before
    the raw tensor bytes; stopping at the first one scanned nothing but the magic number."""
    found, pos = set(), 0
    for _ in range(8):
        if pos >= len(data) or data[pos] != 0x80:          # PROTO opens every pickle here
            break
        pos = _globals_one(data, pos, found)
    return found


def _globals_one(data: bytes, start: int, found: set[str]) -> int:
    import io
    strings, end = [], start
    f = io.BytesIO(data)
    f.seek(start)
    for op, arg, at in pickletools.genops(f):
        end = at + 1
        if op.name in ("GLOBAL", "INST") and isinstance(arg, str):
            found.add(arg.replace(" ", "."))
        elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8", "SHORT_BINSTRING", "BINSTRING", "STRING"):
            strings.append(str(arg))
        elif op.name == "STACK_GLOBAL":
            if len(strings) >= 2:
                found.add(f"{strings[-2]}.{strings[-1]}")
            else:
                found.add("<unresolved STACK_GLOBAL>")
    return end


def scan(path: Path) -> tuple[bool, set[str]]:
    refs = set()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith(".pkl"):
                    refs |= globals_in(z.read(n))
    else:
        refs |= globals_in(path.read_bytes())
    bad = {r for r in refs if DANGER.match(r) or not any(re.fullmatch(p, r) for p in ALLOW)}
    return not bad, refs


def main():
    ok_all = True
    for f in sys.argv[1:]:
        p = Path(f)
        try:
            ok, refs = scan(p)
        except Exception as e:                       # noqa: BLE001 -- unparseable is a fail
            ok, refs = False, {f"<cannot parse: {e}>"}
        ok_all &= ok
        flagged = sorted(r for r in refs if DANGER.match(r) or not any(re.fullmatch(x, r) for x in ALLOW))
        print(f"{'ok  ' if ok else 'FAIL'} {p}  ({len(refs)} globals)" + (f"  NOT ALLOWED: {flagged}" if flagged else ""))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
