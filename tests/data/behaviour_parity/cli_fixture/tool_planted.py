"""tool.py with planted drift: add +1, info length x2, fail exits 4, warn stderr "!"."""

import json
import os
import sys


def main(argv: list[str]) -> int:
    cmd, rest = argv[0], argv[1:]
    if cmd == "add":
        print(int(rest[0]) + int(rest[1]) + 1)
        return 0
    if cmd == "info":
        print(json.dumps({"name": rest[0], "length": len(rest[0]) * 2, "pid": os.getpid()}))
        return 0
    if cmd == "upper":
        sys.stdout.write(sys.stdin.read().upper())
        return 0
    if cmd == "fail":
        print(f"tool: fatal: {rest[0]} (at tool_planted.py)", file=sys.stderr)
        return 4
    if cmd == "warn":
        print("warning: " + rest[0] + "!", file=sys.stderr)
        print("ok")
        return 0
    if cmd == "write":
        path = os.path.join(rest[0], "out.txt")
        with open(path, "w") as fh:
            fh.write(os.environ.get("GREETING", "none"))
        print(path)
        return 0
    if cmd == "sleep":
        import time

        time.sleep(float(rest[0]))
        return 0
    print(f"unknown command {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
