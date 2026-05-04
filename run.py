import sys
from datetime import datetime
from pathlib import Path


def has_out_dir_arg(argv):
    return any(arg == "--out-dir" or arg.startswith("--out-dir=") for arg in argv)


def strip_out_dir_args(argv):
    stripped = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--out-dir":
            skip_next = True
            continue
        if arg.startswith("--out-dir="):
            continue
        stripped.append(arg)
    return stripped


if __name__ == "__main__":
    from STAR.core import build_parser, main as core_main

    # Parse args to check if out-dir was explicitly set
    args, remaining = build_parser().parse_known_args()
    out_dir = Path(args.out_dir)

    # Auto-name with timestamp if out-dir was not explicitly set.
    if not has_out_dir_arg(sys.argv[1:]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = Path("results") / timestamp

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "out.log"

    # Redirect stdout to both terminal and file
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    with open(log_path, "w") as log_file:
        tee = Tee(sys.stdout, log_file)
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            # Override sys.argv to pass the resolved out-dir
            sys.argv = [sys.argv[0]] + strip_out_dir_args(sys.argv[1:])
            sys.argv.extend(["--out-dir", str(out_dir)])
            sys.exit(core_main())
        finally:
            sys.stdout = old_stdout
