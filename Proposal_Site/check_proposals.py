"""Check every proposal folder against the standard Report_Gen format.

Usage (from the repo root or anywhere):
    python Proposal_Site/check_proposals.py                 # all proposals
    python Proposal_Site/check_proposals.py 2026 Fall 8     # one proposal after running json_gen.py + script.py
    python Proposal_Site/check_proposals.py --external      # website submissions too

Exit code 1 if any proposal has errors.
"""
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.WARNING)

from core import config, validate            # noqa: E402
from core.storage import LocalStorage         # noqa: E402


def main(argv):
    store = LocalStorage(config.REPO_ROOT)
    bases = [config.OFFICIAL] + ([config.EXTERNAL] if "--external" in argv else [])
    args = [a for a in argv if not a.startswith("--")]
    folders = sorted({f.rsplit("/", 1)[0] for b in bases for f in store.list_files(b)
                      if re.search(r"/\d{4}/\w+/\d+/input\.json$", f)})
    if args:
        want = "/" + "/".join(args)
        folders = [f for f in folders if f.endswith(want)]
        if not folders:
            print(f"No proposal folder matches {' '.join(args)}")
            return 1
    n_err = n_warn = 0
    for f in folders:
        errors, warnings = validate.check_folder(store, f)
        n_err += bool(errors)
        n_warn += bool(warnings)
        mark = "ERROR" if errors else ("warn " if warnings else "ok   ")
        print(f"[{mark}] {f.replace(config.ARXIV + '/', '')}")
        for e in errors:
            print(f"          x {e}")
        for w in warnings:
            print(f"          ! {w}")
    print(f"\n{len(folders)} proposals checked: {n_err} with errors, {n_warn} with warnings.")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
