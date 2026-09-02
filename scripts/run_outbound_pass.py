"""Manually-triggerable entrypoint for the proactive outbound pass -- a
real OS/cloud cron would call this directly instead of running
scripts/run_outbound_scheduler.py's own always-on loop, once that kind of
hosting exists.

Run: python -m scripts.run_outbound_pass
"""

import json
import sys

from businessflow.outbound.run import run_daily_pass


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    result = run_daily_pass()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        f"\n{len(result['promises_resolved'])} promise(s) resolved, "
        f"{len(result['escalated_for_broken_promises'])} escalated for broken promises, "
        f"{len(result['reminders_sent'])} reminder(s) sent"
    )


if __name__ == "__main__":
    main()
