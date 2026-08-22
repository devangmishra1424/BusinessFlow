"""Manually-triggerable entrypoint for the proactive outbound pass -- no
scheduler exists yet (deferred alongside hosting/Telegram/voice), so
this is what a real cron/cloud-scheduler job would call once one does.

Run: python -m scripts.run_outbound_pass
"""

import json
import sys

from businessflow.outbound.run import run_daily_outbound_pass


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sent = run_daily_outbound_pass()
    print(json.dumps(sent, indent=2, ensure_ascii=False))
    print(f"\n{len(sent)} reminder(s) sent")


if __name__ == "__main__":
    main()
