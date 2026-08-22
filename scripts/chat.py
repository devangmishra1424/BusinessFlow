"""Interactive text chat with the BusinessFlow agent -- the only way to
have a live conversation with it right now (no CLI/UI existed before
this; run_turn/start_conversation in agent/loop.py were pure functions
with nothing wrapping them in a loop). Prints each tool call and its
result inline, not just the final reply -- seeing what got called with
what arguments is the actual thing worth testing before voice.

Run: python -m scripts.chat --account BF-1001 --key 482913 --language en
     python -m scripts.chat                    (no account, general Q&A mode)
"""

import argparse
import sys

from businessflow.agent.loop import AccessDeniedError, run_turn_with_memory, start_conversation, verify_and_start_conversation


def main():
    # Windows consoles default stdout to the legacy codepage (cp1252),
    # which can't encode rupee signs or Devanagari -- both routine output
    # for this product.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--account", default=None, help="account_id, e.g. BF-1001 (omit for no-account mode)")
    parser.add_argument("--key", default=None, help="that account's access key -- required if --account is given")
    parser.add_argument("--language", default="en", choices=["en", "hi"])
    args = parser.parse_args()

    if args.account:
        if not args.key:
            print(f"--key is required to talk about account {args.account}")
            return
        try:
            conversation = verify_and_start_conversation(args.language, args.account, args.key)
        except AccessDeniedError:
            print(f"wrong access key for account {args.account}")
            return
    else:
        conversation = start_conversation(language=args.language, account_id=None)

    print(f"BusinessFlow chat -- account={args.account or 'none'} language={args.language}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            break
        if user_text.lower() in ("exit", "quit"):
            break
        if not user_text:
            continue

        turn_start = len(conversation)
        conversation.append({"role": "user", "content": user_text})
        conversation, reply = run_turn_with_memory(conversation, args.account)

        for msg in conversation[turn_start + 1:]:
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    print(f"  [tool call] {tc['function']['name']}({tc['function']['arguments']})")
            elif msg["role"] == "tool":
                print(f"  [tool result] {msg['content']}")

        print(f"agent> {reply}\n")


if __name__ == "__main__":
    main()
