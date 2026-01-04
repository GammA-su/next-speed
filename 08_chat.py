import os

from infer import SentenceGenerator, split_sents

GEN_STEPS = int(os.environ.get("GEN_STEPS", "6"))
CONTEXT_SENTS = int(os.environ.get("CONTEXT_SENTS", "6"))


def main() -> None:
    gen = SentenceGenerator()
    history = []

    print("Enter a prompt. Use /exit to quit.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue
        if line.lower() in {"/exit", "/quit"}:
            print("Bye.")
            break

        history.append(line)
        prompt = " ".join(history)
        if not split_sents(prompt):
            print("Please enter a longer sentence (5+ chars).")
            history.pop()
            continue

        result = gen.generate(prompt, GEN_STEPS, context_sents=CONTEXT_SENTS)
        for i, sent in enumerate(result["sentences"], start=1):
            match = result["matches"][i - 1]
            print(f"[{i}] match={match}")
            print(sent)
        history.extend(result["sentences"])


if __name__ == "__main__":
    main()
