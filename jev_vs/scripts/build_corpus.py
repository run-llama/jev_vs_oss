"""build-corpus [language|orientation|classify|split|all] [--n N]"""
import argparse

from jev_vs import corpus


def main():
    ap = argparse.ArgumentParser(description="Build the labelled PDF corpora under ./data")
    ap.add_argument("task", choices=[*corpus.BUILDERS, "all"])
    ap.add_argument("--n", type=int, default=None, help="per-language / per-class / total count (task dependent)")
    a = ap.parse_args()
    tasks = list(corpus.BUILDERS) if a.task == "all" else [a.task]
    for t in tasks:
        kw = {}
        if a.n is not None:
            kw = {"language": {"n_per_lang": a.n}, "orientation": {"n": a.n}, "classify": {"n_per_class": a.n}, "split": {"n_bundles": a.n}, "triage": {"n": a.n}}[t]
        print("->", corpus.BUILDERS[t](**kw))


if __name__ == "__main__":
    main()
