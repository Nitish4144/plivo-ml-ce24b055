"""Baseline blend search with progress."""

import argparse
import time

import torch

import synth
import similarity as sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument(
        "--text",
        default="The quick brown fox jumps over the lazy dog."
    )
    ap.add_argument("--out", default="blend_baseline.pt")
    args = ap.parse_args()

    target = sim.target_embedding(args.reference_dir)
    voices = synth.stock_voices()

    print(f"Scoring {len(voices)} stock voices against the target...")

    scores = []
    for name, v in voices.items():
        wav = synth.synthesize(args.text, v)
        s = sim.similarity_to_target(wav, target)
        scores.append((s, name))
        print(f"{name:20s} {s:.4f}")

    scores.sort(reverse=True)

    print("\nTop 5:")
    for s, name in scores[:5]:
        print(f"{name:20s} {s:.4f}")

    top = scores[:5]

    best_score = -1.0
    best_voice = None
    best_desc = ""

    pairs = len(top) * (len(top) - 1) // 2
    total = pairs * 21
    done = 0

    start = time.time()

    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            _, n1 = top[i]
            _, n2 = top[j]

            print(f"\n=== Testing {n1} + {n2} ===", flush=True)

            for k in range(21):
                w = k / 20.0
                done += 1

                print(
                    f"[{done}/{total}] "
                    f"{n1}:{w:.2f}  {n2}:{1.0-w:.2f}",
                    flush=True,
                )

                blend = w * voices[n1] + (1.0 - w) * voices[n2]

                wav = synth.synthesize(args.text, blend)
                score = sim.similarity_to_target(wav, target)

                if score > best_score:
                    best_score = score
                    best_voice = blend.clone()
                    best_desc = (
                        f"{n1} ({w:.2f}) + "
                        f"{n2} ({1.0-w:.2f})"
                    )

                    elapsed = time.time() - start
                    print(
                        f"*** NEW BEST {best_score:.4f} "
                        f"after {elapsed:.1f}s",
                        flush=True,
                    )

    torch.save(best_voice, args.out)

    print("\nDone!")
    print(f"Best blend: {best_desc}")
    print(f"Similarity: {best_score:.4f}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()