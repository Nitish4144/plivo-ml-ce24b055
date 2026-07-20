import argparse
import math

import torch
import soundfile as sf

import synth
import similarity as sim

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please confirm your order number after the beep.",
    "I will call you back tomorrow at three thirty.",
]


def fitness(voice, target_emb):
    scores = []
    for text in SENTENCES:
        wav = synth.synthesize(text, voice)
        scores.append(sim.similarity_to_target(wav, target_emb))
    return sum(scores) / len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--out", default="voice.pt")
    ap.add_argument("--listen_every", type=int, default=5)
    args = ap.parse_args()

    target = sim.target_embedding(args.reference_dir)

    best = synth.load_voice(args.start).clone()
    current = best.clone()

    best_score = fitness(best, target)
    current_score = best_score

    print(f"Start: {best_score:.4f}")

    accepted = 0

    for i in range(1, args.iters + 1):

        step = max(0.005, args.step * (0.995 ** i))

        cand = current.clone()

        rows = torch.randperm(cand.shape[0])[:16]
        cand[rows] += step * torch.randn_like(cand[rows])

        score = fitness(cand, target)

        T = max(0.001, 0.02 * (0.995 ** i))
        delta = score - current_score

        accept = (
            delta > 0
            or torch.rand(1).item() < math.exp(delta / T)
        )

        if accept:
            current = cand
            current_score = score

            if score > best_score:
                best = cand.clone()
                best_score = score
                accepted += 1

                print(
                    f"iter {i:4d} | best={best_score:.4f} | accepted={accepted}"
                )

                if accepted % args.listen_every == 0:
                    sf.write(
                        f"listen_{accepted}.wav",
                        synth.synthesize(SENTENCES[0], best),
                        synth.SR,
                    )

    torch.save(best, args.out)

    sf.write(
        "listen_final.wav",
        synth.synthesize(SENTENCES[0], best),
        synth.SR,
    )

    print(f"\nFINAL BEST = {best_score:.4f}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()