"""Inspect actual skill occupancies and sample their policies in the doorway grid."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from environments import GridEnv, two_room_walls
from utils.blahut_arimoto import build_skill_data
from utils.information import kl_divergence, mutual_information
from utils.occupancy_vertices import occupancy_to_policy


def entropy(probabilities):
    probabilities = np.asarray(probabilities)
    return -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=-1)


def draw_grid(ax, env):
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.tick_params(labelsize=8)
    for a, b in env.walls:
        row, col = a
        ax.plot([col + 0.5, col + 0.5], [row - 0.5, row + 0.5], color="#20252b", lw=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures/mi-lab"))
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--row", type=int, default=4)
    parser.add_argument("--col", type=int, default=3)
    parser.add_argument("--directions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = GridEnv(rows=8, cols=8, walls=two_room_walls(rows=8, wall_col=3, hole_row=4))
    s0 = env.to_s(args.row, args.col)
    data = build_skill_data(env, args.gamma, start_states=[s0], seed=args.seed, n_directions=args.directions)
    channel, prior = data.pz_s0[s0], data.p_prior[s0]
    mixture = prior @ channel
    divergences = kl_divergence(channel, mixture)
    mi = mutual_information(channel, prior)
    marginal_entropy = float(entropy(mixture))
    conditional_entropy = float(prior @ entropy(channel))
    np.testing.assert_allclose(mi, marginal_entropy - conditional_entropy, atol=1e-10)
    np.testing.assert_allclose(channel.sum(axis=1), 1, atol=1e-10)
    target_cells = [(1, 1), (1, 6), (6, 1), (6, 6)]
    available = list(np.flatnonzero(prior > 1e-6))
    selected = []
    for row, col in target_cells:
        if not available:
            break
        index = max(available, key=lambda i: channel[i, env.to_s(row, col)])
        selected.append(index)
        available.remove(index)
    rng = np.random.default_rng(args.seed)
    trajectories = []
    for index in selected:
        policy = occupancy_to_policy(data.vertices_by_s0[s0][index], env.nS, env.nA)
        states = [s0]
        for _ in range(32):
            state = states[-1]
            action_probabilities = policy[state] / policy[state].sum()
            action = rng.choice(env.nA, p=action_probabilities)
            transition = np.asarray(env.P[state, action], dtype=float)
            states.append(int(rng.choice(env.nS, p=transition / transition.sum())))
        trajectories.append(states)

    record = {
        "gamma": args.gamma,
        "start_row_col": [args.row, args.col],
        "search_seed": args.seed,
        "rollout_seed": args.seed,
        "random_lp_directions": args.directions,
        "candidate_skills": len(prior),
        "units": "nats",
        "effective_mi_nats": mi,
        "effective_mi_bits": mi / np.log(2),
        "entropy_marginal_nats": marginal_entropy,
        "expected_conditional_entropy_nats": conditional_entropy,
        "prior": prior.tolist(),
        "channel": channel.tolist(),
        "mixture": mixture.tolist(),
        "kl_per_skill_nats": divergences.tolist(),
        "selected_skills": [int(i) for i in selected],
        "illustrative_target_cells": target_cells,
        "sampled_trajectories": trajectories,
        "interpretation": "Capacity lower bound from sampled occupancy vertices; the rollout GIF is a sample, not the full occupancy distribution.",
    }
    (args.out / "mi-details.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in {"prior", "channel", "mixture", "kl_per_skill_nats", "sampled_trajectories"}}, indent=2), flush=True)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), layout="constrained")
    distributions = [mixture] + [channel[i] for i in selected]
    titles = ["All skills mixed: q(s)"] + [f"Skill {int(i)}: p(s | z)\nPrior = {prior[i]:.3f}" for i in selected]
    vmax = max(float(p.max()) for p in distributions)
    for ax, distribution, title in zip(axes, distributions, titles):
        im = ax.imshow(distribution.reshape(8, 8), cmap="viridis", vmin=0, vmax=vmax)
        draw_grid(ax, env)
        ax.scatter([args.col], [args.row], marker="*", s=90, color="white", edgecolors="black", zorder=5)
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=list(axes), label="Discounted occupancy probability", shrink=0.65)
    fig.suptitle(f"Same start, different skills | gamma = {args.gamma:g}\nI(Z; S+) = H(S+) - H(S+ | Z) = {marginal_entropy:.3f} - {conditional_entropy:.3f} = {mi:.3f} nats", fontsize=13)
    fig.savefig(args.out / "skill-distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout="constrained")
    artists = []
    for ax, index in zip(axes.flat, selected):
        ax.imshow(np.zeros((8, 8)), cmap="Greys", vmin=0, vmax=1)
        draw_grid(ax, env)
        ax.scatter([args.col], [args.row], marker="*", s=110, color="#edaa37", edgecolors="#20252b", zorder=4)
        line, = ax.plot([], [], color="#236fb6", lw=2, alpha=0.75)
        marker, = ax.plot([], [], "o", color="#236fb6", markersize=10)
        ax.set_title(f"Skill {int(index)} | selection probability {prior[index]:.3f}", fontsize=11)
        artists.append((line, marker))
    title = fig.suptitle("", fontsize=13)

    def update(frame):
        title.set_text(f"Samples from four learned skill policies | step {frame}\nSame doorway start; no task reward")
        for states, (line, marker) in zip(trajectories, artists):
            coordinates = np.array([env.to_rc(s) for s in states[:frame + 1]])
            line.set_data(coordinates[:, 1], coordinates[:, 0])
            marker.set_data(coordinates[-1:, 1], coordinates[-1:, 0])
        return [title] + [artist for pair in artists for artist in pair]

    animation = FuncAnimation(fig, update, frames=33, interval=200, blit=False)
    animation.save(args.out / "doorway-skills.gif", writer=PillowWriter(fps=5), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
