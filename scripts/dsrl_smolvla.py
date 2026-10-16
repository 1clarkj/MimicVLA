"""DSRL post-training of a frozen SmolVLA in the 15 x 7 mm no-funnel sim.

Diffusion steering via RL (Wagenmaker et al., CoRL 2025): the VLA's weights
never change. SAC learns a small policy that picks the *initial noise* of the
flow-matching action head instead of sampling it from N(0, I), so every
executed action is still decoded by the policy trained on human data; RL only
chooses among behaviours it can already produce.

One RL step is one action chunk (``n_action_steps`` env steps). The RL
observation is the VLA's own mean-pooled VLM prefix embedding (cameras,
language, joint state) plus the joint state, i.e. nothing the base policy
does not see. Only the noise of the executed chunk steps and real action dims
is steered (``--latent executed``, 10 x 6 = 60 dims) unless ``--latent chunk``;
the remaining noise entries are drawn from N(0, I) as usual.

Training uses the 64 training start states only; held-out clips are never
touched until ``--evaluate``, which writes the same summary.json as
scripts/evaluate_policy.py.

    # train
    PYTHONPATH=scripts MUJOCO_GL=egl .venv/bin/python scripts/dsrl_smolvla.py \
        outputs/train/lowpass_bs32_20k/checkpoints/020000/pretrained_model \
        --output-dir outputs/dsrl/lowpass20k_v1 --decisions 30000
    # evaluate a steering policy (or the unsteered base with --actor none)
    PYTHONPATH=scripts MUJOCO_GL=egl .venv/bin/python scripts/dsrl_smolvla.py CKPT \
        --evaluate outputs/dsrl/lowpass20k_v1/actor_final.pt --split all \
        --output-dir outputs/policy_eval/dsrl_lowpass20k_v1/final_all
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

from evaluate_policy import GEOMETRY, SUCCESS_DEPTH_M, load_policy
from holdout_split import split_indices
from mimicvla.data.sim_export import TASK
from mimicvla.sim.env import PandaUSBEnv, SimConfig, insertion_success_from_metrics


DATASET = Path("outputs/datasets/usb_15x7_no_funnel_full_v2")


# ---------------------------------------------------------------- environment

class InsertionTask:
    """One sim instance that resets to a demonstration's aligned start state."""

    def __init__(self, episodes: list[dict], args, seed: int, focus: list[dict] = ()):
        self.episodes, self.args, self.focus = episodes, args, list(focus)
        self.rng = np.random.default_rng(seed)
        self.env = None

    def reset(self, episode: dict | None = None) -> dict:
        if self.env is not None:
            self.env.close()
        if episode is None:
            pool = (self.focus if self.focus and self.rng.random() < self.args.focus_prob
                    else self.episodes)
            episode = pool[self.rng.integers(len(pool))]
        self.episode = episode
        replay_dir = Path(self.episode.get("replay_dir") or
                          Path("outputs/recalibration/replays") / self.episode["batch"] /
                          self.episode["clip"] / "wide_no_funnel")
        replay = json.loads((replay_dir / "retargeted_replay.json").read_text())
        self.env = PandaUSBEnv(SimConfig(max_steps=10_000, geometry=GEOMETRY,
                                         insertion_depth_m=SUCCESS_DEPTH_M))
        for entry in replay[:self.episode["first_source_step"]]:
            self.env.step(np.asarray(entry["action"], dtype=float))
        info = self.env.metrics()
        self.steps, self.decisions = 0, 0
        self.best_depth = self.start_depth = info["axial_error_m"]
        self.stats = {"max_depth_mm": 1000 * info["axial_error_m"], "peak_wall_force_n": 0.0,
                      "first_bad_collision_step": None, "steps_to_success": None,
                      "return": 0.0}
        self.info = info
        return self.env.observation(render=True)

    def run_chunk(self, actions: np.ndarray) -> tuple[dict, float, bool, bool]:
        """Execute one chunk; returns (obs, reward, terminal, truncated)."""
        a = self.args
        progress = force_cost = collision_steps = 0.0
        success, n = False, 0
        for action in actions:
            n += 1
            _, _, info = self.env.step(action)
            depth = min(info["axial_error_m"], SUCCESS_DEPTH_M)
            progress += max(0.0, depth - self.best_depth)
            self.best_depth = max(self.best_depth, depth)
            force = info["max_wall_normal_force_n"]
            force_cost += max(0.0, force - a.free_force_n) / 100
            collision_steps += info["bad_collision"]
            s = self.stats
            s["max_depth_mm"] = max(s["max_depth_mm"], 1000 * info["axial_error_m"])
            s["peak_wall_force_n"] = max(s["peak_wall_force_n"], force)
            if info["bad_collision"] and s["first_bad_collision_step"] is None:
                s["first_bad_collision_step"] = self.steps
            self.steps += 1
            self.info = info
            if info["insertion_success"]:
                s["steps_to_success"] = self.steps - 1
                success = True
                break
        self.decisions += 1
        span = max(SUCCESS_DEPTH_M - self.start_depth, 1e-3)
        reward = (a.progress_weight * progress / span
                  - a.force_weight * force_cost / n
                  - a.collision_weight * collision_steps / n
                  - a.time_penalty
                  + (a.success_bonus if success else 0.0))
        self.stats["return"] += reward
        aborted = (a.abort_force_n is not None and
                   self.info["max_wall_normal_force_n"] > a.abort_force_n)
        terminal = success or aborted
        truncated = not terminal and self.steps >= a.max_steps
        return self.env.observation(render=True), reward, terminal, truncated

    def result(self) -> dict:
        info, s = self.info, self.stats
        return {
            "episode_index": self.episode["episode_index"], "clip": self.episode["clip"],
            "success": s["steps_to_success"] is not None,
            "steps_to_success": s["steps_to_success"],
            "demo_steps_to_12mm": (None if self.episode.get("first_12mm_step") is None else
                                   self.episode["first_12mm_step"] - self.episode["first_source_step"]),
            "first_bad_collision_step": s["first_bad_collision_step"],
            "max_depth_mm": s["max_depth_mm"],
            "final_depth_mm": 1000 * info["axial_error_m"],
            "final_lateral_mm": [1000 * info["socket_lateral_x_m"],
                                 1000 * info["socket_lateral_y_m"]],
            "final_orientation_error_deg": float(np.rad2deg(info["orientation_error_rad"])),
            "final_tip_in_opening_7mm": insertion_success_from_metrics(
                info, required_depth_m=.007, geometry=GEOMETRY),
            "peak_wall_force_n": s["peak_wall_force_n"],
            "return": s["return"],
        }


# ----------------------------------------------------------- frozen VLA calls

class FrozenVLA:
    """Split SmolVLA inference into encode (prefix + features) and decode (from noise)."""

    def __init__(self, checkpoint: Path, device: str, n_action_steps: int):
        import torch

        self.torch, self.device = torch, device
        self.policy, self.pre, self.post = load_policy(checkpoint, device, n_action_steps)
        for p in self.policy.parameters():
            p.requires_grad_(False)
        cfg = self.policy.config
        self.chunk, self.max_dim = cfg.chunk_size, cfg.max_action_dim
        self.action_dim = cfg.action_feature.shape[0]
        self.n_exec = cfg.n_action_steps

    def encode(self, observations: list[dict]):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        frames = []
        for obs in observations:
            frame = {k: obs[k] for k in ("observation.state", "observation.images.camera1",
                                         "observation.images.camera2")}
            frames.append(self.pre(prepare_observation_for_inference(
                frame, torch.device(self.device), task=TASK, robot_type="franka_panda_usb")))
        batch = {k: torch.cat([f[k] for f in frames]) for k in frames[0]
                 if isinstance(frames[0][k], torch.Tensor)}
        policy, model = self.policy, self.policy.model
        with torch.inference_mode():
            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            embs, pad, att = model.embed_prefix(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                                batch[OBS_LANGUAGE_ATTENTION_MASK], state=state)
            outputs, kv = model.vlm_with_expert.forward(
                attention_mask=make_att_2d_masks(pad, att),
                position_ids=torch.cumsum(pad, dim=1) - 1, past_key_values=None,
                inputs_embeds=[embs, None], use_cache=True)
            hidden = outputs[0].float()
            mask = pad[..., None].float()
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            raw_state = batch["observation.state"].float()
            features = torch.cat([pooled, raw_state], dim=-1)
        return features.clone(), (pad, kv)

    def decode(self, cache, noise) -> np.ndarray:
        import torch
        from lerobot.policies.common.flow_matching import euler_integrate

        pad, kv = cache
        model = self.policy.model
        with torch.inference_mode():
            actions = euler_integrate(
                lambda x_t, t: model.denoise_step(x_t=x_t, prefix_pad_masks=pad,
                                                  past_key_values=kv, timestep=t),
                noise, model.config.num_steps)
            actions = actions[:, :self.n_exec, :self.action_dim]
            actions = self.post(actions)
        return actions.float().cpu().numpy().astype(float)


# -------------------------------------------------------------------- SAC

def build_sac(feature_dim: int, latent_dim: int, args):
    import torch
    from torch import nn

    def mlp(i, o, layer_norm):
        layers, d = [], i
        for _ in range(args.layers):
            layers += [nn.Linear(d, args.hidden)]
            if layer_norm:
                layers += [nn.LayerNorm(args.hidden)]
            layers += [nn.ReLU()]
            d = args.hidden
        return nn.Sequential(*layers, nn.Linear(d, o))

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(feature_dim)
            self.net = mlp(feature_dim, 2 * latent_dim, False)

        def forward(self, x, deterministic=False):
            mean, log_std = self.net(self.norm(x)).chunk(2, -1)
            log_std = log_std.clamp(-5, 2)
            if deterministic:
                return torch.tanh(mean), None
            dist = torch.distributions.Normal(mean, log_std.exp())
            u = dist.rsample()
            a = torch.tanh(u)
            logp = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
            return a, logp

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(feature_dim)
            self.q1 = mlp(feature_dim + latent_dim, 1, True)
            self.q2 = mlp(feature_dim + latent_dim, 1, True)

        def forward(self, x, a):
            h = torch.cat([self.norm(x), a], -1)
            return self.q1(h).squeeze(-1), self.q2(h).squeeze(-1)

    return Actor(), Critic()


class SAC:
    def __init__(self, feature_dim, latent_dim, args, device):
        import torch

        self.torch, self.args, self.device = torch, args, device
        self.actor, self.critic = build_sac(feature_dim, latent_dim, args)
        self.actor.to(device), self.critic.to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(args.init_alpha), device=device, requires_grad=True)
        self.target_entropy = -latent_dim * args.target_entropy_scale
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=args.lr)
        n = args.buffer_size
        self.buf = {"x": torch.zeros(n, feature_dim, device=device),
                    "a": torch.zeros(n, latent_dim, device=device),
                    "r": torch.zeros(n, device=device),
                    "x2": torch.zeros(n, feature_dim, device=device),
                    "d": torch.zeros(n, device=device)}
        self.size = self.ptr = 0

    def add(self, x, a, r, x2, d):
        i = self.ptr
        for k, v in zip("x a r x2 d".split(), (x, a, r, x2, d)):
            self.buf[k][i] = self.torch.as_tensor(v, device=self.device, dtype=self.torch.float32)
        self.ptr = (self.ptr + 1) % self.args.buffer_size
        self.size = min(self.size + 1, self.args.buffer_size)

    def update(self) -> dict:
        torch, args = self.torch, self.args
        idx = torch.randint(0, self.size, (args.batch_size,), device=self.device)
        x, a, r, x2, d = (self.buf[k][idx] for k in "x a r x2 d".split())
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor(x2)
            q_next = torch.min(*self.target(x2, a2)) - alpha * logp2
            y = r + args.gamma * (1 - d) * q_next
        q1, q2 = self.critic(x, a)
        critic_loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        self.opt_critic.zero_grad(), critic_loss.backward(), self.opt_critic.step()
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target.parameters()):
                tp.mul_(1 - args.tau).add_(args.tau * p)
        out = {"critic_loss": float(critic_loss), "q": float(q1.mean())}
        self.updates = getattr(self, "updates", 0) + 1
        if self.updates % args.actor_every == 0:
            a_new, logp = self.actor(x)
            actor_loss = (alpha * logp - torch.min(*self.critic(x, a_new))).mean()
            self.opt_actor.zero_grad(), actor_loss.backward(), self.opt_actor.step()
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.opt_alpha.zero_grad(), alpha_loss.backward(), self.opt_alpha.step()
            out |= {"actor_loss": float(actor_loss), "alpha": float(alpha),
                    "entropy": float(-logp.mean())}
        return out


# ------------------------------------------------------------ noise handling

def make_noise(vla: FrozenVLA, latent, args, generator):
    """Full (B, chunk, max_dim) noise: steered entries from ``latent`` in [-1, 1]."""
    torch = vla.torch
    b = latent.shape[0]
    noise = torch.randn(b, vla.chunk, vla.max_dim, device=vla.device, generator=generator)
    steps = vla.n_exec if args.latent == "executed" else vla.chunk
    noise[:, :steps, :vla.action_dim] = (args.noise_bound * latent).view(b, steps, vla.action_dim)
    return noise


def latent_dim(vla: FrozenVLA, args) -> int:
    return (vla.n_exec if args.latent == "executed" else vla.chunk) * vla.action_dim


def base_latent(vla, n, args, generator):
    """Latent equivalent of ordinary N(0, I) sampling, clipped to the steering bound."""
    torch = vla.torch
    z = torch.randn(n, latent_dim(vla, args), device=vla.device, generator=generator)
    return (z / args.noise_bound).clamp(-.999, .999)


# ---------------------------------------------------------------- evaluation

def evaluate(vla, actor, episodes, args, label) -> dict:
    torch = vla.torch
    generator = torch.Generator(device=vla.device).manual_seed(args.seed)
    results = []
    for episode in episodes:
        task = InsertionTask([episode], args, args.seed)
        obs = task.reset(episode)
        done = False
        while not done:
            features, cache = vla.encode([obs])
            with torch.no_grad():
                if actor == "zero":
                    latent = torch.zeros(1, latent_dim(vla, args), device=vla.device)
                elif actor is None:
                    latent = base_latent(vla, 1, args, generator)
                else:
                    latent, _ = actor(features, deterministic=True)
            actions = vla.decode(cache, make_noise(vla, latent, args, generator))[0]
            obs, _, terminal, truncated = task.run_chunk(actions)
            done = terminal or truncated
        results.append(task.result())
        task.env.close()
        if args.evaluate is not None:
            print(json.dumps(results[-1]), flush=True)
    successes = sum(r["success"] for r in results)
    return {"checkpoint": str(args.checkpoint), "steering_actor": label,
            "dataset": str(args.dataset_dir), "split": args.split, "success_depth_mm": 12,
            "start_noise_mm": 0.0, "max_steps": args.max_steps,
            "n_action_steps": vla.n_exec, "rollouts": len(results), "successes": successes,
            "success_rate": successes / max(len(results), 1),
            "bad_collision_rollouts": sum(r["first_bad_collision_step"] is not None
                                          for r in results),
            "results": results}


# ------------------------------------------------------------------ training

def train(vla, args) -> None:
    import torch

    manifest = json.loads((DATASET / "capture_manifest.json").read_text())["episodes_manifest"]
    split = split_indices(DATASET)
    train_eps = [manifest[i] for i in split["train"]]
    probe = [train_eps[i] for i in np.linspace(0, len(train_eps) - 1, args.probe_starts).astype(int)]
    focus = [manifest[i] for i in args.focus_episodes if i in split["train"]]
    if len(focus) != len(args.focus_episodes):
        raise ValueError("--focus-episodes must be training episodes")
    tasks = [InsertionTask(train_eps, args, args.seed + 1000 * k, focus)
             for k in range(args.num_envs)]
    observations = [t.reset() for t in tasks]
    features, cache = vla.encode(observations)
    sac = SAC(features.shape[1], latent_dim(vla, args), args, vla.device)
    generator = torch.Generator(device=vla.device).manual_seed(args.seed)
    log = (args.output_dir / "train_log.jsonl").open("a")
    (args.output_dir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} |
        {"feature_dim": features.shape[1], "latent_dim": latent_dim(vla, args),
         "train_episodes": split["train"]}, indent=2) + "\n")
    decisions, episodes_done, start, last = 0, 0, time.time(), {}
    recent = []
    while decisions < args.decisions:
        with torch.no_grad():
            if decisions < args.warmup:
                latent = base_latent(vla, len(tasks), args, generator)
            else:
                latent, _ = sac.actor(features)
        actions = vla.decode(cache, make_noise(vla, latent, args, generator))
        next_obs, rewards, terminals, finished = [], [], [], []
        for k, task in enumerate(tasks):
            obs, r, terminal, truncated = task.run_chunk(actions[k])
            next_obs.append(obs), rewards.append(r), terminals.append(terminal)
            finished.append(terminal or truncated)
        next_features, next_cache = vla.encode(next_obs)
        for k in range(len(tasks)):
            sac.add(features[k], latent[k], rewards[k], next_features[k], float(terminals[k]))
        decisions += len(tasks)
        if decisions >= args.warmup and sac.size >= args.batch_size:
            for _ in range(args.utd * len(tasks)):
                last = sac.update()
        reset = [k for k in range(len(tasks)) if finished[k]]
        for k in reset:
            row = tasks[k].result()
            recent = (recent + [row])[-50:]
            episodes_done += 1
            log.write(json.dumps({"type": "episode", "decisions": decisions,
                                  "episode": episodes_done, **row}) + "\n")
            next_obs[k] = tasks[k].reset()
        if reset:
            next_features, next_cache = vla.encode(next_obs)
        features, cache = next_features, next_cache
        if decisions % args.log_every < len(tasks):
            rate = np.mean([r["success"] for r in recent]) if recent else float("nan")
            row = {"type": "train", "decisions": decisions, "episodes": episodes_done,
                   "success_last50": rate,
                   "collision_last50": np.mean([r["first_bad_collision_step"] is not None
                                                for r in recent]) if recent else float("nan"),
                   "elapsed_min": (time.time() - start) / 60, **last}
            log.write(json.dumps(row) + "\n"), log.flush()
            print(json.dumps(row), flush=True)
        if decisions % args.save_every < len(tasks) or decisions >= args.decisions:
            tag = "final" if decisions >= args.decisions else f"{decisions:06d}"
            torch.save({"actor": sac.actor.state_dict(), "critic": sac.critic.state_dict(),
                        "log_alpha": sac.log_alpha.detach().cpu(),
                        "feature_dim": features.shape[1], "latent_dim": latent_dim(vla, args),
                        "args": {k: str(v) if isinstance(v, Path) else v
                                 for k, v in vars(args).items()}},
                       args.output_dir / f"actor_{tag}.pt")
            if args.probe_starts:
                # Deterministic probe on a fixed subset of *training* starts only.
                summary = evaluate(vla, sac.actor.eval(), probe, args, tag)
                sac.actor.train()
                log.write(json.dumps({"type": "probe", "decisions": decisions,
                                      "successes": summary["successes"],
                                      "rollouts": summary["rollouts"],
                                      "collisions": summary["bad_collision_rollouts"]}) + "\n")
                log.flush()
                print(f"probe @ {decisions}: {summary['successes']}/{summary['rollouts']} "
                      f"successes, {summary['bad_collision_rollouts']} deep collisions",
                      flush=True)
    log.close()


def load_actor(path: Path, vla, args):
    import torch

    ckpt = torch.load(path, map_location=vla.device)
    saved = ckpt["args"]
    for key in ("latent", "noise_bound", "hidden", "layers"):
        setattr(args, key, type(getattr(args, key))(saved[key]))
    actor, _ = build_sac(ckpt["feature_dim"], ckpt["latent_dim"], args)
    actor.load_state_dict(ckpt["actor"])
    return actor.to(vla.device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="frozen SmolVLA pretrained_model dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluate", help="actor .pt to evaluate, 'none' for the base policy, "
                                           "or 'zero' for a fixed all-zero steered noise")
    parser.add_argument("--split", choices=["holdout", "train", "all"], default="all")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET,
                        help="manifest whose episodes supply evaluation start states")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400, help="env steps per episode")
    parser.add_argument("--n-action-steps", type=int, default=10)
    # steering space
    parser.add_argument("--latent", choices=["executed", "chunk"], default="executed")
    parser.add_argument("--noise-bound", type=float, default=2.0,
                        help="steered noise lies in [-bound, bound]")
    # reward
    parser.add_argument("--progress-weight", type=float, default=10.0,
                        help="total reward for going from the start depth to 12 mm")
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--force-weight", type=float, default=1.0,
                        help="per 100 N above --free-force-n, averaged over the chunk")
    parser.add_argument("--free-force-n", type=float, default=50.0)
    parser.add_argument("--collision-weight", type=float, default=2.0,
                        help="times the fraction of chunk steps in deep collision")
    parser.add_argument("--time-penalty", type=float, default=.05)
    parser.add_argument("--abort-force-n", type=float, help="end the episode above this force")
    # SAC
    parser.add_argument("--decisions", type=int, default=30_000)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--utd", type=int, default=2, help="gradient steps per decision")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=.97)
    parser.add_argument("--tau", type=float, default=.005)
    parser.add_argument("--actor-every", type=int, default=2)
    parser.add_argument("--init-alpha", type=float, default=.1)
    parser.add_argument("--target-entropy-scale", type=float, default=.5)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--focus-episodes", type=int, nargs="*", default=[],
                        help="training episode indices sampled with probability --focus-prob")
    parser.add_argument("--focus-prob", type=float, default=.5)
    parser.add_argument("--probe-starts", type=int, default=16,
                        help="training starts used for the periodic deterministic probe")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    torch.manual_seed(args.seed)
    vla = FrozenVLA(args.checkpoint, args.device, args.n_action_steps)
    if args.evaluate is None:
        train(vla, args)
        return
    manifest = json.loads((args.dataset_dir / "capture_manifest.json").read_text())["episodes_manifest"]
    episodes = manifest if args.split == "all" else \
        [manifest[i] for i in split_indices(args.dataset_dir)[args.split]]
    actor = {"none": None, "zero": "zero"}.get(args.evaluate) or \
        load_actor(Path(args.evaluate), vla, args)
    summary = evaluate(vla, actor, episodes, args, args.evaluate)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{summary['successes']}/{summary['rollouts']} succeeded "
          f"({100 * summary['success_rate']:.0f}%), "
          f"{summary['bad_collision_rollouts']} with deep collisions")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(Path("outputs/hf_cache").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
