"""Instance-wise optimization of phi (Sec. 3.3, Algorithms 1 and 2).

Round 1 (Algorithm 1): K initializations phi_k each give a candidate trajectory
x^(k) ~ pi_{phi_k} and a reward r_k. The advantages
    A_k = (r_k - mean(r)) / (std(r) + 1e-8),  clipped to [-A_max, A_max],
weight the clipped surrogate of Eq. (3)
    L(phi) = sum_{k,tau} min( rho_{k,tau} A_k, clip_delta(rho_{k,tau}) A_k ),
    rho_{k,tau} = pi_phi(a_tau^(k) | s_tau^(k)) / pi_{phi_k}(a_tau^(k) | s_tau^(k)),
and phi takes E_in passes over the candidates in random order, one optimizer step
per denoising step, decoding A(phi) after every candidate.

Rounds 2..E_max (Algorithm 2): one trajectory under the current phi, whose raw
reward (clipped) is the advantage, one update, one decoding.

Every decoded tour enters the keep-best rule under cost(y) = l(y) + xi V(y), so
the returned tour is the best one evaluated.

The sampling / advantage / clipped-ratio loop follows DDPO
(Black et al., ICLR 2024; https://github.com/kvablack/ddpo-pytorch, MIT License,
scripts/train.py), applied to the single latent phi of one instance instead of
the weights of a diffusion model.

The perturbation controls of Sec. 4.3 share the initialization, decoder, 2-opt,
keep-best rule and the number of reward evaluations with DDRL, and replace only
the update (see `_run_perturbation_control`).
"""
import numpy as np
import torch

from ddrl.config import CONTROL_MODES
from ddrl.geometry import box_crossing_matrix
from ddrl.latent import LatentTourModel
from ddrl.prior_init import load_difusco_heatmap, plug_and_play_init
from ddrl.reward import decode_and_score
from ddrl.sampling import ddim_step_with_logprob, sample_trajectory
from ddrl.solver import TwoOptSolver
from ddrl.utils import Stopwatch, seed_everything


class KeepBest:
    """Best tour seen so far under cost(y) = l(y) + xi V(y), and the cost of every evaluation."""

    def __init__(self, gt_cost, penalty_const):
        self.gt_cost = gt_cost
        self.penalty_const = penalty_const
        self.cost = 10 ** 10
        self.gap = 0
        self.penalty_count = 0
        self.tour = []
        self.epoch = -1
        self.inner_epoch = -1  # -1: the best tour came from a sampling-phase candidate
        self.initial_cost = None  # l(y) of the first evaluated candidate
        self.round_costs, self.round_epochs = [], []
        self.num_evals = 0

    def update(self, info, epoch, inner_epoch):
        """Record one evaluation; returns True if it is the new best."""
        cost = info['basic_cost'] + self.penalty_const * info['penalty_count']
        self.num_evals += 1
        self.round_costs.append(cost)
        self.round_epochs.append(epoch)
        if self.initial_cost is None:
            self.initial_cost = info['basic_cost']
        if cost < self.cost:
            self.cost = cost
            self.gap = 100 * (cost - self.gt_cost) / self.gt_cost
            self.penalty_count = info['penalty_count']
            self.tour = info['solved_tour']
            self.epoch, self.inner_epoch = epoch, inner_epoch
            return True
        return False


class InstanceContext:
    """Everything the search of one instance needs besides the instance itself."""

    def __init__(self, config, device, prior, scheduler, dphi_trace=None):
        self.config = config
        self.device = device
        self.prior = prior
        self.scheduler = scheduler
        self.dphi_trace = dphi_trace or {}


def ppo_clipped_loss(log_prob, old_log_prob, advantages, clip_range, adv_clip_max):
    """Loss for one (k, tau) term of Eq. (3): max(-A rho, -A clip_delta(rho)) = -min(A rho, A clip_delta(rho)).

    DDRL takes one optimizer step per denoising step tau (Algorithm 1: "T optimizer
    steps"), so the sum over tau in Eq. (3) becomes a sequence of steps; the mean is
    over the candidates in the batch (one in the paper).
    """
    advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
    ratio = torch.exp(log_prob - old_log_prob)                       # rho_{k,tau}
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    return torch.mean(torch.maximum(unclipped_loss, clipped_loss))


def _initialize(model, ctx, sample_idx):
    """Draw one initial phi_k (Algorithm 1, 'phi_k <- initialization')."""
    cfg = ctx.config
    model.reset()
    if cfg.init_mode == 'prior':
        plug_and_play_init(model, ctx.prior, steps=cfg.num_steps, device=ctx.device)
    elif cfg.init_mode == 'difusco':
        model.init_from_heatmap(load_difusco_heatmap(cfg.heatmap_path, cfg.num_cities, sample_idx),
                                device=ctx.device)
    # 'none': keep the N(0, I) draw of reset()


def optimize_instance(ctx, img, points, gt_tour, sample_idx, constraint):
    """Run DDRL (or a control) on one instance and return its result row."""
    cfg, device = ctx.config, ctx.device

    # Seed per instance so that a result does not depend on which other instances
    # the same process handled before (i.e. on how the index range was sharded).
    instance_seed = int(cfg.seed) + int(sample_idx)
    seed_everything(instance_seed)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    timer = Stopwatch('total', 'init', 'sample', 'rl', 'reward')
    stats = {'grad_steps': 0, 'dphi_norms': []}

    with timer('total'):
        xT = torch.randn_like(img).float().to(device)  # fixed x_T of every trajectory
        model = LatentTourModel(batch_size=1, num_points=points.shape[0], img_size=cfg.img_size,
                                line_color=cfg.line_color, line_thickness=cfg.line_thickness,
                                xT=xT, use_heatmap=(cfg.init_mode == 'difusco')).to(device)
        model.eval()
        img_query = torch.zeros_like(img)
        img_query[img == 1] = 1  # city pixels
        model.compute_edge_images(points=points, img_query=img_query)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                      betas=(cfg.adam_beta1, cfg.adam_beta2),
                                      weight_decay=cfg.adam_weight_decay, eps=cfg.adam_epsilon)

        if cfg.constraint_type == 'box':
            constraint = box_crossing_matrix(points, constraint)
        solver = TwoOptSolver(points, constraint_type=cfg.constraint_type, constraint=constraint)
        gt_cost = solver.evaluate([i - 1 for i in gt_tour])
        best = KeepBest(gt_cost, cfg.penalty_const)

        dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        penalty_weight = cfg.penalty_const if cfg.penalty_in_reward else 0.0

        def evaluate():
            """Decode A(phi), refine, score: one reward evaluation."""
            adj_mat = model.get_adj_mat()
            with timer('reward'):
                reward, info = decode_and_score(points, adj_mat, dists, cfg.constraint_type,
                                                constraint=constraint, max_iter=cfg.max_iter,
                                                penalty_weight=penalty_weight)
            return reward, info

        if cfg.perturb_mode in CONTROL_MODES:
            _run_perturbation_control(ctx, model, best, evaluate, timer, stats, sample_idx)
        else:
            _run_ddrl(ctx, model, optimizer, best, evaluate, timer, stats, sample_idx, instance_seed)

    dphi = stats['dphi_norms']
    return {
        'sample_idx': int(sample_idx),
        'best_epoch': best.epoch,
        'best_inner_epoch': best.inner_epoch,
        'solved_cost': best.cost,
        'gt_cost': gt_cost,
        'basic_cost': best.initial_cost if best.initial_cost is not None else 0,
        'penalty_count': best.penalty_count,
        'final_gap(%)': best.gap,
        'total_time(s)': timer.total['total'],
        'init_time(s)': timer.total['init'],
        'sample_time(s)': timer.total['sample'],
        'rl_time(s)': timer.total['rl'],
        'reward_time(s)': timer.total['reward'],
        'num_grad_steps': stats['grad_steps'],
        'num_reward_evals': best.num_evals,
        'peak_gpu_mem(MB)': (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                             if torch.cuda.is_available() else 0.0),
        'dphi_median': float(np.median(dphi)) if dphi else 0.0,
        'early_stopped': bool(stats.get('early_stopped', False)),
        'round_costs': ' '.join(f'{c:.6f}' for c in best.round_costs),
        'round_epochs': ' '.join(str(e) for e in best.round_epochs),
        'solved_tour': best.tour,
    }, [float(x) for x in dphi]


# ---------------------------------------------------------------------------- DDRL

def _run_ddrl(ctx, model, optimizer, best, evaluate, timer, stats, sample_idx, instance_seed):
    cfg, device = ctx.config, ctx.device
    epochs_no_improve = 0

    for epoch in range(cfg.num_epochs):
        best_before_epoch = best.cost

        # ---- sampling phase: K candidates in round 1, one afterwards
        num_candidates = cfg.num_init_sample if epoch == 0 else 1
        samples = []
        for _ in range(num_candidates):
            if epoch == 0:
                with timer('init'):
                    _initialize(model, ctx, sample_idx)

            with timer('sample'):
                latents, log_probs = sample_trajectory(
                    ctx.scheduler, ctx.prior, model, num_inference_steps=cfg.num_steps, eta=cfg.eta,
                    device=device, skip_prior=cfg.skip_prior_forward)
            latents = torch.stack(latents, dim=1)      # (1, T + 1, C, H, W)
            log_probs = torch.stack(log_probs, dim=1)  # (1, T): log pi_{phi_k}(a_tau | s_tau)
            timesteps = ctx.scheduler.timesteps.repeat(1, 1)

            reward, info = evaluate()  # r_k = r(phi_k)
            best.update(info, epoch, inner_epoch=-1)
            samples.append({
                "timesteps": timesteps,
                "latents": latents[:, :-1],       # s_tau: x_t
                "next_latents": latents[:, 1:],   # a_tau: x_{t-1}
                "log_probs": log_probs,
                "rewards": torch.as_tensor(reward, device=device),
            })
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # ---- advantages A_k (normalized over the K candidates in round 1; raw afterwards)
        rewards = samples.pop("rewards").cpu().numpy()
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8) if len(rewards) > 1 else rewards
        advantages = _apply_advantage_ablation(advantages, cfg.adv_ablation, epoch, instance_seed)
        samples["advantages"] = torch.as_tensor(advantages).to(device)

        # ---- policy update phase
        num_candidates, num_timesteps = samples["timesteps"].shape
        num_inner_epochs = cfg.num_inner_epochs if epoch == 0 else 1
        for inner_epoch in range(num_inner_epochs):
            # candidates in random order, and each candidate's steps in random order
            perm = torch.randperm(num_candidates, device=device)
            samples = {k: v[perm] for k, v in samples.items()}
            perms = torch.stack([torch.randperm(num_timesteps, device=device) for _ in range(num_candidates)])
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][torch.arange(num_candidates, device=device)[:, None], perms]

            batch_size = min(cfg.batch_size_train, num_candidates)
            batched = {k: v.reshape(-1, batch_size, *v.shape[1:]) for k, v in samples.items()}
            batched = [dict(zip(batched, x)) for x in zip(*batched.values())]

            for batch in batched:
                with timer('rl'):
                    phi_before = model.latent.detach().clone()
                    for j in range(num_timesteps):  # T optimizer steps, one per denoising step
                        if cfg.skip_prior_forward:
                            noise_pred = torch.zeros_like(batch["latents"][:, j])
                        else:
                            noise_pred = ctx.prior(batch["latents"][:, j], batch["timesteps"][:, j])
                        _, log_prob = ddim_step_with_logprob(
                            ctx.scheduler, noise_pred, batch["timesteps"][:, j], batch["latents"][:, j],
                            model, eta=cfg.eta, prev_sample=batch["next_latents"][:, j])
                        loss = ppo_clipped_loss(log_prob, batch["log_probs"][:, j], batch["advantages"],
                                                cfg.clip_range, cfg.adv_clip_max)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        stats['grad_steps'] += 1
                    stats['dphi_norms'].append(float(torch.linalg.vector_norm(
                        model.latent.detach() - phi_before)))

                _, info = evaluate()  # decode A(phi) after the update; keep-best
                best.update(info, epoch, inner_epoch)

        if cfg.use_early_stop:
            epochs_no_improve = 0 if best.cost < best_before_epoch - 1e-9 else epochs_no_improve + 1
            if epochs_no_improve >= cfg.patience:
                stats['early_stopped'] = True
                break


def _apply_advantage_ablation(advantages, mode, epoch, instance_seed):
    """Round-1 controls of the supplementary material ("Round-1 Controls").

    shuffle: permute the round-1 advantages across candidates (breaks the
        reward-to-candidate link, keeps the distribution). A local generator keeps
        the global random streams identical to the unablated run.
    zero: round 1 carries no reward signal.
    late_zero: zero rounds 2..E_max instead, where the objective is constant.
    """
    if epoch == 0 and mode == "shuffle":
        return advantages[np.random.default_rng(instance_seed + 104729).permutation(len(advantages))]
    if epoch == 0 and mode == "zero":
        return np.zeros_like(advantages)
    if epoch >= 1 and mode == "late_zero":
        return np.zeros_like(advantages)
    return advantages


# ---------------------------------------------------------------------------- controls

def _run_perturbation_control(ctx, model, best, evaluate, timer, stats, sample_idx):
    """Replace the policy update by a perturbation of phi with a matched budget.

    The number of reward evaluations is DDRL's under the same config:
    K + E_in * (K // batch_size_train) in round 1 and two per later round. Modes:
      isotropic:        phi <- phi + s * u,   u uniform on the unit sphere (random walk)
      independent:      phi <- phi_0 + s * u  (no accumulation)
      isotropic_replay: as isotropic, with DDRL's own per-update step length
      structured:       u ~ N(0, J_f^T J_f) normalized, i.e. the anisotropy of the
                        exploration without the reward guidance
    """
    cfg = ctx.config
    n_batches_ep0 = max(1, cfg.num_init_sample // min(cfg.batch_size_train, cfg.num_init_sample))
    rounds = cfg.num_init_sample + cfg.num_inner_epochs * n_batches_ep0 + (cfg.num_epochs - 1) * 2

    with timer('init'):
        _initialize(model, ctx, sample_idx)
    phi0 = model.latent.detach().clone()

    replay = None
    if cfg.perturb_mode == 'isotropic_replay':
        replay = ctx.dphi_trace.get(int(sample_idx))
        if not replay:
            raise SystemExit(f'--dphi_trace has no entry for sample_idx {int(sample_idx)}')

    for rnd in range(rounds):
        if rnd > 0:
            with timer('rl'):
                if cfg.perturb_mode == 'structured':
                    # Cov(J_f^T eps) = J_f^T J_f for eps ~ N(0, I) in image space
                    img = model.encode()
                    eps = torch.randn_like(img)
                    structured = torch.autograd.grad(img, model.latent, grad_outputs=eps)[0].detach()
                with torch.no_grad():
                    step = structured if cfg.perturb_mode == 'structured' else torch.randn_like(model.latent)
                    if replay is not None:
                        # matched on reward evaluations, not on update count, so DDRL's
                        # trace is stretched onto this control's round grid
                        scale = replay[min(int((rnd - 1) * len(replay) / max(rounds - 1, 1)), len(replay) - 1)]
                    else:
                        scale = cfg.perturb_scale
                    step *= scale / (torch.linalg.vector_norm(step) + 1e-12)
                    if cfg.perturb_mode == 'independent':
                        model.latent.copy_(phi0 + step)
                    else:
                        model.latent.add_(step)
                    stats['dphi_norms'].append(float(torch.linalg.vector_norm(step)))
                stats['grad_steps'] += 1

        _, info = evaluate()
        if best.update(info, epoch=0, inner_epoch=0):
            best.epoch = rnd  # controls report the round index of the winning evaluation
