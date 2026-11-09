"""DDIM denoising with f(phi) as the clean-image estimate, and per-step log-probabilities.

The denoising process is the policy pi_phi of the MDP (Sec. 3.1). One step is
Eq. (1) of the paper:

    x_{t-1} = sqrt(alpha_{t-1}) f(phi) + sqrt(1 - alpha_{t-1} - sigma_t^2) eps_prior(x_t, t) + sigma_t eps,

i.e. the Gaussian transition of Supp. Definition D.1,

    p_phi(x_{t-1} | x_t) = N( mu(x_t, phi), sigma_t^2 I ),
    mu(x_t, phi)         = sqrt(alpha_{t-1}) f(phi) + sqrt(1 - alpha_{t-1} - sigma_t^2) eps_prior(x_t, t),

with sigma_t^2 = eta^2 (1 - alpha_{t-1}) / (1 - alpha_t) (1 - alpha_t / alpha_{t-1})
and eta = 1 (Supp. Sec. A). alpha_t is the cumulative product of the schedule
(`alphas_cumprod` in diffusers).

`ddim_step_with_logprob` and `sample_trajectory` are adapted from DDPO
(Black et al., "Training Diffusion Models with Reinforcement Learning", ICLR 2024;
https://github.com/kvablack/ddpo-pytorch, MIT License,
ddpo_pytorch/diffusers_patch/{ddim_with_logprob,pipeline_with_logprob}.py),
which patch the DDIM step of diffusers (Apache License 2.0). DDRL's changes:
the clean-image estimate is f(phi) instead of the U-Net's prediction, the
pipeline is reduced to the scheduler, and the prior's forward pass can be
skipped (see `sample_trajectory`).
"""
import math

import torch
from diffusers import DDIMScheduler
from diffusers.utils import randn_tensor

# DDIM scheduler of Stable Diffusion v1.5 (scheduler/scheduler_config.json of
# runwayml/stable-diffusion-v1-5). The scheduler is the only part of that model
# DDRL uses: the noise predictor is the TSP prior and no VAE or text encoder is
# involved. 1000 training steps, scaled-linear betas; T = 50 visited steps.
SD15_SCHEDULER_CONFIG = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "num_train_timesteps": 1000,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "trained_betas": None,
    "clip_sample": False,
}


def build_scheduler():
    return DDIMScheduler.from_config(SD15_SCHEDULER_CONFIG)


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(scheduler, timestep, prev_timestep):
    """sigma_t^2 / eta^2 = (1 - alpha_{t-1}) / (1 - alpha_t) * (1 - alpha_t / alpha_{t-1})."""
    alpha_prod_t = torch.gather(scheduler.alphas_cumprod, 0, timestep.cpu().type(torch.int64)).to(
        timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        scheduler.alphas_cumprod.gather(0, prev_timestep.cpu().type(torch.int64)),
        scheduler.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    return (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)


def ddim_step_with_logprob(scheduler: DDIMScheduler, model_output, timestep, sample, model,
                           eta=0.0, generator=None, prev_sample=None):
    """One step of Eq. (1) and the log-density of the resulting x_{t-1}.

    Args:
        scheduler: DDIMScheduler with `set_timesteps` already called.
        model_output: eps_prior(x_t, t) (zeros when the prior forward is skipped).
        timestep: (B,) current t.
        sample: x_t.
        model: LatentTourModel; `model.encode()` supplies f(phi).
        eta: stochasticity (1 in the paper).
        prev_sample: if given, the log-probability of this x_{t-1} is returned
            (used by the policy update); otherwise x_{t-1} is sampled.

    Returns:
        (x_{t-1}, log p_phi(x_{t-1} | x_t)), the log-density averaged over pixels, shape (B,).
    """
    assert isinstance(scheduler, DDIMScheduler)
    if scheduler.num_inference_steps is None:
        raise ValueError("call scheduler.set_timesteps before stepping")

    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timestep = torch.clamp(prev_timestep, 0, scheduler.config.num_train_timesteps - 1)

    # alpha_t and alpha_{t-1} (cumulative products)
    alpha_prod_t = scheduler.alphas_cumprod.gather(0, timestep.cpu().type(torch.int64))
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        scheduler.alphas_cumprod.gather(0, prev_timestep.cpu().type(torch.int64)),
        scheduler.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)

    # DDRL: the clean-image estimate x_0 is f(phi), not the network's prediction
    pred_original_sample = model.encode()
    pred_epsilon = model_output

    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)

    variance = _get_variance(scheduler, timestep, prev_timestep)
    std_dev_t = eta * variance ** 0.5                     # sigma_t
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    # mu(x_t, phi) of Supp. Definition D.1
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * pred_epsilon
    prev_sample_mean = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

    if prev_sample is not None and generator is not None:
        raise ValueError("pass either generator or prev_sample, not both")

    if prev_sample is None:
        variance_noise = randn_tensor(model_output.shape, generator=generator,
                                      device=model_output.device, dtype=model_output.dtype)
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # Gaussian log-density log N(x_{t-1}; mu, sigma_t^2 I), averaged over pixels
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t ** 2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), log_prob


@torch.no_grad()
def sample_trajectory(scheduler, prior, model, num_inference_steps=50, eta=1.0,
                      device=None, skip_prior=False):
    """Draw one trajectory x_T -> x_0 under pi_phi (the sampling phase, Algorithm 1).

    x_T is the model's fixed noise `model.xT`. With `skip_prior=True` the prior's
    eps_prior(x_t, t) is replaced by zeros. This leaves the policy update
    unchanged: both the sampling and the evaluated policy use the same
    eps_prior(x_t, t), so it cancels from the likelihood ratio (Supp. Lemma D.2),
    and the forward pass consumes no random numbers. The paper's runs use it.

    Returns:
        latents: list of T + 1 tensors [x_T, ..., x_0].
        log_probs: list of T tensors, log pi_phi(a_tau | s_tau) per step.
    """
    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = model.xT
    batch_size = latents.shape[0]

    all_latents = [latents]
    all_log_probs = []
    for t in scheduler.timesteps:
        t = t.float().view(batch_size)
        if skip_prior:
            noise_pred = torch.zeros_like(latents)
        else:
            noise_pred = prior(latents, t)
        latents, log_prob = ddim_step_with_logprob(scheduler, noise_pred, t, latents, model, eta=eta)
        all_latents.append(latents)
        all_log_probs.append(log_prob)
    return all_latents, all_log_probs
