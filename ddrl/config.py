"""Command-line interface: YAML defaults overridden by flags."""
import argparse

import yaml

from ddrl.utils import Config

PERTURB_MODES = ("ddrl", "isotropic", "independent", "isotropic_replay", "structured")
CONTROL_MODES = PERTURB_MODES[1:]  # every mode except the DDRL update itself


def build_parser():
    p = argparse.ArgumentParser(description="DDRL: instance-wise diffusion-driven RL for the TSP")
    BOOL = argparse.BooleanOptionalAction

    g = p.add_argument_group("instances")
    g.add_argument("--constraint_type", default="basic", choices=["basic", "box", "path", "cluster"],
                   help="basic TSP or one of the constrained benchmarks (box = obstacle)")
    g.add_argument("--num_cities", type=int, default=20)
    g.add_argument("--start_idx", type=int, default=0, help="first instance (inclusive)")
    g.add_argument("--end_idx", type=int, default=1, help="last instance (exclusive)")
    g.add_argument("--data_path", default="./data",
                   help="directory with tsp{N}_test_concorde.txt and the constrained files")
    g.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")

    g = p.add_argument_group("method")
    g.add_argument("--config", default="./configs/default.yaml", help="YAML with the default settings")
    g.add_argument("--init_mode", default="prior", choices=["prior", "difusco", "none"],
                   help="initialization of phi: plug-and-play prior sampling (DDRL-prior), a DIFUSCO "
                        "heatmap (DDRL-DIFUSCO), or random N(0, I)")
    g.add_argument("--img_size", type=int, default=64, help="canvas size of the edge images")
    g.add_argument("--use_prior", action=BOOL, default=True,
                   help="load the pre-trained prior weights (otherwise a randomly initialized U-Net)")
    g.add_argument("--skip_prior_forward", action=BOOL, default=True,
                   help="replace eps_prior(x_t, t) by zeros inside the RL stage; the prior's term cancels "
                        "from the update (Supp. Lemma D.2), so results are identical and runs are faster")
    g.add_argument("--ckpt_path", default="./ckpt", help="directory with the prior checkpoints")
    g.add_argument("--prior_ckpt", default=None,
                   help="prior checkpoint file under --ckpt_path (default: chosen by --num_cities)")
    g.add_argument("--heatmap_path", default="./heatmaps",
                   help="DIFUSCO heatmaps {heatmap_path}/tsp{N}/numpy_heatmap/test-heatmap-{i}.npy")

    g = p.add_argument_group("overrides of configs/default.yaml")
    for name, typ in [("seed", int), ("num_init_sample", int), ("num_epochs", int),
                      ("num_inner_epochs", int), ("num_steps", int), ("patience", int),
                      ("max_iter", int), ("penalty_const", float), ("batch_size_train", int)]:
        g.add_argument(f"--{name}", type=typ, default=None)
    g.add_argument("--use_early_stop", action=BOOL, default=None)

    g = p.add_argument_group("controls (Sec. 4.3 and the supplementary Extended Ablation)")
    g.add_argument("--perturb_mode", default="ddrl", choices=PERTURB_MODES,
                   help="ddrl: the policy-gradient update. The others replace it by a perturbation "
                        "of phi with the same number of reward evaluations: isotropic (random walk), "
                        "independent (resample around the initial phi), isotropic_replay (random "
                        "direction, DDRL's per-round step length from --dphi_trace), structured "
                        "(direction ~ N(0, J_f^T J_f), no reward signal)")
    g.add_argument("--perturb_scale", type=float, default=None,
                   help="step length of the perturbation controls; match it to the dphi_median "
                        "column of the DDRL run")
    g.add_argument("--dphi_trace", default=None,
                   help="JSON written by --dump_dphi; required by --perturb_mode isotropic_replay")
    g.add_argument("--dump_dphi", action="store_true",
                   help="write the per-update ||delta phi|| of every instance next to the result CSV")
    g.add_argument("--adv_ablation", default="none", choices=["none", "shuffle", "zero", "late_zero"],
                   help="shuffle / zero: permute or zero the round-1 advantages; "
                        "late_zero: zero the advantages of rounds 2..E_max")

    g = p.add_argument_group("output")
    g.add_argument("--run_name", default="ddrl", help="results go to Results/<constraint>/<run_name>/")
    return p


def load_config(argv=None):
    args = build_parser().parse_args(argv)
    with open(args.config) as f:
        config = Config(yaml.safe_load(f))
    for key, value in vars(args).items():
        if value is not None or key not in config:
            config[key] = value

    if config.perturb_mode == "isotropic_replay" and not config.dphi_trace:
        raise SystemExit("--perturb_mode isotropic_replay requires --dphi_trace "
                         "(the per-update ||delta phi|| written by a DDRL run with --dump_dphi)")
    if config.perturb_mode in ("isotropic", "independent", "structured") and config.perturb_scale is None:
        raise SystemExit(f"--perturb_scale is required with --perturb_mode {config.perturb_mode} "
                         "(use the DDRL run's dphi_median so both move equally far)")
    return config
