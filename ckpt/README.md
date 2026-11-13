# Checkpoints

Nothing in this directory is tracked by git. Place the files listed below here
(or point `--ckpt_path` elsewhere). Checksums are MD5.

## Diffusion priors (DDRL-prior and the `--use_prior` initialization)

| file | used for | origin | md5 |
|---|---|---|---|
| `unet50_64_8.pth` | `N <= 200`, 64x64 images | Graikos et al., *Diffusion models as plug-and-play priors* (NeurIPS 2022): <https://diffusion-priors.s3.amazonaws.com/unet50_64_8.pth> | `4cc6f50d2fecbbfb4ea0b18bbebb53c2` |
| `unet500_128_8.pth` | `N >= 500`, 128x128 images (loaded at 96x96 for `N = 1000`) | trained by us with the diffusion_priors training script on TSP-500 instances; attached to this repository's GitHub release | `e01dfe8bcca191735873b075064d11e4` |

`main.py` picks the first for `--num_cities <= 200` and the second above that;
`--prior_ckpt <file>` overrides the choice.

## DIFUSCO heatmaps (DDRL-DIFUSCO, `--init_mode difusco`)

DDRL-DIFUSCO does not run DIFUSCO itself. It reads heatmaps produced beforehand by
DIFUSCO's inference script and expects them at

    {heatmap_path}/tsp{N}/numpy_heatmap/test-heatmap-{i}.npy      (i = instance index)

Produce them with the official DIFUSCO repository
(<https://github.com/Edward-Sun/DIFUSCO>, commit `65eb3b7`), running
`difusco/train.py --do_test --save_numpy_heatmap` on the same
`tsp{N}_test_concorde.txt` file with the checkpoint for that size and the
per-size inference settings of its `reproducing_scripts.md`
(`--sparse_factor 50` at `N = 500`, `100` at `N = 1000`; dense below that).

| size | checkpoint | origin | md5 |
|---|---|---|---|
| 50, 100, 500, 1000 | `tsp{N}_categorical.ckpt` | DIFUSCO's released checkpoints (Google Drive folder linked from its README) | see that README |
| 20 | `tsp20_categorical.ckpt` | trained by us with DIFUSCO's own `train.py` (50 epochs, batch 128, lr 2e-4, cosine decay) because no official checkpoint exists at this size; attached to the GitHub release | `a30c481bffd5cfcda0dd0cd0f7144b53` |
| 200 | `tsp100_categorical.ckpt` decoded zero-shot | the official `N = 100` checkpoint applied to `N = 200` instances; no checkpoint is distributed for 200 | see that README |

The paper's Table 1 and Table 2 rows labelled DIFUSCO are these heatmaps decoded
once (greedy + 2-opt); the DDRL-DIFUSCO rows run DDRL's search from the same
heatmaps.
