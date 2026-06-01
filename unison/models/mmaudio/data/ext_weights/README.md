# MMAudio VAE weights

Place the following files here before running UNISON:

| File | Description |
|------|-------------|
| `v1-44.pth` | MMAudio ToD VAE, 44 kHz (used by default, paper results) |
| `v1-16.pth` | MMAudio ToD VAE, 16 kHz |
| `best_netG.pt` | BigVGAN vocoder checkpoint for 16 kHz decoding |

Download from the [MMAudio](https://github.com/hkchengrex/MMAudio) project or copy from a local installation.

For the 44 kHz config (`vae_config_44k.yaml`), the BigVGAN vocoder is loaded
automatically from HuggingFace (`nvidia/bigvgan_v2_44khz_128band_512x`) when
`bigvgan_vocoder_ckpt` is `null`.
