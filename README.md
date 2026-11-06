# ProDM — Property-aware Progressive Correction Diffusion Model

Coronary artery calcium (CAC) scoring from chest CT helps refine cardiovascular
risk estimation, but accurate quantification depends on clean delineation of
calcified lesions. ECG-gated cardiac CT minimizes motion artifacts yet is not
widely available, while routine **non-gated** chest CT — an accessible alternative
for incidental CAC — suffers from more severe cardiac and respiratory motion.
**ProDM** is a generative diffusion framework that restores motion-free calcified
lesions from non-gated CT, improving CAC scoring accuracy, lesion fidelity, and
risk stratification.

Because real paired (motion-corrupted / motion-free) scans do not exist, ProDM is
trained on **simulated** non-gated acquisitions synthesized from cardiac-gated CTs.
The repository implements this as a four-stage pipeline; each directory has its own
README with details and usage.

## Pipeline

| Stage | Directory | In → Out |
|---|---|---|
| 0. Raw data | — | Gated (motion-free) cardiac CT volumes in HU + per-artery coronary-calcium masks, this comes from the Stanford COCA dataset|
| 1. Motion simulation | [`motion_simulation/`](motion_simulation/) | Gated volume + mask → synthesized non-gated-style motion-corrupted reconstructions (only the calcium deposits are displaced), giving paired training data |
| 2. Data preprocessing | [`data_preprocessing/`](data_preprocessing/) | Recon + gated + mask → HU-normalized calcium patch datasets (train set + optimized validation shortlist) |
| 3. Train / evaluate | [`src/`](src/) | Patch datasets → trained ProDM checkpoints, motion-suppressed predictions, and evaluation based on Agatston scores and calcified lesion properties|

ProDM's property-aware learning (a differentiable calcium-consistency loss) and
progressive correction across diffusion steps live in the training/evaluation code
under [`src/`](src/).

## Note

The underlying diffusion model (BBDM) package is not vendored here — see
[`src/README.md`](src/README.md) for obtaining it and placing it under `src/model/`.
