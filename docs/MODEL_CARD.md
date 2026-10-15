# Model card: Geo Enhance residual reconstruction model

## Intended use

Research and prototyping with continuous optical satellite rasters. The pipeline
supports synthetic blur/downsample training and aligned LR/HR-pair training. It is
not a general-purpose pretrained satellite model. Categorical maps, alpha/palette
bands, complex SAR and domain-specific physical inversions are outside scope.

## Model and data

Residual CNN, pixel-shuffle upsampling, bicubic skip, no batch normalization.
Defaults: 64 features, eight residual blocks, 4× per-axis scaling. Training uses
Charbonnier and spatial-gradient losses. Paired observations receive synchronized
rotation/flip augmentation; synthetic pairs additionally use randomized blur/noise.
The default degradation is approximate and is not a calibrated sensor model.

## Evidence and limitations

No independent real-satellite benchmark has been performed. The local demo uses
artificial patterns and must not be used as production weights. Validation selects
the best model, so final performance requires an untouched test region. Geographic
bounding-box separation reduces leakage but cannot verify acquisition independence,
cloud quality, co-registration, spectral response or product correctness.

Nominal 10 m pixels do not establish effective 10 m resolution. Never infer newly
observed buildings, roads, boundaries or physical measurements from sharpened
appearance alone. There is no calibrated uncertainty map. Pixels near missing
data depend on nearest-valid computational context and are less reliable.

## Before applying a trained model

- Record the sensor, product level, bands/order, units, offsets and native resolution.
- Use geographically separated training/validation/test scenes with compatible dates.
- Mask clouds/shadows; co-register real pairs externally and inspect alignment.
- Evaluate against an independent reference, including per-band errors and spectral angle.
- Compare bicubic and the model under the same inference settings, including ensembling.
- Validate the actual downstream task before relying on model-derived measurements.

New checkpoints retain normalization and band metadata; older checkpoints without
these fields remain readable but cannot provide the same compatibility checks.
Use trusted checkpoints only. Restricted weight loading reduces pickle risk but is
not a resource sandbox for hostile model files.
