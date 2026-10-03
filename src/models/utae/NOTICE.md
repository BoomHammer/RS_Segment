Adapted from VSainteuf/utae-paps, MIT license (see LICENSE).
Upstream commit: 987874e27a98bb399277f2d55eb010744d26f881.
https://github.com/VSainteuf/utae-paps/tree/987874e27a98bb399277f2d55eb010744d26f881

Retained the full U-TAE convolutional encoder, bottleneck L-TAE, grouped
attention skip aggregation and convolutional decoder. Removed unrelated
RecUNet/ConvLSTM code. Imports, default tuples and formatting were adapted.
Positional denominators are device-following buffers. L-TAE accepts spatial
validity masks and zeroes attention for fully absent time series.
Attention normalization uses FP32 for FP16 stability. Positional encodings
are shared across pixels and queries use tensor expansion instead of a
Python list of repeated parameters; model parameters and scales are unchanged.

The hybrid wrapper in ../pretrained_utae.py uses the original blocks with
frame-chunk checkpointing, removes padded dates per sample, supplies missing
feature indicators and renormalizes masked grouped skips. It retains all
encoder and decoder scales; its final classifier is replaced by multiscale
static fusion, a SegFormer MLP decoder and the project's hierarchical heads.
Frame checkpoint inputs use the active autocast dtype. Fully present sequences
use views, and grouped skips are recomputed during backward to bound memory.
It does not claim to be a bitwise reproduction of the PASTIS experiment, or
to load PASTIS U-TAE pretrained weights (the input channels differ).
