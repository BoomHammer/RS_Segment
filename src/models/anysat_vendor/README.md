# Official AnySat components

Source: https://github.com/gastruc/AnySat
Commit: `5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a`
Upstream paths: `src/models/networks/encoder/utils/{utils,utils_ViT,irpe,pos_embed}.py`.
License: MIT, reproduced in `LICENSE`.

Only the attention classes used by the downstream encoder are retained from
`utils_ViT.py`. Imports use this package's namespace; Ruff formatting and lint
fixes remove unused variables, mutable defaults and an empty optional-extension
warning. Mathematical operators and parameter names are preserved.

Project-specific masking, SDPA, spatial chunking and checkpointing live in
`../anysat_core.py`; project-specific sensor projectors live in `../anysat.py`.
The official unmasked forwards remain available for numerical regression tests.
