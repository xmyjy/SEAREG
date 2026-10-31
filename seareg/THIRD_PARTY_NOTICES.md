# Third-party components

## NVIDIA CUDA Toolkit ptxas 11.8.89

SEAREG includes NVIDIA Corporation's unmodified Linux x86_64 `ptxas` executable
at `_compiler/cuda11/ptxas` for the optional CUDA 11 scoring path. This NVIDIA
component is distributed under the separate NVIDIA CUDA Toolkit End User
License Agreement, including its supplement and third-party notices. It is
not covered by SEAREG's MIT license.

The complete terms supplied with NVIDIA's official CUDA 11.8 archive are
included at [`_compiler/cuda11/LICENSE`](_compiler/cuda11/LICENSE).
The corresponding official documentation is the
[CUDA Toolkit 11.8 EULA](https://docs.nvidia.com/cuda/archive/11.8.0/eula/index.html).

The bundled executable is 19,369,040 bytes with SHA-256
`c0ad17863f2b18bba806c39381977df75ff5f255855ddaed44f0e7206dc69166`.
Its contents match `bin/ptxas` in NVIDIA's official
[`cuda_nvcc-linux-x86_64-11.8.89-archive.tar.xz`](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-11.8.89-archive.tar.xz).
NVIDIA publishes the archive's checksum in the
[CUDA 11.8 redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_11.8.0.json).
Component, archive, binary and license identities are recorded in
[`_compiler/cuda11/provenance.json`](_compiler/cuda11/provenance.json).
