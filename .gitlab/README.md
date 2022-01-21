# GitLab CI

## Pipelines

The following pipelines are supported:

- Merge Request - runs when a merge request is created/updated. Note that this
  pipeline won't run for draft MRs. The same pipeline is also run when the
  default branch is updated (e.g. when an MR is merged).
- Nightly - pipeline used for nightly releases. This pipeline is scheduled to
  run daily. It can also be triggered manually.
- RISC-V - pipeline used for testing on the RISC-V simulator with ComputeAorta.
  This pipeline can be triggered manually.

## Nightly

The nightly pipeline is scheduled to run automatically. It can also be run
manually. It can be triggered
[here](https://git.office.codeplay.com/research-and-development/sycl-onnx/sycl-onnx-runtime/onnxruntime/-/pipelines/new)
by defining variable `ORT_PIPELINE` with value `nightly`.

The nightly pipeline deploys an onnxruntime package. Packages can be found in
the [Package
Registry](https://git.office.codeplay.com/research-and-development/sycl-onnx/sycl-onnx-runtime/onnxruntime/-/packages).

## RISC-V

The RISC-V pipeline is run manually. It can be triggered
[here](https://git.office.codeplay.com/research-and-development/sycl-onnx/sycl-onnx-runtime/onnxruntime/-/pipelines/new)
by defining the variable `ORT_PIPELINE` with value `riscv`.

## Additional options

The `nightly` and `riscv` pipelines support the following additional variables:

- `COMPUTECPP_VERSION` - which release of ComputeCpp to use. This variable
  should be set to the direct URL to download the .tar.gz package.
- `COMPUTEAORTA_URL` - which release of ComputeAorta to use. This variable
  should be set to the direct URL to download the .tar.gz package.
- `SYCLBLAS_URL` - the URL of the SYCL-BLAS repository to clone.
- `SYCLDNN_URL` - the URL of the SYCL-DNN repository to clone.
- `SYCLBLAS_COMMIT` - which commit of sycl-blas to use. The value should be set
  to the SHA of the commit.
- `SYCLDNN_COMMIT` - which commit of SYCL-DNN to use. The value should be set to
  the SHA of the commit.
