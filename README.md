# GENERIC GPU PROJECT

## Overview

GENERIC GPU PROJECT (GGP) is a library for performing calculations on graphics
processing units (GPUs), leveraging NVIDIA's CUDA platform, as well as support
for AMD, INTEL, and CPU architectures. It is a 'pruned' version of the
[QUDA library](https://github.com/lattice/quda) which itself is a highly optimised library for
lattice gauge theory computations, capable of running accross different architectures.

At this development stage we reatain the much of the QUDA, Quda, and quda naming conventions
in the C++ code so that we may track any future addition to the QUDA library and
integerate with ease. This will change at a future date.

In making this library we attempt to leverarge several desirable
aspects of QUDA while leaving the library open to more than lattice gauge
theory application codes. The aspects we retain are:

1. Robust mixed-precision combinations of double, single, half and quarter
precisions (where the latter two are 16-bit and 8-bit "block floating
point", respectively).

2. Use of many GPUs in parallel is supported throughout, with communication
handled by QMP or MPI.

3. Automatic kernel profiling and optimisation via the autotuner.

4. Power monitoring for the GPUs, and in future, individual kernels

5. A robust cross platform and cross compilier build and installation system

6. Robust error checking, memory management, and API interaction.


## Software Compatibility:

The library has been tested under Linux (CentOS 7 and Ubuntu 18.04)
using releases 10.1 through 11.4 of the CUDA toolkit.  Earlier versions
of the CUDA toolkit will not work, and we highly recommend the use of
11.x.  QUDA has been tested in conjunction with x86-64, IBM
POWER8/POWER9 and ARM CPUs.  Both GCC and Clang host compilers are
supported, with the minimum recommended versions being 7.x and 6, respectively.
CMake 3.15 or greater to required to build QUDA.


## Hardware Compatibility:

For a list of supported devices, see

http://developer.nvidia.com/cuda-gpus

Before building the library, you should determine the "compute
capability" of your card, either from NVIDIA's documentation or by
running the deviceQuery example in the CUDA SDK, and pass the
appropriate value to the `GGP_GPU_ARCH` variable in cmake.

See also "Known Issues" below.

## Installation:

It is recommended to build GGP in a separate directory from the
source directory.  For instructions on how to build QUDA using cmake
see this page
https://gitlab.com/dsa-2000/code/generic-GPU-project/-/wikis/GGP%20Build%20With%20CMake. Note
that this requires cmake version 3.15 or later. You can obtain cmake
from https://cmake.org/download/. On Linux the binary tar.gz archives
unpack into a cmake directory and usually run fine from that
directory.

The basic steps for building with cmake are:

1. Create a build dir, outside of the quda source directory. 
2. In your build-dir run `cmake <path-to-GGP-src>` 
3. It is recommended to set options by calling `ccmake` in
your build dir. Alternatively you can use the `-DVARIABLE=value`
syntax in the previous step.
4. run 'make -j <N>' to build with N
parallel jobs. 

You are most likely to want to specify the GPU architecture of the
machine you are building for. Either configure `GGP_GPU_ARCH` in step 3
or specify e.g. -DGGP_GPU_ARCH=sm_60 for a Pascal GPU in step 2.

### Multi-GPU support

GGP supports using multiple GPUs through MPI and QMP, together with
the optional use of NVSHMEM GPU-initiated communication for improved
strong scaling of the Dirac operators.  To enable multi-GPU support
either set `GGP_MPI` or `GGP_QMP` to ON when configuring GGP
through cmake.

Note that in any case cmake will automatically try to detect your MPI
installation. If you need to specify a particular MPI please set
`MPI_C_COMPILER` and `MPI_CXX_COMPILER` in cmake.  See also
https://cmake.org/cmake/help/v3.9/module/FindMPI.html for more help.

For QMP please set `GGP_QMP_HOME` to the installation directory of QMP.

For more details see https://github.com/lattice/quda/wiki/Multi-GPU-Support
(DMH: FIXME to be GGP specific.)

To enable NVSHMEM support set `QUDA_NVSHMEM` to ON, and set the
location of the local NVSHMEM installation with `QUDA_NVSHMEM_HOME`.
For more details see
https://github.com/lattice/quda/wiki/Multi-GPU-with-NVSHMEM
(DMH: FIXME to enable GGP NVSHMEM.)

## Tuning

Throughout the library, auto-tuning is used to select optimal launch
parameters for most performance-critical kernels.  This tuning process
takes some time and will generally slow things down the first time a
given kernel is called during a run.  To avoid this one-time overhead in
subsequent runs the optimal parameters are cached to disk.  For this to work, the
`GGP_RESOURCE_PATH` environment variable must be set, pointing to a
writable directory.  Note that since the tuned parameters are hardware-
specific, this "resource directory" should not be shared between jobs
running on different systems (e.g., two clusters with different GPUs
installed).  Attempting to use parameters tuned for one card on a
different card may lead to unexpected errors.

This autotuning information can also be used to build up a first-order
kernel profile: since the autotuner measures how long a kernel takes
to run, if we simply keep track of the number of kernel calls, from
the product of these two quantities we have a time profile of a given
job run.  If `GGP_RESOURCE_PATH` is set, then this profiling
information is output to the file "profile.tsv" in this specified
directory.  Optionally, the output filename can be specified using the
`GGP_PROFILE_OUTPUT` environment variable, to avoid overwriting
previously generated profile outputs.  In addition to the kernel
profile, a policy profile, e.g., collections of kernels and/or other
algorithms that are auto-tuned, is also output to the file
"profile_async.tsv".  The policy profile for example includes
the entire multi-GPU dslash, whose style and order of communication is
autotuned.  Hence while the dslash kernel entries appearing the kernel
profile do include communication time, the entries in the policy
profile include all constituent parts (halo packing, interior update,
communication and exterior update).

## Using the Library:

Include the header file include/ggp.h in your application, link against
lib/libggp.so.


## CUTLASS Interface Build Configuration (GB10 / SM120)

The FTD XEngine uses the `cutlass_interface` library for HERK (Hermitian rank-K update)
correlation. On GB10 (SM120), the CUTLASS library must be built with HERK-optimized
settings because SM120 has only 99 KB shared memory per block.

**Build `libcutlass_gemm_api.a` for FTD/XEngine:**

```bash
cd /path/to/cutlass_interface
mkdir build && cd build
cmake .. -DCUTLASS_DIR=/path/to/cutlass \
         -DCOMPLEX_FP8_ARCH=120a \
         -DCOMPLEX_SM100_ENABLE_FP6=ON \
         -DCOMPLEX_SM100_ENABLE_FP4=ON \
         -DCOMPLEX_FP8_SM100_STAGES=3
make cutlass_gemm_api -j
```

This auto-selects the 128×64 FP8 tile with 3-stage pipeline (86 KB SharedStorage).
The XEngine's HERK path uses a custom direct kernel (24 KB SMEM) that bypasses
CUTLASS tiles entirely, so HERK throughput is unaffected by the FP8 tile choice.

> **Note:** Dedispersion and voltage beamforming workloads use standalone GEMM (not HERK)
> and benefit from a wider 128×128 FP8 tile. If building the CUTLASS library for those
> consumers, add `-DCOMPLEX_SM100_FP8_TILE_N=128`. See the `cutlass_interface/README.md`
> and `dedisp_tcfdd/README.md` for details.

See `cutlass_interface/README.md` § "Production Deployment Summary" for full benchmark
data, runtime parameters, and the dual-build rationale.

## Known Issues:

* When the auto-tuner is active in a multi-GPU run it may cause issues
with binary reproducibility of this run if domain-decomposition
preconditioning is used. This is caused by the possibility of
different launch configurations being used on different GPUs in the
tuning run simultaneously. If binary reproducibility is strictly
required make sure that a run with active tuning has completed. This
will ensure that the same launch configurations for a given kernel is
used on all GPUs and binary reproducibility.

## Authors:

This author list is inherited from QUDA as of Aug 20 2024 and additions will be
made based on contributions to GGP.

### QUDA Contributors

*  Ronald Babich (NVIDIA)
*  Simone Bacchio (Cyprus)
*  Michael Baldhauf (Regensburg)
*  Kipton Barros (Los Alamos National Laboratory)
*  Richard Brower (Boston University) 
*  Nuno Cardoso (NCSA) 
*  Kate Clark (NVIDIA)
*  Michael Cheng (Boston University)
*  Carleton DeTar (Utah University)
*  Justin Foley (NIH)
*  Arjun Gambhir (William and Mary)
*  Marco Garofalo (HISKP, University of Bonn)
*  Joel Giedt (Rensselaer Polytechnic Institute) 
*  Steven Gottlieb (Indiana University) 
*  Anthony Grebe (Fermilab)
*  Kyriakos Hadjiyiannakou (Cyprus)
*  Ben Hoerz (Intel)
*  Dean Howarth (Caltech, Department of Astronomy)
*  Hwancheol Jeong (Indiana University)
*  Xiangyu Jiang (ITP, Chinese Academy of Sciences)
*  Balint Joo (OLCF, Oak Ridge National Laboratory, formerly Jefferson Lab)
*  Hyung-Jin Kim (Samsung Advanced Institute of Technology)
*  Bartosz Kostrzewa (HPC/A-Lab, University of Bonn)
*  Damon McDougall (AMD)
*  Colin Morningstar (Carnegie Mellon University)
*  James Osborn (Argonne National Laboratory)
*  Ferenc Pittler (Cyprus)
*  Claudio Rebbi (Boston University) 
*  Eloy Romero (William and Mary)
*  Hauke Sandmeyer (Bielefeld)
*  Mario Schröck (INFN)
*  Aniket Sen (HISKP, University of Bonn)
*  Guochun Shi (NCSA)
*  James Simone (Fermi National Accelerator Laboratory)
*  Alexei Strelchenko (Fermi National Accelerator Laboratory)
*  Jiqun Tu (NVIDIA)
*  Carsten Urbach (HISKP, University of Bonn)
*  Alejandro Vaquero (Utah University)
*  Michael Wagman (Fermilab)
*  Mathias Wagner (NVIDIA)
*  Andre Walker-Loud (Lawrence Berkley Laboratory)
*  Evan Weinberg (NVIDIA)
*  Frank Winter (Jefferson Lab)
*  Yi-Bo Yang (ITP, Chinese Academy of Sciences)

### GGP Contributors

* Dean Howarth (Caltech, Department of Astronomy)
* Vikram Ravi (Caltech, Department of Astronomy)

Portions of this software were developed at the Innovative Systems Lab,
National Center for Supercomputing Applications
http://www.ncsa.uiuc.edu/AboutUs/Directorates/ISL.html

Development was supported in part by the U.S. Department of Energy under
grants DE-FC02-06ER41440, DE-FC02-06ER41449, and DE-AC05-06OR23177; the
National Science Foundation under grants DGE-0221680, PHY-0427646,
PHY-0835713, OCI-0946441, and OCI-1060067; as well as the PRACE project
funded in part by the EUs 7th Framework Programme (FP7/2007-2013) under
grants RI-211528 and FP7-261557.  Any opinions, findings, and
conclusions or recommendations expressed in this material are those of
the authors and do not necessarily reflect the views of the Department
of Energy, the National Science Foundation, or the PRACE project.

