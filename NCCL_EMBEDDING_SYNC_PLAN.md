# NCCL Embedding Synchronization Plan

## Goal

Implement optional CUDA-aware embedding synchronization backends while keeping the current MPI backend as the default and as the correctness/performance baseline.

The first target adds two GPU-buffer full-sync backends:

- `mpi-cuda`: CUDA-aware MPI `Allreduce` directly on `d_syn0`.
- `nccl`: NCCL `AllReduce` directly on `d_syn0`.

The default `mpi` backend remains the existing selected-row host-buffer MPI path. Selective GPU-row synchronization, hierarchical communicators, and topology-aware scheduling are later extensions and should not be mixed into this pass.

## Current Baseline

- Process launch and distributed runtime use MPI.
- Graph loading, graph shuffle, walker/message exchange, barriers, evaluation flags, and statistics use MPI.
- GPU training keeps embeddings in `d_syn0`.
- Current default embedding synchronization uses MPI on a host staging buffer for selected embedding rows.
- The `mpi-cuda` and `nccl` backends should replace only the embedding synchronization path in the first stage.

## Branch

Work branch:

```bash
feature/nccl-embedding-sync
```

Baseline branch:

```bash
feat-compress
```

## Non-Goals For First Pass

- Do not replace MPI as the launcher or general communication substrate.
- Do not rewrite graph/walker communication.
- Do not implement selective row synchronization yet.
- Do not implement node-local/inter-node hierarchical NCCL yet.
- Do not change corpus compression or sampling behavior.

## Implementation Plan

### 1. Add Build-Time NCCL And CUDA-Aware MPI Support

- Add a CMake option:

```cmake
option(WITH_NCCL "Enable NCCL embedding synchronization" OFF)
option(WITH_MPI_CUDA "Enable CUDA-aware MPI embedding synchronization" OFF)
```

- When `WITH_NCCL=ON`, verify NCCL headers and library are available.
- Link `nccl` only for the CUDA `felog` target.
- Define `WITH_NCCL` and `WITH_MPI_CUDA` for conditional compilation.
- Keep the default build unchanged when both options are `OFF`.

Acceptance checks:

- `cmake ..` without NCCL or CUDA-aware MPI still succeeds.
- `cmake -DWITH_MPI_CUDA=ON ..` succeeds with a CUDA-aware MPI toolchain.
- `cmake -DWITH_NCCL=ON ..` either succeeds with NCCL or fails with a clear error.

### 2. Add Runtime Sync Backend Selection

- Add a runtime option:

```bash
--sync-backend mpi
--sync-backend mpi-cuda
--sync-backend nccl
```

- Default should be `mpi`.
- If `--sync-backend mpi-cuda` is requested in a binary built without `WITH_MPI_CUDA`, fail early with a clear message.
- If `--sync-backend nccl` is requested in a binary built without `WITH_NCCL`, fail early with a clear message.
- Log the selected sync backend at startup.

Acceptance checks:

- Existing README command still runs without adding `--sync-backend`.
- Passing `--sync-backend mpi` uses the existing MPI path.
- Passing `--sync-backend mpi-cuda` uses CUDA-aware MPI full `d_syn0` sync when compiled in.
- Passing `--sync-backend nccl` uses NCCL full `d_syn0` sync when compiled in.
- Passing an invalid backend exits with a readable error.

### 3. Extract Embedding Sync Interface

Introduce a small internal abstraction around embedding synchronization:

```cpp
init_embedding_sync();
sync_embedding();
finalize_embedding_sync();
```

Expected behavior:

- MPI backend preserves current behavior.
- MPI-CUDA backend uses the same MPI communicator but passes CUDA device memory to MPI.
- NCCL backend owns NCCL initialization and cleanup.
- Existing training loop should call the abstracted sync function rather than directly calling MPI synchronization logic.

Acceptance checks:

- MPI behavior remains unchanged after refactor.
- No NCCL symbols are required in non-NCCL builds.
- No MPI-CUDA backend is selectable unless the binary is built with `WITH_MPI_CUDA=ON`.

### 4. Implement NCCL Bootstrap

Use MPI only to bootstrap NCCL:

- Duplicate or reuse the embedding MPI communicator for rank/size.
- Use `MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)` to compute node-local rank.
- Use `cudaGetDeviceCount`.
- Select GPU with `local_rank % device_count`.
- Call `cudaSetDevice`.
- Rank 0 calls `ncclGetUniqueId`.
- Broadcast `ncclUniqueId` with MPI.
- All ranks call `ncclCommInitRank`.
- Create or reuse a CUDA stream for NCCL synchronization.

Acceptance checks:

- Single-node multi-rank run maps ranks to local GPUs deterministically.
- Single process does not create unnecessary NCCL state.
- Errors from CUDA/NCCL calls are checked and reported.

### 5. Implement Full GPU-Buffer Embedding Sync

Initial NCCL sync should operate on full `d_syn0`:

```cpp
ncclAllReduce(d_syn0, d_syn0, vocab_size * layer1_size, ncclFloat, ncclSum, nccl_comm, nccl_stream);
```

The CUDA-aware MPI sync should operate on the same full `d_syn0` buffer:

```cpp
MPI_Allreduce(MPI_IN_PLACE, d_syn0, vocab_size * layer1_size, MPI_FLOAT, MPI_SUM, MPI_SYNC_COMM);
```

Then divide by `num_procs` on GPU. This can be a simple CUDA kernel.

Important constraints:

- Do not copy full `d_syn0` to host for `mpi-cuda` or NCCL sync.
- Synchronize the NCCL stream before training uses the synchronized embeddings.
- Preserve MPI backend as the baseline.

Acceptance checks:

- `mpi-cuda` and NCCL full sync complete without deadlock for `np=2`.
- GPU memory remains valid after sync.
- Embedding output is finite and has reasonable values.

### 6. Add Instrumentation

Record and print:

- selected sync backend;
- local rank and selected GPU id;
- number of sync calls;
- total sync time;
- average sync time;
- full sync payload size in bytes;
- whether backend is `mpi`, `mpi-cuda`, or `nccl`.

Acceptance checks:

- MPI, MPI-CUDA, and NCCL runs all report comparable sync metrics.
- Metrics do not require extra build flags.

## Feedback-Oriented Test Loop

After each implementation phase, run tests and use the results to update the code before continuing.

### Test Set A: Build Tests

Run:

```bash
mkdir -p build_mpi
cd build_mpi
cmake ..
make -j
```

Expected:

- Build succeeds with default MPI backend.
- No NCCL dependency is required.

If this fails:

- Fix the non-NCCL build before continuing.
- Do not proceed to NCCL work until the baseline build is clean.

Run:

```bash
mkdir -p build_nccl
cd build_nccl
cmake -DWITH_NCCL=ON ..
make -j
```

Expected:

- Build succeeds on machines with NCCL installed.
- If NCCL is missing, CMake fails with a clear message.

If this fails unexpectedly:

- Check include paths, link paths, and target-level linkage.
- Avoid hard-coding one machine's CUDA/NCCL path if CMake can discover it.

Run:

```bash
mkdir -p build_mpi_cuda
cd build_mpi_cuda
cmake -DWITH_MPI_CUDA=ON ..
make -j
```

Expected:

- Build succeeds with a CUDA-aware MPI toolchain.
- Requesting `--sync-backend mpi-cuda` in a non-`WITH_MPI_CUDA` binary fails early with a clear message.

### Test Set B: Baseline MPI Runtime

Run a small existing workload with:

```bash
mpirun -np 2 ./bin/felog ... --sync-backend mpi
```

Expected:

- Behavior matches the pre-NCCL baseline.
- No NCCL initialization logs appear.
- No sync backend regression.

If this fails:

- Treat it as a refactor regression.
- Fix before testing NCCL.

### Test Set C: NCCL Runtime Smoke Test

Run:

```bash
NCCL_DEBUG=INFO mpirun -np 2 ./bin/felog ... --sync-backend nccl
```

Expected:

- NCCL initializes on all ranks.
- Each rank reports its selected GPU.
- Full embedding sync runs without deadlock.
- Training reaches output generation.

If this fails:

- Check GPU assignment first.
- Then check NCCL unique id broadcast.
- Then check stream synchronization and communicator lifetime.

### Test Set C2: CUDA-Aware MPI Runtime Smoke Test

Run:

```bash
mpirun -np 2 ./bin/felog ... --sync-backend mpi-cuda
```

Expected:

- The binary is linked to a CUDA-aware MPI implementation.
- The backend reports `mpi-cuda`.
- Full embedding sync runs without deadlock.
- Training reaches output generation.

If this fails:

- First verify `MPIX_Query_cuda_support()` or the MPI implementation's equivalent CUDA-support check.
- Then check whether the selected MPI transport can handle CUDA device buffers on this node set.
- Then check count limits and whether full `d_syn0` is resident on GPU.

### Test Set D: Correctness Sanity

Compare MPI, MPI-CUDA, and NCCL outputs on the same small workload.

Expected:

- Outputs do not need to be bitwise identical.
- Embeddings should be finite.
- Training should complete the same high-level workflow.
- Evaluation/sampling control should not hang.

If outputs are invalid:

- Check divide-by-`num_procs` after GPU-buffer sum.
- Check that all ranks use the same `vocab_size` and `layer1_size`.
- Check that `d_syn0` is initialized before first NCCL sync.

### Test Set E: Performance Feedback

Compare:

- MPI sync time;
- MPI-CUDA sync time;
- NCCL sync time;
- total training time;
- GPU transfer-back time;
- end-to-end runtime.

Expected:

- MPI-CUDA and NCCL should reduce or eliminate full embedding host round trips in sync.
- If total runtime does not improve, identify whether the bottleneck is sampling, evaluation, barriers, or corpus transfer.

If GPU-buffer sync does not improve performance:

- Do not immediately add selective sync.
- First verify NCCL and MPI-CUDA use the intended network path on multi-node runs.
- Check whether sync time is a significant portion of total runtime.

## Completion Criteria

The first NCCL implementation is complete only when:

- default MPI build still works;
- CUDA-aware MPI build works on the NFS CUDA-aware MPICH environment;
- NCCL build works on a NCCL-capable machine;
- runtime backend switch works;
- MPI backend remains the default;
- MPI-CUDA and NCCL full embedding sync run without explicit host staging of full `d_syn0`;
- small MPI, MPI-CUDA, and NCCL runs all complete;
- instrumentation reports backend and sync timing;
- known limitations are documented.

## Later Extensions

Only after the full NCCL backend is stable:

- activity-aware selective embedding synchronization;
- compact GPU gather/scatter buffers for selected rows;
- node-local NCCL communicator;
- cross-node communicator;
- topology-aware rank-to-GPU mapping;
- integration with active-frontier feedback signals.

## Implementation Notes

### 2026-07-07 Progress

Implemented in branch `feature/nccl-embedding-sync`:

- Added `WITH_NCCL` CMake option.
- Added NCCL header/library discovery when `WITH_NCCL=ON`.
- Linked `libnccl` only for the CUDA `felog` target when enabled.
- Added runtime `--sync-backend mpi|mpi-cuda|nccl` option.
- Kept `mpi` as the default runtime backend.
- Registered `--sync-backend` in the main random-walk option parser so the flag is accepted before the training thread starts.
- Added an entry-point pre-scan for `--sync-backend` in `src/examples/felog.cpp` so invalid values and unsupported NCCL requests fail before graph loading starts. This is needed because the generic args parser intentionally swallows parse errors to tolerate word2vec-specific flags.
- Moved the entry-point `--sync-backend` pre-scan before MPI initialization so invalid backend values, missing backend arguments, and unsupported NCCL requests fail without noisy MPI startup logs.
- Added clear error handling for invalid sync backend values.
- Added clear error when `--sync-backend nccl` is requested from a non-NCCL build.
- Added embedding sync backend state and instrumentation.
- Added `init_embedding_sync()`, `sync_embedding()`, and `finalize_embedding_sync()`.
- Preserved the existing MPI selected-embedding synchronization path as the MPI backend.
- Added a deterministic round-end embedding synchronization point at the end of each training round, before the existing training-end host embedding write-back and before evaluation. This pauses the periodic sync thread, aligns ranks with MPI barriers, then calls the selected backend once from the training thread. The periodic sync thread remains in place, but short workloads now exercise the selected sync backend reliably.
- Kept the host `syn0` mirror consistent after round-end sync. MPI selected sync now updates both host `syn0` and resident GPU rows for selected embeddings. NCCL round-end sync runs before the existing full `d_syn0` to `syn0` training-end copy, so CPU evaluation and save paths read synchronized values without adding an extra post-sync full D2H refresh.
- Added NCCL bootstrap using MPI broadcast of `ncclUniqueId`.
- Added node-local rank discovery with `MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)`.
- Added local GPU selection as `local_rank % cudaGetDeviceCount()`.
- Added full `d_syn0` NCCL AllReduce path with GPU-side averaging.
- Added an explicit runtime error for NCCL full sync when `vocab_size > syn_size`, because the first pass only supports a full resident `d_syn0`.
- Added a diagnostic before the graph partition consistency assertion so partition test failures report `last partition end`, `vertex_num`, `partition_num`, and `partition_path`.
- Added `cudaDeviceSynchronize()` before NCCL full AllReduce so the NCCL stream reads embeddings after preceding training kernels complete.
- Reset embedding sync metrics and sync-thread control flags at sync/training startup.
- Fixed a potential out-of-bounds write in the MPI selected-sync degree range buffer by allocating `vocab_size + 1` entries.
- Note: the NCCL collective itself does not stage full `d_syn0` through host memory. The current CPU evaluation/output paths still depend on the pre-existing training-end device-to-host write-back, but the NCCL sync path no longer adds a second full host refresh after the collective.

Test feedback:

- Default non-NCCL CMake configure passed:

```bash
cmake -S . -B build_mpi_nccl_plan
```

- Default non-NCCL build passed:

```bash
cmake --build build_mpi_nccl_plan -j 8
```

This was re-run after the NCCL/MPI backend audit fixes and still passed.

- Before installing real NCCL into the NFS environment, `WITH_NCCL=ON` configure was tested and failed clearly as expected:

```bash
cmake -S . -B build_nccl_plan -DWITH_NCCL=ON
```

Observed expected error:

```text
WITH_NCCL=ON requires nccl.h and libnccl. Set NCCL_HOME or install NCCL.
```

- Non-NCCL binary with `--sync-backend nccl` exits with a clear error:

```text
[ Error ] --sync-backend nccl requires a binary built with -DWITH_NCCL=ON
```

- Invalid backend exits with a clear parser error:

```text
invalid --sync-backend 'bad'. Expected 'mpi', 'mpi-cuda', or 'nccl'.
```

- After the entry-point pre-scan fix, invalid backend values and unsupported `nccl` requests fail before graph loading. A non-NCCL binary now reports:

```text
--sync-backend nccl requires a binary built with -DWITH_NCCL=ON
```

- After moving the pre-scan before MPI initialization, invalid backend values and missing backend arguments now exit cleanly without MPI startup noise:

```text
invalid --sync-backend 'bad'. Expected 'mpi', 'mpi-cuda', or 'nccl'.
Argument missing for --sync-backend
```

- Default runtime behavior without `--sync-backend` was re-tested after moving the deterministic sync point. Single-rank and two-rank tiny smoke tests both passed. The two-rank run used the MPI backend by default and reported a real selected-sync call:

```text
[ Embedding Sync ] backend=mpi ranks=2
[ 0 ] Embedding sync summary backend=mpi calls=1 total=0.000472s avg=0.000472s payload=192 bytes
[ 1 ] Embedding sync summary backend=mpi calls=1 total=0.000397s avg=0.000397s payload=192 bytes
```

The generated default-path embedding files did not contain `NaN` or `Inf`.

- Single-rank MPI smoke test on a tiny generated graph passed and reported embedding sync instrumentation:

```text
[ Embedding Sync ] backend=mpi ranks=1
[ 0 ] Embedding sync local_rank=0 local_size=1 gpu=0 backend=mpi
[ 0 ] Embedding sync summary backend=mpi calls=0 total=0.000000s avg=0.000000s payload=0 bytes
```

- Multi-rank MPI smoke testing initially failed because `/usr/bin/mpirun` is MPICH/Hydra while the binary is linked against OpenMPI. This made each launched process see `MPI_COMM_WORLD` size 1. The partition diagnostic exposed this clearly:

```text
partition mismatch: last partition end=3, vertex_num=6, partition_num=1
```

- Re-running with the matching OpenMPI launcher passed:

```bash
mpirun.openmpi --allow-run-as-root -np 2 ./build_mpi_nccl_plan/bin/felog ...
```

Observed instrumentation:

```text
[ Embedding Sync ] backend=mpi ranks=2
[ 0 ] Embedding sync local_rank=0 local_size=2 gpu=0 backend=mpi
[ 1 ] Embedding sync local_rank=1 local_size=2 gpu=0 backend=mpi
```

- On this machine OpenMPI's OFI/IB path can fail for larger local tests. The stable local smoke command used:

```bash
mpirun.openmpi --allow-run-as-root --mca pml ob1 --mca btl self,tcp -np 2 ./build_mpi_nccl_plan/bin/felog ...
```

- After the backend audit fixes, both single-rank and two-rank MPI smoke tests passed with the stable OpenMPI TCP/self transport command. At that early stage, the `WITH_NCCL=ON` configure check was also re-run and still failed clearly because real NCCL had not yet been installed into the NFS environment.

- A short two-rank MPI smoke test now exercises a real selected-embedding sync call because of the round-end synchronization point:

```bash
mpirun.openmpi --allow-run-as-root --mca pml ob1 --mca btl self,tcp -np 2 \
  ./build_mpi_nccl_plan/bin/felog ... --sync-backend mpi
```

Observed instrumentation:

```text
[ 0 ] Embedding sync summary backend=mpi calls=1 total=0.000289s avg=0.000289s payload=192 bytes
[ 1 ] Embedding sync summary backend=mpi calls=1 total=0.000395s avg=0.000395s payload=192 bytes
```

- A local NCCL stub was added under `tmp_launch_test/nccl_stub` to compile-check the NCCL code path on this non-NCCL machine. With `NCCL_HOME` pointed at the stub, `WITH_NCCL=ON` configure and build both passed:

```bash
NCCL_HOME=$PWD/tmp_launch_test/nccl_stub cmake -S . -B build_nccl_stub_plan -DWITH_NCCL=ON
cmake --build build_nccl_stub_plan -j 8
```

- The stub NCCL build also passed single-rank and two-rank smoke tests with `--sync-backend nccl`, confirming runtime option selection, MPI bootstrap, node-local rank discovery, and instrumentation are wired correctly:

```bash
LD_LIBRARY_PATH=$PWD/tmp_launch_test/nccl_stub/lib \
mpirun.openmpi --allow-run-as-root --mca pml ob1 --mca btl self,tcp -np 2 \
  ./build_nccl_stub_plan/bin/felog ... --sync-backend nccl
```

Observed instrumentation:

```text
[ Embedding Sync ] backend=nccl ranks=2
[ 0 ] Embedding sync local_rank=0 local_size=2 gpu=0 backend=nccl
[ 1 ] Embedding sync local_rank=1 local_size=2 gpu=0 backend=nccl
```

- After adding the round-end synchronization point, the stub NCCL two-rank smoke test also exercises the NCCL full-sync call path:

```text
[ 0 ] Embedding sync summary backend=nccl calls=1 total=0.000211s avg=0.000211s payload=384 bytes
[ 1 ] Embedding sync summary backend=nccl calls=1 total=0.000118s avg=0.000118s payload=384 bytes
```

- After fixing host `syn0` consistency for round-end sync, the default MPI build and stub NCCL build were rebuilt successfully. Two-rank MPI and two-rank stub NCCL tiny smoke tests both passed again with `calls=1`, and the generated embedding files did not contain `NaN` or `Inf`.
- The deterministic round-end sync was then moved before the existing training-end host embedding write-back, removing the extra post-sync full host refresh while preserving CPU evaluation/output correctness. After this move:

```bash
cmake --build build_mpi_nccl_plan -j 8
cmake --build build_nccl_stub_plan -j 8
```

both passed. Two-rank MPI and two-rank stub NCCL tiny smoke tests also passed again:

```text
[ 0 ] Embedding sync summary backend=mpi calls=1 total=0.000326s avg=0.000326s payload=192 bytes
[ 1 ] Embedding sync summary backend=mpi calls=1 total=0.000439s avg=0.000439s payload=192 bytes
[ 0 ] Embedding sync summary backend=nccl calls=1 total=0.000208s avg=0.000208s payload=384 bytes
[ 1 ] Embedding sync summary backend=nccl calls=1 total=0.000116s avg=0.000116s payload=384 bytes
```

The generated embedding files from this verification pass did not contain `NaN` or `Inf`.

The stub test does not prove real NCCL collective correctness or performance; it only proves the `WITH_NCCL` code path compiles, links, reaches NCCL bootstrap calls, and calls the NCCL full-sync API path.

Known remaining verification gap:

- Real NCCL compile and runtime smoke tests still need a machine with real NCCL installed.
- The round-end synchronization point now gives short local fixtures `calls=1`. Longer workloads are still needed to evaluate periodic sync behavior and performance.

### MPI Runtime Compatibility Notes

The cluster nodes currently have mixed MPI defaults. A quick SSH audit over the FeLoG hostfiles showed:

- `g123` and `g124`: default `/usr/bin/mpirun` is OpenMPI 4.0.3, and default `mpicxx` is OpenMPI.
- `g125`, `g127`, and `g130`: default `/usr/bin/mpirun` is MPICH/HYDRA 3.3.2, but default `mpicxx` points to OpenMPI.
- `g126`, `g128`, and `g129`: default `/usr/bin/mpirun` is MPICH/HYDRA 3.3.2, and default `mpicxx` points to MPICH.

Existing local build directories are also mixed:

- MPICH-linked: `build/bin/felog`, `build_legacy_ref/bin/felog`, `build_thr065_mpich/bin/felog`, `build_thr09_mpich/bin/felog`.
- OpenMPI-linked: `build_mpi_nccl_plan/bin/felog`, `build_nccl_stub_plan/bin/felog`, `build_thr065/bin/felog`, `build_thr09/bin/felog`.

Because several production hostfiles use nodes whose default launcher is MPICH, keep an MPICH-aligned build for compatibility with the existing run style:

```bash
cmake -S . -B build_mpi_nccl_plan_mpich_clean \
  -DMPI_C_COMPILER=/usr/bin/mpicc.mpich \
  -DMPI_CXX_COMPILER=/usr/bin/mpicxx.mpich \
  -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun.mpich
cmake --build build_mpi_nccl_plan_mpich_clean -j 8
```

Verification:

- `build_mpi_nccl_plan_mpich_clean/bin/felog` links only `libmpich.so.12`.
- `mpirun.mpich -np 2 ./build_mpi_nccl_plan_mpich_clean/bin/felog ... --sync-backend mpi` passed on the tiny fixture.
- Observed MPICH instrumentation:

```text
[ Embedding Sync ] backend=mpi ranks=2
[ 0 ] Embedding sync summary backend=mpi calls=1 total=0.000361s avg=0.000361s payload=192 bytes
[ 1 ] Embedding sync summary backend=mpi calls=1 total=0.000452s avg=0.000452s payload=192 bytes
```

Rule for all future tests: the MPI launcher must match the MPI library used at build time. Use `mpirun.mpich` for MPICH-linked binaries and `mpirun.openmpi` for OpenMPI-linked binaries.

### NFS Environment Layout

The shared environment root is:

```text
/home/lzl/nfs.d/env
```

Created and verified:

```text
/home/lzl/nfs.d/env/
  README.md
  mpi/
    mpich-3.3.2-system/
      bin/mpicc
      bin/mpicxx
      bin/mpiexec
      bin/mpiexec.hydra
      bin/hydra_pmi_proxy
      bin/mpirun
      include/
      lib/
  nccl/
  scripts/
    env-mpich-system.sh
    env-nccl-mpich.sh
```

`mpich-3.3.2-system` is a user-space NFS copy of the current system MPICH 3.3.2 runtime/development files from `g130`. This is not a CUDA-aware MPI build; it is intended for the default `mpi-host` backend and for NCCL launcher/bootstrap compatibility. It avoids relying on each node's `/etc/alternatives` state.

Use it with:

```bash
source /home/lzl/nfs.d/env/scripts/env-mpich-system.sh

cmake -S . -B build_mpi_nfs_mpich \
  -DMPI_C_COMPILER=$MPICC \
  -DMPI_CXX_COMPILER=$MPICXX \
  -DMPIEXEC_EXECUTABLE=$MPIRUN
cmake --build build_mpi_nfs_mpich -j 8
```

Verification:

- `$MPIRUN -np 2 hostname` works locally with the NFS MPICH launcher.
- `$MPIRUN -hostfile build/hosts_4n_safe -np 4 hostname` reached `g130`, `g125`, `g129`, and `g127` through the NFS launcher.
- `build_mpi_nfs_mpich/bin/felog` links `/home/lzl/nfs.d/env/mpi/mpich-3.3.2-system/lib/libmpich.so.12`.
- `build_mpi_nfs_mpich` built successfully.
- `source /home/lzl/nfs.d/env/scripts/env-mpich-system.sh && $MPIRUN -np 2 ./build_mpi_nfs_mpich/bin/felog ... --sync-backend mpi` passed on the tiny fixture:

```text
[ Embedding Sync ] backend=mpi ranks=2
[ 0 ] Embedding sync summary backend=mpi calls=1 total=0.000265s avg=0.000265s payload=192 bytes
[ 1 ] Embedding sync summary backend=mpi calls=1 total=0.000362s avg=0.000362s payload=192 bytes
```

The generated embedding file from the NFS MPICH smoke test did not contain `NaN` or `Inf`.

### Real NCCL And CUDA-Aware MPI Environment Verification

NCCL is now installed under:

```text
/home/lzl/nfs.d/env/nccl/nccl-2.21.5-defaults
```

The newer conda-forge NCCL install under:

```text
/home/lzl/nfs.d/env/nccl/nccl-conda
```

was not selected for the current build because it linked against a newer
`libstdc++/CXXABI` than the current GCC 9 toolchain provides:

```text
undefined reference to __cxa_call_terminate@CXXABI_1.3.15
```

The validated NCCL build is:

```bash
source /home/lzl/nfs.d/env/scripts/env-nccl-mpich.sh
cmake -S . -B build_nccl_nfs_mpich_221 \
  -DWITH_NCCL=ON \
  -DMPI_C_COMPILER=$MPICC \
  -DMPI_CXX_COMPILER=$MPICXX \
  -DMPIEXEC_EXECUTABLE=$MPIRUN \
  -DCMAKE_EXE_LINKER_FLAGS="-L$NCCL_HOME/lib -Wl,-rpath,$NCCL_HOME/lib"
cmake --build build_nccl_nfs_mpich_221 -j 8
```

Real NCCL over TCP smoke passed on `g130/g129` with:

```bash
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=ens17f0
$MPIRUN -hostfile build/hosts_2n_safe -np 2 \
  ./build_nccl_nfs_mpich_221/bin/felog ... --sync-backend nccl
```

Observed:

```text
NCCL_SOCKET_IFNAME set by environment to ens17f0
NET/Socket : Using ens17f0:192.168.6.130
NET/Socket : Using ens17f0:192.168.6.129
Using network Socket
ncclCommInitRank ... Init COMPLETE
[ 0 ] Embedding sync summary backend=nccl calls=1 total=0.007931s avg=0.007931s payload=384 bytes
[ 1 ] Embedding sync summary backend=nccl calls=1 total=0.008092s avg=0.008092s payload=384 bytes
```

The generated embedding file did not contain `NaN` or `Inf`.

A CUDA-aware MPI environment is now installed under:

```text
/home/lzl/nfs.d/env/ucx/ucx-1.14.0-cuda
/home/lzl/nfs.d/env/mpi/mpich-4.2.3-cuda-ucx
```

UCX verification:

```text
Library version: 1.14.0
Configured with: --with-cuda=/usr/local/cuda
Memory domain: cuda_cpy
Transport: cuda_copy
Memory domain: cuda_ipc
Transport: cuda_ipc
```

MPICH verification:

```text
MPICH Version: 4.2.3
MPICH Device: ch4:ucx
MPICH configure: --with-device=ch4:ucx --with-ucx=/home/lzl/nfs.d/env/ucx/ucx-1.14.0-cuda --with-cuda=/usr/local/cuda
```

Direct CUDA device-buffer MPI smoke passed locally and across `g130/g129`:

```bash
source /home/lzl/nfs.d/env/scripts/env-mpich-cuda-ucx.sh
$MPIRUN -hostfile build/hosts_2n_safe -np 2 \
  ./tmp_launch_test/mpi_cuda/cuda_allreduce
```

Observed:

```text
rank=0 size=2 cuda_query=1 expected=3.0 got0=3.0 ok=1
rank=1 size=2 cuda_query=1 expected=3.0 got0=3.0 ok=1
```

FeLoG built and ran with CUDA-aware MPICH:

```bash
source /home/lzl/nfs.d/env/scripts/env-mpich-cuda-ucx.sh
cmake -S . -B build_mpi_cuda_ucx \
  -DWITH_MPI_CUDA=ON \
  -DMPI_C_COMPILER=$MPICC \
  -DMPI_CXX_COMPILER=$MPICXX \
  -DMPIEXEC_EXECUTABLE=$MPIRUN
cmake --build build_mpi_cuda_ucx -j 8
```

Tiny two-node FeLoG smoke passed with `--sync-backend mpi`:

```text
[ 0 ] Embedding sync summary backend=mpi calls=1 total=0.000255s avg=0.000255s payload=192 bytes
[ 1 ] Embedding sync summary backend=mpi calls=1 total=0.000204s avg=0.000204s payload=192 bytes
```

The generated embedding file did not contain `NaN` or `Inf`.

After adding the `mpi-cuda` backend, `--sync-backend mpi-cuda` performs full
`d_syn0` AllReduce directly on the CUDA device buffer through CUDA-aware MPI,
then averages on GPU. It does not stage the full embedding through host memory.

FeLoG also built and ran with CUDA-aware MPICH plus NCCL:

```bash
source /home/lzl/nfs.d/env/scripts/env-nccl-mpich-cuda.sh
cmake -S . -B build_nccl_mpi_cuda_ucx \
  -DWITH_MPI_CUDA=ON \
  -DWITH_NCCL=ON \
  -DMPI_C_COMPILER=$MPICC \
  -DMPI_CXX_COMPILER=$MPICXX \
  -DMPIEXEC_EXECUTABLE=$MPIRUN \
  -DCMAKE_EXE_LINKER_FLAGS="-L$NCCL_HOME/lib -Wl,-rpath,$NCCL_HOME/lib"
cmake --build build_nccl_mpi_cuda_ucx -j 8
```

Tiny two-node FeLoG smoke passed with `--sync-backend nccl`:

```text
NCCL_SOCKET_IFNAME set by environment to ens17f0
NET/Socket : Using [0]ens17f0:192.168.6.130
NET/Socket : Using [0]ens17f0:192.168.6.129
Using network Socket
ncclCommInitRank ... Init COMPLETE
[ 0 ] Embedding sync summary backend=nccl calls=1 total=0.008482s avg=0.008482s payload=384 bytes
[ 1 ] Embedding sync summary backend=nccl calls=1 total=0.008594s avg=0.008594s payload=384 bytes
```

The generated embedding file did not contain `NaN` or `Inf`.

Tiny two-node FeLoG smoke passed with `--sync-backend mpi-cuda` from the same
combined binary:

```text
[ Embedding Sync ] backend=mpi-cuda ranks=2
[ 0 ] Embedding sync summary backend=mpi-cuda calls=1 total=0.536859s avg=0.536859s payload=384 bytes
[ 1 ] Embedding sync summary backend=mpi-cuda calls=1 total=0.536874s avg=0.536874s payload=384 bytes
```

The generated embedding file did not contain `NaN` or `Inf`.

Final incremental build verification on 2026-07-07:

```bash
cmake --build build_mpi_nfs_mpich -j 8
cmake --build build_mpi_cuda_ucx -j 8
cmake --build build_nccl_mpi_cuda_ucx -j 8
```

All three builds completed successfully. The final combined NCCL and MPI-CUDA
embedding outputs were checked again:

```bash
rg -n -i '(^|[^a-z])(nan|inf)([^a-z]|$)' \
  tmp_launch_test/nccl_plan/emb_nccl_combined_after_mpicuda.txt \
  tmp_launch_test/nccl_plan/emb_mpicuda_combined_after_nccl.txt
```

No matches were found.

Fresh combined runtime verification on 2026-07-07 used:

```bash
source /home/lzl/nfs.d/env/scripts/env-nccl-mpich-cuda.sh
$MPIRUN -hostfile build/hosts_2n_safe -np 2 \
  ./build_nccl_mpi_cuda_ucx/bin/felog ... --sync-backend nccl
$MPIRUN -hostfile build/hosts_2n_safe -np 2 \
  ./build_nccl_mpi_cuda_ucx/bin/felog ... --sync-backend mpi-cuda
```

Environment:

```text
MPIRUN=/home/lzl/nfs.d/env/mpi/mpich-4.2.3-cuda-ucx/bin/mpirun
NCCL_HOME=/home/lzl/nfs.d/env/nccl/nccl-2.21.5-defaults
UCX_TLS=tcp,cuda_copy,cuda_ipc,self,sm
UCX_NET_DEVICES=ens17f0
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=ens17f0
MPICH_GPU_SUPPORT_ENABLED=1
```

Observed NCCL smoke result:

```text
[ Embedding Sync ] backend=nccl ranks=2
[ 0 ] Embedding sync summary backend=nccl calls=1 total=0.009797s avg=0.009797s payload=384 bytes
[ 1 ] Embedding sync summary backend=nccl calls=1 total=0.009842s avg=0.009842s payload=384 bytes
```

Observed MPI-CUDA smoke result:

```text
[ Embedding Sync ] backend=mpi-cuda ranks=2
[ 0 ] Embedding sync summary backend=mpi-cuda calls=1 total=0.528100s avg=0.528100s payload=384 bytes
[ 1 ] Embedding sync summary backend=mpi-cuda calls=1 total=0.528026s avg=0.528026s payload=384 bytes
```

The fresh generated files:

```text
tmp_launch_test/nccl_plan/emb_nccl_final_verify.txt
tmp_launch_test/nccl_plan/emb_mpicuda_final_verify.txt
```

both reported `6 16` in the embedding header and did not contain `NaN` or `Inf`.

After this verification, backend parsing was tightened so both accepted CLI
forms select the same backend:

```bash
--sync-backend nccl
--sync-backend=nccl
```

The same applies to `mpi` and `mpi-cuda`. This avoids the help-text form
silently falling back to the default backend.

Final parser/build/runtime verification after that fix:

```text
default non-MPI-CUDA build with --sync-backend=mpi-cuda:
--sync-backend mpi-cuda requires a binary built with -DWITH_MPI_CUDA=ON

default non-NCCL/non-MPI-CUDA build with --sync-backend=bad:
invalid --sync-backend 'bad'. Expected 'mpi', 'mpi-cuda', or 'nccl'.
```

All three builds completed successfully again:

```bash
cmake --build build_mpi_nfs_mpich -j 8
cmake --build build_mpi_cuda_ucx -j 8
cmake --build build_nccl_mpi_cuda_ucx -j 8
```

All three backend smoke tests passed with the `--sync-backend=value` form:

```text
backend=mpi calls=1 payload=192 bytes avg=0.0004s
backend=nccl calls=1 payload=384 bytes avg=0.0084s
backend=mpi-cuda calls=1 payload=384 bytes avg=0.5184s
```

The generated embedding files:

```text
tmp_launch_test/nccl_plan/emb_mpi_equals_verify.txt
tmp_launch_test/nccl_plan/emb_nccl_equals_verify.txt
tmp_launch_test/nccl_plan/emb_mpicuda_equals_verify.txt
```

all reported `6 16` in the embedding header and did not contain `NaN` or `Inf`.

### Explicit Selected/Full Sync Scope

The runtime interface now separates the communication backend from the embedding
sync range:

```text
--sync-backend mpi|mpi-cuda|nccl
--sync-scope selected|full
```

`selected` is the default scope. It synchronizes only the vocabulary rows used
by the current compressed training corpus. `full` synchronizes
`vocab_size * layer1_size` floats. The intended matrix is:

| Backend | `selected` | `full` |
|---|---|---|
| `mpi` | host packed selected rows + `MPI_Allreduce` | host full `syn0` + `MPI_Allreduce` |
| `mpi-cuda` | device packed selected rows + CUDA-aware `MPI_Allreduce` | full `d_syn0` CUDA-aware `MPI_Allreduce` |
| `nccl` | device packed selected rows + `ncclAllReduce` | full `d_syn0` `ncclAllReduce` |

The code validates both `--sync-scope value` and `--sync-scope=value` forms in
the entry point and the training parser. Startup and summary logs now include
both fields, for example:

```text
[ Embedding Sync ] backend=mpi-cuda scope=selected ranks=2
[ 0 ] Embedding sync summary backend=mpi-cuda scope=selected ...
```

Incremental build verification after adding `--sync-scope` passed:

```bash
cmake --build build_mpi_nfs_mpich -j 8
cmake --build build_mpi_cuda_ucx -j 8
cmake --build build_nccl_mpi_cuda_ucx -j 8
git diff --check
```

The two-node smoke test initially exposed a common random-walk crash when `-o`
was not parsed before word2vec-specific options: `internal_walk_epoch()` wrote
footprints through a null `PathCollector`. That path is now guarded so footprint
collection only happens when path output is enabled. The final smoke test passed
all six backend/scope combinations on the latest binaries:

```text
mpi selected      payload=192 bytes
mpi full          payload=384 bytes
mpi-cuda selected payload=192 bytes
mpi-cuda full     payload=384 bytes
nccl selected     payload=192 bytes
nccl full         payload=384 bytes
```

The final logs and outputs are under:

```text
tmp_launch_test/nccl_plan/scope_smoke_final/log/
tmp_launch_test/nccl_plan/scope_smoke_final/out/
```

All six generated embedding files reported `6 16` and did not contain `NaN` or
`Inf`.

## 8-node LJ phase-order fix

The first 8-node LJ runs with CUDA-aware MPI and NCCL failed before embedding
sync with:

```text
Fatal error in internal_Allreduce: Message truncated
MPI_Allreduce(... count=1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD)
```

The failure was not an NCCL collective error. It came from training-side MPI
operations and walk-side MPI operations entering MPI in different phases across
ranks. In particular, the training thread duplicated communicators from
`MPI_COMM_WORLD` after it was started, while the main thread could already be in
random-walk MPI calls. The background periodic sync thread also made MPI phase
ordering harder to reason about.

Applied fix:

- Initialize training-side communicators and NCCL communicator in the main
  thread before random walk starts.
- Reuse that initialization in `train_corpus_cuda()` instead of duplicating
  communicators from the training thread.
- Disable the background periodic sync thread; keep the explicit per-round
  embedding sync after training.
- Block a new walk round while the previous corpus is still owned by the
  training side.

Build verification:

```bash
cmake --build build_mpi_nfs_mpich -j 8
cmake --build build_mpi_cuda_ucx -j 8
cmake --build build_nccl_mpi_cuda_ucx -j 8
```

All three builds passed.

8-node LJ verification with `dataset/LJ-8.part`, `np=8`, TCP-only:

```text
mpi:      pass, real 18.71s, sync 0.036812s, payload 84,544 B
mpi-cuda: pass, real 29.25s, sync 4.643503s, payload 143,278,784 B
nccl:     pass, real 21.64s, sync 2.159594s, payload 143,278,784 B
```

The same phase-order fix also cleared the previous 4-node `np=8` CUDA-aware MPI
timeout when using the original 8-way LJ partition:

```text
4 nodes / 8 ranks / mpi-cuda: pass, real 23.38s, sync 3.576635s, payload 143,278,784 B
```

The 4-node NCCL case still needs a different launch/data layout: this run uses
two ranks per node while each node exposes one GPU, so both local ranks map to
GPU 0. A valid 4-way LJ partition with `np=4`, or two usable GPUs per node, is
needed for a meaningful 4-node NCCL measurement.

A repository search found the original partitioning path:

- `README.md` documents `./bin/mpgp -i [train_data] -e [test_data] -v [vertex_num] -p [partition_num] -t [float:0, integer:1]`.
- `src/tools/mpgp.cpp` implements `partition_relabel()`.
- The partitioner first converts the text graph to undirected form, computes
  degrees, builds/sorts CSR, applies a DFS-degree stream order, assigns vertices
  with an LDG-style score, then relabels vertices so each partition becomes a
  contiguous ID range.
- It writes a matching set of outputs: relabeled train graph, relabeled test
  edges, and a `.part` file. The `.part` file must match the relabeled graph.
- `src/tools/gconverter.cpp` is then used to convert the relabeled text graph to
  the binary `.data-r` consumed by FeLoG.

Existing generated LJ partition data was found under:

```text
/home/lzl/nfs.d/code/6.1Huge/dataset/binary/mpad/LJ-4.data-r
/home/lzl/nfs.d/code/6.1Huge/dataset/binary/mpad/LJ-4.part
```

The 4-way LJ partition ranges are:

```text
0 478837
478837 1041066
1041066 1674564
1674564 2238731
```

Using this matching 4-way data, 4 nodes / 4 ranks / NCCL passed:

```text
real 28.46s, rank0 whole 27.119251s, sync 4.317366s, payload 143,278,784 B
```

Generated embedding files all reported `2238731 16` and had no `NaN`/`Inf`
matches. Detailed logs and result table are in:

```text
tmp_launch_test/lj_backend_compare/RESULTS.md
```

One final runtime fix was applied after review: `init_embedding_sync()` now calls
`cudaSetDevice(local_rank % cudaGetDeviceCount())` for all backends when CUDA
devices are present. Previously only the NCCL branch called `cudaSetDevice`,
which made the logged GPU selection unreliable for `mpi` and `mpi-cuda`.

Final verification after that fix:

```bash
cmake --build build_mpi_nfs_mpich -j 8
cmake --build build_mpi_cuda_ucx -j 8
cmake --build build_nccl_mpi_cuda_ucx -j 8
```

All three builds passed. Tiny smoke tests also passed with the
`--sync-backend=value` form:

```text
backend=mpi ranks=2 calls=1 avg=0.000262s/0.000359s payload=192 bytes
backend=nccl ranks=2 calls=1 avg=0.008816s/0.008901s payload=384 bytes
backend=mpi-cuda ranks=2 calls=1 avg=0.497050s/0.497056s payload=384 bytes
```

The final generated embedding files:

```text
tmp_launch_test/nccl_plan/emb_mpi_final_device.txt
tmp_launch_test/nccl_plan/emb_nccl_final_device.txt
tmp_launch_test/nccl_plan/emb_mpicuda_final_device.txt
```

all reported `6 16` in the embedding header and did not contain `NaN` or `Inf`.

### Latest Status: Sync Scope Matrix Complete

Current implementation status:

- `--sync-backend mpi|mpi-cuda|nccl` selects the communication backend.
- `--sync-scope selected|full` selects partial-row or full-embedding sync.
- Default remains `--sync-backend mpi --sync-scope selected`.
- The backend/scope matrix is implemented through named functions for all six
  combinations.
- `internal_walk_epoch()` now guards footprint writes so missing path output
  cannot dereference a null `PathCollector` during smoke tests.

Latest verification:

```text
cmake --build build_mpi_nfs_mpich -j 8
cmake --build build_mpi_cuda_ucx -j 8
cmake --build build_nccl_mpi_cuda_ucx -j 8
git diff --check
```

Final two-node tiny smoke on the latest binaries:

```text
mpi selected      pass, payload=192 bytes
mpi full          pass, payload=384 bytes
mpi-cuda selected pass, payload=192 bytes
mpi-cuda full     pass, payload=384 bytes
nccl selected     pass, payload=192 bytes
nccl full         pass, payload=384 bytes
```

Logs:

```text
tmp_launch_test/nccl_plan/scope_smoke_final/log/
tmp_launch_test/nccl_plan/scope_smoke_final/out/
```

All six generated embedding files reported `6 16` and had no `NaN` or `Inf`.
