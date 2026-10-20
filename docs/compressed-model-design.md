# Compressed GGUF models (xz and gzip)

A model file that is xz- or gzip-compressed is accepted anywhere llama.cpp takes a model path, with no flag and no frontend code.
The format is decided by the content of the file, not its name. A plain `.gguf` takes exactly the code path it always did.

Status: implemented and tested on macOS (Apple Silicon, Metal) for both formats. The Linux implementation is written and compile-checked, but has **never been run**. See "Verified and not verified".

## Data flow

```
 path
  |
  v
 llama_open_model_source()                        src/llama-model-source.cpp
  |  sniff the first bytes
  |  "GGUF" or unknown --> plain llama_file + gguf_init_from_file          (the old code)
  |  a known compression magic (xz, gzip)  --> llama_decoder (llama-codec)
  |       |
  |       |-- metadata only (no_alloc, vocab_only, fit):
  |       |       gguf_init_from_callback pulls the header from the decoder
  |       |       llama_file(virtual size)    nothing stored, nothing to map
  |       |
  |       '-- full load:
  |               llama_backing_store::create(size)           src/llama-backing-store.cpp
  |               decoder.read(store.write_addr() + off, 8 MiB slices)     <- the final destination
  |               store.seal()
  |               llama_file(store)           mappable, not readable
  |               gguf_init_from_buffer(temporary mapping)    header, no second decode
  v
 llama_model_loader                               src/llama-model-loader.cpp
  |  init_mappings: llama_mmap(file)   mmap(PROT_READ, MAP_SHARED, store fd)   same pages
  v
 load_tensors: tensor->data = mapping + offset, or a backend buffer over the mapping,
               or (repack, discrete GPU) a separate allocation filled from the mapping
```

## Responsibilities

| Concern | Where | Knows about | Does not know about |
|---|---|---|---|
| Model source | `llama_file` (extended), `llama_open_model_source` | what the loader needs: size, a descriptor to map, metadata | codec internals, shm, memfd |
| Backing store | `llama_backing_store` | pages, descriptors, sealing | codecs, GGUF |
| Codec | `llama_decoder`, `llama_codec` table (`llama-codec-xz.cpp`, `llama-codec-gz.cpp`) | bytes in, bytes out, the decoded size | GGUF, storage, backends |
| Backend placement | `llama_model_base::load_tensors` (unchanged) | mapped host memory vs copies | where the bytes came from |

Nothing was added to `include/llama.h`. One staging function was added to `src/llama-ext.h`: `llama_gguf_open_metadata(path)` (see "Callers outside the model loader").

## Interfaces

```cpp
struct llama_decoder {                 // src/llama-codec.h
    virtual uint64_t size() const = 0;                    // decoded size, known before decoding
    virtual size_t   read(void * dst, size_t n) = 0;      // sequential, 0 at the end, throws on bad input
};
struct llama_codec { const char * name, * suffix; const uint8_t * magic; size_t magic_len;
                     bool available; const char * enable;       // "-DLLAMA_ZLIB=ON", for the error when it is not built in
                     std::unique_ptr<llama_decoder> (*open)(llama_file & in, bool parallel); };

struct llama_backing_store {           // src/llama-backing-store.h
    static std::unique_ptr<llama_backing_store> create(size_t size);   // throws
    void * write_addr();   void seal();   int fd() const;   bool read_only_enforced() const;
};

struct llama_file {                    // src/llama-mmap.h, two new kinds
    explicit llama_file(std::unique_ptr<llama_backing_store>);   // mappable, not readable
    explicit llama_file(size_t virtual_size);                    // neither: for validation of sizes only
    bool readable() const;  bool mappable() const;
};
```

A new codec is one entry in the table in `src/llama-codec.cpp` and one file. It must report its decoded size before decoding, because the backing store has a fixed size (macOS allows `ftruncate` on shared memory only once). A format that does not store its size has to recover it itself, see the next section.

## Detection

By content. `llama_codec_find` compares the first bytes with each codec's magic (xz: `FD 37 7A 58 5A 00`, gzip: `1F 8B 08`). A file that matches nothing is handed to the GGUF parser as before.

* A file named `.gguf` that holds compressed data is decompressed. A file named `.gguf.xz` or `.gguf.gz` that holds a plain GGUF is loaded as plain.
* Decoded output is never sniffed again, so there is one level of decoding.
* The file name is used in one place, split discovery (below), and only to find sibling file names.
* Security: the decoded size comes from the file, so a small file can request a large allocation. There is no RAM-based limit. The size is validated (non-zero, fits `size_t` and `off_t`, no overflow in page rounding), and failures of `ftruncate`, `mmap` and `fallocate` are reported with the requested size. This does not remove every out-of-memory failure mode, see "Not verified".

## The decoded size, and what gzip needed

The backing store has a fixed size, so a codec has to say how big the output is before it produces any.

* xz stores it in an index at the end of the file, together with the container structure. The codec reads it, and checks that there is one stream and nothing after it.
* gzip has no index. The size is in the last 4 bytes of the file, modulo 2^32, so it is wrong for any model that decodes to 4 GiB or more. `llama-codec-gz.cpp` solves this inside the codec, with no change anywhere else:
  * up to 1 GiB compressed, the trailer is used. Every decode is checked against it, and zlib checks the CRC and length itself. If the data is larger than the trailer says, the load fails with a message that names the 4 GiB limit, never with a wrong size;
  * from 1 GiB compressed, the file is decoded once and the output counted, nothing is stored. A 1.12 GiB gzip that decodes to 4.49 GiB (trailer: 0.49 GiB) was sized correctly this way, and its decoded bytes then matched the original on all 4,822,495,747 bytes. That pass costs about 0.4 s per GiB here, and it is paid by a metadata-only open too, so the fit pass of a large gzip file decodes it once. xz does not have this cost.
* The 1 GiB threshold is a rule of thumb, not a proof: a gzip file under 1 GiB that decodes to 4 GiB or more (a very compressible model) is refused with the error above. A 5 GiB file of zeros (23 MB compressed) was refused that way.
* Only a single gzip member is accepted. A file made by concatenating members (and block-gzip tools that write several) is rejected, like concatenated xz streams. A zlib or raw deflate stream is not recognised.
* A metadata-only open of a gzip file checks less than one of an xz file. A truncated `.gz` can pass the header-only open and is refused at the real load, because there is no index to check the container with.

## Backing store

| | macOS (POSIX shm) | Linux (`memfd_create`) |
|---|---|---|
| Implemented | yes | yes |
| Run | yes | **no** |
| Create | `shm_open(O_EXCL)`, one `ftruncate`, RW `mmap`, a second `O_RDONLY` descriptor opened by name, `shm_unlink` at once | `memfd_create(MFD_CLOEXEC \| MFD_ALLOW_SEALING)`, `ftruncate`, `fallocate`, RW `mmap` |
| `seal()` | unmap the writer, close the RW descriptor | unmap the writer, `F_ADD_SEALS` (write, shrink, grow, seal) |
| After `seal()` | the remaining descriptor cannot be mapped writable, `mprotect` to RW fails, `ftruncate` fails (measured) | writes refused by the kernel (documented behaviour, not run here) |
| `read`, `lseek` on the descriptor | fail (measured). Only `fstat` and `mmap` are relied on | work, but are not relied on |
| Pages after the store is destroyed | live while any mapping exists (measured) | same (assumed) |
| Out-of-memory | not known | `fallocate` should report it before decoding (assumed) |
| Close-on-exec | `fcntl(FD_CLOEXEC)` | `MFD_CLOEXEC` |

`read_only_enforced()` reports whether sealing took effect. Closing the writable mapping alone does not make the object read-only. On macOS it is the closing of the RW descriptor, together with the read-only one opened before `shm_unlink`, that does.

Other platforms get a clear "not supported on this platform" error. There is no silent fallback to a temporary file. The three branches (`SHM`, `MEMFD`, `NONE`) were each parsed and type-checked; only `SHM` was run.

Residual window: between `shm_open` and `shm_unlink` the name exists for microseconds. A `kill -9` in that window could leak the name until reboot. Mode is 0600.

## Loader paths

| Path | Behaviour |
|---|---|
| `llama_model_load_from_file`, `_splits` (`llama-cli`, `llama-server`, `llama-completion`, ...) | Compressed input works through the shared loader |
| `no_alloc`, `vocab_only`, the memory-fit pass (`common/fit.cpp`, `LOAD_MODE_NONE`) | Metadata only: the header is decoded and parsed, nothing is stored or mapped. No decode of the body |
| Full load with mmap | One decode into the store, then the normal mapping |
| `--no-mmap`, direct I/O (`LOAD_MODE_NONE`, `DIRECT_IO`) on a real load | Error: "compressed models are loaded through memory mapping, which is disabled". A copy would hold the model twice |
| `llama-quantize` | Works. Its default (`NONE` on macOS) is an internal choice, so for compressed input it uses `MMAP`. Plain input is unchanged |
| Split models | Each file is opened by `llama_open_model_source`, so each can be plain or compressed, in any mix |
| `llama_adapter_lora_init` | Explicit error: compressed adapters are not supported |
| Build without the format's library | The file is recognised and refused, with the way to enable it: "gzip-compressed, but this build has no gzip support (configure with -DLLAMA_ZLIB=ON)" |

Metadata-only mode validates the container (index, footer, single stream, size) and decodes the header. It does not check block checksums of data beyond the header, so a corrupt body is found at the real load, not at the fit step.

### Split models

`llama_get_list_splits` finds the siblings from the name. It now sets one compression suffix (`.xz`) aside before calling the unchanged public `llama_split_prefix` and `llama_split_path`, and puts it back on the generated names. So `model-00001-of-00003.gguf.xz` finds `model-00002-of-00003.gguf.xz`, and the same for `.gz`. The suffixes come from the codec table. The suffix is naming only. Each sibling is still sniffed, so files that are compressed but named `.gguf`, and mixed sets, load too. An explicit split list (`llama_model_load_from_splits`) needs no naming at all. Each compressed split has its own backing store, so memory is the sum, as for plain files.

### Callers outside the model loader

| Caller | Outcome |
|---|---|
| `common_get_gguf_info` (router modalities, trained context) | Uses `llama_gguf_open_metadata`, so a compressed model is described correctly. Before, it returned empty info |
| `common_speculative_types_from_gguf` (draft model auto-detection) | Same. Before, a compressed draft model was silently not detected |
| `tools/imatrix` (two model-path readers) | Same |
| LoRA | Explicit error |
| Control vectors, `mtmd` clip (mmproj), `export-lora`, `gguf-split`, `gguf-hash` | **Not supported.** They read tensor data with `fread` and need a readable file. They report the parser's "invalid magic" error. Not changed |
| `tools/results` | Opens the model's GGUF but never uses it, not changed |

`llama_gguf_open_metadata` lives in `src/llama-ext.h`, the repository's staging header, and is not in `llama.h`. It is the only exported addition.

## Progress and cancellation

`llama_progress_callback` keeps its documented contract: a value in [0, 1], and returning false cancels.

* The value is monotonic. A plain load reports what it did before.
* For a compressed file, decoding takes the first half of the range and tensor placement the second half. For a split model the decode half is divided by the number of files (from the name, or from the list). This weighting is a heuristic, not an estimate of remaining time.
* The decoder is driven in 8 MiB slices. Progress is reported and cancellation is noticed after each slice.
* A cancelled decode throws `llama_load_cancelled`, which `llama_model_load` turns into the existing status `-2` ("cancelled model load"). The store is destroyed on the way out.

## Errors and ownership

* The store owns its descriptors. The `llama_file` that wraps it owns the store. The loader owns the files only while loading. The model owns the mappings.
* Any exception in the open path destroys the store (RAII) and the kernel frees the pages once the last mapping goes. Nothing is retried in another way.
* Success of a decode requires: stream end, exactly the decoded size, all input consumed, and one stream. Truncated data, trailing bytes (including valid stream padding), concatenated streams, a zero-size result and a corrupt block are all errors with a specific message.
* A non-GGUF payload inside a valid xz stream is reported by the GGUF parser as usual.

## Optional dependencies

`LLAMA_LZMA` (xz, liblzma) and `LLAMA_ZLIB` (gzip, zlib) are each `AUTO` (default), `ON` or `OFF`. They are independent. One CMake function, `llama_optional_codec`, does the work for both.

* `AUTO` uses the library if `find_package` finds it, and prints `llama: <format> support for compressed models: ON/OFF` at configure time. The macOS SDK ships `liblzma.tbd` but no `lzma.h`, so on a machine without Homebrew's header `AUTO` builds without xz. It does ship `zlib.h`, so gzip is on.
* `ON` fails the configuration if the library is missing.
* The dependencies are `PRIVATE` to `llama`. A shared `libllama` needs nothing from consumers.
* A static `libllama.a` needs the consumer to link what it was built with. `cmake/llama-config.cmake.in` adds `LibLZMA::LibLZMA` and `ZLIB::ZLIB` to the link interface when the build was static and the format was enabled. `cmake/llama.pc.in` has `Libs.private: -llzma -lz` (only the enabled ones).

## Memory behaviour (measured)

All numbers are macOS, Apple Silicon, `phys_footprint` from `task_info`, one load per fresh process, 2 to 3 runs per cell. Models: `stories15M` f32 (98,357,920 B, 93.8 MiB), `stories15M` Q4_0 (19,077,344 B, 18.2 MiB), `stories260K` (1.1 MiB). They are small. **No multi-gigabyte model was available, so nothing here says how a large model behaves.**

Footprint growth caused by the load:

| Model, backend | plain `.gguf` | `.gguf.xz` (xz -6, 4 blocks) |
|---|---|---|
| f32, Metal (`-ngl 99`), buffers `CPU_Mapped` 35.2 + `MTL0_Mapped` 58.0 MiB | +2.8 to +4.9 MiB | +103.9 to +108.3 MiB |
| f32, CPU (`-ngl 0`), `CPU_Mapped` 93.1 MiB | +2.9 to +4.1 MiB | +105.1 to +107.9 MiB |
| Q4_0, Metal | +2.0 to +4.5 MiB | +28.8 to +29.6 MiB |
| Q4_0, CPU, `CPU_Mapped` 4.9 + `CPU_REPACK` 12.5 MiB | +16.6 to +17.4 MiB | +40.9 to +41.3 MiB |

What this shows and does not show:

* The xz result is the plain result plus about one decoded model. For f32 that is 93.8 MiB plus roughly 8 to 12 MiB. The remainder is consistent with the 8 MiB LZMA2 dictionary of preset 6, but I did not isolate it.
* The backing pages are dirty shared memory, so they count in the footprint. With a plain file the mapped pages are clean file pages and are not counted. That difference, not a copy, is why the xz footprint is higher.
* Q4_0 on the CPU: the repacked weights are a separate 12.5 MiB buffer in both cases. With xz, the source pages stay resident as dirty memory. This is a backend allocation, not decoder overhead, and is the same repack that happens with a plain file.
* `vocab_only` and `no_alloc` loads of the compressed 98 MB model grew the footprint by 0.0 to 2.1 MiB. A full load grew it by 93.2 MiB.
* The footprint after the load equals the peak within about 1 MiB, so there is no large temporary at load time.

Decoder buffers, found and fixed. liblzma's multi-threaded decoder decodes each block into buffers of its own and copies it out. With a fixed 512 MiB budget this small model, with all four 24 MiB blocks in flight, grew the footprint by **292 MiB (about 3 times the model)** and kept it. Changing only `memlimit_threading` (8, 24, 64, 128, 512 MiB) gave +108, +104, +169, +239, +292 MiB, with wall time falling from 3.1 s to 1.0 s. The budget is now `min(decoded size / 16, 512 MiB)`. When no worker fits in it, liblzma decodes straight into the destination. This is a speed-for-memory choice made on a measured cause, not a proof that the destination is the only buffer for large files: for a multi-gigabyte model the cap is 512 MiB of worker buffers.

| `llama-completion -n 1`, 98 MB model | wall | peak footprint | max RSS |
|---|---|---|---|
| plain | 0.12 to 0.14 s | 54 to 58 MB | 190 to 194 MB |
| xz, now | 3.24 to 3.29 s | 155 to 158 MB | 198 to 202 MB |

A small xz model is decoded on one thread, about 30 MB/s here. A large, many-block file gets a larger budget and parallelism, but I did not measure that.

gzip, same model (89.6 MB compressed), fresh process each, 3 runs:

| | wall | peak footprint | footprint growth by the load |
|---|---|---|---|
| plain | 0.12 to 0.15 s | 53 to 54 MB | +2.8 to +4.9 MiB |
| xz | 3.26 to 3.30 s | 155 to 159 MB | +104 to +108 MiB |
| gzip | 0.33 to 0.34 s | 147 MB | +97.3 to +101.0 MiB (CPU and Metal) |

gzip adds only the decoded model: its decoder state is a 32 KiB window, there are no worker buffers, and the growth is the model (93.8 MiB) plus about 3 to 7 MiB. It is also ten times faster to decode than xz here, at a larger file. The full-load footprint check in `test-model-load-compressed` gave +93.2 MiB for gzip and for xz.

## Tests

| Test | What it establishes | What it does not |
|---|---|---|
| `test-model-source` (no model needed) | The source layer on a repo vocab file, for each format built in: plain and compressed give the same metadata and the same bytes. Metadata-only is not mappable. The mapping outlives the store. The descriptor is close-on-exec and, on macOS, refuses writable mappings. A dozen damaged containers are rejected (truncations, trailing byte, trailing zero bytes, concatenation, empty payload, flipped byte), and a valid stream of non-GGUF data is not accepted as a model. For gzip also a trailer size that is too small or too large. Progress is monotonic and cancellation frees the store. The codec table, the suffix helper and the "not built in" error for each format. `llama_gguf_open_metadata` | A real model. gzip files of 1 GiB or more (see below) |
| `test-model-load-compressed` (needs a model) | For each format: public API only, same parameter count, size and logits. Progress monotonic and ends at 1.0. Cancellation early and late. Real load without mmap refused. `vocab_only` and `no_alloc` load. On macOS with a model of 32 MiB or more: they do not grow the footprint, and a full load grows it by about the model size and not much more. The check failed with the old fixed 512 MiB xz budget (129 MiB growth, against a bound of the model plus 32 MiB when I ran it; the bound is now the model plus 20 MiB and was not re-run against the old budget) | Other platforms, large models |
| `test-compressed-model.sh` (needs a model; runs each format whose tool, `xz` or `gzip`, is installed) | `llama-completion`, `llama-tokenize`, `llama-fit-params`, `llama-quantize`, `llama-gguf-split` on the compressed file. Same text. Quantized output byte-identical to the plain one. Split models named `.gguf.<ext>`, named `.gguf`, and mixed. Four bad inputs rejected with no output. No file over 64 KiB appears in the working or temp directory | The Linux store |

The two model tests skip when no model is given (`LLAMACPP_TEST_MODELFILE`), like the other model tests. They never download. The unit test builds its fixtures with the encoders of liblzma and zlib, so no command line tool is needed there.

The gzip counting pass (compressed size of 1 GiB or more) is not in the test suite, because it needs a file that large. It was run by hand on generated files: 1.15 GiB of random data (the trailer happens to be right), and 1.12 GiB compressed that decodes to 4.49 GiB, whose size and every byte were correct.

Also run once by hand:

* A sandbox that forbids file writes under `/Users`, `/tmp` and `/private/tmp`: the compressed model loaded and generated text, and no file of 1 MiB or more appeared in the allowed temp areas.
* `kill -9` at 0.4 s, 1.2 s and 2.0 s into a decode: exit 137, no files left, the next run fine.
* `llama-server` on the compressed model over loopback: `/health` ok, `/completion` produced text. `llama-cli` produced the same text as the plain file.
* A build with `LLAMA_LZMA=OFF`, and another with `LLAMA_ZLIB=OFF`: the explicit error for that format with the flag to enable it, exit 1; plain models and the other format still load; `LLAMA_LZMA=ON` with the package disabled fails to configure.
* Install audit: a shared install and a static install, each used by a small external CMake project with `find_package(llama)`. Both load a compressed model. Both formats load through the static one. Removing the liblzma line from the static config makes the consumer fail with `Undefined symbols: _lzma_code`, and removing the zlib line with `_inflate`. `pkg-config --static --libs` adds `-llzma -lz`, plain `--libs` does not.
* The full suite: 63 of 65 passed before gzip was added; the new tests were run again afterwards. `test-tokenizers-ggml-vocabs` clones a repository from the network and was not counted, and `test-backend-sampler` fails identically on unmodified `HEAD` with the same model (it expects a specific model).

## Checking the design: adding gzip

gzip was added after the rest was finished, as a test of the layering.

New: `src/llama-codec-gz.cpp`, 184 lines. It is the whole decoder, including the size recovery above.

Changed for the format, outside that file:

* `src/llama-codec.cpp`: a magic and one table entry per build variant.
* `src/llama-codec.h`: one prototype, and one new field, `enable`.
* `src/llama-model-source.cpp`: one string.
* CMake: the option, and the xz block turned into a function used by both. The installed config and `.pc` file list the new library.
* Tests: the two C++ tests and the script run once per format.

Not changed: the backing store, `llama_file`, `llama_mmap`, the model loader, `llama.cpp`, progress and cancellation, split discovery (the suffix comes from the table), quantize, `llama_gguf_open_metadata`, and every caller outside the model loader.

Where the design pushed back:

1. **A hard-coded xz hint.** The error for a format that is not built in said `-DLLAMA_LZMA=ON` whatever the format. It is now a field of the codec.
2. **Size before decoding.** The contract came from xz, which has an index. gzip has none, and the 4 GiB limit of its trailer is exactly the case that matters for models. It was solved inside the codec (trailer, or one counting pass), which is what the interface allows, but the cost is real: the fit pass of a large gzip file decodes it once.
3. **Weaker early validation.** The metadata-only open relies on the container being checkable from its end. xz can, gzip cannot, so a truncated gzip file is found at the real load. The unit test now states this per codec instead of assuming xz behaviour.
4. **Copy-pasted build logic**, which would have been a third copy for a third library. Now one function.
5. **Tests that assumed xz** in names and fixtures. They are parameterised.

The `parallel` argument of `open` means nothing to gzip, which cannot decode one stream in parallel. It is ignored. If more formats have no use for it, the argument is a candidate for removal.

## Verified and not verified

Verified here (macOS, Apple Silicon, Metal and CPU, small models):

* Compressed and plain give the same output through `llama-completion`, `llama-cli`, `llama-server`, a tiny external program, and `llama-simple` (unmodified).
* No uncompressed file is written (sandbox), nothing remains after `kill -9`, errors are reported and clean up.
* `vocab_only`, `no_alloc` and the fit pass do not decode the body and do not create or map a store (footprint, and the source type refuses mapping).
* Split models work in three naming styles. Quantize works and matches.
* The xz footprint is the plain footprint plus about one model, for f32 and Q4_0, after the decoder-budget fix.
* gzip: same output as plain through the same binaries; a gzip of 1.12 GiB that decodes to 4.49 GiB was sized and decoded correctly, which is also the largest decode through the store so far.
* Builds without liblzma and without zlib; shared and static installs.

Not verified:

* **Linux.** `memfd`, `fallocate` and sealing were never run. The branch is parsed and type-checked against a shim header on macOS, which says nothing about runtime behaviour. Treat it as untested.
* **Large models.** The largest model tested is 98 MB. The speed and the memory of a multi-gigabyte model, the effect of the 512 MiB cap and parallel decoding are unmeasured. The planned slope test over several large sizes is **inconclusive**.
* **Memory pressure.** What the system does when the decoded size exceeds free memory (swap, compression, `SIGBUS`) was not tested.
* **Discrete GPUs.** The copy to device memory and the release of the host pages after load are untested here. Metal wrapping of ranges split at `max_buffer_size` is untested.
* **Windows**, and all other platforms: a clear error, not built.
* `llama-embedding` (the test model is not an embedding model and fails in plain form too), the server router's use of `common_get_gguf_info`, `imatrix`, speculative auto-detection and the LoRA error were not run end to end. They are covered by the unit tests of `llama_gguf_open_metadata` and by reading the code.
* Single-block xz files of any size beyond 98 MB, and xz files made with `-T` and large preset dictionaries.
* gzip: real models of several GiB (only generated files were used), the 1 GiB threshold as a policy, and files made by block-gzip tools.

## Rejected alternatives

| Alternative | Why not |
|---|---|
| Decoding in each frontend | Duplicates logic and misses every other consumer |
| A public `llama_model_load_from_fd` | Not needed once the loader opens the source itself. Removed |
| External launcher with an inherited descriptor | `/dev/fd/N` on macOS shm gives `EBADF`, and `fread` cannot read it |
| Temporary or unlinked file | Writes the uncompressed model to disk |
| Compression inside `gguf_init_from_file` | Couples the lowest layer to a platform-specific, optional dependency |
| Keeping the decoded source between the fit pass and the real load | Hidden global state; metadata-only decoding removes the need and was shown to be correct |
| Buffering the header with a retry loop | Cannot tell truncation from corruption; the streaming reader does not guess |
| Decoding straight into per-tensor backend buffers | Couples the codec to tensor layout. Would save the host copy on discrete GPUs, not done |
| Always trusting the gzip trailer | Wrong for 4 GiB or more, and a wrong size must not be silent |
| Always counting a gzip file first | Doubles the decode for files that do not need it. Kept for 1 GiB and over |
| FUSE or another virtual filesystem | Needs a system extension and a daemon |
| `MAP_ANON` memory | No descriptor for `llama_mmap` to map |
| A RAM-based size cap, or a `progress_callback(0.0f)` cancel poll | Rejected as arbitrary and as an abuse of the callback contract |
