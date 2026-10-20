# Learning guide: compressed models in the loader (xz and gzip)

This guide follows the code in this checkout. Read [compressed-model-design.md](compressed-model-design.md) first for the decisions, the measured numbers and the list of what is not verified.
Search for the function names if line numbers have moved.

## 1. Architecture

```
model.gguf.xz  (a regular file, compressed)
      |
      v
llama_open_model_source(path)                  src/llama-model-source.cpp
      |  read the first bytes: "GGUF" -> plain; xz magic -> compressed
      |
      |-- plain: gguf_init_from_file + llama_file(path)         as it always was
      |
      '-- xz or gzip: llama_decoder                             src/llama-codec-xz.cpp, llama-codec-gz.cpp
                   1. read the xz index at the END of the file -> decoded size N
                   2. llama_backing_store::create(N)            src/llama-backing-store.cpp
                        macOS: shm_open + ftruncate + mmap RW, a read-only descriptor, shm_unlink
                   3. decoder.read(store.write_addr() + off)     the decoder writes into the final pages
                   4. store.seal()                               the writer goes away
                   5. llama_file(store): mappable, not readable
      v
llama_model_loader                             src/llama-model-loader.cpp
      |  init_mappings: llama_mmap(file) -> mmap(PROT_READ, MAP_SHARED, fd)    src/llama-mmap.cpp
      v
tensor->data = mapping + offset                the same physical pages the decoder wrote
      |
      v
CPU reads them directly; Metal wraps the same pages ("MTL0_Mapped"); other backends or repacking copy
```

Three questions, three pieces, and a fourth that stays where it was:

* Where do the uncompressed bytes live? `llama_backing_store`.
* How are they produced from the compressed bytes? `llama_decoder` (xz here).
* How does the loader get hold of them? `llama_file`, extended, plus `llama_open_model_source`.
* Which backend uses them directly, and which copies? `load_tensors`, unchanged.

## 2. Source tour

`git status` shows all of it. About 950 new lines in `src/` (the codec for gzip is one file of 184 lines), and about 20 modified files.

### `src/llama-codec.h`, `llama-codec.cpp`

`llama_decoder` is a pull interface: `size()` tells the decoded size, `read(dst, n)` fills the next bytes. The `llama_codec` table lists each format with its magic bytes, its customary suffix, whether this build can decode it, and a factory. The xz entry is always listed. In a build without liblzma `available` is false, which is how the loader can say "this file is xz, but this build cannot read it".

Why pull and not "decode everything into this buffer"? The same decoder serves two uses: filling the backing store, and feeding just the GGUF header to the parser. Section 4 of this guide returns to that.

### `src/llama-codec-xz.cpp`

* `xz_decoded_size` reads the index at the end of the file (`lzma_file_info_decoder`). It is the only place the decoded size is stored, and it is needed first, because macOS lets you `ftruncate` shared memory once. It also checks there is exactly one stream and no bytes after it.
* `llama_decoder_xz::read` drives `lzma_code` straight into the caller's pointer. The call that completes the output goes on until liblzma reports the end of the stream, so the checks on the index and footer are not skipped.
* `memlimit_threading = min(decoded size / 16, 512 MiB)`. This line has a history: see section 5.

### `src/llama-codec-gz.cpp`

The second codec, written after everything else, as a check on the design. Its decoder is the shorter of the two because zlib drives straight into the caller's pointer and has no worker threads. The interesting part is `gz_decoded_size`: gzip has no index, so the size comes from a 4-byte trailer that wraps at 4 GiB, and the code has to decide when to trust it (section 4 explains the policy).

### `src/llama-backing-store.cpp`

Two small implementations in one file, chosen at compile time.

* macOS: `shm_open(O_EXCL)`, `ftruncate`, a writable `mmap`, then a second descriptor opened **read-only** by name, and only then `shm_unlink`. `seal()` unmaps the writer and closes the writable descriptor. What is left cannot be mapped writable, even by this process (measured: `mmap(PROT_WRITE)` and `mprotect` both fail).
* Linux: `memfd_create` with sealing, `fallocate` so that running out of memory is an error up front, and `F_ADD_SEALS` in `seal()`. **Written from documentation and never run.**
* Other platforms: `create` throws a clear error.

### `src/llama-mmap.h`, `llama-mmap.cpp`

`llama_mmap` only needs `size()` and `file_id()` from a `llama_file`. Two new kinds of `llama_file` supply them without a path: one that owns a backing store, and one that has only a size (for a model that was never materialised). `readable()` and `mappable()` tell them apart. `tell`, `seek` and `read` on a file that cannot do them throw a clear error, instead of failing later with a confusing errno.

### `src/llama-model-source.cpp`

`llama_open_model_source` is the one place that decides how a file is provided. It also holds the progress and cancellation rules (`llama_load_progress`). In metadata-only mode, `llama_decoder_reader` feeds the GGUF parser from the decoder. It insists the parser reads strictly in order, which was measured to be true (section 4).

### `src/llama-model-loader.cpp`

The two places that used to open a file by name, the main file and each split, now call `llama_open_model_source`. `init_mappings` skips a file that cannot be mapped. `llama_get_list_splits` sets a compression suffix aside, matches the usual split name, and puts the suffix back on the sibling names.

### Smaller changes

* `src/llama.cpp`: passes `no_alloc || vocab_only` as "metadata only", and turns `llama_load_cancelled` into the existing cancelled status.
* `src/llama-quant.cpp`: uses mmap for compressed input.
* `src/llama-adapter.cpp`: an explicit error for a compressed LoRA file.
* `src/llama-ext.h`, `common/`, `tools/imatrix`: `llama_gguf_open_metadata` replaces a direct `gguf_init_from_file` where only the metadata is wanted.
* `CMakeLists.txt`, `src/CMakeLists.txt`, `cmake/*.in`: the `LLAMA_LZMA` option and what a static install must export.

## 3. Systems concepts, as used here

**File descriptor.** A small integer per process that indexes a table of open objects. `dup` makes a second number for the same object. Closing a number does not destroy the object while another number or a mapping still refers to it. In this design `llama_file` owns the store, and the model's mappings keep the pages alive after the loader and the store are gone (the unit test checks this).

**`fstat` and `mmap` against `read` and `lseek`.** A POSIX shared memory object on macOS answers `fstat` and `mmap` but refuses `read`, `pread` and `lseek`. This is why the loader parses the header from a temporary mapping, and why the file type says "mappable, not readable".

**`mmap`.** Makes a range of your address space show the pages of an object instead of copying them. With `MAP_SHARED` every mapping of the object sees the same physical pages. Here the decoder's writable mapping and the loader's read-only mapping are two views of the same pages. Nothing is copied between them.

**Read-only is a property of a handle.** Unmapping the writer does not by itself make an object read-only; anyone holding a writable descriptor can map it again. On macOS the store closes the writable descriptor and keeps only one opened read-only. On Linux a seal does it in the kernel. `read_only_enforced()` says whether it worked.

**Page cache, clean and dirty pages.** Pages of a regular file are clean: they match the disk, and the kernel can drop and re-read them. That is why mapping a normal model costs little "memory footprint". A shared memory object has no disk behind it, so its pages are dirty and can only be compressed or swapped. This, and not a copy, is why the compressed path shows a higher footprint than the plain one.

**Anonymous against unlinked.** An anonymous object never had a filesystem name (our shared memory after `shm_unlink`). An unlinked file is a regular file whose name was removed; it lived in the filesystem and may have been written to disk. `mkstemp` plus `unlink` is the second kind.

**Descriptor inheritance.** A descriptor without close-on-exec is passed to child processes. `llama-server` in router mode starts child servers, so the store's descriptors are marked close-on-exec.

**Allocator retention.** A program can free memory and still show it in its footprint. Before concluding that something is "freed", know which allocator made it and measure after the free. The decoder section below is an example of why.

**Misconceptions to drop**

* "Closing the fd unmaps the memory." No. A mapping is independent of the descriptor it came from.
* "mmap loads the file into RAM." No. It maps lazily. Here the pages are already in RAM because we put them there.
* "No file on disk means no memory use." The opposite: the model stays in RAM, dirty.
* "A different footprint means a copy." Not by itself. Clean and dirty pages are counted differently, and the backend can allocate for repacking.
* "A category name in a memory tool tells you what was freed." No. Trace the allocation.

## 4. Why the design looks like this

**The loader looks at content.** `llama_open_model_source` reads a few bytes. A compressed file is compressed whatever it is called. This also keeps the old path intact: unknown content goes to the old code.

**The decoder pulls.** In metadata-only mode (fit, `vocab_only`, `no_alloc`) the loader needs the header and nothing else. `gguf_init_from_callback` exists for this. A probe measured how it reads: with `no_alloc`, strictly sequentially and only up to the end of the header: 727,058 bytes of a 98,357,920-byte file, in 128,471 calls of at most 48 bytes. So a streaming decoder with a small buffer is enough, and decoding stops there. That is why the fit pass does not decode the whole model, and why it does not need a cache between the fit and the real load.

**Why the second codec was a good test.** A design that works for one example may only fit that example. xz hands over its decoded size from an index at the end of the file. The store and the loader both depend on that number before any decoding. gzip does not give it: its trailer holds the size modulo 2^32. The fix lives entirely in the codec, as it should: the trailer is used up to 1 GiB compressed and verified by the decode, and above that the data is decoded once and counted. Everything else stayed as it was. What did leak was small, and is a useful list of the ways a layer can hide an assumption: an error message that named xz's build flag, build logic copied for one library, and tests that said "xz" in their fixtures.

**What a missing index costs.** With xz the container can be checked from its end, so even the header-only open used by the fit pass notices a truncated file. With gzip it cannot, so that check moves to the real load. And a large gzip file is decoded once, with the output discarded, just to count it. An interface can hide a difference like this, but it cannot remove it.

**Why `--no-mmap` is an error.** Without mmap the loader reads tensors with `read`, which the store does not support. Making it work would copy the whole model to a second place, which is the cost the design avoids. It is better to refuse and say why.

**Why quantize picks mmap.** Its default on macOS (no mmap) is a performance choice by the tool, not by the user. For a compressed file it chooses mmap, which is what Linux and Windows already do.

**Why split discovery strips a suffix.** The public `llama_split_prefix` requires a name ending in `-NNNNN-of-NNNNN.gguf`. Changing it would change a public function. Setting `.xz` aside inside the loader leaves the public one alone and keeps the format decision in the content.

**Why the callers outside the loader are different.** Control vectors, the multimodal projector and a few tools read tensor data with `fread`. They need a readable file and are left alone, with the parser's error. The two callers that only read key-value pairs and tensor names go through `llama_gguf_open_metadata`, because a silent empty answer (for example, a draft model not detected) is worse than an error.

## 5. The decoder buffer that tripled the footprint

This is a worked example of measuring before explaining.

1. The first compressed load of the 98 MB model grew the process footprint by about 292 MiB, three times the model. It stayed there after the load.
2. A guess was available ("a copy somewhere in llama.cpp"). The data flow in the source said no: the decoder writes into the store, the loader maps the same pages, Metal wraps them, and a plain `.gguf` run grew by only about 4 MiB.
3. The next step was to change one thing. liblzma's multi-threaded decoder decodes each block into its own buffers and copies out. Its memory budget was a fixed 512 MiB, more than this whole model. Changing only that budget gave growth of +108, +104, +169, +239, +292 MiB for 8, 24, 64, 128 and 512 MiB. With a small budget liblzma decodes straight into the destination.
4. The fix is a budget of one sixteenth of the model, capped at 512 MiB. The cost is speed: a small model now decodes on one thread, about 3.3 s for 98 MB instead of 1.0 s.
5. A test now guards it. It makes a file like `xz -6` does, with several large blocks, and checks that a full load grows the footprint by about the model size and not much more. The first version of the test used 4 MiB blocks and would never have seen the problem.

What the example does not prove: that large models have no extra buffers. The cap is 512 MiB for them, and no large model was available to measure.

## 6. Testing: what each result means

| Test | Establishes | Does not establish |
|---|---|---|
| `test-model-source` | For each format: metadata and bytes equal between plain and compressed. Metadata-only is not mappable. The mapping outlives the store. Damaged containers are rejected, including concatenated streams, trailing bytes and stream padding. Cancellation and monotonic progress | Anything about a real model |
| `test-model-load-compressed` | For each format: same logits through the public API. Cancellation early and late. `vocab_only`, `no_alloc` do not grow the footprint. A full load grows it by about one model | Other platforms, large models |
| `test-compressed-model.sh` | For each format, the command line tools agree with the plain file, including split models in three naming styles and quantize | The Linux store |
| A sandbox forbidding writes under `/Users`, `/tmp` | No file was created there. It was checked that the sandbox blocks a probe write | That nothing is written in the allowed temp areas, beyond a scan for files over 1 MiB |
| `kill -9` during the decode | No leftover files, the next run is fine | That the kernel returned the memory (inferred from `shm_unlink` coming first) |
| Footprint of plain against compressed, per model and backend | The extra is about one model, and the Q4_0 repack buffer is the same in both | A cause for the last 8 to 12 MiB. Preset 6 uses an 8 MiB dictionary, which fits, but it was not isolated |

A caution about tools: an `LD_PRELOAD`-style interposer that worked on the first prototype saw nothing from inside `libllama.dylib`. It was dropped in favour of the sandbox and the footprint measurements. If a check reports nothing, first ask whether it can see the code in question.

## 7. Guided walkthrough

1. **Read the diff.** `git status`, then `git diff src/llama-model-loader.cpp src/llama-mmap.cpp`. Predict which function calls which before opening them.
2. **Trace a load by reading.** Start at `llama_model_load_from_file_impl` in `src/llama.cpp`, go to `llama_model_load`, then the `llama_model_loader` constructor, then `llama_open_model_source`, then `llama_backing_store::create`. Write down at each step which object owns the descriptor.
3. **Watch it happen.** Run `llama-completion -m stories15M.gguf.xz -n 1 -p hi` and in another terminal `lsof -p <pid>` during the first seconds. Find the shared-memory descriptor (type `PSXSHM`). Repeat with the plain file and find the regular file.
4. **Read the log.** Find `decoding ... into ... MiB of memory`, `load_mode = mmap`, and the buffer lines (`CPU_Mapped`, `MTL0_Mapped`). Add `-ngl 0`. For a Q4_0 model, find `CPU_REPACK`.
5. **Probe the OS yourself.** Write 20 lines of C: `shm_open`, `ftruncate`, a writable `mmap`, a second `O_RDONLY` open by name, `shm_unlink`, close the first descriptor. Then try `mmap(PROT_WRITE)` on the one left, `mprotect`, `ftruncate` and `read`, and print `strerror`. Compare with the table in the design note.
6. **See the metadata-only path.** Run `llama-fit-params -m stories15M.gguf.xz` and `llama-tokenize -m ... --ids -p hi`. Note they return in about 0.1 s while a full load takes about 3 s.
7. **Repeat the decoder experiment.** Temporarily change `memlimit_threading` in `src/llama-codec-xz.cpp` to a constant, rebuild, and measure the footprint of a compressed load with `/usr/bin/time -l`. Restore the line.
8. **Break the split naming on purpose.** Compress three split files, give the first a name with `.xz`, delete the second, and read the error. Repeat with `.gz`.
9. **Fool the gzip size.** Take a gzip file and change its last 4 bytes. Predict which message appears for a trailer that is too small and for one that is too large, then run `llama-completion` on it. Then read `gz_decoded_size` and find the 1 GiB rule.
10. **Compare the two formats.** Compress the same model with `xz -6` and `gzip -6` and time a load of each. Which is faster, which is smaller, and what does each do to the footprint? The design note has the numbers to check against.

## 8. Check your understanding

1. Why does the loader parse the GGUF header from a temporary mapping of the store, and why not for a metadata-only open?
2. After the loader and the store are destroyed, what keeps the pages alive, and what ends their life?
3. The compressed run has a larger footprint than the plain one, with a similar maximum RSS. Why, and what would a second copy of the model look like in the numbers?
4. Why must the decoded size be known before decoding, and what stops a format without a stored size from being added?
5. A reviewer proposes `mkstemp` followed by `unlink` "because it is simpler". Which requirement does that break, and what experiment shows it?
6. The footprint tripled and stayed. Why was "freed memory still counts" not an acceptable explanation, and what single measurement separated the possibilities?
7. Why does the fit pass not need to keep the decoded source for the real load?
8. Which callers still reject a compressed file, and for which reason does each need a readable file?
9. gzip keeps its size modulo 2^32. Why is a gzip model under 1 GiB compressed that decodes to 5 GiB refused instead of loaded, and why is that better than loading it?
10. A third format has no way to give its size without decoding, and a decode is slow. What in the current design would you change, and what would you leave alone?
