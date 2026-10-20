#!/usr/bin/env bash
# Loads a compressed GGUF through the command line tools and compares it with the plain file:
# generation, vocab_only, the memory fit, quantization, split models in every naming style, bad input.
# Runs for each format whose tool is installed (xz, gzip). Without a model (arg or LLAMACPP_TEST_MODELFILE) it is skipped.
set -euo pipefail

if [ $# -lt 1 ]; then
    printf "Usage: %s <bin-dir> [<model.gguf>]\n" "$0"
    exit 1
fi

bin=$(cd "$1" && pwd)
model=${2:-${LLAMACPP_TEST_MODELFILE:-}}

if [ -z "$model" ] || [ ! -f "$model" ]; then
    printf "SKIP: no model, give one or set LLAMACPP_TEST_MODELFILE\n"
    exit 0
fi

base=$(mktemp -d "${TMPDIR:-/tmp}/test-compressed-model.XXXXXX")
trap 'rm -rf "$base"' EXIT
fail=0
n_formats=0

check() { # check <description> <command...>
    local what=$1
    shift
    if "$@"; then
        printf "  ok    %s\n" "$what"
    else
        printf "  FAIL  %s\n" "$what"
        fail=1
    fi
}

gen() { # gen <model> : greedy text of the model, stderr dropped
    "$bin/llama-completion" -m "$1" -n 20 --temp 0 -no-cnv -p "Once upon a time" 2>/dev/null
}

run_format() { # run_format <name> <extension> <command that compresses the file given as its last argument to stdout...>
    local fmt=$1 ext=$2
    shift 2
    local comp=("$@")
    local work="$base/$fmt"
    mkdir -p "$work/cwd" "$work/tmp"

    printf "== %s\n" "$fmt"

    # the working directory and the temp directory must stay free of anything model-sized
    run_clean() { (cd "$work/cwd" && TMPDIR="$work/tmp" "$@"); }
    no_big_files() { [ -z "$(find "$work/cwd" "$work/tmp" -type f -size +64k 2>/dev/null)" ]; }

    printf "plain and compressed\n"
    "${comp[@]}" "$model" > "$work/model.gguf$ext"
    gen "$model" > "$work/plain.txt"
    run_clean "$bin/llama-completion" -m "$work/model.gguf$ext" -n 20 --temp 0 -no-cnv -p "Once upon a time" 2>/dev/null > "$work/c.txt"
    check "the same text" cmp -s "$work/plain.txt" "$work/c.txt"
    check "the text is not empty" test -s "$work/c.txt"
    check "nothing model-sized in the working or temp directory" no_big_files

    cp "$work/model.gguf$ext" "$work/renamed.bin"
    gen "$work/renamed.bin" > "$work/renamed.txt"
    check "a compressed file is recognised by content, whatever its name" cmp -s "$work/plain.txt" "$work/renamed.txt"

    printf "vocab_only, fit\n"
    check "tokenize gives the same ids" cmp -s \
        <("$bin/llama-tokenize" -m "$model" --ids -p "Hello world" 2>/dev/null) \
        <("$bin/llama-tokenize" -m "$work/model.gguf$ext" --ids -p "Hello world" 2>/dev/null)
    fit_runs() { "$bin/llama-fit-params" -m "$work/model.gguf$ext" >/dev/null 2>&1; }
    check "fit-params runs" fit_runs

    printf "mapping is required\n"
    refused() {
        ! "$bin/llama-completion" -m "$work/model.gguf$ext" -lm none -n 1 -p hi >/dev/null 2>"$work/lm.txt" && grep -q 'memory mapping' "$work/lm.txt"
    }
    check "load mode none is refused, with the reason" refused

    printf "quantize\n"
    "$bin/llama-quantize" "$model" "$work/q-plain.gguf" Q8_0 >/dev/null 2>&1
    "$bin/llama-quantize" "$work/model.gguf$ext" "$work/q-c.gguf" Q8_0 >/dev/null 2>&1
    check "a plain model quantizes" test -s "$work/q-plain.gguf"
    check "a compressed model quantizes to the same file" cmp -s "$work/q-plain.gguf" "$work/q-c.gguf"

    printf "split models\n"
    mkdir -p "$work/split/a" "$work/split/b" "$work/split/c"
    "$bin/llama-gguf-split" --split-max-tensors 20 "$model" "$work/split/model" >/dev/null 2>&1
    local n_splits
    n_splits=$(ls "$work"/split/model-0000?-of-0000?.gguf | wc -l | tr -d ' ')
    if [ "$n_splits" -ge 2 ]; then
        local first f b i=0
        first=$(basename "$(ls "$work"/split/model-00001-of-*.gguf)")
        gen "$work/split/$first" > "$work/split-plain.txt"
        for f in "$work"/split/model-0000?-of-0000?.gguf; do
            b=$(basename "$f")
            "${comp[@]}" "$f" > "$work/split/a/$b$ext"                     # named .gguf<ext>
            "${comp[@]}" "$f" > "$work/split/b/$b"                         # compressed, named .gguf
            if [ $((i % 2)) -eq 1 ]; then "${comp[@]}" "$f" > "$work/split/c/$b"; else cp "$f" "$work/split/c/$b"; fi   # mixed
            i=$((i + 1))
        done
        check "split, plain"                           test -s "$work/split-plain.txt"
        check "split, every file named .gguf$ext"      cmp -s "$work/split-plain.txt" <(gen "$work/split/a/$first$ext")
        check "split, compressed files named .gguf"    cmp -s "$work/split-plain.txt" <(gen "$work/split/b/$first")
        check "split, some compressed and some plain"  cmp -s "$work/split-plain.txt" <(gen "$work/split/c/$first")
    else
        printf "  skip  the model is too small to split\n"
    fi

    printf "bad input\n"
    local size bad rc ok out
    size=$(wc -c < "$work/model.gguf$ext" | tr -d ' ')
    head -c $((size / 2)) "$work/model.gguf$ext" > "$work/trunc.bad"
    { cat "$work/model.gguf$ext"; printf 'x'; } > "$work/trailing.bad"
    cat "$work/model.gguf$ext" "$work/model.gguf$ext" > "$work/concat.bad"
    cp "$work/model.gguf$ext" "$work/flip.bad"
    printf '\x55' | dd of="$work/flip.bad" bs=1 seek=$((size / 2)) conv=notrunc 2>/dev/null
    for bad in trunc trailing concat flip; do
        rc=0
        out=$("$bin/llama-completion" -m "$work/$bad.bad" -n 4 -p hi 2>/dev/null </dev/null) || rc=$?
        ok=0
        if [ "$rc" -ne 0 ] && [ -z "$out" ]; then ok=1; fi
        check "$bad is rejected with no output" test "$ok" -eq 1
    done
}

if command -v xz >/dev/null; then
    run_format xz .xz xz -c -6
    n_formats=$((n_formats + 1))
else
    printf "skip xz: the tool is not installed\n"
fi
if command -v gzip >/dev/null; then
    run_format gzip .gz gzip -c -6
    n_formats=$((n_formats + 1))
else
    printf "skip gzip: the tool is not installed\n"
fi

if [ $n_formats -eq 0 ]; then
    printf "SKIP: neither xz nor gzip is installed\n"
    exit 0
fi
if [ $fail -ne 0 ]; then
    printf "FAILED\n"
    exit 1
fi
printf "all checks passed\n"
