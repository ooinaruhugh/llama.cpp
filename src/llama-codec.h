#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct llama_file;

// Sequential decoder of one compressed stream.
// It knows bytes only: not GGUF, not where the output lives, not the backends.
struct llama_decoder {
    virtual ~llama_decoder() = default;

    // decoded size, known before any output is produced
    virtual uint64_t size() const = 0;

    // writes up to n bytes to dst and returns the count, 0 at the end of the stream
    // throws if the input is corrupt, truncated or has trailing data, or if the decoded size does not match size()
    virtual size_t read(void * dst, size_t n) = 0;
};

struct llama_codec {
    const char *    name;
    const char *    suffix; // customary file name suffix, used only to find sibling files of a split model
    const uint8_t * magic;
    size_t          magic_len;
    bool            available; // false: the format is recognised but this build cannot decode it
    const char *    enable;    // how to build with it, for the error when it is not available

    // `in` must outlive the decoder, parallel allows worker threads (for decoding a whole model)
    std::unique_ptr<llama_decoder> (*open)(llama_file & in, bool parallel);
};

constexpr size_t LLAMA_CODEC_MAGIC_MAX = 8;

#ifdef LLAMA_USE_LZMA
std::unique_ptr<llama_decoder> llama_codec_xz_open(llama_file & in, bool parallel);
#endif

// nullptr if the leading bytes match no known compressed format
const llama_codec * llama_codec_find(const uint8_t * head, size_t n);

// removes the suffix of a known compressed format from a file name and returns it, empty if there is none
// this is naming only, whether a file is compressed is decided by its content
std::string llama_codec_strip_suffix(std::string & name);
