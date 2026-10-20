#pragma once

#include "llama.h"

#include "ggml-cpp.h"
#include "llama-mmap.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

// thrown when the progress callback asks to cancel while the model source is being prepared
struct llama_load_cancelled : std::runtime_error {
    llama_load_cancelled() : std::runtime_error("model load cancelled") {}
};

// Maps the phases of a load onto the single [0, 1] range of llama_progress_callback.
// The value is monotonic and reaches 1.0 on success. The weights are a heuristic, not a time estimate:
// when a file had to be decoded, decoding takes the first half of the range and tensor placement the second.
struct llama_load_progress {
    llama_progress_callback callback  = nullptr;
    void *                  user_data = nullptr;

    static constexpr float decode_share = 0.5f;

    bool  decoded = false; // set once a file is decoded, which also shrinks the placement range
    float last    = 0.0f;

    // returns false if the callback asked to cancel
    bool report(float value) {
        last = std::max(last, std::min(value, 1.0f));
        return callback == nullptr || callback(last, user_data);
    }

    // file `index` of `count` is `frac` decoded
    bool report_decode(size_t index, size_t count, float frac) {
        return report(decode_share * ((float) index + frac) / (float) count);
    }

    // tensor placement is `frac` done
    bool report_load(float frac) {
        const float base = decoded ? decode_share : 0.0f;
        return callback == nullptr ? true : report(base + (1.0f - base) * frac);
    }
};

struct llama_model_source_params {
    bool metadata_only  = false; // no_alloc or vocab_only: the tensor data is never read
    bool use_mmap       = false;
    bool use_direct_io  = false;
    llama_load_progress * progress = nullptr;
    size_t index = 0; // position among the files of a split model, for progress
    size_t count = 1;
};

struct llama_model_source {
    std::unique_ptr<llama_file> file;
    gguf_context_ptr            meta; // nullptr if the file could not be parsed
    struct ggml_context *       ctx = nullptr; // tensor shells created by the gguf parser
};

// The one place that decides how the bytes of a model file are provided, by looking at the content and not the name:
//   plain GGUF        -> a regular file, as before
//   compressed, full  -> decoded straight into an anonymous backing store that llama_mmap maps
//   compressed, metadata only -> the header is decoded and parsed, nothing is stored
// throws with a reason if the file is compressed and cannot be provided, meta == nullptr if it is not a valid GGUF
llama_model_source llama_open_model_source(const std::string & path, const llama_model_source_params & params);

// true if the file starts with the magic of a known compressed format, whether or not this build can decode it
bool llama_model_source_is_compressed(const std::string & path);
