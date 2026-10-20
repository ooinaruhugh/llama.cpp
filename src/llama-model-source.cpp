#include "llama-model-source.h"

#include "llama-ext.h"

#include "llama-backing-store.h"
#include "llama-codec.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstring>
#include <vector>

// output is decoded in slices, which is also how often progress is reported and cancellation is noticed
static constexpr size_t LLAMA_DECODE_SLICE = 8u << 20;

static size_t llama_sniff(const std::string & path, uint8_t * head, size_t cap) {
    FILE * f = ggml_fopen(path.c_str(), "rb");
    if (!f) {
        return 0;
    }
    const size_t n = fread(head, 1, cap, f);
    fclose(f);
    return n;
}

bool llama_model_source_is_compressed(const std::string & path) {
    uint8_t head[LLAMA_CODEC_MAGIC_MAX];
    return llama_codec_find(head, llama_sniff(path, head, sizeof(head))) != nullptr;
}

// feeds the sequential reads of the gguf parser from a decoder
struct llama_decoder_reader {
    explicit llama_decoder_reader(llama_decoder & dec) : dec(dec) {}

    llama_decoder & dec;
    uint64_t pos = 0;
    std::vector<uint8_t> buf = std::vector<uint8_t>(1 << 16);
    size_t buf_pos = 0;
    size_t buf_len = 0;
    std::string error;

    static size_t read(void * ud, void * out, uint64_t offset, size_t len) {
        auto * r = (llama_decoder_reader *) ud;
        if (offset != r->pos) {
            r->error = "the GGUF header is not read sequentially";
            return 0;
        }
        size_t done = 0;
        try {
            while (done < len) {
                if (r->buf_pos == r->buf_len) {
                    r->buf_len = r->dec.read(r->buf.data(), r->buf.size());
                    r->buf_pos = 0;
                    if (r->buf_len == 0) {
                        break;
                    }
                }
                const size_t k = std::min(len - done, r->buf_len - r->buf_pos);
                memcpy((uint8_t *) out + done, r->buf.data() + r->buf_pos, k);
                r->buf_pos += k;
                done += k;
            }
        } catch (const std::exception & e) {
            r->error = e.what();
            return 0;
        }
        r->pos += done;
        return done;
    }
};

llama_model_source llama_open_model_source(const std::string & path, const llama_model_source_params & params) {
    llama_model_source src;

    struct gguf_init_params gparams = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &src.ctx,
    };

    uint8_t head[LLAMA_CODEC_MAGIC_MAX];
    const llama_codec * codec = llama_codec_find(head, llama_sniff(path, head, sizeof(head)));

    if (codec == nullptr) {
        // plain GGUF, or anything else that the gguf parser reports on
        src.meta.reset(gguf_init_from_file(path.c_str(), gparams));
        if (src.meta) {
            src.file = std::make_unique<llama_file>(path.c_str(), "rb", params.use_direct_io);
        }
        return src;
    }

    if (!codec->available) {
        throw std::runtime_error(format("%s is %s-compressed, but this build has no %s support (configure with %s)",
                    path.c_str(), codec->name, codec->name, codec->enable));
    }
    if (!params.metadata_only && !params.use_mmap) {
        throw std::runtime_error(format("%s is %s-compressed, and compressed models are loaded through memory mapping, "
                    "which is disabled (--no-mmap or direct I/O)", path.c_str(), codec->name));
    }

    try {
        llama_file in(path.c_str(), "rb");
        std::unique_ptr<llama_decoder> dec = codec->open(in, !params.metadata_only);
        const size_t size = (size_t) dec->size();

        if (params.metadata_only) {
            // only the header is decoded: enough to know the tensors, and cheap
            llama_decoder_reader reader(*dec);
            src.meta.reset(gguf_init_from_callback(llama_decoder_reader::read, &reader, 0, size, gparams));
            if (!src.meta && !reader.error.empty()) {
                throw std::runtime_error(reader.error);
            }
            src.file = std::make_unique<llama_file>(size);
            return src;
        }

        LLAMA_LOG_INFO("%s: decoding %s (%s) into %.2f MiB of memory\n", __func__, path.c_str(), codec->name, size / 1024.0 / 1024.0);

        std::unique_ptr<llama_backing_store> store = llama_backing_store::create(size);

        // the decoder writes straight into the final storage
        uint8_t * dst = (uint8_t *) store->write_addr();
        if (params.progress) {
            params.progress->decoded = true;
        }
        for (size_t off = 0; off < size; ) {
            const size_t n = dec->read(dst + off, std::min(LLAMA_DECODE_SLICE, size - off));
            if (n == 0) {
                throw std::runtime_error(format("the stream ended after %zu of %zu bytes", off, size));
            }
            off += n;
            if (params.progress && !params.progress->report_decode(params.index, params.count, (float) off / (float) size)) {
                throw llama_load_cancelled();
            }
        }
        store->seal();

        src.file = std::make_unique<llama_file>(std::move(store));

        // the header is parsed from a temporary mapping of the same pages
        llama_mmap tmp(src.file.get(), 0);
        src.meta.reset(gguf_init_from_buffer(tmp.addr(), tmp.size(), gparams));
    } catch (const llama_load_cancelled &) {
        throw;
    } catch (const std::exception & e) {
        throw std::runtime_error(format("failed to load %s: %s", path.c_str(), e.what()));
    }
    return src;
}

struct gguf_context * llama_gguf_open_metadata(const char * path) {
    try {
        llama_model_source_params params;
        params.metadata_only = true;
        llama_model_source src = llama_open_model_source(path, params);
        if (src.ctx) {
            ggml_free(src.ctx);
        }
        return src.meta.release();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        return nullptr;
    }
}
