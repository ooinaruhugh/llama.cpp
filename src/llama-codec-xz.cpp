#ifdef LLAMA_USE_LZMA

#include "llama-codec.h"

#include "llama-impl.h"
#include "llama-mmap.h"

#include <lzma.h>

#include <algorithm>
#include <stdexcept>
#include <string>

static std::string xz_error(lzma_ret ret) {
    switch (ret) {
        case LZMA_MEM_ERROR:      return "out of memory";
        case LZMA_MEMLIMIT_ERROR: return "decoder memory limit exceeded";
        case LZMA_FORMAT_ERROR:   return "not an xz file";
        case LZMA_OPTIONS_ERROR:  return "unsupported xz options";
        case LZMA_DATA_ERROR:     return "corrupt data";
        case LZMA_BUF_ERROR:      return "truncated or inconsistent input";
        default:                  return format("lzma error %d", (int) ret);
    }
}

// the index at the end of the file holds the decoded size, which a fixed-size backing store needs up front
static uint64_t xz_decoded_size(llama_file & in) {
    const uint64_t in_size = in.size();

    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_index * idx = nullptr;
    uint8_t buf[1 << 16];
    uint64_t pos = 0;

    in.seek(0, SEEK_SET);

    lzma_ret ret = lzma_file_info_decoder(&strm, &idx, UINT64_MAX, in_size);
    while (ret == LZMA_OK) {
        if (strm.avail_in == 0) {
            const size_t n = (size_t) std::min<uint64_t>(sizeof(buf), in_size - pos);
            if (n > 0) {
                in.read_raw(buf, n);
            }
            pos += n;
            strm.next_in  = buf;
            strm.avail_in = n;
        }
        ret = lzma_code(&strm, LZMA_RUN);
        if (ret == LZMA_SEEK_NEEDED) {
            pos = strm.seek_pos;
            in.seek(pos, SEEK_SET);
            strm.avail_in = 0;
            ret = LZMA_OK;
        }
    }
    lzma_end(&strm);

    if (ret != LZMA_STREAM_END) {
        if (idx) {
            lzma_index_end(idx, nullptr);
        }
        throw std::runtime_error("invalid xz file (" + xz_error(ret) + "), it must be one complete stream with no trailing data");
    }

    const uint64_t n_streams = lzma_index_stream_count(idx);
    const uint64_t file_size = lzma_index_file_size(idx);
    const uint64_t out_size  = lzma_index_uncompressed_size(idx);
    lzma_index_end(idx, nullptr);

    if (n_streams != 1) {
        throw std::runtime_error("xz file with several concatenated streams is not supported");
    }
    if (file_size != in_size) {
        throw std::runtime_error("xz file has trailing data after the stream");
    }
    if (out_size == 0) {
        throw std::runtime_error("xz file decodes to zero bytes");
    }
    if (out_size > (uint64_t) SIZE_MAX) {
        throw std::runtime_error(format("xz file decodes to %llu bytes, which does not fit in the address space", (unsigned long long) out_size));
    }
    return out_size;
}

class llama_decoder_xz : public llama_decoder {
public:
    llama_decoder_xz(llama_file & in, bool parallel) : in(in), in_size(in.size()), out_size(xz_decoded_size(in)) {
        in.seek(0, SEEK_SET);

        lzma_ret ret;
        if (parallel) {
            // blocks of a multi-threaded xz file decode in parallel, but the workers hold the blocks in buffers of their own
            //     and copy them out. Their budget is a fraction of the model, so that a small model, whose blocks would all
            //     fit, is decoded straight into the destination instead. liblzma does that when no worker fits in the budget
            lzma_mt mt = {};
            mt.threads            = std::max<uint32_t>(lzma_cputhreads(), 1);
            mt.memlimit_threading = std::min<uint64_t>(out_size / 16, 512ull << 20);
            mt.memlimit_stop      = UINT64_MAX;
            ret = lzma_stream_decoder_mt(&strm, &mt);
        } else {
            ret = lzma_stream_decoder(&strm, UINT64_MAX, 0);
        }
        if (ret != LZMA_OK) {
            throw std::runtime_error("cannot start the xz decoder: " + xz_error(ret));
        }
    }

    ~llama_decoder_xz() override {
        lzma_end(&strm);
    }

    uint64_t size() const override {
        return out_size;
    }

    size_t read(void * dst, size_t n) override {
        if (done) {
            return 0;
        }
        n = (size_t) std::min<uint64_t>(n, out_size - produced);

        // the call that completes the output also has to reach the end of the stream (index, footer)
        const bool last = produced + n == out_size;

        strm.next_out  = (uint8_t *) dst;
        strm.avail_out = n;

        for (;;) {
            if (strm.avail_in == 0 && in_pos < in_size) {
                const size_t k = (size_t) std::min<uint64_t>(sizeof(buf), in_size - in_pos);
                in.read_raw(buf, k);
                in_pos += k;
                strm.next_in  = buf;
                strm.avail_in = k;
            }
            const bool eof = strm.avail_in == 0 && in_pos >= in_size;

            const lzma_ret ret = lzma_code(&strm, eof ? LZMA_FINISH : LZMA_RUN);
            if (ret == LZMA_STREAM_END) {
                done = true;
                break;
            }
            if (ret != LZMA_OK) {
                throw std::runtime_error("failed to decode xz data: " + xz_error(ret));
            }
            if (strm.avail_out == 0 && !last) {
                break;
            }
        }

        const size_t got = n - strm.avail_out;
        produced += got;

        if (done) {
            if (produced != out_size) {
                throw std::runtime_error(format("xz stream ended after %llu of %llu bytes",
                            (unsigned long long) produced, (unsigned long long) out_size));
            }
            if (in_pos != in_size || strm.avail_in != 0) {
                throw std::runtime_error("xz file has trailing data after the stream");
            }
        }
        return got;
    }

private:
    llama_file & in;
    const uint64_t in_size;
    const uint64_t out_size;
    uint64_t in_pos   = 0;
    uint64_t produced = 0;
    bool     done     = false;
    lzma_stream strm  = LZMA_STREAM_INIT;
    uint8_t  buf[1 << 16];
};

std::unique_ptr<llama_decoder> llama_codec_xz_open(llama_file & in, bool parallel) {
    return std::make_unique<llama_decoder_xz>(in, parallel);
}

#endif // LLAMA_USE_LZMA
