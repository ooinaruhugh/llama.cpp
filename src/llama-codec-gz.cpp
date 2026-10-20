#ifdef LLAMA_USE_ZLIB

#include "llama-codec.h"

#include "llama-impl.h"
#include "llama-mmap.h"

#include <zlib.h>

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

// gzip only: a zlib or raw deflate stream is not a model file format
static constexpr int LLAMA_GZ_WINDOW_BITS = 16 + MAX_WBITS;

// gzip files up to this compressed size have their decoded size taken from the trailer (see gz_decoded_size)
static constexpr uint64_t LLAMA_GZ_TRUST_TRAILER_BELOW = 1ull << 30;

static std::string gz_error(int ret, const z_stream & strm) {
    if (strm.msg) {
        return strm.msg;
    }
    switch (ret) {
        case Z_MEM_ERROR:  return "out of memory";
        case Z_DATA_ERROR: return "corrupt data";
        case Z_BUF_ERROR:  return "truncated or inconsistent input";
        default:           return format("zlib error %d", ret);
    }
}

// gzip has no index. The size of the data is in the last 4 bytes, modulo 2^32, and that is not enough to size
// a model that is 4 GiB or more. Up to 1 GiB compressed the trailer is used, and the decoder checks it, so a
// larger file is an explicit error and not a wrong size. Above that the data is decoded once and counted.
static uint64_t gz_decoded_size(llama_file & in) {
    const uint64_t in_size = in.size();
    if (in_size < 18) { // 10 bytes of header, 8 of trailer
        throw std::runtime_error("invalid gzip file (too short)");
    }

    uint64_t out_size = 0;

    if (in_size < LLAMA_GZ_TRUST_TRAILER_BELOW) {
        uint8_t t[4];
        in.seek(in_size - 4, SEEK_SET);
        in.read_raw(t, sizeof(t));
        out_size = (uint64_t) t[0] | (uint64_t) t[1] << 8 | (uint64_t) t[2] << 16 | (uint64_t) t[3] << 24;
    } else {
        z_stream strm = {};
        if (inflateInit2(&strm, LLAMA_GZ_WINDOW_BITS) != Z_OK) {
            throw std::runtime_error("cannot start the gzip decoder");
        }
        std::vector<uint8_t> ibuf(1 << 16);
        std::vector<uint8_t> obuf(1 << 16);
        uint64_t pos = 0;
        int ret = Z_OK;
        in.seek(0, SEEK_SET);
        while (ret == Z_OK) {
            if (strm.avail_in == 0) {
                const size_t k = (size_t) std::min<uint64_t>(ibuf.size(), in_size - pos);
                if (k == 0) {
                    ret = Z_BUF_ERROR;
                    break;
                }
                in.read_raw(ibuf.data(), k);
                pos += k;
                strm.next_in  = ibuf.data();
                strm.avail_in = k;
            }
            strm.next_out  = obuf.data();
            strm.avail_out = obuf.size();
            ret = inflate(&strm, Z_NO_FLUSH);
            out_size += obuf.size() - strm.avail_out;
        }
        const bool whole = ret == Z_STREAM_END && pos == in_size && strm.avail_in == 0;
        const std::string msg = whole ? "" : gz_error(ret, strm);
        inflateEnd(&strm);
        if (!whole) {
            throw std::runtime_error(ret == Z_STREAM_END ? "gzip file has trailing data after the stream"
                                                         : "invalid gzip file (" + msg + ")");
        }
    }

    if (out_size == 0) {
        throw std::runtime_error("gzip file decodes to zero bytes (or to a multiple of 4 GiB, which its trailer cannot tell apart)");
    }
    if (out_size > (uint64_t) SIZE_MAX) {
        throw std::runtime_error(format("gzip file decodes to %llu bytes, which does not fit in the address space", (unsigned long long) out_size));
    }
    return out_size;
}

class llama_decoder_gz : public llama_decoder {
public:
    llama_decoder_gz(llama_file & in) : in(in), in_size(in.size()), out_size(gz_decoded_size(in)) {
        in.seek(0, SEEK_SET);
        const int ret = inflateInit2(&strm, LLAMA_GZ_WINDOW_BITS);
        if (ret != Z_OK) {
            throw std::runtime_error("cannot start the gzip decoder: " + gz_error(ret, strm));
        }
    }

    ~llama_decoder_gz() override {
        inflateEnd(&strm);
    }

    uint64_t size() const override {
        return out_size;
    }

    size_t read(void * dst, size_t n) override {
        if (done) {
            return 0;
        }
        n = (size_t) std::min<uint64_t>(n, out_size - produced);

        // the call that completes the output also has to reach the end of the stream (the trailer)
        const bool last = produced + n == out_size;

        size_t got = 0;
        while (got < n || (last && !done)) {
            if (strm.avail_in == 0 && in_pos < in_size) {
                const size_t k = (size_t) std::min<uint64_t>(sizeof(buf), in_size - in_pos);
                in.read_raw(buf, k);
                in_pos += k;
                strm.next_in  = buf;
                strm.avail_in = k;
            }

            const size_t want = std::min<size_t>(n - got, UINT_MAX);
            strm.next_out  = (Bytef *) dst + got;
            strm.avail_out = (uInt) want;

            const int ret = inflate(&strm, Z_NO_FLUSH);
            got += want - strm.avail_out;

            if (ret == Z_STREAM_END) {
                done = true;
                break;
            }
            if (ret == Z_BUF_ERROR && strm.avail_in == 0 && in_pos >= in_size) {
                throw std::runtime_error("failed to decode gzip data: the input ends before the stream does");
            }
            if (ret == Z_BUF_ERROR && strm.avail_out == 0 && last) {
                throw std::runtime_error("failed to decode gzip data: the output is larger than the size in the trailer "
                                         "(corrupt data, or an uncompressed size of 4 GiB or more, which a gzip trailer cannot record)");
            }
            if (ret != Z_OK && ret != Z_BUF_ERROR) {
                throw std::runtime_error("failed to decode gzip data: " + gz_error(ret, strm));
            }
        }
        produced += got;

        if (done) {
            if (produced != out_size) {
                throw std::runtime_error(format("gzip stream ended after %llu of %llu bytes",
                            (unsigned long long) produced, (unsigned long long) out_size));
            }
            if (in_pos != in_size || strm.avail_in != 0) {
                throw std::runtime_error("gzip file has trailing data after the stream (several members are not supported)");
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
    z_stream strm     = {};
    uint8_t  buf[1 << 16];
};

std::unique_ptr<llama_decoder> llama_codec_gz_open(llama_file & in, bool parallel) {
    GGML_UNUSED(parallel); // one deflate stream cannot be decoded in parallel
    return std::make_unique<llama_decoder_gz>(in);
}

#endif // LLAMA_USE_ZLIB
