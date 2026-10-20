#include "llama-codec.h"

#include <cstring>

static const uint8_t llama_magic_xz[] = { 0xFD, '7', 'z', 'X', 'Z', 0x00 };

// every known format is listed, so a build without its library can say why it refuses the file
static const llama_codec llama_codecs[] = {
#ifdef LLAMA_USE_LZMA
    { "xz", ".xz", llama_magic_xz, sizeof(llama_magic_xz), true,  "-DLLAMA_LZMA=ON", llama_codec_xz_open },
#else
    { "xz", ".xz", llama_magic_xz, sizeof(llama_magic_xz), false, "-DLLAMA_LZMA=ON", nullptr },
#endif
};

const llama_codec * llama_codec_find(const uint8_t * head, size_t n) {
    for (const auto & codec : llama_codecs) {
        if (n >= codec.magic_len && memcmp(head, codec.magic, codec.magic_len) == 0) {
            return &codec;
        }
    }
    return nullptr;
}

std::string llama_codec_strip_suffix(std::string & name) {
    for (const auto & codec : llama_codecs) {
        const std::string suffix = codec.suffix;
        if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            name.resize(name.size() - suffix.size());
            return suffix;
        }
    }
    return "";
}
