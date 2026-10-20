// Tests of the model source layer: plain and compressed GGUF files, the backing store and the codecs.
// The fixtures are made from a GGUF file given on the command line (the vocab files in models/ are enough).

#include "../src/llama-backing-store.h"
#include "../src/llama-codec.h"
#include "../src/llama-mmap.h"
#include "../src/llama-model-source.h"

#include "ggml-cpp.h"
#include "gguf.h"

#ifdef TEST_HAVE_LZMA
#include <lzma.h>
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static std::vector<uint8_t> read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    std::vector<uint8_t> data;
    if (f) {
        fseek(f, 0, SEEK_END);
        data.resize(ftell(f));
        fseek(f, 0, SEEK_SET);
        if (fread(data.data(), 1, data.size(), f) != data.size()) {
            data.clear();
        }
        fclose(f);
    }
    return data;
}

static void write_file(const std::string & path, const std::vector<uint8_t> & data) {
    FILE * f = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
    }
}

// true if fn throws
static bool throws(const std::function<void()> & fn, std::string * what = nullptr) {
    try {
        fn();
    } catch (const std::exception & e) {
        if (what) {
            *what = e.what();
        }
        return true;
    }
    return false;
}

// open a file as a source, true if it was provided as a valid GGUF
static bool opens(const std::string & path, bool metadata_only, std::string * what = nullptr) {
    llama_model_source_params p;
    p.metadata_only = metadata_only;
    p.use_mmap      = true;
    bool ok = false;
    const bool t = throws([&] {
        llama_model_source src = llama_open_model_source(path, p);
        ok = src.meta != nullptr;
    }, what);
    return !t && ok;
}

static void test_backing_store() {
    printf("backing store\n");

    CHECK(throws([] { llama_backing_store::create(0); }));
    CHECK(throws([] { llama_backing_store::create(SIZE_MAX); }));

    const size_t size = (1u << 20) + 123; // not a multiple of the page size
    auto store = llama_backing_store::create(size);
    CHECK(store->size() == size);
    CHECK(throws([&] { store->fd(); })); // not sealed yet

    uint8_t * w = (uint8_t *) store->write_addr();
    for (size_t i = 0; i < size; i++) {
        w[i] = (uint8_t) (i * 31 + 7);
    }
    store->seal();
    store->seal(); // idempotent
    CHECK(throws([&] { store->write_addr(); }));

    const int fd = store->fd();
    CHECK(fd >= 0);
    CHECK((fcntl(fd, F_GETFD) & FD_CLOEXEC) != 0);

    printf("  read-only enforced by the OS: %d\n", (int) store->read_only_enforced());
    if (store->read_only_enforced()) {
        void * rw = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        CHECK(rw == MAP_FAILED);
        if (rw != MAP_FAILED) {
            munmap(rw, size);
        }
    }

    void * ro = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    CHECK(ro != MAP_FAILED);

    // the pages outlive the store and its descriptor
    store.reset();
    bool same = ro != MAP_FAILED;
    for (size_t i = 0; same && i < size; i++) {
        same = ((uint8_t *) ro)[i] == (uint8_t) (i * 31 + 7);
    }
    CHECK(same);
    if (ro != MAP_FAILED) {
        munmap(ro, size);
    }
}

static void test_plain(const std::string & path, const std::vector<uint8_t> & orig) {
    printf("plain\n");

    llama_model_source_params p;
    p.use_mmap = true;
    llama_model_source src = llama_open_model_source(path, p);
    CHECK(src.meta != nullptr);
    CHECK(src.file != nullptr && src.file->readable() && src.file->mappable());
    CHECK(src.file->size() == orig.size());

    CHECK(!llama_model_source_is_compressed(path));
    CHECK(!opens("/nonexistent/path/model.gguf", false));
}

#ifdef TEST_HAVE_LZMA

// a compressed format under test, with an encoder made from the same library the codec uses
struct fixture_codec {
    const char * name;
    const char * ext;
    bool         has_index; // the container is checked without reading the data, not only by a trailer
    std::vector<uint8_t> (*encode)(const std::vector<uint8_t> &, size_t block);
};

#ifdef TEST_HAVE_LZMA
// several blocks, so that the parallel decoder has work to share
static std::vector<uint8_t> xz_encode(const std::vector<uint8_t> & in, size_t block) {
    lzma_mt mt = {};
    mt.threads    = 2;
    mt.block_size = block;
    mt.preset     = 1;
    mt.check      = LZMA_CHECK_CRC32;

    lzma_stream strm = LZMA_STREAM_INIT;
    CHECK(lzma_stream_encoder_mt(&strm, &mt) == LZMA_OK);

    std::vector<uint8_t> out(lzma_stream_buffer_bound(in.size()) + (1 << 16));
    strm.next_in   = in.data();
    strm.avail_in  = in.size();
    strm.next_out  = out.data();
    strm.avail_out = out.size();
    lzma_ret ret;
    do {
        ret = lzma_code(&strm, LZMA_FINISH);
    } while (ret == LZMA_OK);
    CHECK(ret == LZMA_STREAM_END);
    out.resize(out.size() - strm.avail_out);
    lzma_end(&strm);
    return out;
}
#endif


static const std::vector<fixture_codec> & fixture_codecs() {
    static const std::vector<fixture_codec> codecs = {
#ifdef TEST_HAVE_LZMA
        { "xz",   ".xz", true,  xz_encode },
#endif
    };
    return codecs;
}

static bool same_kv(const gguf_context * a, const gguf_context * b) {
    if (!a || !b || gguf_get_n_kv(a) != gguf_get_n_kv(b) || gguf_get_n_tensors(a) != gguf_get_n_tensors(b)) {
        return false;
    }
    for (int64_t i = 0; i < gguf_get_n_kv(a); i++) {
        if (strcmp(gguf_get_key(a, i), gguf_get_key(b, i)) != 0) {
            return false;
        }
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(a); i++) {
        if (strcmp(gguf_get_tensor_name(a, i), gguf_get_tensor_name(b, i)) != 0 || gguf_get_tensor_offset(a, i) != gguf_get_tensor_offset(b, i)) {
            return false;
        }
    }
    return true;
}

static void test_compressed(const fixture_codec & c, const std::string & dir, const std::vector<uint8_t> & orig, const std::string & plain_path) {
    printf("compressed (%s)\n", c.name);

    const std::string xz_path = dir + "/model.gguf" + c.ext;
    write_file(xz_path, c.encode(orig, 64 << 10));

    CHECK(llama_model_source_is_compressed(xz_path));

    llama_model_source_params pp;
    pp.use_mmap = true;
    llama_model_source plain = llama_open_model_source(plain_path, pp);

    // full: decoded into a backing store, mappable and not readable, the same metadata and bytes
    {
        llama_model_source src = llama_open_model_source(xz_path, pp);
        CHECK(src.meta != nullptr);
        CHECK(src.file && !src.file->readable() && src.file->mappable());
        CHECK(src.file->size() == orig.size());
        CHECK(same_kv(plain.meta.get(), src.meta.get()));
        CHECK(throws([&] { src.file->seek(0, SEEK_SET); }));

        // the mapping keeps the pages after the source is gone
        std::unique_ptr<llama_mmap> map(new llama_mmap(src.file.get(), 0));
        src.file.reset();
        CHECK(map->size() == orig.size() && memcmp(map->addr(), orig.data(), orig.size()) == 0);
    }

    // metadata only: nothing stored, nothing to map, the size is still known
    {
        llama_model_source_params mp = pp;
        mp.metadata_only = true;
        llama_model_source src = llama_open_model_source(xz_path, mp);
        CHECK(src.meta != nullptr);
        CHECK(same_kv(plain.meta.get(), src.meta.get()));
        CHECK(src.file && !src.file->readable() && !src.file->mappable());
        CHECK(src.file->size() == orig.size());
        CHECK(throws([&] { llama_mmap m(src.file.get(), 0); }));
    }

    // compressed input needs mmap unless only the metadata is wanted
    {
        std::string what;
        llama_model_source_params np = pp;
        np.use_mmap = false;
        CHECK(throws([&] { llama_open_model_source(xz_path, np); }, &what));
        CHECK(what.find("memory mapping") != std::string::npos);
    }

    // the format comes from the content, not from the name
    {
        const std::string renamed = dir + "/renamed.bin";
        std::filesystem::copy_file(xz_path, renamed, std::filesystem::copy_options::overwrite_existing);
        CHECK(llama_model_source_is_compressed(renamed));
        CHECK(opens(renamed, false));

        const std::string misnamed = dir + "/plain.gguf" + c.ext; // a plain GGUF with a compressed name
        std::filesystem::copy_file(plain_path, misnamed, std::filesystem::copy_options::overwrite_existing);
        CHECK(!llama_model_source_is_compressed(misnamed));
        CHECK(opens(misnamed, false));
    }
}

static void test_malformed(const fixture_codec & c, const std::string & dir, const std::vector<uint8_t> & orig) {
    printf("malformed (%s)\n", c.name);

    const std::vector<uint8_t> good = c.encode(orig, 64 << 10);
    const std::string path = dir + "/bad" + c.ext;

    // damage to the container is also found by a metadata-only open when the format has an index to check it with
    auto expect_rejected = [&](const char * name, const std::vector<uint8_t> & data, bool metadata_too) {
        write_file(path, data);
        std::string what;
        const bool full = opens(path, false, &what);
        CHECK(!full);
        if (metadata_too) {
            CHECK(!opens(path, true));
        }
        printf("  %-28s rejected=%d  %.90s\n", name, (int) !full, what.c_str());
    };

    for (double frac : { 0.0, 0.25, 0.5, 0.9 }) {
        expect_rejected(("truncated at " + std::to_string((int) (frac * 100)) + "%").c_str(),
                std::vector<uint8_t>(good.begin(), good.begin() + (size_t) (good.size() * frac) + 6), c.has_index);
    }
    expect_rejected("truncated by one byte", std::vector<uint8_t>(good.begin(), good.end() - 1), c.has_index);
    expect_rejected("truncated footer",      std::vector<uint8_t>(good.begin(), good.end() - 12), c.has_index);
    expect_rejected("magic only",            std::vector<uint8_t>(good.begin(), good.begin() + 6), true);

    std::vector<uint8_t> v = good;
    v.push_back(0x42);
    expect_rejected("trailing byte", v, c.has_index);

    v = good;
    v.insert(v.end(), 4, 0); // valid stream padding for xz, still not accepted
    expect_rejected("trailing zero bytes", v, true);

    v = good;
    v.insert(v.end(), good.begin(), good.end());
    expect_rejected("two concatenated streams", v, c.has_index);

    expect_rejected("empty payload", c.encode({}, 64 << 10), true);

    // a valid stream of something that is not a GGUF
    write_file(path, c.encode(std::vector<uint8_t>(4096, 'x'), 64 << 10));
    CHECK(!opens(path, false));
    CHECK(!opens(path, true));

    // damage inside the data: a full decode checks the checksums, a metadata-only open does not read that far
    v = good;
    v[v.size() / 2] ^= 0x55;
    expect_rejected("flipped byte in the data", v, false);
}

static void test_progress(const fixture_codec & c, const std::string & dir, const std::vector<uint8_t> & orig) {
    printf("progress and cancellation (%s)\n", c.name);

    // large enough for several decode slices: the GGUF followed by zeros, which the parser does not look at
    std::vector<uint8_t> big = orig;
    big.resize(orig.size() + (20u << 20), 0);
    const std::string path = dir + "/big.gguf" + c.ext;
    write_file(path, c.encode(big, 1 << 20));

    struct state { std::vector<float> values; float cancel_above = 2.0f; } st;
    llama_load_progress progress;
    progress.user_data = &st;
    progress.callback  = [](float v, void * ud) {
        auto * s = (state *) ud;
        s->values.push_back(v);
        return v <= s->cancel_above;
    };

    llama_model_source_params p;
    p.use_mmap  = true;
    p.progress  = &progress;

    {
        llama_model_source src = llama_open_model_source(path, p);
        CHECK(src.meta != nullptr);
        CHECK(st.values.size() >= 3);
        bool monotonic = true;
        for (size_t i = 1; i < st.values.size(); i++) {
            monotonic = monotonic && st.values[i] >= st.values[i - 1];
        }
        CHECK(monotonic);
        CHECK(!st.values.empty() && st.values.back() <= llama_load_progress::decode_share + 1e-6f);
        CHECK(progress.decoded);

        // placement takes the rest of the range, up to 1.0
        progress.report_load(0.5f);
        progress.report_load(1.0f);
        CHECK(st.values.back() == 1.0f);
    }

    // cancelled in the middle of the decode
    {
        st = state();
        st.cancel_above = 0.1f;
        progress = llama_load_progress();
        progress.user_data = &st;
        progress.callback  = [](float v, void * ud) {
            auto * s = (state *) ud;
            s->values.push_back(v);
            return v <= s->cancel_above;
        };
        p.progress = &progress;
        bool cancelled = false;
        try {
            llama_open_model_source(path, p);
        } catch (const llama_load_cancelled &) {
            cancelled = true;
        }
        CHECK(cancelled);
        CHECK(!st.values.empty() && st.values.back() > 0.1f && st.values.back() < llama_load_progress::decode_share);
    }
}

#endif // TEST_HAVE_LZMA

static void test_unavailable() {
    printf("codec table\n");

    const uint8_t xz_magic[] = { 0xFD, '7', 'z', 'X', 'Z', 0x00, 0, 0 };
    const llama_codec * xz = llama_codec_find(xz_magic, sizeof(xz_magic));
    CHECK(xz != nullptr && std::string(xz->name) == "xz");
    CHECK(llama_codec_find((const uint8_t *) "GGUF\x03\0\0\0", 8) == nullptr);
    CHECK(llama_codec_find(xz_magic, 3) == nullptr);

    // a build without a library recognises its format and says why it cannot load the file
    if (xz && !xz->available) {
        const std::string path = std::filesystem::temp_directory_path().string() + "/llama-test-unavailable" + xz->suffix;
        write_file(path, std::vector<uint8_t>(xz_magic, xz_magic + 8));
        std::string what;
        CHECK(throws([&] { llama_model_source_params p; p.use_mmap = true; llama_open_model_source(path, p); }, &what));
        CHECK(what.find(std::string("no ") + xz->name + " support") != std::string::npos);
        CHECK(what.find(xz->enable) != std::string::npos);
        std::filesystem::remove(path);
    }

    std::string name = "model-00001-of-00003.gguf.xz";
    CHECK(llama_codec_strip_suffix(name) == ".xz" && name == "model-00001-of-00003.gguf");
    name = "model.gguf";
    CHECK(llama_codec_strip_suffix(name).empty() && name == "model.gguf");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }
    const std::string plain_path = argv[1];
    const std::vector<uint8_t> orig = read_file(plain_path);
    if (orig.empty()) {
        fprintf(stderr, "cannot read %s\n", plain_path.c_str());
        return 2;
    }

    const std::string dir = (std::filesystem::temp_directory_path() / ("llama-test-source-" + std::to_string(getpid()))).string();
    std::filesystem::create_directories(dir);

    test_backing_store();
    test_plain(plain_path, orig);
    test_unavailable();
#ifdef TEST_HAVE_LZMA
    for (const fixture_codec & c : fixture_codecs()) {
        test_compressed(c, dir, orig, plain_path);
        test_malformed(c, dir, orig);
        test_progress(c, dir, orig);
    }
#else
    printf("built without compression libraries: the compressed cases are skipped\n");
#endif

    std::filesystem::remove_all(dir);
    printf(n_fail ? "%d check(s) FAILED\n" : "all checks passed\n", n_fail);
    return n_fail ? 1 : 0;
}
