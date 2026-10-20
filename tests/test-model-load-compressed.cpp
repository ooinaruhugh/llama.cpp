// Loads a model plain and compressed through the public API only and compares them, for every format built in.
// The compressed copies are made here with the encoders of the libraries, so the test needs no xz binary.

#include "llama.h"
#include "common.h"

#ifdef TEST_HAVE_LZMA
#include <lzma.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

// one compressed format under test
struct codec_case {
    const char * name;
    const char * ext;
    // a file with several large blocks where the format has blocks, made the way the command line tool makes it
    bool (*compress)(const std::string & src, const std::string & dst, bool like_cli_tool);
};

#ifdef TEST_HAVE_LZMA
static bool xz_compress(const std::string & src, const std::string & dst, bool like_cli_tool) {
    FILE * in  = fopen(src.c_str(), "rb");
    FILE * out = fopen(dst.c_str(), "wb");
    if (!in || !out) {
        return false;
    }
    lzma_mt mt = {};
    mt.threads    = 4;
    mt.block_size = like_cli_tool ? 24u << 20 : 4u << 20;
    mt.preset     = like_cli_tool ? 6 : 1;
    mt.check      = LZMA_CHECK_CRC32;
    lzma_stream strm = LZMA_STREAM_INIT;
    if (lzma_stream_encoder_mt(&strm, &mt) != LZMA_OK) {
        return false;
    }
    std::vector<uint8_t> ibuf(1 << 20), obuf(1 << 20);
    lzma_ret ret = LZMA_OK;
    bool eof = false;
    while (ret == LZMA_OK) {
        if (strm.avail_in == 0 && !eof) {
            strm.next_in  = ibuf.data();
            strm.avail_in = fread(ibuf.data(), 1, ibuf.size(), in);
            eof = strm.avail_in == 0;
        }
        strm.next_out  = obuf.data();
        strm.avail_out = obuf.size();
        ret = lzma_code(&strm, eof ? LZMA_FINISH : LZMA_RUN);
        fwrite(obuf.data(), 1, obuf.size() - strm.avail_out, out);
    }
    lzma_end(&strm);
    fclose(in);
    fclose(out);
    return ret == LZMA_STREAM_END;
}
#endif


static std::vector<codec_case> codec_cases() {
    std::vector<codec_case> cases;
#ifdef TEST_HAVE_LZMA
    cases.push_back({ "xz",   ".xz", xz_compress });
#endif
    return cases;
}

// physical footprint of this process, or 0 if unknown
static size_t footprint() {
#ifdef __APPLE__
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count) == KERN_SUCCESS) {
        return (size_t) info.phys_footprint;
    }
#endif
    return 0;
}

// logits of the first token, to compare two loads of the same model
static std::vector<float> first_logits(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return {};
    }
    llama_token tok = llama_vocab_bos(llama_model_get_vocab(model));
    if (tok < 0) {
        tok = 1;
    }
    std::vector<float> res;
    if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) == 0) {
        const float * l = llama_get_logits_ith(ctx, 0);
        res.assign(l, l + llama_vocab_n_tokens(llama_model_get_vocab(model)));
    }
    llama_free(ctx);
    return res;
}

static void test_codec(const codec_case & c, const char * model_path, const std::string & dir) {
    printf("== %s\n", c.name);

    const std::string comp_path = dir + "/model.gguf" + c.ext;
    if (!c.compress(model_path, comp_path, false)) {
        fprintf(stderr, "cannot compress the model with %s\n", c.name);
        n_fail++;
        return;
    }
    const size_t model_bytes = std::filesystem::file_size(model_path);
    printf("model %zu bytes, compressed %zu bytes\n", model_bytes, (size_t) std::filesystem::file_size(comp_path));

    // 1. plain and compressed give the same model and the same logits
    {
        llama_model_params mp = llama_model_default_params();
        llama_model * plain = llama_model_load_from_file(model_path, mp);
        llama_model * comp  = llama_model_load_from_file(comp_path.c_str(), mp);
        CHECK(plain != nullptr);
        CHECK(comp  != nullptr);
        if (plain && comp) {
            CHECK(llama_model_n_params(plain) == llama_model_n_params(comp));
            CHECK(llama_model_size(plain) == llama_model_size(comp));
            const auto a = first_logits(plain);
            const auto b = first_logits(comp);
            CHECK(!a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
        }
        llama_model_free(comp);
        llama_model_free(plain);
    }

    // 2. the progress is monotonic and finishes at 1.0, for both
    for (const std::string & path : { std::string(model_path), comp_path }) {
        struct state { float last = 0.0f; bool monotonic = true; int calls = 0; } st;
        llama_model_params mp = llama_model_default_params();
        mp.progress_callback = [](float v, void * ud) {
            auto * s = (state *) ud;
            s->monotonic = s->monotonic && v >= s->last;
            s->last = v;
            s->calls++;
            return true;
        };
        mp.progress_callback_user_data = &st;
        llama_model * m = llama_model_load_from_file(path.c_str(), mp);
        CHECK(m != nullptr);
        CHECK(st.monotonic);
        CHECK(st.last == 1.0f);
        printf("progress %s: %d calls, monotonic=%d, last=%.2f\n", path == comp_path ? c.name : "plain", st.calls, (int) st.monotonic, st.last);
        llama_model_free(m);
    }

    // 3. cancellation, early (while decoding) and late (while placing tensors)
    for (float limit : { 0.01f, 0.75f }) {
        llama_model_params mp = llama_model_default_params();
        mp.progress_callback = [](float v, void * ud) { return v <= *(float *) ud; };
        mp.progress_callback_user_data = &limit;
        llama_model * m = llama_model_load_from_file(comp_path.c_str(), mp);
        CHECK(m == nullptr);
        llama_model_free(m);
    }

    // 4. compressed input without mmap is an explicit error, not a silent copy
    {
        llama_model_params mp = llama_model_default_params();
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        CHECK(llama_model_load_from_file(comp_path.c_str(), mp) == nullptr);
    }

    // 5. vocab_only and the memory-fit load (no_alloc, no mmap) work, and do not materialise the model
    const bool big = model_bytes >= (32u << 20);
    {
        const size_t before = footprint();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        llama_model * m = llama_model_load_from_file(comp_path.c_str(), mp);
        CHECK(m != nullptr);
        const size_t after = footprint();
        if (m) {
            llama_model * p = llama_model_load_from_file(model_path, mp);
            CHECK(p != nullptr && llama_vocab_n_tokens(llama_model_get_vocab(p)) == llama_vocab_n_tokens(llama_model_get_vocab(m)));
            llama_model_free(p);
        }
        llama_model_free(m);
        printf("vocab_only %s: footprint grew by %.1f MiB (model %.1f MiB)\n", c.name, (after - before) / 1048576.0, model_bytes / 1048576.0);
        if (big && before) {
            CHECK(after - before < model_bytes / 2);
        }
    }
    {
        const size_t before = footprint();
        llama_model_params mp = llama_model_default_params();
        mp.no_alloc  = true;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        llama_model * m = llama_model_load_from_file(comp_path.c_str(), mp);
        CHECK(m != nullptr);
        const size_t after = footprint();
        llama_model_free(m);
        printf("no_alloc %s  : footprint grew by %.1f MiB (model %.1f MiB)\n", c.name, (after - before) / 1048576.0, model_bytes / 1048576.0);
        if (big && before) {
            CHECK(after - before < model_bytes / 2);
        }
    }

    // a full compressed load holds the model once. The file is made like the command line tool makes it, with
    // several large blocks, which a multi-threaded decoder would buffer outside the destination if it were allowed to
    if (big && footprint()) {
        const std::string blocks_path = dir + "/model-blocks.gguf" + c.ext;
        CHECK(c.compress(model_path, blocks_path, true));

        const size_t before = footprint();
        llama_model_params mp = llama_model_default_params();
        llama_model * m = llama_model_load_from_file(blocks_path.c_str(), mp);
        const size_t after = footprint();
        const double grew = (after - before) / 1048576.0;
        printf("full load %s : footprint grew by %.1f MiB (model %.1f MiB)\n", c.name, grew, model_bytes / 1048576.0);
        CHECK(m != nullptr);
        CHECK(after - before > model_bytes * 3 / 4);             // it does hold the model: the measure can see it
        CHECK(after - before < model_bytes + (20u << 20));       // and not much more than once
        llama_model_free(m);
    } else {
        printf("footprint checks skipped (needs macOS and a model of at least 32 MiB)\n");
    }
}

int main(int argc, char ** argv) {
    const char * model_path = common_get_model_or_exit(argc, argv);
    if (!std::filesystem::exists(model_path)) {
        fprintf(stderr, "no model at '%s' found\n", model_path);
        return EXIT_FAILURE;
    }

    const std::string dir = (std::filesystem::temp_directory_path() / ("llama-test-compressed-" + std::to_string(getpid()))).string();
    std::filesystem::create_directories(dir);

    llama_backend_init();
    for (const codec_case & c : codec_cases()) {
        test_codec(c, model_path, dir);
    }
    llama_backend_free();

    std::filesystem::remove_all(dir);
    printf(n_fail ? "%d check(s) FAILED\n" : "all checks passed\n", n_fail);
    return n_fail ? EXIT_FAILURE : EXIT_SUCCESS;
}
