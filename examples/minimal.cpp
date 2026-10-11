// C++ over the C ABI: RAII wrappers and synthesis to stdout (raw float32 PCM, mono).
//   c++ -std=c++17 examples/minimal.cpp -Iinclude -Lbuild -lvox -o minimal_cpp
//   LD_LIBRARY_PATH=build ./minimal_cpp model.tmod voice.tvoice "Hello!" > out.f32
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vox.h"

static void check(vox_status s, const char *what) {
    if (s < 0) throw std::runtime_error(std::string(what) + ": " + vox_last_error());
}
template <class T, void (*F)(T *)> struct Deleter { void operator()(T *p) const { F(p); } };
using Engine = std::unique_ptr<vox_engine, Deleter<vox_engine, vox_engine_close>>;
using Voice = std::unique_ptr<vox_voice, Deleter<vox_voice, vox_voice_free>>;
using Session = std::unique_ptr<vox_session, Deleter<vox_session, vox_session_free>>;

int main(int argc, char **argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s model.tmod voice.tvoice text\n", argv[0]); return 1; }
    try {
        vox_engine *e = nullptr;
        check(vox_engine_open(argv[1], nullptr, &e), "open");
        Engine engine(e);
        vox_voice *v = nullptr;
        check(vox_voice_load(e, argv[2], &v), "voice");
        Voice voice(v);

        vox_gen_params p;
        vox_gen_params_default(&p);
        p.seed = 42;
        vox_session *s = nullptr;
        check(vox_session_create(e, v, &p, &s), "session");
        Session session(s);
        check(vox_session_feed_text(s, argv[3]), "feed");
        check(vox_session_end_text(s), "end");

        std::vector<float> buf(4096);
        size_t n = 0, total = 0;
        vox_status st;
        while ((st = vox_session_pull(s, buf.data(), buf.size(), &n)) == VOX_OK) {
            std::fwrite(buf.data(), sizeof(float), n, stdout);
            total += n;
        }
        check(st, "pull");
        std::fprintf(stderr, "%zu samples @ %u Hz\n", total, vox_engine_sample_rate(e));
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
    return 0;
}
