/* Minimal C ABI example in plain C: synthesize a phrase to raw float32 PCM.
 *   cc examples/minimal.c -Iinclude -Lbuild -lvox -o minimal
 *   ./minimal model.tmod voice.tvoice "Hello!" > out.f32
 */
#include <stdio.h>
#include "vox.h"

#define CHECK(x) do { vox_status s_ = (x); if (s_ < 0) { fprintf(stderr, "%s: %s\n", #x, vox_last_error()); return 1; } } while (0)

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.tmod voice.tvoice text\n", argv[0]); return 1; }
    vox_engine *e; vox_voice *v; vox_session *s;
    CHECK(vox_engine_open(argv[1], NULL, &e));
    CHECK(vox_voice_load(e, argv[2], &v));

    vox_gen_params p;
    vox_gen_params_default(&p);
    p.seed = 42;
    CHECK(vox_session_create(e, v, &p, &s));
    CHECK(vox_session_feed_text(s, argv[3]));
    CHECK(vox_session_end_text(s));

    float buf[4096];
    size_t n, total = 0;
    vox_status st;
    while ((st = vox_session_pull(s, buf, 4096, &n)) == VOX_OK) {
        fwrite(buf, sizeof(float), n, stdout);
        total += n;
    }
    CHECK(st);
    fprintf(stderr, "%zu samples @ %u Hz\n", total, vox_engine_sample_rate(e));
    vox_session_free(s);
    vox_voice_free(v);
    vox_engine_close(e);
    return 0;
}
