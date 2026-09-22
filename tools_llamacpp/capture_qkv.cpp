// mzsae capture_qkv: dump per-layer Qcur/Kcur/Vcur/kqv_out via cb_eval.
// Pinned llama.cpp a894dae939d426954ce54bb604824f1ae918a0c5, Metal build.
// Usage:
//   capture_qkv --model M.gguf --list-names
//   capture_qkv --model M.gguf --text doc.txt --tokens 8192 --decode 20 --chunk 2048 --out DIR
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

struct Cap {
    // name -> {shape(ne), dtype, bytes of LAST seen node this eval}
    struct Rec { std::vector<int64_t> ne; ggml_type type; std::vector<uint8_t> data; };
    std::map<std::string, Rec> last;
    bool list_mode = false;
    bool collect_kq = false; // only during decode (prefill kq is huge)
    bool want(const std::string & n) {
        if (n.rfind("Qcur-", 0) == 0 || n.rfind("Kcur-", 0) == 0 ||
            n.rfind("Vcur-", 0) == 0 || n.rfind("kqv_out-", 0) == 0) return true;
        if (collect_kq && n.rfind("kq-", 0) == 0) return true;
        return false;
    }
};

static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    Cap * c = (Cap *) user_data;
    if (ask) {
        if (c->list_mode) {
            char sh[128];
            snprintf(sh, sizeof(sh), "%lld,%lld,%lld,%lld", (long long) t->ne[0], (long long) t->ne[1],
                     (long long) t->ne[2], (long long) t->ne[3]);
            printf("NODE %s [%s] %s\n", ggml_get_name(t), sh, ggml_type_name(t->type));
        }
        return true;
    }
    std::string n = ggml_get_name(t);
    if (!c->want(n)) return true;
    Cap::Rec r;
    r.ne = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    r.type = t->type;
    r.data.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, r.data.data(), 0, r.data.size());
    c->last[n] = std::move(r); // last-seen wins within one eval
    return true;
}

// minimal .npy v1.0 writer, C-order, dtype '<f4' (FP32: model Q/K magnitudes ~1e2
// make fp16 capture lossy beyond tolerance; FP32 preserves exact replay)
static void write_npy(const std::string & path, const std::vector<float> & d,
                      const std::vector<int64_t> & shape) {
    std::ostringstream ss;
    ss << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); i++) {
        ss << shape[i];
        if (shape.size() == 1 || i + 1 < shape.size()) ss << ", ";
    }
    ss << "), }";
    std::string h = ss.str();
    size_t pad = 64 - ((10 + h.size() + 1) % 64);
    h += std::string(pad, ' ');
    h += "\n";
    FILE * f = fopen(path.c_str(), "wb");
    fwrite("\x93NUMPY", 1, 6, f);
    uint8_t maj = 1, min = 0;
    fwrite(&maj, 1, 1, f);
    fwrite(&min, 1, 1, f);
    uint16_t hl = (uint16_t) h.size();
    fwrite(&hl, 2, 1, f);
    fwrite(h.data(), 1, h.size(), f);
    fwrite(d.data(), 4, d.size(), f);
    fclose(f);
}

static float f2f(float x) { return x; }

// tensor bytes (F32/F16) -> fp32 vector in logical [ne0,ne1,ne2] order (ggml row-major flat)
static std::vector<float> to_f32(const Cap::Rec & r) {
    size_t n = 1;
    for (auto e : r.ne) n *= (size_t) e;
    std::vector<float> o(n);
    if (r.type == GGML_TYPE_F32) {
        memcpy(o.data(), r.data.data(), n * 4);
    } else if (r.type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *) r.data.data();
        for (size_t i = 0; i < n; i++) o[i] = ggml_fp16_to_fp32(s[i]);
    } else {
        fprintf(stderr, "unsupported tensor dtype %s\n", ggml_type_name(r.type));
        exit(1);
    }
    return o;
}

// ggml flat index for logical (i0,i1,i2): idx = i0 + ne0*(i1 + ne1*i2)
int main(int argc, char ** argv) {
    std::string model, text, outdir;
    int tokens = 0, decode = 0, chunk = 2048, n_layer = 0;
    bool list_names = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&](std::string dflt) { return (i + 1 < argc) ? argv[++i] : dflt.c_str(); };
        if (a == "--model") model = nxt("");
        else if (a == "--text") text = nxt("");
        else if (a == "--out") outdir = nxt("");
        else if (a == "--tokens") tokens = atoi(nxt("0"));
        else if (a == "--decode") decode = atoi(nxt("0"));
        else if (a == "--chunk") chunk = atoi(nxt("2048"));
        else if (a == "--list-names") list_names = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model.empty()) { fprintf(stderr, "need --model\n"); return 1; }

    llama_model_params mp = llama_model_default_params();
    llama_model * mdl = llama_model_load_from_file(model.c_str(), mp);
    if (!mdl) { fprintf(stderr, "model load failed\n"); return 1; }
    n_layer = llama_model_n_layer(mdl);
    printf("layers=%d\n", n_layer);

    Cap cap;
    cap.list_mode = list_names;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = list_names ? 64 : (tokens + decode + 64);
    cp.n_batch = list_names ? 64 : chunk;
    cp.n_ubatch = list_names ? 64 : chunk;
    cp.n_threads = 8;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(mdl, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    if (list_names) {
        const llama_vocab * vocab = llama_model_get_vocab(mdl);
        std::vector<llama_token> t = {llama_vocab_bos(vocab), 1, 2, 3, 4, 5, 6, 7};
        llama_batch b = llama_batch_init(8, 0, 1);
        for (int i = 0; i < 8; i++) { b.token[i] = t[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = (i == 7); }
        b.n_tokens = 8;
        if (llama_decode(ctx, b) != 0) fprintf(stderr, "decode failed\n");
        llama_batch_free(b);
        llama_free(ctx);
        llama_model_free(mdl);
        return 0;
    }

    // load + tokenize text natively
    std::ifstream f(text, std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const llama_vocab * vocab = llama_model_get_vocab(mdl);
    std::vector<llama_token> ids(s.size() + 16);
    int32_t n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), ids.data(), (int32_t) ids.size(), true, true);
    if (n < 0) { ids.resize(-n + 16); n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), ids.data(), (int32_t) ids.size(), true, true); }
    ids.resize(n);
    printf("doc tokens=%d taking first %d\n", (int) ids.size(), tokens);
    if ((int) ids.size() < tokens) { fprintf(stderr, "doc too short\n"); return 1; }
    ids.resize(tokens);
    system(("mkdir -p " + outdir).c_str());

    // prefill in chunks; collect Kcur/Vcur per layer (+Qcur layer 0 for rope diagnosis)
    struct PerLayer { std::vector<float> K, V; };
    std::vector<PerLayer> acc(n_layer);
    std::vector<float> Q0pre;
    int pos = 0;
    llama_batch b = llama_batch_init(chunk, 0, 1);
    for (int s0 = 0; s0 < tokens; s0 += chunk) {
        int c = std::min(chunk, tokens - s0);
        cap.last.clear();
        b.n_tokens = c;
        for (int i = 0; i < c; i++) {
            b.token[i] = ids[s0 + i];
            b.pos[i] = pos + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (s0 + c == tokens && i == c - 1); // need last-token logits for decode #1
        }
        double t0 = llama_time_us();
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill decode failed\n"); return 1; }
        double dt = (llama_time_us() - t0) / 1e6;
        printf("prefill [%d,%d) %.1fs\n", s0, s0 + c, dt);
        for (int il = 0; il < n_layer; il++) {
            char nm[32];
            snprintf(nm, 32, "Kcur-%d", il);
            auto it = cap.last.find(nm);
            if (it == cap.last.end()) { fprintf(stderr, "missing %s\n", nm); return 1; }
            std::vector<float> fl = to_f32(it->second);
            // reshape [d,nh,T] -> append [T,nh,d]
            int64_t d = it->second.ne[0], nh = it->second.ne[1], T = it->second.ne[2];
            for (int64_t t = 0; t < T; t++)
                for (int64_t h = 0; h < nh; h++)
                    for (int64_t i = 0; i < d; i++)
                        acc[il].K.push_back(fl[(size_t) i + (size_t) d * (h + (size_t) nh * t)]);
            snprintf(nm, 32, "Vcur-%d", il);
            it = cap.last.find(nm);
            if (it == cap.last.end()) { fprintf(stderr, "missing %s\n", nm); return 1; }
            fl = to_f32(it->second);
            d = it->second.ne[0];
            nh = it->second.ne[1];
            T = it->second.ne[2];
            for (int64_t t = 0; t < T; t++)
                for (int64_t h = 0; h < nh; h++)
                    for (int64_t i = 0; i < d; i++)
                        acc[il].V.push_back(fl[(size_t) i + (size_t) d * (h + (size_t) nh * t)]);
            if (il == 0 && s0 == 0)
                printf("Kcur shape [%lld,%lld,%lld] Vcur shape [%lld,%lld,%lld]\n", (long long) it->second.ne[0],
                       (long long) it->second.ne[1], (long long) it->second.ne[2], (long long) it->second.ne[0],
                       (long long) it->second.ne[1], (long long) it->second.ne[2]);
            if (il == 0) { // diagnosis: prefill Qcur layer 0
                auto jt = cap.last.find("Qcur-0");
                if (jt == cap.last.end()) { fprintf(stderr, "missing Qcur-0 prefill\n"); return 1; }
                std::vector<float> qf = to_f32(jt->second);
                int64_t qd = jt->second.ne[0], qnh = jt->second.ne[1], qT = jt->second.ne[2];
                for (int64_t t = 0; t < qT; t++)
                    for (int64_t h = 0; h < qnh; h++)
                        for (int64_t i = 0; i < qd; i++)
                            Q0pre.push_back(qf[(size_t) i + (size_t) qd * (h + (size_t) qnh * t)]);
            }
        }
        pos += c;
    }

    // decode steps, greedy; collect Qcur all layers + kqv_out all layers + kq layer 0
    std::vector<std::vector<float>> Qs(n_layer), Os(n_layer);
    std::vector<std::vector<float>> KQs; // raw kq bytes per step (layer 0)
    std::vector<std::string> KQshapes;
    std::vector<llama_token> gen;
    {
        // first token from last prefill logits
        float * lg = llama_get_logits(ctx);
        int nvoc = llama_vocab_n_tokens(llama_model_get_vocab(mdl));
        int bt = 0;
        for (int i = 1; i < nvoc; i++) if (lg[i] > lg[bt]) bt = i;
        gen.push_back(bt);
    }
    for (int st = 0; st < decode; st++) {
        cap.last.clear();
        cap.collect_kq = true;
        b.n_tokens = 1;
        b.token[0] = gen.back();
        b.pos[0] = pos;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = true;
        double t0 = llama_time_us();
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        double dt = (llama_time_us() - t0) / 1e6;
        for (int il = 0; il < n_layer; il++) {
            char nm[32];
            snprintf(nm, 32, "Qcur-%d", il);
            auto it = cap.last.find(nm);
            if (it == cap.last.end()) { fprintf(stderr, "missing %s step %d\n", nm, st); return 1; }
            std::vector<float> fl = to_f32(it->second);
            int64_t d = it->second.ne[0], nh = it->second.ne[1];
            for (int64_t h = 0; h < nh; h++)
                for (int64_t i = 0; i < d; i++)
                    Qs[il].push_back(fl[(size_t) i + (size_t) d * h]);
            snprintf(nm, 32, "kqv_out-%d", il);
            it = cap.last.find(nm);
            if (it == cap.last.end()) { fprintf(stderr, "missing %s step %d\n", nm, st); return 1; }
            fl = to_f32(it->second);
            // kqv_out: [n_embd, T=1] (2d) or [d,nh,1]
            if (it->second.ne[2] > 1 || it->second.ne[3] > 1) {
                d = it->second.ne[0];
                nh = it->second.ne[1];
                for (int64_t h = 0; h < nh; h++)
                    for (int64_t i = 0; i < d; i++)
                        Os[il].push_back(fl[(size_t) i + (size_t) d * h]);
            } else {
                for (size_t k = 0; k < fl.size(); k++) Os[il].push_back(fl[k]);
            }
        }
        float * lg = llama_get_logits_ith(ctx, 0);
        int nvoc = llama_vocab_n_tokens(llama_model_get_vocab(mdl));
        int bt = 0;
        for (int i = 1; i < nvoc; i++) if (lg[i] > lg[bt]) bt = i;
        gen.push_back(bt);
        pos++;
        cap.collect_kq = false;
        { // stash kq-0 raw + shape
            auto it = cap.last.find("kq-0");
            if (it != cap.last.end()) {
                std::vector<float> kf = to_f32(it->second);
                KQs.push_back(std::move(kf));
                char sh[128];
                snprintf(sh, 128, "%lld,%lld,%lld,%lld", (long long) it->second.ne[0],
                         (long long) it->second.ne[1], (long long) it->second.ne[2],
                         (long long) it->second.ne[3]);
                KQshapes.push_back(sh);
            } else KQs.emplace_back();
        }
        if (st == 0) printf("decode step time %.3fs\n", dt);
    }
    llama_batch_free(b);

    // discover dims from acc
    int64_t Ld = tokens, Dd = 0, NHd = 0, KVd = 0;
    {
        // K per layer has Ld*KVd*Dd entries
        size_t n = acc[0].K.size();
        // Dd=64 assumed from arch; solve KVd
        Dd = 64;
        KVd = n / (size_t)(Ld * Dd);
        NHd = Qs[0].size() / (size_t)(decode * Dd);
    }
    printf("dims: L=%lld D=%lld Qheads=%lld KVheads=%lld decode=%d\n", (long long) Ld, (long long) Dd,
           (long long) NHd, (long long) KVd, decode);
    for (int il = 0; il < n_layer; il++) {
        char p[512];
        snprintf(p, 512, "%s/L%02d_K.npy", outdir.c_str(), il);
        write_npy(p, acc[il].K, {Ld, KVd, Dd});
        snprintf(p, 512, "%s/L%02d_V.npy", outdir.c_str(), il);
        write_npy(p, acc[il].V, {Ld, KVd, Dd});
        snprintf(p, 512, "%s/L%02d_Q.npy", outdir.c_str(), il);
        write_npy(p, Qs[il], {(int64_t) decode, NHd, Dd});
        snprintf(p, 512, "%s/L%02d_O.npy", outdir.c_str(), il);
        // O stored flat per head if 3d else whole embd: normalize to [decode, NH, D]
        int64_t per = Os[il].size() / decode;
        if (per == NHd * Dd) write_npy(p, Os[il], {(int64_t) decode, NHd, Dd});
        else write_npy(p, Os[il], {(int64_t) decode, per});
    }
    { // diagnosis files: prefill Q layer 0 + kq layer 0 per decode step
        char p[512];
        snprintf(p, 512, "%s/L00_Qpre.npy", outdir.c_str());
        write_npy(p, Q0pre, {Ld, NHd, Dd});
        for (size_t st = 0; st < KQs.size(); st++) {
            if (KQs[st].empty()) continue;
            snprintf(p, 512, "%s/L00_KQ_s%02d.npy", outdir.c_str(), (int) st);
            // flat save with shape in sidecar
            FILE * sf = fopen((std::string(p) + ".shape").c_str(), "w");
            fprintf(sf, "%s\n", KQshapes[st].c_str());
            fclose(sf);
            write_npy(p, KQs[st], {(int64_t) KQs[st].size()});
        }
    }
    {
        char p[512];
        snprintf(p, 512, "%s/positions.txt", outdir.c_str());
        FILE * f = fopen(p, "w");
        fprintf(f, "ctx_tokens=%d decode=%d decode_positions=", tokens, decode);
        for (int st = 0; st < decode; st++) fprintf(f, "%s%d", st ? "," : "", tokens + st);
        fprintf(f, "\n");
        fclose(f);
    }
    printf("SAVED to %s\n", outdir.c_str());
    llama_free(ctx);
    llama_model_free(mdl);
    return 0;
}
