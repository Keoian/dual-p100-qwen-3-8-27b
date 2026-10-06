#include "jev.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

using json = nlohmann::json;

static const int JEV_SLOT_LO[3] = { 0, 2,  8 };
static const int JEV_SLOT_HI[3] = { 2, 8, 24 };

bool jev_kind_from_str(const std::string & s, jev_kind & kind) {
    if (s == "noul")   { kind = JEV_KIND_NOUL;   return true; }
    if (s == "score")  { kind = JEV_KIND_SCORE;  return true; }
    if (s == "choice") { kind = JEV_KIND_CHOICE; return true; }
    return false;
}

const char * jev_kind_str(jev_kind kind) {
    switch (kind) {
        case JEV_KIND_NOUL:   return "noul";
        case JEV_KIND_SCORE:  return "score";
        case JEV_KIND_CHOICE: return "choice";
    }
    return "?";
}

bool jev_head_load(const std::string & head_path, const std::string & calib_path, jev_head & head, std::string & err) {
    std::ifstream f(head_path, std::ios::binary);
    if (!f) {
        err = "cannot open " + head_path;
        return false;
    }
    uint64_t n_hdr = 0;
    f.read((char *) &n_hdr, sizeof(n_hdr));
    if (!f || n_hdr == 0 || n_hdr > (1u << 20)) {
        err = "bad safetensors header in " + head_path;
        return false;
    }
    std::string hdr(n_hdr, '\0');
    f.read(hdr.data(), n_hdr);
    const size_t data_start = 8 + n_hdr;

    try {
        const json j = json::parse(hdr);
        auto read = [&](const char * name, size_t n_expected, std::vector<float> & out, std::vector<int64_t> & shape) {
            const json & t = j.at(name);
            if (t.at("dtype").get<std::string>() != "F32") {
                throw std::runtime_error(std::string(name) + " is not F32");
            }
            shape = t.at("shape").get<std::vector<int64_t>>();
            const auto offs = t.at("data_offsets").get<std::vector<uint64_t>>();
            const size_t n = (offs.at(1) - offs.at(0)) / sizeof(float);
            if (n_expected && n != n_expected) {
                throw std::runtime_error(std::string(name) + " has an unexpected size");
            }
            out.resize(n);
            f.seekg(data_start + offs[0]);
            f.read((char *) out.data(), n * sizeof(float));
            if (!f) {
                throw std::runtime_error(std::string(name) + ": short read");
            }
        };
        std::vector<int64_t> shape;
        read("proj.weight", 0, head.w, shape);
        if (shape.size() != 2 || shape[0] != JEV_N_SLOTS) {
            throw std::runtime_error("proj.weight must be [24, n_embd]");
        }
        head.n_embd = (int) shape[1];
        read("proj.bias", JEV_N_SLOTS, head.b, shape);
    } catch (const std::exception & e) {
        err = head_path + ": " + e.what();
        return false;
    }

    if (!calib_path.empty()) {
        try {
            std::ifstream fc(calib_path);
            const json pk = json::parse(fc).at("per_kind");
            head.temp[JEV_KIND_NOUL]   = pk.value("noul",   1.0f);
            head.temp[JEV_KIND_SCORE]  = pk.value("score",  1.0f);
            head.temp[JEV_KIND_CHOICE] = pk.value("choice", 1.0f);
        } catch (const std::exception & e) {
            err = calib_path + ": " + e.what();
            return false;
        }
    }
    return true;
}

bool jev_check_options(jev_kind kind, std::vector<std::string> & options, std::string & err) {
    switch (kind) {
        case JEV_KIND_NOUL: {
            const std::vector<std::string> want = { "false", "true" };
            if (options.empty()) {
                options = want;
            }
            if (options != want) {
                err = "noul options must be [\"false\", \"true\"]";
                return false;
            }
        } break;
        case JEV_KIND_SCORE: {
            const std::vector<std::string> want = { "0", "1", "2", "3", "4", "5" };
            if (options.empty()) {
                options = want;
            }
            if (options != want) {
                err = "score options must be [\"0\" .. \"5\"]";
                return false;
            }
        } break;
        case JEV_KIND_CHOICE: {
            if (options.size() < 2 || options.size() > JEV_MAX_CHOICES) {
                err = "choice needs 2-16 options, got " + std::to_string(options.size());
                return false;
            }
            for (const auto & o : options) {
                if (o.find('\n') != std::string::npos) {
                    err = "choice options must not contain newlines";
                    return false;
                }
            }
        } break;
    }
    return true;
}

std::string jev_build_prompt(jev_kind kind, const std::string & state, const std::string & question,
                             const std::vector<std::string> & options) {
    std::string p = "[kind] ";
    p += jev_kind_str(kind);
    p += "\n[state] " + state + "\n[question] " + question + "\n[options]\n";
    for (size_t i = 0; i < options.size(); i++) {
        if (i > 0) {
            p += "\n";
        }
        if (kind == JEV_KIND_CHOICE) {
            p += (char) ('A' + i);
            p += ") ";
        }
        p += options[i];
    }
    p += "\n[decision]:";
    return p;
}

std::vector<float> jev_decide(const jev_head & head, jev_kind kind, const float * hidden, size_t n_options,
                              std::vector<float> * raw_logits) {
    const int lo = JEV_SLOT_LO[kind];
    const int n  = (int) n_options;
    std::vector<float> z(n);
    for (int i = 0; i < n; i++) {
        const float * w = head.w.data() + (size_t) (lo + i) * head.n_embd;
        double acc = 0.0;
        for (int k = 0; k < head.n_embd; k++) {
            acc += (double) w[k] * hidden[k];
        }
        z[i] = (float) acc + head.b[lo + i];
    }
    if (raw_logits) {
        *raw_logits = z;
    }
    const float t = head.temp[kind];
    float zmax = -INFINITY;
    for (float & v : z) {
        v /= t;
        zmax = std::max(zmax, v);
    }
    double sum = 0.0;
    for (float & v : z) {
        v = std::exp(v - zmax);
        sum += v;
    }
    for (float & v : z) {
        v = (float) (v / sum);
    }
    return z;
}
