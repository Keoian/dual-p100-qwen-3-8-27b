#pragma once

// JEV System 1 decisions (autotrust/JEV-27B, template bare-v1): a backbone LoRA plus a linear 24-slot head over the
// last token's final-norm hidden state, with a per-kind temperature.
//   noul   -> slots 0..1  (false, true)
//   score  -> slots 2..7  (0..5)
//   choice -> slots 8..23 (A..P)

#include <string>
#include <vector>

enum jev_kind {
    JEV_KIND_NOUL   = 0,
    JEV_KIND_SCORE  = 1,
    JEV_KIND_CHOICE = 2,
};

constexpr int JEV_N_SLOTS      = 24;
constexpr int JEV_MAX_CHOICES  = 16;

struct jev_head {
    int                n_embd = 0;
    std::vector<float> w;          // [JEV_N_SLOTS, n_embd], row-major (proj.weight)
    std::vector<float> b;          // [JEV_N_SLOTS]          (proj.bias)
    float              temp[3] = { 1.0f, 1.0f, 1.0f }; // per kind, indexed by jev_kind
};

bool        jev_kind_from_str(const std::string & s, jev_kind & kind);
const char * jev_kind_str(jev_kind kind);

// head.safetensors (proj.weight F32 [24, n_embd], proj.bias F32 [24]) and calibration.json ("per_kind"); calib may be empty
bool jev_head_load(const std::string & head_path, const std::string & calib_path, jev_head & head, std::string & err);

// options: noul -> must be {"false","true"} (or empty), score -> {"0".."5"} (or empty), choice -> 2..16 strings
bool jev_check_options(jev_kind kind, std::vector<std::string> & options, std::string & err);

// "[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n{lines}\n[decision]:", tokenized as one string
// without BOS; choice lines are "A) option"
std::string jev_build_prompt(jev_kind kind, const std::string & state, const std::string & question,
                             const std::vector<std::string> & options);

// hidden: n_embd floats; returns calibrated probabilities over the options and (optionally) the raw slot logits
std::vector<float> jev_decide(const jev_head & head, jev_kind kind, const float * hidden, size_t n_options,
                              std::vector<float> * raw_logits = nullptr);
