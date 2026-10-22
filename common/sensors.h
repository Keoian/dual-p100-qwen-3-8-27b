#pragma once

// System-1 semantic sensor bank: linear heads over a hidden row of the frozen model (no extra transformer pass).
// bank.json: {"n_embd", "heads": [{"id", "row", "layer" ("final" or a layer index), "bias", "platt_a", "platt_b", ...}]}
// bank.f32:  n_heads x n_embd float32 weights, row-major (row = head "row").
//   p = sigmoid(platt_a * (dot(w, h) + bias) + platt_b)

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

struct sensor_head {
    std::string        id;
    int                layer = -1; // -1: final-norm row (the row MTP / JEV read), else the input of this layer
    std::vector<float> w;
    float              bias    = 0.0f;
    float              platt_a = 1.0f;
    float              platt_b = 0.0f;
    nlohmann::ordered_json meta; // the head's registry entry as loaded
};

struct sensor_bank {
    int                      n_embd = 0;
    std::vector<sensor_head> heads;
    nlohmann::ordered_json   info; // bank.json without the heads

    int find(const std::string & id) const; // -1 if unknown
    std::vector<int> layers() const;      // distinct layers used (-1 = final)
};

// bank_path is the .json; the weights are the same path with .f32 instead of .json
bool sensor_bank_load(const std::string & bank_path, sensor_bank & bank, std::string & err);

float sensor_eval(const sensor_head & head, const float * h, int n_embd);
