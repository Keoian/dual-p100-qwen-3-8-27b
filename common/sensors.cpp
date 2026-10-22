#include "sensors.h"

#include <algorithm>
#include <cmath>
#include <fstream>

using json = nlohmann::ordered_json;

int sensor_bank::find(const std::string & id) const {
    for (size_t i = 0; i < heads.size(); i++) {
        if (heads[i].id == id) {
            return (int) i;
        }
    }
    return -1;
}

std::vector<int> sensor_bank::layers() const {
    std::vector<int> out;
    for (const auto & h : heads) {
        if (std::find(out.begin(), out.end(), h.layer) == out.end()) {
            out.push_back(h.layer);
        }
    }
    return out;
}

bool sensor_bank_load(const std::string & bank_path, sensor_bank & bank, std::string & err) {
    std::ifstream fj(bank_path);
    if (!fj) {
        err = "cannot open " + bank_path;
        return false;
    }
    json j;
    try {
        j = json::parse(fj);
    } catch (const std::exception & e) {
        err = bank_path + ": " + e.what();
        return false;
    }
    const std::string ext = ".json";
    if (bank_path.size() <= ext.size() || bank_path.compare(bank_path.size() - ext.size(), ext.size(), ext) != 0) {
        err = "sensor bank path must end in .json: " + bank_path;
        return false;
    }
    const std::string w_path = bank_path.substr(0, bank_path.size() - ext.size()) + ".f32";
    std::ifstream fw(w_path, std::ios::binary | std::ios::ate);
    if (!fw) {
        err = "cannot open " + w_path;
        return false;
    }
    try {
        bank.n_embd = j.at("n_embd").get<int>();
        const auto & hs = j.at("heads");
        const size_t row_bytes = (size_t) bank.n_embd * sizeof(float);
        const size_t n_bytes   = (size_t) fw.tellg();
        bank.heads.clear();
        for (const auto & e : hs) {
            sensor_head h;
            h.id      = e.at("id").get<std::string>();
            const auto & ly = e.at("layer");
            h.layer   = ly.is_string() ? -1 : ly.get<int>();
            h.bias    = e.at("bias").get<float>();
            h.platt_a = e.value("platt_a", 1.0f);
            h.platt_b = e.value("platt_b", 0.0f);
            h.meta    = e;
            const size_t row = e.at("row").get<size_t>();
            if ((row + 1) * row_bytes > n_bytes) {
                err = "head " + h.id + ": row " + std::to_string(row) + " out of range in " + w_path;
                return false;
            }
            h.w.resize(bank.n_embd);
            fw.seekg((std::streamoff) (row * row_bytes));
            fw.read((char *) h.w.data(), (std::streamsize) row_bytes);
            if (!fw) {
                err = "short read in " + w_path;
                return false;
            }
            if (bank.find(h.id) >= 0) {
                err = "duplicate head id " + h.id;
                return false;
            }
            bank.heads.push_back(std::move(h));
        }
        bank.info = j;
        bank.info.erase("heads");
    } catch (const std::exception & e) {
        err = bank_path + ": " + e.what();
        return false;
    }
    return true;
}

float sensor_eval(const sensor_head & head, const float * h, int n_embd) {
    double z = head.bias;
    for (int i = 0; i < n_embd; i++) {
        z += (double) head.w[i] * h[i];
    }
    const double t = head.platt_a * z + head.platt_b;
    return (float) (1.0 / (1.0 + std::exp(-t)));
}
