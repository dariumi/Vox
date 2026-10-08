#pragma once
// Детерминированный сэмплинг, одинаковый на всех бэкендах.
//
// Philox4x32-10 — счётчиковый генератор: случайное число вычисляется из пары
// (ключ, счётчик), а не из внутреннего состояния. Счётчик = (кадр, кодбук),
// поэтому результат не зависит от порядка вычислений, батчинга и железа.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace vox {

struct Philox4x32 {
    using ctr_t = std::array<uint32_t, 4>;
    using key_t = std::array<uint32_t, 2>;

    static ctr_t round(ctr_t c, key_t k) {
        constexpr uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
        uint64_t p0 = uint64_t(M0) * c[0], p1 = uint64_t(M1) * c[2];
        return {uint32_t(p1 >> 32) ^ c[1] ^ k[0], uint32_t(p1), uint32_t(p0 >> 32) ^ c[3] ^ k[1], uint32_t(p0)};
    }
    static ctr_t block(ctr_t c, key_t k) {
        constexpr uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
        for (int i = 0; i < 10; ++i) {
            c = round(c, k);
            if (i < 9) { k[0] += W0; k[1] += W1; }
        }
        return c;
    }
};

// Генератор одной сессии. uniform(frame, codebook) всегда даёт одно и то же число.
class DetRng {
public:
    explicit DetRng(uint64_t seed) : key_{uint32_t(seed), uint32_t(seed >> 32)} {}
    // Равномерное число в [0, 1) с 24 битами точности (точно представимо во float).
    float uniform(uint64_t frame, uint32_t codebook, uint32_t draw = 0) const {
        auto r = Philox4x32::block({uint32_t(frame), uint32_t(frame >> 32), codebook, draw}, key_);
        return float(r[0] >> 8) * (1.0f / 16777216.0f);
    }

    uint32_t key0() const { return key_[0]; }
    uint32_t key1() const { return key_[1]; }

private:
    Philox4x32::key_t key_;
};

struct SampleParams {
    float temperature = 0.9f;
    float top_p = 1.0f;
    int32_t top_k = 50;
    float repetition_penalty = 1.0f;
};

// Сэмплирует индекс из логитов. history — ранее выбранные токены (для repetition penalty).
// Порядок операций как в HF transformers: penalty → temperature → top-k → top-p.
inline int32_t sample_logits(std::vector<float> &logits, const SampleParams &p, float u,
                             const std::vector<int32_t> *history = nullptr) {
    const int32_t n = int32_t(logits.size());
    if (n == 0) return -1;

    if (history && p.repetition_penalty != 1.0f) {
        std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
        for (int32_t t : *history)
            if (t >= 0 && t < n && !seen[size_t(t)]) {
                seen[size_t(t)] = 1;
                float &l = logits[size_t(t)];
                l = l > 0 ? l / p.repetition_penalty : l * p.repetition_penalty;
            }
    }

    // Жадный выбор: при равенстве — меньший индекс (детерминированно).
    if (p.temperature <= 0.0f) {
        return int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    std::vector<int32_t> idx(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    auto better = [&](int32_t a, int32_t b) {
        return logits[size_t(a)] > logits[size_t(b)] || (logits[size_t(a)] == logits[size_t(b)] && a < b);
    };
    int32_t k = (p.top_k > 0 && p.top_k < n) ? p.top_k : n;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), better);
    idx.resize(size_t(k));

    // softmax по отобранным, в double для стабильности
    const double inv_t = 1.0 / double(p.temperature);
    const double mx = double(logits[size_t(idx[0])]);
    std::vector<double> probs(idx.size());
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        probs[i] = std::exp((double(logits[size_t(idx[i])]) - mx) * inv_t);
        sum += probs[i];
    }
    for (auto &q : probs) q /= sum;

    // top-p: минимальный префикс с суммой >= top_p (минимум один токен)
    size_t keep = probs.size();
    if (p.top_p > 0.0f && p.top_p < 1.0f) {
        double c = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            c += probs[i];
            if (c >= double(p.top_p)) { keep = i + 1; break; }
        }
    }
    double total = 0;
    for (size_t i = 0; i < keep; ++i) total += probs[i];
    double target = double(u) * total, acc = 0;
    for (size_t i = 0; i < keep; ++i) {
        acc += probs[i];
        if (target < acc) return idx[i];
    }
    return idx[keep - 1];
}

}  // namespace vox
