#pragma once
#ifndef COLLECTION_H
#define COLLECTION_H
#include <vector>
#include <torch/torch.h>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <unordered_map>
#include <iostream>
#include "DNA.h"
#include "alignment.h"

using std::vector;
using std::pair;
using std::unordered_map;
using std::move;
using std::cout;
using std::endl;

template <typename T>
class dna_collection {
public:
    torch::Tensor data;
    torch::Tensor alignments;
    int num_sequences;
    vector<float> mutation_rates;

    virtual ~dna_collection() = default;

    torch::Tensor getData() const { return data; }
    torch::Tensor getAlignments() const { return alignments; }

protected:
    torch::TensorOptions getOptions() {
        if constexpr (std::is_same<T, int32_t>::value) {
            return torch::TensorOptions().dtype(torch::kInt32);
        } else if constexpr (std::is_same<T, int64_t>::value) {
            return torch::TensorOptions().dtype(torch::kInt64);
        } else if constexpr (std::is_same<T, int8_t>::value) {
            return torch::TensorOptions().dtype(torch::kInt8);
        } else if constexpr (std::is_same<T, float>::value) {
            return torch::TensorOptions().dtype(torch::kFloat32);
        } else if constexpr (std::is_same<T, double>::value) {
            return torch::TensorOptions().dtype(torch::kFloat64);
        } else {
            throw std::runtime_error("Unsupported type for TensorOptions");
        }
    }

    virtual void init(vector<vector<T>> anchor_seq,
                      int drop,
                      int add,
                      int sub,
                      int seq_length,
                      bool semiglobal,
                      bool identity,
                      int count_ends,
                      float match_score,
                      float mismatch_penalty,
                      float gap_open_penalty,
                      float gap_extend_penalty) = 0;

    virtual vector<T> flatten(const vector<vector<vector<T>>>& vec) {
        vector<T> result;
        result.reserve(vec.size() * vec[0].size() * vec[0][0].size());
    
        for (const auto& matrix : vec) {     
            for (const auto& row : matrix) {  
                result.insert(result.end(), row.begin(), row.end());
            }
        }
        return result;
    }
};

template <typename T>
class kmer_collection : public dna_collection<T> {
public:
    kmer_collection(const torch::Tensor tensor, vector<float> rates,
                    int drop,
                    int add,
                    int sub,
                    int seq_length = 60,
                    bool semiglobal = false,
                    bool identity = true,
                    int count_ends = -1,
                    float match_score = 1,
                    float mismatch_penalty = -1.5,
                    float gap_open_penalty  = -2.5,
                    float gap_extend_penalty = -1);

protected:
    void init(vector<vector<T>> anchor_seq,
              int drop,
              int add,
              int sub,
              int seq_length,
              bool semiglobal,
              bool identity,
              int count_ends,
              float match_score,
              float mismatch_penalty,
              float gap_open_penalty,
              float gap_extend_penalty) override;

    vector<T> flatten(const vector<vector<vector<T>>>& vec) override;
};



template <typename T>
class cgr_collection : public dna_collection<T> {
public:
    cgr_collection(const torch::Tensor tensor, vector<float> rates,
                   int drop, int add, int sub,
                   int seq_length = 60, bool semiglobal = false,
                   bool identity = true, int count_ends = -1,
                   float match_score = 1, float mismatch_penalty = -1.5,
                   float gap_open_penalty = -2.5, float gap_extend_penalty = -1);

protected:
    void init(vector<vector<T>> anchor_seq,
              int drop, int add, int sub,
              int seq_length, bool semiglobal, bool identity,
              int count_ends, float match_score,
              float mismatch_penalty, float gap_open_penalty,
              float gap_extend_penalty) override;

    vector<float> cgr(vector<T> seq, int seq_length);

    vector<float> flatten(const vector<vector<vector<float>>>& vec);
};

#include "collection.tpp"
#endif