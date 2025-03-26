#ifndef ALIGNMENT_H
#define ALIGNMENT_H
#pragma once
#include "DNA.h"
#include <vector>
#include <limits>
#include <algorithm>
#include <string>

using std::vector;
using std::string;
using std::max;
using std::numeric_limits;


template <typename T>
class alignment {
private:
    float total_score;
    bool semiglobal = false;
    bool identity = true;
    int count_ends = -1;
    float match_score = 1;
    float mismatch_penalty = -1.5;
    float gap_open_penalty = -2.5;
    float gap_extend_penalty = -1;
    int seq_length = 60;

    // Alignment function
    float align(const DNA<T>& dna1, const DNA<T>& dna2);

public:
    // Default constructor
    alignment() = default;

    // Constructor with parameters
    alignment(bool semiglobal, bool identity, int count_ends, float match_score, 
              float mismatch_penalty, float gap_open_penalty, float gap_extend_penalty, int seq_length);

    // Constructor that performs alignment on two DNA sequences
    alignment(DNA<T>& seq1, DNA<T>& seq2, bool semiglobal, bool identity, int count_ends, 
              float match_score, float mismatch_penalty, float gap_open_penalty, 
              float gap_extend_penalty, int seq_length);

    // Destructor
    ~alignment() = default;

    // Perform global alignment
    float global_alignment(DNA<T>& seq1, DNA<T>& seq2);

    // Get alignment score
    float get_score() const;
};

#include "alignment.tpp"
#endif