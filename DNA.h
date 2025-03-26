#ifndef DNA_H
#define DNA_H
#pragma once
#include <vector>
#include <iostream>
#include <string>
#include <torch/torch.h>     // PyTorch C++ Extensions
#include <torch/script.h>
#include <omp.h>
#include <random>
#include <iostream>
using std::vector;
using std::unordered_map;
using std::string;
using std::random_device;
using std::mt19937;
using std::uniform_int_distribution;
using std::iota;
using std::cout;
using std::endl;
template <typename T>
struct DNA {
    // Data
    vector<T> seq;

    // Default Constructor
    DNA() = default;

    // Constructor from a vector
    DNA(const vector<T>& sequence);

    // Constructor from a string
    DNA(const string& sequence);

    // Destructor
    ~DNA() = default;

    // Convert base integer to character
    char toChar(T base) const;

    // Convert character to integer
    int toInt(char base) const;

    // Check if sequence is empty
    void checkEmpty() const;

    // Print stored sequence
    void printSequences() const;

    // Access elements safely
    int operator[](size_t index) const;

    // Get sequence length
    size_t size() const;

    // Get sequence
    vector<T> getSeq() const;

    // Mutate sequence
    DNA<T> mutate(double mutation_probability, int drop, int add, int sub, int seq_length );

private:
    // Mappings for DNA bases
    unordered_map<char, T> charToInt = {
        {'A', static_cast<T>(1)},
        {'T', static_cast<T>(2)},
        {'C', static_cast<T>(3)},
        {'G', static_cast<T>(4)},
        {'-', static_cast<T>(0)}
    };
    
    unordered_map<T, char> intToChar = {
        {static_cast<T>(1), 'A'},
        {static_cast<T>(2), 'T'},
        {static_cast<T>(3), 'C'},
        {static_cast<T>(4), 'G'},
        {static_cast<T>(0), '_'}
    };

    // Generate a mutated sequence
    vector<T> mutate_vec(double p, int drop, int add, int sub, int seq_length);
};

#include "DNA.tpp"
#endif