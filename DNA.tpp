#include "DNA.h"

template <typename T>
DNA<T>::DNA(const vector<T>& sequence) :  seq(sequence) {
    checkEmpty();
}

template <typename T>
DNA<T>::DNA(const string& sequence) {
    // Preallocate memory
    seq.resize(sequence.size());
    for (size_t i = 0; i < sequence.size(); ++i) {
        seq[i] = charToInt.at(sequence[i]);
    }
    checkEmpty();
}

template <typename T>
char DNA<T>::toChar(T base) const {
    return this->intToChar.at(base);
}

template <typename T>
int DNA<T>::toInt(char base) const {
    return this->charToInt.at(base);
}

template <typename T>
void DNA<T>::checkEmpty() const {
    if (seq.empty()) {
        throw std::runtime_error("Input cannot be empty");
    }
}

template <typename T>
void DNA<T>::printSequences() const {
    for (size_t i = 0; i < seq.size(); ++i) {
        cout << seq[i] << " ";
    }
    cout << endl;
}

template <typename T>

int DNA<T>::operator[](size_t index) const {
    if (index >= seq.size()) {
        throw std::out_of_range("Index out of range");
    }
    return seq[index];
}

template <typename T>
size_t DNA<T>::size() const {
    return seq.size();
}

template <typename T>
vector<T> DNA<T>::getSeq() const {
    return seq;
}

template <typename T>
DNA<T> DNA<T>::mutate(
    double mutation_probability,
    int drop,
    int add,
    int sub,
    int seq_length) {
        
    const vector<T>& mutated_seq = mutate_vec( mutation_probability, drop, add, sub, seq_length);        
    
    // Move makes the l value become the r value, so the memory is not copied
    DNA<T> mutated(move(mutated_seq));
    return mutated;
}

    
template <typename T>
vector<T> DNA<T>::mutate_vec(double p, int drop, int add, int sub, int seq_length) {
    size_t max_length = seq.size();
    vector<T> mutated_seq = seq;

    assert(p >= 0 && p <= 1);
    size_t num_mutation =  static_cast<size_t>(std::round(seq_length * p));

    // initialize RNG for each thread
    
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<T> dist_alphabet(0, 3);
    uniform_int_distribution<int> dist_mutation(0, drop + add + sub - 1);



    // edge case check
    if (mutated_seq.empty()) {
        return std::vector<T>(seq_length, static_cast<T>(1));
    }
    size_t start_index = 0;

    // Decide which indices to mutate
    vector<size_t> possible_indices(seq_length);
    iota(possible_indices.begin(), possible_indices.end(), start_index);
    shuffle(possible_indices.begin(), possible_indices.end(), rng);
    vector<size_t> indices(possible_indices.begin(), possible_indices.begin() + num_mutation);
    std::sort(indices.begin(), indices.end(), std::greater<size_t>());

    // Mutation Loop
    for (auto index: indices) {
        // safety check
        if (index >= max_length) continue;  
        int mutation_type = dist_mutation(rng);

        if (mutation_type < drop){
            mutated_seq.erase(mutated_seq.begin() + index);
        } else if (mutation_type < drop + add) {
            T new_base = dist_alphabet(rng)+1;
            mutated_seq.insert(mutated_seq.begin() + index, new_base);
        } else {
            T current =mutated_seq[index];
            T substitute_base = dist_alphabet(rng)+1;
            while (substitute_base == current) {
                substitute_base = dist_alphabet(rng)+1;
            }
            mutated_seq[index] = substitute_base;
        }
    }
    if (mutated_seq.size() > seq_length) {
        mutated_seq.resize(seq_length);
    }
    while (mutated_seq.size() < seq_length) {
        T new_base = dist_alphabet(rng)+1;
        mutated_seq.push_back(new_base);
    }

    return mutated_seq;

}


    

