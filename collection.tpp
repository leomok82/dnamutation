#include "collection.h"

// Redirect cerr to the file
template <typename T>
kmer_collection<T>::kmer_collection(const torch::Tensor tensor, vector<float> rates,
                                    int drop, int add, int sub, int seq_length,
                                    bool semiglobal, bool identity, int count_ends,
                                    float match_score, float mismatch_penalty,
                                    float gap_open_penalty, float gap_extend_penalty) {
    this->mutation_rates = rates;
    assert(tensor.dim() == 2);
    this->num_sequences = tensor.size(0);
    int max_length = tensor.size(1);
    auto tensor_cp = tensor.to(torch::kInt8);

    vector<vector<T>> anchor_seq;
    anchor_seq.resize(this->num_sequences, vector<T>(max_length));

    #pragma omp parallel for
    for (int i = 0; i < this->num_sequences; ++i) {
        auto row_tensor = tensor_cp[i].contiguous();
        anchor_seq[i] = vector<T>(row_tensor.data_ptr<T>(), row_tensor.data_ptr<T>() + max_length);
    }

    init(anchor_seq, drop, add, sub, seq_length, semiglobal, identity,
         count_ends, match_score, mismatch_penalty, gap_open_penalty, gap_extend_penalty);
}


template <typename T>
void kmer_collection<T>::init(vector<vector<T>> anchor_seq,
                              int drop, int add, int sub,
                              int seq_length, bool semiglobal, bool identity,
                              int count_ends, float match_score,
                              float mismatch_penalty, float gap_open_penalty,
                              float gap_extend_penalty) {
    vector<T> flat_data;
    flat_data.resize((this->mutation_rates.size() + 1) * (this->num_sequences) * seq_length);
    
    this->alignments = torch::zeros({static_cast<int32_t>(this->mutation_rates.size()),
                                     static_cast<int64_t>(this->num_sequences)},
                                     torch::kFloat32);
    

    #pragma omp parallel for
    for (int j = 0; j < this->num_sequences; j++) {
        auto flat_ptr = flat_data.data() +  (j* (this->mutation_rates.size()+1)) *seq_length ; 
        std::memcpy(flat_ptr, anchor_seq[j].data(), seq_length * sizeof(T));
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < this->mutation_rates.size(); ++i) {
        for (int j = 0; j < this->num_sequences; ++j) {
            DNA<T> seqs(anchor_seq[j]);  

            auto mutated_dna = seqs.mutate(this->mutation_rates[i], drop, add, sub, seq_length);
            auto mutated_seq = mutated_dna.getSeq();
            assert(mutated_seq.size() == seq_length);
            // copy mutated sequence into flat vector
            auto flat_ptr = flat_data.data() +  (j* (this->mutation_rates.size()+1)+(i+1)) *seq_length ; 
            std::memcpy(flat_ptr, mutated_seq.data(), seq_length * sizeof(T));

            alignment<T> align(mutated_dna, seqs, semiglobal, identity, count_ends,
                               match_score, mismatch_penalty, gap_open_penalty, gap_extend_penalty, seq_length);
            this->alignments.index({i, j}) = align.get_score();
        }
    }


    std::vector<int64_t> sizes = {
        static_cast<int64_t>(this->num_sequences),
        static_cast<int64_t>(this->mutation_rates.size() + 1),
        static_cast<int64_t>(seq_length)
    };
    std::vector<int64_t> strides = {
        static_cast<int64_t>((this->mutation_rates.size() + 1) * seq_length),
        static_cast<int64_t>(seq_length),
        1
    };

    auto options = this->getOptions();
    this->dna_collection<T>::data = torch::from_blob(flat_data.data(), sizes, strides, options).clone();
}

template <typename T>
vector<T> kmer_collection<T>::flatten(const vector<vector<vector<T>>>& vec) {
    vector<T> result;
    result.reserve(vec.size() * vec[0].size() * vec[0][0].size());

    for (const auto& matrix : vec) {    
        for (const auto& row : matrix) {  
            result.insert(result.end(), row.begin(), row.end());
        }
    }

    return result;
}



template <typename T>
cgr_collection<T>::cgr_collection(const torch::Tensor tensor, vector<float> rates,
                                  int drop, int add, int sub,
                                  int seq_length, bool semiglobal,
                                  bool identity, int count_ends,
                                  float match_score, float mismatch_penalty,
                                  float gap_open_penalty, float gap_extend_penalty) {
    this->mutation_rates = rates;
    assert(tensor.dim() == 2);
    this->num_sequences = tensor.size(0);
    int max_length = tensor.size(1);

    auto tensor_cp = tensor.to(torch::kInt8);

    vector<vector<T>> anchor_seq;
    anchor_seq.resize(this->num_sequences, vector<T>(max_length));

    #pragma omp parallel for
    for (int i = 0; i < this->num_sequences; ++i) {
        auto row_tensor = tensor_cp[i].contiguous();
        anchor_seq[i] = vector<T>(row_tensor.data_ptr<T>(), row_tensor.data_ptr<T>() + max_length);
    }

    init(anchor_seq, drop, add, sub, seq_length, semiglobal, identity,
         count_ends, match_score, mismatch_penalty, gap_open_penalty, gap_extend_penalty);
}

template <typename T>
void cgr_collection<T>::init(vector<vector<T>> anchor_seq,
                             int drop, int add, int sub,
                             int seq_length, bool semiglobal, bool identity,
                             int count_ends, float match_score,
                             float mismatch_penalty, float gap_open_penalty,
                             float gap_extend_penalty) {
    vector<float> flat_data;
    
    flat_data.resize((this->mutation_rates.size() + 1)*(this->num_sequences) *seq_length*2);

    this->alignments = torch::zeros({static_cast<int64_t>(this->mutation_rates.size()),
                                     static_cast<int64_t>(this->num_sequences)},
                                     torch::kFloat32);

    #pragma omp parallel for
    for (int  j = 0; j < this->num_sequences; j++) {
        auto cgr_seq = cgr(anchor_seq[j], seq_length);
        auto flat_ptr = flat_data.data()  +  (j* (this->mutation_rates.size()+1)) *seq_length * 2;
        std::memcpy(flat_ptr, cgr_seq.data(), seq_length * sizeof(float)*2);
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < this->mutation_rates.size(); ++i) {
        for (int  j = 0; j < this->num_sequences; ++j) {
            DNA<T> seqs(anchor_seq[j]);

            auto mutated_dna = seqs.mutate(this->mutation_rates[i], drop, add, sub, seq_length);
            auto mutated_seq = cgr(mutated_dna.getSeq(), seq_length);

            auto flat_ptr = flat_data.data() +  (j* (this->mutation_rates.size()+1)+(i+1)) *seq_length * 2; 
            std::memcpy(flat_ptr, mutated_seq.data(), seq_length * sizeof(float)*2);
            alignment<T> align(mutated_dna, seqs, semiglobal, identity, count_ends,
                               match_score, mismatch_penalty, gap_open_penalty, gap_extend_penalty, seq_length);
            this->alignments.index({i, j}) = align.get_score();
        }
    }

    assert(flat_data.size() == (this->mutation_rates.size() + 1) * this->num_sequences * seq_length * 2);

    std::vector<int64_t> sizes = {
        static_cast<int64_t>(this->num_sequences),                  
        static_cast<int64_t>(this->mutation_rates.size() + 1),        
        static_cast<int64_t>(seq_length),                           
        2                                                         
    };
    std::vector<int64_t> strides = {
        static_cast<int64_t>((this->mutation_rates.size() + 1) * seq_length * 2),
        static_cast<int64_t>(seq_length * 2),
        2,
        1
    };
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    this->dna_collection<T>::data = torch::from_blob(flat_data.data(), sizes, strides, options).clone();
}

template <typename T>
vector<float> cgr_collection<T>::cgr(vector<T> seq, int seq_length) {
    unordered_map<T, pair<float, float>> nucleoCoords = {
        {static_cast<T>(1), {-1.0, 1.0}},  {static_cast<T>(2), {1.0, 1.0}},
        {static_cast<T>(3), {-1.0, -1.0}}, {static_cast<T>(4), {1.0, -1.0}}
    };

    float x = 0;
    float y = 0;
    vector<float> coords;
    coords.reserve(seq.size() * 2);

    for (int i = 0; i < seq_length; ++i) {
        if (nucleoCoords.find(seq[i]) == nucleoCoords.end()) {
            throw std::runtime_error("Invalid nucleotide found in sequence (Note: Gaps are not allowed with cgr!)");
        }
        auto coord = nucleoCoords[seq[i]];
        float vx = coord.first;
        float vy = coord.second;

        x = x + 0.5 * (vx - x);
        y = y + 0.5 * (vy - y);

        coords.push_back(x);
        coords.push_back(y);
    }

    assert(coords.size() == seq_length * 2);
    return coords;
}

template <typename T>
vector<float> cgr_collection<T>::flatten(const vector<vector<vector<float>>>& vec) {
    vector<float> result;
    result.reserve(vec.size() * vec[0].size() * vec[0][0].size() * 2);

    for (const auto& matrix : vec) {    
        for (const auto& row : matrix) {  
            result.insert(result.end(), row.begin(), row.end());
        }
    }
    return result;
}