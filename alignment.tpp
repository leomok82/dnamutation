#include "alignment.h"


template <typename T>
alignment<T>::alignment(bool semiglobal, bool identity, int count_ends, float match_score,
                        float mismatch_penalty, float gap_open_penalty, float gap_extend_penalty, int seq_length)
    : semiglobal(semiglobal), identity(identity), count_ends(count_ends),
      match_score(match_score), mismatch_penalty(mismatch_penalty),
      gap_open_penalty(gap_open_penalty), gap_extend_penalty(gap_extend_penalty),
      seq_length(seq_length) {}


template <typename T>
alignment<T>::alignment(DNA<T>& seq1, DNA<T>& seq2, bool semiglobal, bool identity, int count_ends, float match_score, float mismatch_penalty, 
    float gap_open_penalty, float gap_extend_penalty, int seq_length)
    : semiglobal(semiglobal), identity(identity), count_ends(count_ends),
      match_score(match_score), mismatch_penalty(mismatch_penalty),
      gap_open_penalty(gap_open_penalty), gap_extend_penalty(gap_extend_penalty),
      seq_length(seq_length) {

        global_alignment(seq1, seq2);
        }

template <typename T>
float alignment<T>::align(const DNA<T>& dna1, const DNA<T>& dna2) {
    // auto seq1 = dna1.getSeq();
    auto seq1 = dna1;
    auto seq2 = dna2;
    // auto seq2 = dna2.getSeq();
    vector<int> scores(seq_length);
    int m = seq_length;
    int n = seq_length;

    // Score matrice
    vector<vector<float>> M(m + 1, vector<float>(n + 1, numeric_limits<float>::lowest() / 2));
    vector<vector<float>> X(m + 1, vector<float>(n + 1, numeric_limits<float>::lowest() / 2));
    vector<vector<float>> Y(m + 1, vector<float>(n + 1, numeric_limits<float>::lowest() / 2));
    vector<vector<char>> traceback(m + 1, vector<char>(n + 1, 'M'));

    // traceback matrices
    M[0][0] = 0;
    X[0][0] = Y[0][0] = numeric_limits<float>::lowest() / 2;

    for (int i = 1; i <= m; ++i) {
        M[i][0] = numeric_limits<float>::lowest() / 2;
        X[i][0] = gap_open_penalty + (i - 1) * gap_extend_penalty;
        Y[i][0] = numeric_limits<float>::lowest() / 2;
        if (semiglobal) {
            X[i][0] = 0;
        }
        traceback[i][0] = 'U'; // Up (gap in seq2)
    }
    for (int j = 1; j <= n; ++j) {
        M[0][j] = numeric_limits<float>::lowest() / 2;
        X[0][j] = numeric_limits<float>::lowest() / 2;
        Y[0][j] = gap_open_penalty + (j - 1) * gap_extend_penalty;
        if (semiglobal) {
            Y[0][j] = 0;
        }
        traceback[0][j] = 'L'; // Left (gap in seq1)
    }

    // Compute scores and fill matrices
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            // Match or mismatch
            float score_substitution = (seq1[i - 1] == seq2[j - 1]) ? match_score : mismatch_penalty;

            float M_score = M[i - 1][j - 1] + score_substitution;
            float X_score = X[i - 1][j - 1] + score_substitution;
            float Y_score = Y[i - 1][j - 1] + score_substitution;
            M[i][j] = max({M_score, X_score, Y_score});

            // Gap in seq1
            float open_gap_Y = max(M[i][j - 1] + gap_open_penalty, X[i][j - 1] + gap_open_penalty);
            float extend_gap_Y = Y[i][j - 1] + gap_extend_penalty;
            if (semiglobal && i == m) {
                open_gap_Y = max(M[i][j - 1], X[i][j - 1]);
                extend_gap_Y = Y[i][j - 1];
            }
            Y[i][j] = max(open_gap_Y, extend_gap_Y);

            // Gap in seq2
            float open_gap_X = max(M[i - 1][j] + gap_open_penalty, Y[i - 1][j] + gap_open_penalty);
            float extend_gap_X = X[i - 1][j] + gap_extend_penalty;
            if (semiglobal && j == n) {
                open_gap_X = max(M[i - 1][j], Y[i - 1][j]);
                extend_gap_X = X[i - 1][j];
            }
            X[i][j] = max(open_gap_X, extend_gap_X);

            // Traceback
            float max_score = std::max({M[i][j], X[i][j], Y[i][j]});
            if (max_score == M[i][j]) {
                traceback[i][j] = 'M'; // Match/Mismatch
            } else if (max_score == X[i][j]) {
                traceback[i][j] = 'U'; // Up (gap in seq2)
            } else {
                traceback[i][j] = 'L'; // Left (gap in seq1)
            }
        }
    }
    
    // Start traceback from the cell with the maximum score at (m, n)
    int i = m;
    int j = n;

    // Find the maximum score in the last row and column
    double max_score = max({M[i][j], X[i][j], Y[i][j]});

    string align1 = "";
    string align2 = "";

    // Variables to calculate sequence identity
    int matches = 0;

    // Traceback to get the aligned sequences and calculate sequence identity
    while (i > 0 || j > 0) {
        if (traceback[i][j] == 'M') {
            align1 = seq1.toChar(seq1[i - 1]) + align1;
            align2 = seq2.toChar(seq2[j - 1]) + align2;

            // Update sequence identity counters
            if (seq1[i - 1] == seq2[j - 1]) matches++;
            --i;
            --j;
        } else if (traceback[i][j] == 'U') {
            align1 = seq1.toChar(seq1[i - 1]) + align1;
            align2 = "-" + align2;
            --i;
        } else if (traceback[i][j] == 'L') {
            align1 = "-" + align1;
            align2 = seq2.toChar(seq2[j - 1]) + align2;
            --j;
        } else {
            break;
        }
    }
    int total_length = align1.length();
    int denominator = 1;
    if (count_ends == 0) {
        int start_pos = 0;
        
        //Find the first non-continuous gap position in align1 or align2
        while (start_pos < total_length && (
            (start_pos == 0 && (align1[start_pos] == '-' || align2[start_pos] == '-')) || 
            (start_pos > 0 && (align1.substr(start_pos-1,2) == "--" || align2.substr(start_pos-1,2) == "--"))
        ))
        {
            start_pos++;
        }

        //Find the last non-continuous gap position in align1 or align2
        int end_pos = total_length - 1;
        while (end_pos > start_pos && (
            (end_pos == total_length - 1 && (align1[end_pos] == '-' || align2[end_pos] == '-')) || 
            (end_pos < total_length - 1 && (align1.substr(end_pos,2) == "--" || align2.substr(end_pos,2) == "--"))
        ))
        {
            end_pos--;
        }
        denominator = -start_pos + end_pos + 1;
    } else if (count_ends == 1) {
        denominator = total_length;
    } else {
        denominator = seq1.size();
    }

    // Return either best alignment score or sequence identity
    max_score = max_score /denominator;
    if (identity == false) {
        return max_score;
    } else {
        double sequence_identity = (matches > 0) ? (static_cast<double>(matches) / (denominator)) : 0.0;
        return sequence_identity;
    }
    return -1;
}
      
      


template <typename T>
float alignment<T>::global_alignment(DNA<T>& seq1, DNA<T>& seq2) {
        total_score = align(seq1, seq2);
        return total_score;
    }

template <typename T>
float alignment<T>::get_score() const {
    return total_score;
}
      
