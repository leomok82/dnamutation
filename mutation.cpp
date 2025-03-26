#include "mutation.h"

using namespace std;




int main() {
    torch::Tensor tensor = torch::randint(1,5, {1,60});
    vector<float> rates = {0.1, 0.2, 0.3};

    kmer_collection<int8_t> collection(tensor, rates, 1, 1, 8, 60);
    // cout << collection.getData() << endl;

    cgr_collection<int8_t> collection2(tensor, rates, 1, 1, 8, 30);
    // cout << collection2.getData() << endl;
    return 0;
}

PYBIND11_MODULE(mutation_new, m ) {
    m.doc() = "Python bindings for multithread DNA mutation, alignment and chaos game representations";

    py::class_<DNA<int8_t>>(m,"DNA")
        .def(py::init<const std::vector<int8_t>&>(), "Construct from a vector of int")
        .def(py::init<const std::string&>(), "Construct from a string")
        .def("printSequences", &DNA<int8_t>::printSequences, "Print the sequence")
        .def("__getitem__", &DNA<int8_t>::operator[], "Access element at an index")
        .def("size", &DNA<int8_t>::size, "Get the sequence length")
        .def("getSeq", &DNA<int8_t>::getSeq, "Get the sequence vector")
        .def("mutate", &DNA<int8_t>::mutate, "Mutate the sequence");
    
    py::class_<alignment<int8_t>>(m, "alignment")
        .def(py::init<bool, bool, int, float, float, float, float, int>(), "Constructor with parameters")
        .def(py::init<DNA<int8_t>&, DNA<int8_t>&, bool, bool, int, float, float, float, float, int>(), "Constructor that performs alignment on two DNA sequences")
        .def("global_alignment", &alignment<int8_t>::global_alignment, "Perform global alignment")
        .def("get_score", &alignment<int8_t>::get_score, "Get alignment score");

    py::class_<kmer_collection<int8_t>>(m, "kmer_collection")
        .def(py::init<
            torch::Tensor,
            std::vector<float>,
            int,
            int,
            int,
            int,
            bool,
            bool,
            int,
            float,
            float,
            float,
            float>(),
            py::arg("tensor"),
            py::arg("rates"),
            py::arg("drop"),
            py::arg("add"),
            py::arg("sub"),
            py::arg("seq_length") = 60,
            py::arg("semiglobal") = false,
            py::arg("identity") = true,
            py::arg("count_ends") = -1,
            py::arg("match_score") = 1.0,
            py::arg("mismatch_penalty") = -1.5,
            py::arg("gap_open_penalty") = -2.5,
            py::arg("gap_extend_penalty") = -1.0)
        .def("getData", &kmer_collection<int8_t>::getData, "Get the data tensor")
        .def("getAlignments", &kmer_collection<int8_t>::getAlignments, "Get the alignment tensor")
        .def_readwrite("data", &kmer_collection<int8_t>::data)
        .def_readwrite("alignments", &kmer_collection<int8_t>::alignments);

    py::class_<cgr_collection<int8_t>>(m, "cgr_collection")
        .def(py::init<
                torch::Tensor,
                std::vector<float>,
                int,
                int,
                int,
                int,
                bool,
                bool,
                int,
                float,
                float,
                float,
                float>(),
                py::arg("tensor"),
                py::arg("rates"),
                py::arg("drop"),
                py::arg("add"),
                py::arg("sub"),
                py::arg("seq_length") = 60,
                py::arg("semiglobal") = false,
                py::arg("identity") = true,
                py::arg("count_ends") = -1,
                py::arg("match_score") = 1.0,
                py::arg("mismatch_penalty") = -1.5,
                py::arg("gap_open_penalty") = -2.5,
                py::arg("gap_extend_penalty") = -1.0)
        .def("getData", &cgr_collection<int8_t>::getData, "Get the data tensor")
        .def("getAlignments", &cgr_collection<int8_t>::getAlignments, "Get the alignment tensor")
        .def_readwrite("data", &cgr_collection<int8_t>::data)
        .def_readwrite("alignments", &cgr_collection<int8_t>::alignments);

    
    }