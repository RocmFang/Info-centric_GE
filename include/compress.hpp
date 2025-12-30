#pragma once
#include "type.hpp"
#include <cstddef>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

extern int compress_size;
extern int avg_length;

class Bitmap {
private:
    size_t bitSize = 0;
    const int EXPANDSIZE = 1;
    // Get the byte index for the target bit
    size_t getByteIndex(size_t bitIndex) const {
        return bitIndex / 8;
    }
    // Get the bit position inside the byte
    size_t getBitOffset(size_t bitIndex) const {
        return bitIndex % 8;
    }
    // Grow storage to make sure the requested bit index is addressable
    void expandToFit(size_t bitIndex) {
        size_t requiredBytes = getByteIndex(bitIndex) + 1;
        if(data.size()<requiredBytes){
            data.resize(requiredBytes + EXPANDSIZE, 0);
        }
    }
public:
    vector<char> data;
    Bitmap(){
        data.resize(2,0);
    }
    Bitmap(size_t bitNum){
        data.resize(bitNum/ sizeof(char) +1);
    }
    // Set the specified bit to one
    void set(size_t bitIndex) {
        expandToFit(bitIndex);
        size_t byteIndex = getByteIndex(bitIndex);
        size_t bitOffset = getBitOffset(bitIndex);
        data[byteIndex] |= (1 << bitOffset);
        this->bitSize = max(bitSize,bitIndex+1);
    }

    void unset(size_t bitIndex) {
        expandToFit(bitIndex);
        size_t byteIndex = getByteIndex(bitIndex);
        size_t bitOffset = getBitOffset(bitIndex);
        data[byteIndex] &= ~(1 << bitOffset);
        this->bitSize = max(bitSize,bitIndex + 1);
    }

    // Check whether the specified bit equals one
    bool check(size_t bitIndex)const {
        size_t byteIndex = getByteIndex(bitIndex);
        size_t bitOffset = getBitOffset(bitIndex);
        if(byteIndex >= data.size()){
            cout << "bitmap out of range\n";
            return false;
        }
        return (data[byteIndex] & (1 << bitOffset)) != 0;
    }
    
    size_t size() {
        return this-> bitSize;
    }

    size_t mem_size(){
        return data.size();
    }

    void printData(){
        for(size_t i = 0; i < data.size() * 8; i++){
            cout<< check(i);
        }
        cout << endl;
    }
};

class CoreCompressedSequence {
public:
    vector<Bitmap> coreMap;
    vector<vertex_id_t> misc_data;
    int original_size;
};

using compress_t = vector<CoreCompressedSequence>; 

class CorpusCompressor {
public:
    void compressSequence(const vector<vertex_id_t> &seq, CoreCompressedSequence& hms) {
        hms.original_size = seq.size();

        map<vertex_id_t,int> freq;
        vector<vertex_id_t> topK_nodes;
        for(size_t i = 0; i < seq.size(); i++){
            freq[seq[i]]++;
        }

        topKQuickSelect(freq, topK_nodes, compress_size - 1);
        hms.misc_data.insert(hms.misc_data.end(), topK_nodes.begin(), topK_nodes.end());
        for(int i = 0;i < compress_size;i++) hms.coreMap.emplace_back();

        for(size_t  i = 0; i < seq.size(); i++) {
            bool hasFound = false;
            for(int j = 0;j < compress_size - 1 && j < topK_nodes.size();j++) { // [TODO]
                if(seq[i] == topK_nodes[j]) {
                    hms.coreMap[j].set(i);
                    hasFound = true;
                    break;
                }
            }

            if(!hasFound) {
                hms.misc_data.push_back(seq[i]);
                hms.coreMap[compress_size - 1].set(i);
            }
        }
    }

    void uncompressSequence(vector<vertex_id_t> &seq, CoreCompressedSequence& hms) {
        int p = compress_size - 1;// p misc_data point; the first one is Freq_Peak
        int q = 0; // map point
        while(q < hms.original_size) {
            for(int i = 0;i < compress_size;i++) {
                if(hms.coreMap[i].check(q)) {
                    if(i == compress_size - 1) {
                        seq.push_back(hms.misc_data[p]);
                        p++;
                    } else {
                        seq.push_back(hms.misc_data[i]);
                    }
                    q++;
                    break;
                }
            }
        }
    }

    void compressCorpus(const corpus_t &cor, compress_t &cp) {
        cp.resize(cor.size());

        #pragma omp parallel for
        for(size_t i = 0; i < cor.size(); i++){
            compressSequence(cor[i], cp[i]);
        }
    }

    void uncompressCorpus(corpus_t &cor, compress_t& cp) {
        cor.resize(cp.size());

        #pragma omp parallel for
        for(size_t i = 0; i < cp.size(); i++){
            uncompressSequence(cor[i], cp[i]);
        }
    }

    void printCorpus(vector<vector<vertex_id_t>> &corpus){
        cout << "=== corpus print === " << endl;
        for(size_t i = 0; i < corpus.size(); i++){
            for(size_t j = 0; j < corpus[i].size(); j++){
                cout << corpus[i][j] <<" ";
            }
            cout << endl;
        }
        cout << "====================" << endl;
    }

    void topKQuickSelect(map<vertex_id_t,int> &m, vector<vertex_id_t> &ans, int k){
        if(k <= 0) return;

        std::vector<std::pair<vertex_id_t, int>> vec(m.begin(), m.end());

        int pivot_index = 0, left = 0, right = vec.size() - 1;
        while(left <= right){
            pivot_index = partition(vec, left, right);

            if(pivot_index == k - 1) break;
            else if(pivot_index < k - 1) left = pivot_index + 1;
            else right = pivot_index - 1;
        }

        for(int i = 0;i < k && i < vec.size();i++) ans.push_back(vec[i].first);
    }

    int partition(vector<pair<vertex_id_t,int>> &vec, int left, int right){
        int pivot = vec[right].second;
        int i = left;
        for(int j = left; j < right; j++){
            if(vec[j].second >= pivot){
                std::swap(vec[i], vec[j]);
                i++;
            }
        }
        std::swap(vec[i], vec[right]);
        return i;
    }

    //===================GPU Uncompress===================
    void uncompressCorpusGPU(int *misc_data, int *misc_data_len, char *bitmap, int *bitmap_len, int *d_sen, int *d_sent_len, int cnt_sentence);

    bool check(const corpus_t &cor, int *sen, int *sen_len, int corpus_index, int cnt_sentence, const vector<vertex_id_t> &id2offset) {
        int cnt = 0, all = 0;
        for(size_t i = corpus_index;i < corpus_index + cnt_sentence;i++){
            if(cor[i].size() != (sen_len[i+1-corpus_index] - sen_len[i-corpus_index])){
                cout << "sentence " << i << " length mismatch: " << cor[i].size() << " != " << sen_len[i+1-corpus_index] - sen_len[i-corpus_index] << endl;
                return false;
            }

            for(size_t j = 0; j < cor[i].size(); j++){
                auto word = id2offset[cor[i][j]];
                if(word != sen[sen_len[i-corpus_index] + j]){
                    cout << "sentence " << i << " word " << j << " mismatch: " << word << " != " << sen[sen_len[i-corpus_index] + j] << endl;
                    cnt++;
                    // return false;
                }
                all++;
            }
        }
        cout<< "cnt: " << cnt << " all: " << all << endl; 

        if(cnt > 0) return false;

        cout << "corpus check passed!" << endl;
        return true;
    }

    bool check(const corpus_t &corA, const corpus_t &corB) {
        if(corA.size() != corB.size()) {
            cout << "corpus size mismatch: " << corA.size() << " != " << corB.size() << endl;
            return false;
        }

        for(size_t i = 0; i < corA.size(); i++) {
            if(corA[i].size() != corB[i].size()) {
                cout << "sentence " << i << " length mismatch: " << corA[i].size() << " != " << corB[i].size() << endl;
                return false;
            }

            for(size_t j = 0; j < corA[i].size(); j++) {
                if(corA[i][j] != corB[i][j]) {
                    cout << "sentence " << i << " word " << j << " mismatch: " << corA[i][j] << " != " << corB[i][j] << endl;
                    return false;
                }
            }
        }

        cout << "corpus check passed!" << endl;
        return true;
    }
};
