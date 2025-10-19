#pragma once
#include "type.hpp"
#include <cstddef>
#include <iostream>
#include <map>
#include <vector>

using namespace std;
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
    Bitmap coreMap;
    vector<vertex_id_t> misc_data;
};

using compress_t = vector<CoreCompressedSequence>; 

class CorpusCompressor {
public:
    void compressSequence(vector<vertex_id_t> &seq,CoreCompressedSequence& hms) {
        map<vertex_id_t,int> freq;
        for(size_t i = 0; i < seq.size(); i++){
            freq[seq[i]]++;
        }
        vertex_id_t freq_max_node;
        int max_freq = 0;
        for(const auto& pair: freq){
            if(pair.second > max_freq){
                freq_max_node = pair.first;
                max_freq = pair.second;
            } 
        }
        hms.misc_data.push_back(freq_max_node);
        for(size_t  i = 0; i < seq.size(); i++) {
            if(seq[i] == freq_max_node){
                hms.coreMap.set(i);
                continue;
            }else{
                hms.misc_data.push_back(seq[i]);
            }
        }
    }

    void uncompressSequence(vector<vertex_id_t> &seq,CoreCompressedSequence& hms) {
        vertex_id_t freq_max_node = hms.misc_data[0];
        int p = 1;// p misc_data point; the first one is Freq_Peak
        int q = 0; // map point
        while(q < hms.coreMap.size()){
            if(hms.coreMap.check(q)){
                seq.push_back(freq_max_node);
            }else {
                seq.push_back(hms.misc_data[p]);
                p++;
            }
            q++;
        }
        while(p < hms.misc_data.size()){
            seq.push_back(hms.misc_data[p]);
            p++;
        }
    }

    void compressCorpus(corpus_t &cor, compress_t &cp) {
        for(size_t i = 0; i < cor.size(); i++){
            CoreCompressedSequence hms;
            compressSequence(cor[i], hms);
            cp.push_back(hms);
        }
    }

    void uncompressCorpus(corpus_t &cor,compress_t& cp) {
        for(size_t i = 0; i < cp.size(); i++){
            vector<vertex_id_t> seq;
            uncompressSequence(seq, cp[i]);
            cor.push_back(seq);
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
};
