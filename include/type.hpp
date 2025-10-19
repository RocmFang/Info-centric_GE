
#pragma once

#include <stdint.h>
#include <queue>
#include <mutex>
#include <string>
#include <vector>

using std::queue;
using std::mutex;
using std::lock_guard;
using std::string;
using std::vector;

typedef uint32_t vertex_id_t;
typedef uint64_t com_neighbour_t;
typedef uint64_t edge_id_t;
typedef uint8_t partition_id_t;
typedef uint32_t task_counter_t;
typedef uint32_t walker_id_t;
typedef uint32_t step_t;
typedef float real_t;
typedef uint32_t dist_counter_t;
using corpus_t = vector<vector<vertex_id_t>>;

struct EmptyData
{
};

class SyncQueue
{
private:
   queue<corpus_t> base_queue;  // Changed to directly store corpus data
   mutex mutx; 
   bool isClose ;
public:
    SyncQueue(){
        this->isClose = false;
    }
    void push(corpus_t&& corpus_data){  // Accept rvalue reference for move semantics
        mutx.lock();
        base_queue.push(std::move(corpus_data));
        mutx.unlock();
    }
    corpus_t pop() {
        mutx.lock();
        corpus_t top = base_queue.front();
        base_queue.pop();
        mutx.unlock();
        return top;
    }
    void closeQueue(){
        lock_guard<std::mutex>(this->mutx);
        this->isClose = true;
    }
    bool isClosed() {
        return this->isClose;
    }
    bool isEmpty() {
        return this->base_queue.empty();
    }
};

template <typename edge_data_t>
struct Edge
{
    vertex_id_t src;
    vertex_id_t dst;
    edge_data_t data;
    Edge() {}
    Edge(vertex_id_t _src, vertex_id_t _dst, edge_data_t _data) : src(_src), dst(_dst), data(_data) {}
    bool friend operator == (const Edge<edge_data_t> &a, const Edge<edge_data_t> &b)
    {
        return (a.src == b.src
            && a.dst == b.dst
            && a.data == b.data
        );
    }

    void transpose()
    {
        std::swap(src, dst);
    }
};

template <>
struct Edge <EmptyData>
{
    vertex_id_t src;
    union
    {
        vertex_id_t dst;
        EmptyData data;
    };
    Edge() {}
    Edge(vertex_id_t _src, vertex_id_t _dst) : src(_src), dst(_dst) {}
    bool friend operator == (const Edge<EmptyData> &a, const Edge<EmptyData> &b)
    {
        return (a.src == b.src
            && a.dst == b.dst
        );
    }
    void transpose()
    {
        std::swap(src, dst);
    }
};

union U{
    struct
    {
        vertex_id_t key1;
        vertex_id_t key2;
    };
    com_neighbour_t key;
};
