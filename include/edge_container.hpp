
#include "type.hpp"

struct myAdjUnit
{
    vertex_id_t neighbour;
    real_t data;
};

struct myAdjList
{

    myAdjUnit *begin;
    myAdjUnit *end;
    void init()
    {
        begin = nullptr;
        end = nullptr;
    }
    void printmyAdjList()
    {
        for(myAdjUnit *p = begin; p < end; p++)
        {
            std::cout << p->neighbour << " " ;
        }
        std::cout << std::endl;
    }
};

struct myEdgeContainer
{

    myAdjList *adj_lists;

    myAdjUnit *adj_units;
    myEdgeContainer() : adj_lists(nullptr), adj_units(nullptr) {}
    ~myEdgeContainer()
    {
        if (adj_lists != nullptr)
        {
            delete []adj_lists;
        }
        if (adj_units != nullptr)
        {
            delete []adj_units;
        }
    }
};
