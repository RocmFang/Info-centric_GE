#include "type.hpp"
#include "walk.hpp"
#include "option_helper.hpp"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include "edge_container.hpp"
#include <sys/stat.h>
#include <cstdio>
#include "compress.hpp"
#include <map>


// template struct EdgeContainer<real_t>;
using namespace std;

struct TrainingConfig {
    int init_round;
    int batch_size;
    
    TrainingConfig(int init_round = 1, int batch_size = 16384) 
        : init_round(init_round), batch_size(batch_size) {}
};

int train_corpus_cuda(int argc, char **argv,const vector<vertex_id_t>& degrees,SyncQueue& corpus_q,int _my_rank,myEdgeContainer *csr, const TrainingConfig& config);

struct Empty
{
};

// ./bin/simple_walk -g ./karate.data -v 34 -w 34 -o ./out/walks.txt > perf_dist.txt
int main(int argc, char **argv)
{
    umask(0);
    Timer timer;
    double load_graph_time = 0.0;
    double data_conversion_time = 0.0;
    double walk_setup_time = 0.0; 
    double walk_execution_time = 0.0;
    double corpus_output_time = 0.0;
    double corpus_compression_time = 0.0;
    // double training_time = 0.0;
    MPI_Instance mpi_instance(&argc, &argv);
    int my_rank = get_mpi_rank();


    RandomWalkOptionHelper opt;
    opt.parse(argc, argv);

    WalkEngine<real_t, uint32_t> graph;

    //=============== annotation line ===================
    graph.set_init_round(opt.init_round);
    // printf("opt min length: %d\n",opt.min_length);
    graph.set_minLength(opt.min_length);
    // printf("init_round = %d, min_length = %d\n", graph.init_round, graph.minLength);
    // printf("graph path: %s\n",opt.graph_path.c_str());
    Timer load_timer;
    graph.load_graph(opt.v_num, opt.graph_path.c_str(), opt.partition_path.c_str(), opt.make_undirected);
    load_graph_time = load_timer.duration();
    printf("[ %d ] load_graph ok!\n",my_rank);
    graph.vertex_cn.resize(graph.get_vertex_num());
    // graph.load_commonNeighbors(opt.graph_common_neighbour.c_str());
    vector<vertex_id_t> vertex_degree(graph.v_num,0);
    for (vertex_id_t v = 0; v < graph.v_num; v++){
        vertex_degree[v] = graph.vertex_out_degree[v];
    }
        
    //myEdgeContainer* myec = reinterpret_cast<myEdgeContainer*>(&graph.g_csr);
    //cout <<"myec access " << myec-> adj_lists[0].begin->neighbour<<endl; 
    myEdgeContainer* myec = new myEdgeContainer();
    myec->adj_lists = new myAdjList[graph.v_num];
    myec->adj_units = new myAdjUnit[graph.e_num];
    edge_id_t chunk_edge_idx = 0;
    printf("[ %d ] malloc ok\n",my_rank);
    for(vertex_id_t v_i = 0; v_i < graph.v_num; v_i++){
      myec->adj_lists[v_i].begin = myec->adj_units + chunk_edge_idx;
      chunk_edge_idx += graph.csr->adj_lists[v_i].end -graph.csr->adj_lists[v_i].begin; 
      myec->adj_lists[v_i].end = myec->adj_units + chunk_edge_idx;
    }
    for(edge_id_t e_i = 0; e_i < graph.e_num; e_i++){
     myec->adj_units[e_i].neighbour = graph.csr->adj_units[e_i].neighbour;
     myec->adj_units[e_i].data = graph.csr->adj_units[e_i].data;
    }
    // cout <<my_rank <<" myec access " << myec-> adj_lists[110].begin->neighbour<<endl; 
    // cout << my_rank <<" graph.csr access " << graph.csr-> adj_lists[110].begin->neighbour<<endl; 

    // =============== Start Training Thread ===============
    Timer training_start_timer;
    printf("[ %d ] Starting training thread...\n", my_rank);
    TrainingConfig train_config(graph.init_round, opt.batch_size);
    thread train_thread(train_corpus_cuda,argc,argv,std::ref(vertex_degree),std::ref(graph.out_queue), my_rank,myec, train_config);
    printf("[ %d ] Training thread started\n", my_rank);

    auto extension_comp = [&](Walker<uint32_t> &walker, vertex_id_t current_v)
    {
        // return 0.995;
        return walker.step >= 40 ? 0.0 : 1.0;
    };
    auto static_comp = [&](vertex_id_t v, AdjUnit<real_t> *edge)
    {
        return 1.0; /*edge->data is a real number denoting edge weight*/
    };
    auto dynamic_comp = [&](Walker<uint32_t> &walker, vertex_id_t current_v, AdjUnit<real_t> *edge)
    {
        return 1.0;
    };
    auto dynamic_comp_upperbound = [&](vertex_id_t v_id, AdjList<real_t> *adj_lists)
    {
        return 1.0;
    };

    // =============== Walk Configuration Setup ===============

    WalkerConfig<real_t, uint32_t> walker_conf(opt.walker_num);
    TransitionConfig<real_t, uint32_t> tr_conf(extension_comp);
    
    for (int i = 0; i < 1; i++) // ???????????? for(int i = 0; i < 1; i++) 
    {
        int pid = get_mpi_rank();
        WalkConfig walk_conf;
        
        if (!opt.output_path.empty())
        {
            std::cout<< opt.output_path <<std::endl;
            walk_conf.set_output_file(opt.output_path.c_str());
        }
        if (opt.set_rate)
        {
            walk_conf.set_walk_rate(opt.rate);
        }
        // =============== Random Walk Execution ===============
        Timer walk_timer;
        printf("=================[ %d ] RANDOM WALK EXECUTION ================\n",my_rank);
        graph.random_walk(&walker_conf, &tr_conf, &walk_conf);
        // double sum_time = walk_timer.duration();
        // walk_execution_time = sum_time;
        // double walk_time = sum_time - graph.other_time;
        // printf("[p%u][WALK EXECUTION] Total: %lf s, Pure walk: %lf s, Other: %lf s\n", 
        //        graph.get_local_partition_id(), sum_time, walk_time, graph.other_time);
        
        // // Use in-memory pipeline (no disk I/O)
        // if (!opt.output_path.empty()) {
        //     printf("[ %d ] *** USING MEMORY PIPELINE (NO DISK I/O) *** \n", my_rank);
        // }
    }
    printf("> [p%d RANDOM WALKING TIME:] %lf \n",get_mpi_rank(), timer.duration());

    // * Close the task queue
    graph.out_queue.closeQueue();

    if(get_mpi_rank()==0){
        cout<<"============partion table=========="<<endl;
        for(int p=0;p<get_mpi_size();p++){
            cout<<"part: "<<p<<" "<<graph.vertex_partition_begin[p]<<" ~ "<<graph.vertex_partition_end[p]<<endl;
        }
    }

    train_thread.join();

    double total_time = timer.duration();
    
    
    printf("> [p%d WHOLE TIME:] %lf \n",get_mpi_rank(), total_time);
    printf("msgTime： %lf \n",graph.msg_time);
    printf("load graph time： %lf \n",load_graph_time);
    
    // train_corpus_cuda(argc,argv,vertex_degree,graph.out_queue);
    // dsgl(argc, argv,&graph.vertex_cn,&graph.new_sort,&graph);
    return 0;
}
