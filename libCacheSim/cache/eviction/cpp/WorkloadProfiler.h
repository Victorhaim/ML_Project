#ifndef WORKLOAD_PROFILER_H
#define WORKLOAD_PROFILER_H

#include <string>
#include <vector>
#include <map>
#include <fstream>

class WorkloadProfiler {
public:
    WorkloadProfiler();
    ~WorkloadProfiler();

    void init(const std::string& trace_name,
              const std::string& cache_size_str,
              const std::string& algorithm_name,
              unsigned long total_requests);

    void add_request(unsigned long req_id,
                     unsigned int size,
                     bool hit,
                     double current_miss_ratio,
                     bool is_warmup,
                     bool is_test,
                     const std::vector<double>& features,
                     std::vector<double>& rewards);

private:
    // Add private members for your profiler here
    std::ofstream _output_file;
    bool _is_pretrain = false;
};

#endif // WORKLOAD_PROFILER_H