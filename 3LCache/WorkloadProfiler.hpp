#ifndef WORKLOAD_PROFILER_HPP
#define WORKLOAD_PROFILER_HPP

#include <vector>
#include <deque>
#include <unordered_map>
#include <cstdint>
#include <string>

// שני מצבי עבודה אונליין: TRAIN (אימון ובניית טבלאות) ו-TEST (שימוש חוזר)
enum class ProfilerMode {
    TRAIN,
    TEST
};

struct WorkloadFeatures {
    double write_ratio = 0.0;
    double avg_request_size = 0.0;
    double size_variance = 0.0;
    double singleton_ratio = 0.0;
    double popularity_skewness = 0.0;
    double avg_frequency = 0.0;
    double object_diversity = 0.0;
    double avg_reuse_distance = 0.0;
    double sequentiality_ratio = 0.0;
    double out_cache_hit_rate = 0.0;
    double request_rate = 0.0;
    double burstiness_index = 0.0;
    double arrival_time_variance = 0.0;
    double working_set_byte_delta = 0.0;
    double scan_ratio = 0.0;
};

struct RequestLog {
    uint64_t id;
    uint32_t size;
    bool is_write;
    double timestamp;
    bool out_cache_hit;
    
    double reuse_dist;
    bool is_sequential;
    double dt;
    bool first_time;
};

struct PolicyRecord {
    WorkloadFeatures features;
    std::vector<double> weights;
    double best_miss_rate = 1.0;
};

class WorkloadProfiler {
private:
    ProfilerMode current_mode;
    size_t window_size;
    std::deque<RequestLog> window;
    
    std::unordered_map<uint64_t, size_t> id_counts;
    std::unordered_map<uint64_t, uint32_t> unique_id_sizes;
    std::unordered_map<uint64_t, uint64_t> last_seen_seq;
    
    WorkloadFeatures current_features;
    WorkloadFeatures phase_start_features;
    WorkloadFeatures prev_window_features;

    std::string policy_filename;
    std::string config_filename;
    int cooldown_counter;
    size_t flush_interval; // number of requests between periodic persistence to disk
    uint64_t request_counter; // explicit request counter used for heartbeat flushes

    // מנגנון סף דינמי בזמן אמת (Online Adaptive Thresholding)
    double shift_threshold; 
    double matrix_epsilon;  
    double ema_delta_mean; // ממוצע זז של שינוי הפיצ'רים
    double ema_delta_var;  // שונות זזה של שינוי הפיצ'רים
    const double alpha = 0.05; // מקדם למידה ל-EMA

    // משתני ניטור פאזה
    uint64_t phase_requests;
    uint64_t phase_misses;
    WorkloadFeatures last_stable_features;
    std::vector<double> last_stable_weights;

    // מונים גלובליים
    uint64_t local_seq;
    uint64_t last_id;
    double last_timestamp;

    size_t write_count;
    double size_sum;
    double size_sq_sum;
    size_t singleton_count;
    double freq_sq_sum;
    double total_reuse_dist;
    size_t reuse_count;
    size_t sequential_count;
    size_t out_cache_hits;
    double dt_sum;
    double dt_sq_sum;
    size_t dt_count;
    double working_set_bytes;
    size_t first_time_seen_count;

    std::vector<PolicyRecord> policy_matrix;

    void recalculate_features();
    double calculate_aggregate_distance(const WorkloadFeatures& f1, const WorkloadFeatures& f2);
    void update_online_threshold(double current_delta);
    void save_config();
    void load_config();
    void save_policy_matrix();
    void load_policy_matrix();
    std::vector<double> find_closest_policy(const WorkloadFeatures& current);

public:
    WorkloadProfiler();
    ~WorkloadProfiler();
    
    void init(const std::string& mode_str, 
              const std::string& policy_file = "meta_policy.txt", 
              const std::string& config_file = "profiler_config.txt", 
              size_t cache_size = 10000);
    
    bool add_request(uint64_t id, uint32_t size, bool is_write, double timestamp, 
                     bool out_cache_hit, bool is_miss, 
                     const std::vector<double>& current_weights,
                     std::vector<double>& out_new_weights);

    void flush_to_disk();

    WorkloadFeatures get_features() const { return current_features; }
};

#endif // WORKLOAD_PROFILER_HPP