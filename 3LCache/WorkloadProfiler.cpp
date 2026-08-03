#include "WorkloadProfiler.hpp"
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace {

bool ensure_parent_dir(const std::string& filepath) {
    if (filepath.empty()) return false;
    std::filesystem::path parent = std::filesystem::path(filepath).parent_path();
    if (parent.empty()) return true;
    std::error_code ec;
    if (std::filesystem::exists(parent, ec)) return true;
    return std::filesystem::create_directories(parent, ec);
}

} // namespace

WorkloadProfiler::WorkloadProfiler() : 
    current_mode(ProfilerMode::TRAIN), window_size(20000), cooldown_counter(0), flush_interval(0), // flush_interval = 0 (ביטול 50k)
    shift_threshold(0.15), matrix_epsilon(0.08),
    ema_delta_mean(0.02), ema_delta_var(0.0004),
    phase_requests(0), phase_misses(0), local_seq(0), last_id(0), last_timestamp(0.0),
    write_count(0), size_sum(0.0), size_sq_sum(0.0), singleton_count(0),
    freq_sq_sum(0.0), total_reuse_dist(0.0), reuse_count(0), sequential_count(0),
    out_cache_hits(0), dt_sum(0.0), dt_sq_sum(0.0), dt_count(0),
    working_set_bytes(0.0), first_time_seen_count(0), request_counter(0) {
    current_features = WorkloadFeatures();
    phase_start_features = WorkloadFeatures();
    prev_window_features = WorkloadFeatures();
}

WorkloadProfiler::~WorkloadProfiler() {
    if (current_mode == ProfilerMode::TRAIN) {
        save_policy_matrix();
        save_config();
    }
}

void WorkloadProfiler::init(const std::string& mode_str, 
                            const std::string& policy_file, 
                            const std::string& config_file, 
                            size_t cache_size) {
    policy_filename = policy_file;
    config_filename = config_file;

    if (!policy_filename.empty()) ensure_parent_dir(policy_filename);
    if (!config_filename.empty()) ensure_parent_dir(config_filename);
    
    size_t estimated_entries = cache_size;
    if (cache_size > 100000) { 
        estimated_entries = std::max((size_t)1, cache_size / 4096);
    }
    window_size = std::max(estimated_entries * 2, (size_t)10000);

    if (mode_str == "test") {
        current_mode = ProfilerMode::TEST;
        load_config();
        load_policy_matrix();
    } else { // train default
        current_mode = ProfilerMode::TRAIN;
        load_config();
        load_policy_matrix();
    }
}

void WorkloadProfiler::update_online_threshold(double current_delta) {
    double diff = current_delta - ema_delta_mean;
    ema_delta_mean += alpha * diff;
    ema_delta_var = (1.0 - alpha) * (ema_delta_var + alpha * diff * diff);
    
    double std_dev = std::sqrt(std::max(0.0, ema_delta_var));
    shift_threshold = ema_delta_mean + (3.0 * std_dev);
    matrix_epsilon = ema_delta_mean;
}

double WorkloadProfiler::calculate_aggregate_distance(const WorkloadFeatures& f1, const WorkloadFeatures& f2) {
    auto norm = [](double x, double scale) { return x / (x + scale); };

    double sum = 0.0;
    sum += std::abs(f1.write_ratio - f2.write_ratio);
    sum += std::abs(norm(f1.avg_request_size, 64000.0) - norm(f2.avg_request_size, 64000.0));
    sum += std::abs(norm(f1.size_variance, 1e8) - norm(f2.size_variance, 1e8));
    sum += std::abs(f1.singleton_ratio - f2.singleton_ratio);
    sum += std::abs(norm(f1.popularity_skewness, 10.0) - norm(f2.popularity_skewness, 10.0));
    sum += std::abs(norm(f1.avg_frequency, 10.0) - norm(f2.avg_frequency, 10.0));
    sum += std::abs(f1.object_diversity - f2.object_diversity);
    sum += std::abs(norm(f1.avg_reuse_distance, 10000.0) - norm(f2.avg_reuse_distance, 10000.0));
    sum += std::abs(f1.sequentiality_ratio - f2.sequentiality_ratio);
    sum += std::abs(f1.out_cache_hit_rate - f2.out_cache_hit_rate);
    sum += std::abs(norm(f1.request_rate, 10000.0) - norm(f2.request_rate, 10000.0));
    sum += std::abs(norm(f1.burstiness_index, 5.0) - norm(f2.burstiness_index, 5.0));
    sum += std::abs(norm(f1.arrival_time_variance, 100.0) - norm(f2.arrival_time_variance, 100.0));
    sum += std::abs(norm(f1.working_set_byte_delta, 1e7) - norm(f2.working_set_byte_delta, 1e7));
    sum += std::abs(f1.scan_ratio - f2.scan_ratio);
    return sum / 15.0;
}

void WorkloadProfiler::save_config() {
    if (config_filename.empty()) return;
    if (!ensure_parent_dir(config_filename)) return;
    std::string tmp = config_filename + ".tmp";
    std::ofstream file(tmp, std::ios::trunc);
    if (!file.is_open()) return;
    file << shift_threshold << "\n";
    file << matrix_epsilon << "\n";
    file.close();

    int ret = std::rename(tmp.c_str(), config_filename.c_str());
    if (ret != 0) {
        std::ifstream src(tmp, std::ios::binary);
        std::ofstream dst(config_filename, std::ios::binary | std::ios::trunc);
        if (src.is_open() && dst.is_open()) {
            dst << src.rdbuf();
            dst.close();
        }
        src.close();
        std::remove(tmp.c_str());
    }
}

void WorkloadProfiler::load_config() {
    std::ifstream file(config_filename);
    if (!file.is_open()) return;
    file >> shift_threshold;
    file >> matrix_epsilon;
    file.close();
}

void WorkloadProfiler::save_policy_matrix() {
    if (policy_filename.empty()) return;

    if (policy_matrix.empty() && current_mode == ProfilerMode::TRAIN) {
        std::vector<double> weights_to_save = last_stable_weights.empty() ? std::vector<double>{0.2, 0.2, 0.2, 0.2, 0.2} : last_stable_weights;
        WorkloadFeatures feats_to_save = (local_seq > 0) ? current_features : phase_start_features;
        double current_miss_rate = (phase_requests > 0) ? (static_cast<double>(phase_misses) / phase_requests) : 1.0;
        
        PolicyRecord baseline_rec = {feats_to_save, weights_to_save, current_miss_rate};
        policy_matrix.push_back(baseline_rec);
    }

    if (!ensure_parent_dir(policy_filename)) return;

    std::string tmp = policy_filename + ".tmp";
    std::ofstream file(tmp, std::ios::trunc);
    if (!file.is_open()) return;

    for (const auto& record : policy_matrix) {
        const auto& feat = record.features;
        file << feat.write_ratio << "," << feat.avg_request_size << "," << feat.size_variance << ","
             << feat.singleton_ratio << "," << feat.popularity_skewness << "," << feat.avg_frequency << ","
             << feat.object_diversity << "," << feat.avg_reuse_distance << "," << feat.sequentiality_ratio << ","
             << feat.out_cache_hit_rate << "," << feat.request_rate << "," << feat.burstiness_index << ","
             << feat.arrival_time_variance << "," << feat.working_set_byte_delta << "," << feat.scan_ratio
             << "," << record.best_miss_rate;
        
        for (double w : record.weights) {
            file << "," << w;
        }
        file << "\n";
    }
    file.close();

    int ret = std::rename(tmp.c_str(), policy_filename.c_str());
    if (ret != 0) {
        std::ifstream src(tmp, std::ios::binary);
        std::ofstream dst(policy_filename, std::ios::binary | std::ios::trunc);
        if (src.is_open() && dst.is_open()) {
            dst << src.rdbuf();
            dst.close();
        }
        src.close();
        std::remove(tmp.c_str());
    }
}

void WorkloadProfiler::load_policy_matrix() {
    policy_matrix.clear();
    std::ifstream file(policy_filename);
    if (!file.is_open()) return;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;
        PolicyRecord record;
        std::vector<double> all_nums;

        while (std::getline(ss, val, ',')) {
            all_nums.push_back(std::stod(val));
        }

        if (all_nums.size() < 16) continue;

        record.features.write_ratio = all_nums[0];
        record.features.avg_request_size = all_nums[1];
        record.features.size_variance = all_nums[2];
        record.features.singleton_ratio = all_nums[3];
        record.features.popularity_skewness = all_nums[4];
        record.features.avg_frequency = all_nums[5];
        record.features.object_diversity = all_nums[6];
        record.features.avg_reuse_distance = all_nums[7];
        record.features.sequentiality_ratio = all_nums[8];
        record.features.out_cache_hit_rate = all_nums[9];
        record.features.request_rate = all_nums[10];
        record.features.burstiness_index = all_nums[11];
        record.features.arrival_time_variance = all_nums[12];
        record.features.working_set_byte_delta = all_nums[13];
        record.features.scan_ratio = all_nums[14];
        record.best_miss_rate = all_nums[15];

        for (size_t i = 16; i < all_nums.size(); ++i) { // size_t
            record.weights.push_back(all_nums[i]);
        }
        policy_matrix.push_back(record);
    }
    file.close();
}

std::vector<double> WorkloadProfiler::find_closest_policy(const WorkloadFeatures& current) {
    if (policy_matrix.empty()) return std::vector<double>();
    size_t best_index = 0;
    double min_dist = 9999999.0;
    for (size_t i = 0; i < policy_matrix.size(); ++i) {
        double dist = calculate_aggregate_distance(current, policy_matrix[i].features);
        if (dist < min_dist) {
            min_dist = dist;
            best_index = i;
        }
    }
    return policy_matrix[best_index].weights;
}

bool WorkloadProfiler::add_request(uint64_t id, uint32_t size, bool is_write, double timestamp, 
                                   bool out_cache_hit, bool is_miss,
                                   const std::vector<double>& current_weights,
                                   std::vector<double>& out_new_weights) {
    
    double dt = (local_seq > 0) ? (timestamp - last_timestamp) : 0.0;
    bool is_seq = (local_seq > 0 && id == last_id + 1);
    double r_dist = 0.0;
    if (last_seen_seq.find(id) != last_seen_seq.end()) {
        r_dist = static_cast<double>(local_seq - last_seen_seq[id]);
    }
    bool first_time = (id_counts[id] == 0);

    if (is_write) write_count++;
    size_sum += size;
    size_sq_sum += static_cast<double>(size) * size;
    if (out_cache_hit) out_cache_hits++;
    if (is_seq) sequential_count++;
    if (r_dist > 0.0) { total_reuse_dist += r_dist; reuse_count++; }
    if (dt > 0.0) { dt_sum += dt; dt_sq_sum += dt * dt; dt_count++; }
    if (first_time) { 
        first_time_seen_count++; 
        unique_id_sizes[id] = size;
        working_set_bytes += size; 
    }

    size_t current_f = id_counts[id];
    if (current_f > 0) freq_sq_sum -= static_cast<double>(current_f) * current_f;
    if (current_f == 1) singleton_count--;
    
    size_t new_f = current_f + 1;
    freq_sq_sum += static_cast<double>(new_f) * new_f;
    if (new_f == 1) singleton_count++;

    id_counts[id]++;
    last_seen_seq[id] = local_seq;
    window.push_back({id, size, is_write, timestamp, out_cache_hit, r_dist, is_seq, dt, first_time});
    
    last_id = id;
    last_timestamp = timestamp;
    local_seq++;

    if (window.size() > window_size) {
        RequestLog old = window.front();
        window.pop_front();

        if (old.is_write) write_count--;
        size_sum -= old.size;
        size_sq_sum -= static_cast<double>(old.size) * old.size;
        if (old.out_cache_hit) out_cache_hits--;
        if (old.is_sequential) sequential_count--;
        if (old.reuse_dist > 0.0) { total_reuse_dist -= old.reuse_dist; reuse_count--; }
        if (old.dt > 0.0) { dt_sum -= old.dt; dt_sq_sum -= old.dt * old.dt; dt_count--; }
        if (old.first_time) first_time_seen_count--;

        size_t old_f = id_counts[old.id];
        freq_sq_sum -= static_cast<double>(old_f) * old_f;
        if (old_f == 1) singleton_count--;

        size_t old_new_f = old_f - 1;
        if (old_new_f > 0) freq_sq_sum += static_cast<double>(old_new_f) * old_new_f;
        if (old_new_f == 1) singleton_count++;

        id_counts[old.id]--;
        if (id_counts[old.id] == 0) {
            id_counts.erase(old.id);
            working_set_bytes -= unique_id_sizes[old.id];
            unique_id_sizes.erase(old.id);
        }
    }

    recalculate_features();

    if (local_seq % 1000 == 0 && local_seq > 1000) {
        double delta = calculate_aggregate_distance(current_features, prev_window_features);
        update_online_threshold(delta);
        prev_window_features = current_features;
    }

    request_counter++;
    if (flush_interval > 0 && request_counter > 0 && (request_counter % flush_interval == 0)) {
        save_policy_matrix();
        save_config();
    }

    double current_shift = calculate_aggregate_distance(current_features, phase_start_features);

    if (current_mode == ProfilerMode::TRAIN) {
        phase_requests++;
        if (is_miss) phase_misses++;

        if (phase_requests % 1000 == 0) {
            last_stable_features = current_features;
            last_stable_weights = current_weights;
        }

        size_t min_phase_len = window_size / 2;

        if (current_shift > shift_threshold) {
            if (phase_requests > min_phase_len && !last_stable_weights.empty()) {
                double current_miss_rate = static_cast<double>(phase_misses) / phase_requests;

                size_t best_index = 0;
                double min_dist = 9999999.0;
                for (size_t i = 0; i < policy_matrix.size(); ++i) {
                    double dist = calculate_aggregate_distance(last_stable_features, policy_matrix[i].features);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_index = i;
                    }
                }

                if (min_dist < matrix_epsilon && !policy_matrix.empty()) {
                    if (current_miss_rate < policy_matrix[best_index].best_miss_rate) {
                        for (size_t i = 0; i < last_stable_weights.size(); ++i) {
                            policy_matrix[best_index].weights[i] = 0.7 * policy_matrix[best_index].weights[i] + 0.3 * last_stable_weights[i];
                        }
                        policy_matrix[best_index].best_miss_rate = current_miss_rate;
                    }
                } else {
                    PolicyRecord new_rec = {last_stable_features, last_stable_weights, current_miss_rate};
                    policy_matrix.push_back(new_rec);
                }
                save_policy_matrix();
            }

            phase_start_features = current_features;
            phase_requests = 0;
            phase_misses = 0;
        }
        return false;
    }

    else { // ProfilerMode::TEST
        if (cooldown_counter > 0) {
            cooldown_counter--;
            return false;
        }

        if (current_shift > shift_threshold) {
            std::vector<double> matched_weights = find_closest_policy(current_features);
            if (!matched_weights.empty()) {
                out_new_weights.resize(current_weights.size());
                for (size_t i = 0; i < current_weights.size(); ++i) {
                    double target = (i < matched_weights.size()) ? matched_weights[i] : current_weights[i];
                    out_new_weights[i] = 0.5 * current_weights[i] + 0.5 * target;
                }
                phase_start_features = current_features;
                cooldown_counter = static_cast<int>(window_size / 4);
                return true;
            }
        }
        return false;
    }
}

void WorkloadProfiler::flush_to_disk() {
    save_policy_matrix();
    save_config();
}

void WorkloadProfiler::recalculate_features() {
    if (window.empty()) return;
    size_t total = window.size();
    size_t unique = id_counts.size();

    current_features.write_ratio = static_cast<double>(write_count) / total;
    current_features.avg_request_size = size_sum / total;
    
    double mean_sz = current_features.avg_request_size;
    double var_sz = (size_sq_sum / total) - (mean_sz * mean_sz);
    current_features.size_variance = (var_sz > 0.0) ? var_sz : 0.0;

    current_features.singleton_ratio = static_cast<double>(singleton_count) / total;
    current_features.avg_frequency = static_cast<double>(total) / unique;

    double mean_f = current_features.avg_frequency;
    double var_f = (freq_sq_sum / unique) - (mean_f * mean_f);
    current_features.popularity_skewness = (var_f > 0.0) ? std::sqrt(var_f) : 0.0;

    current_features.object_diversity = static_cast<double>(unique) / total;
    current_features.avg_reuse_distance = (reuse_count > 0) ? (total_reuse_dist / reuse_count) : 0.0;
    current_features.sequentiality_ratio = static_cast<double>(sequential_count) / total;
    current_features.out_cache_hit_rate = static_cast<double>(out_cache_hits) / total;

    double time_span = window.back().timestamp - window.front().timestamp;
    current_features.request_rate = (time_span > 0.0) ? (total / time_span) : 0.0;

    if (dt_count > 0) {
        double mean_dt = dt_sum / dt_count;
        double var_dt = (dt_sq_sum / dt_count) - (mean_dt * mean_dt);
        current_features.arrival_time_variance = (var_dt > 0.0) ? var_dt : 0.0;
        double std_dt = std::sqrt(current_features.arrival_time_variance);
        current_features.burstiness_index = (mean_dt > 0.0) ? (std_dt / mean_dt) : 0.0;
    } else {
        current_features.arrival_time_variance = 0.0;
        current_features.burstiness_index = 0.0;
    }

    current_features.working_set_byte_delta = working_set_bytes;
    current_features.scan_ratio = static_cast<double>(first_time_seen_count) / total;
}