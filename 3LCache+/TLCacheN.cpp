#include "TLCacheN.h"
#include <algorithm>
#include "utils.h"
#include <chrono>
#include <random>
#include <cmath>

using namespace chrono;
using namespace std;
using namespace TLCacheN;

// ========================================================================
// CDT (Continuous Dynamic Tuning) Helper Implementations
// ========================================================================
static std::mt19937 cdt_gen(1337);
static std::normal_distribution<double> standard_normal(0.0, 1.0);

// Generate clipped normal sample to prevent underestimation of good arms
double TLCacheNCache::get_clipped_normal_sample() {
    double z = standard_normal(cdt_gen);
    double lower_bound = 1.0 / std::sqrt(2.0 * M_PI);
    return std::max(lower_bound, z);
}

// Update the reward for the currently active hyperparameter configuration
void TLCacheNCache::update_cdt_reward(double reward) {
    if (cdt_current_arm_idx == -1) return;
    
    CDTArm& current_arm = cdt_arms[cdt_current_arm_idx];
    current_arm.n_pulls += 1;
    
    // Online mean update 
    current_arm.mean_reward = (current_arm.mean_reward * (current_arm.n_pulls - 1) + reward) / current_arm.n_pulls;
    cdt_step_counter++;
}

// Zooming TS Algorithm with Restarts 
void TLCacheNCache::select_next_cdt_arm() {
    // Restart logic for switching environments 
    if (cdt_step_counter >= CDT_EPOCH_LENGTH || cdt_arms.empty()) {
        cdt_arms.clear();
        cdt_step_counter = 0;
        
        // Initialize the space with a random starting point (Activation)
        std::uniform_real_distribution<double> dist_f(1.0, 50.0);
        std::uniform_real_distribution<double> dist_q(1.0, 40.0);
        
        cdt_arms.push_back({dist_f(cdt_gen), dist_q(cdt_gen), 0.0, 1, true});
        cdt_current_arm_idx = 0;
        
        f = cdt_arms[0].f_val;
        Q = cdt_arms[0].Q_val;
        return;
    }

    double max_I = -1e9;
    int best_arm_idx = -1;
    
    // s_0 parameter: sqrt(52 * pi * tau_0^2 * ln(T))
    double s_0 = std::sqrt(52.0 * M_PI * (CDT_TAU * CDT_TAU) * std::log(CDT_EPOCH_LENGTH));
    
    // Removal and Selection logic
    for (size_t i = 0; i < cdt_arms.size(); ++i) {
        if (!cdt_arms[i].active) continue;
        
        // Confidence radius calculation r_t(v)
        double r_t = std::sqrt((13.0 * (CDT_TAU * CDT_TAU) * std::log(CDT_EPOCH_LENGTH)) / (2.0 * cdt_arms[i].n_pulls));
        
        // TS standard deviation s_t(v)
        double s_t = s_0 * std::sqrt(1.0 / cdt_arms[i].n_pulls);
        
        // Perturbed estimate I_t(v) 
        double z_val = get_clipped_normal_sample();
        double I_t = cdt_arms[i].mean_reward + s_t * z_val;
        
        if (I_t > max_I) {
            max_I = I_t;
            best_arm_idx = i;
        }
    }
    
    // Activation logic: randomly explore new regions to cover the space
    std::uniform_real_distribution<double> rand_prob(0.0, 1.0);
    if (rand_prob(cdt_gen) < 0.1) { // 10% chance to activate a new arm
        std::uniform_real_distribution<double> dist_f(1.0, 50.0);
        std::uniform_real_distribution<double> dist_q(1.0, 40.0);
        cdt_arms.push_back({dist_f(cdt_gen), dist_q(cdt_gen), 0.0, 1, true});
        best_arm_idx = cdt_arms.size() - 1;
    }
    
    cdt_current_arm_idx = best_arm_idx;
    f = cdt_arms[best_arm_idx].f_val;
    Q = cdt_arms[best_arm_idx].Q_val;
}
// ========================================================================


// model training
void TLCacheNCache::train() {
    MAX_EVICTION_BOUNDARY[0] = MAX_EVICTION_BOUNDARY[1];
    auto timeBegin = chrono::system_clock::now();
    if (booster) LGBM_BoosterFree(booster);
    // create training dataset
    DatasetHandle trainData;
    LGBM_DatasetCreateFromCSR(
            static_cast<void *>(training_data->indptr.data()),
            C_API_DTYPE_INT32,
            training_data->indices.data(),
            static_cast<void *>(training_data->data.data()),
            C_API_DTYPE_FLOAT64,
            training_data->indptr.size(),
            training_data->data.size(),
            n_feature,  //remove future t
            training_params,
            nullptr,
            &trainData);
    LGBM_DatasetSetField(trainData,
                         "label",
                         static_cast<void *>(training_data->labels.data()),
                         training_data->labels.size(),
                         C_API_DTYPE_FLOAT32);
    // init booster
    LGBM_BoosterCreate(trainData, training_params, &booster);
    // train
    for (int i = 0; i < stoi(training_params["num_iterations"]); i++) {
        int isFinished;
        LGBM_BoosterUpdateOneIter(booster, &isFinished);
        if (isFinished) {
            break;
        }
    }
    LGBM_DatasetFree(trainData);
    training_time += chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - timeBegin).count();  
    if (window_hit[0] * 1.0 / window_hit[1] / hsw > 0.01) {
        if (hsw < (window_hit[2] - window_hit[1]) / window_hit[0] - 1)
            hsw = fmin(hsw + 1, 4);
    }
    window_hit[0] = 0, window_hit[1] = 0; window_hit[2] = 0;
}

// Acquire training samples
void TLCacheNCache::get_sample(Meta &meta, uint64_t cur_seq, bool isHit) {
    if (MAX_EVICTION_BOUNDARY[1] < (cur_seq - meta._past_timestamp))
        MAX_EVICTION_BOUNDARY[1] = (cur_seq - meta._past_timestamp);
    uint64_t future_distance = cur_seq - meta._past_timestamp;
    if (_distribution(_generator) % 4 != 0 && booster)
        return;
    uint32_t window_size = (in_cache_metas.size() + out_cache_metas.size());
    uint64_t waitting_time, future_interval;
    uint32_t age;
    if (future_distance > UINT32_MAX)
        future_distance = UINT32_MAX;

    if (future_distance > window_size * 4)
        window_size = future_distance / 2 + 1;
    else 
        window_size *= 2;

    for (uint64_t i = 0; i < future_distance; i += window_size) {
        age = _distribution(_generator) % window_size + i;
        if (age >= future_distance)
            break;
        if (isHit) 
            future_interval = future_distance - age;
        else
            future_interval = future_distance + MAX_EVICTION_BOUNDARY[0];
        training_data->emplace_back(meta, age, future_interval);    
        if (training_data->labels.size() >= batch_size) {
            train();
            training_data->clear();
        }
    }  
}

void TLCacheNCache::update_stat_periodic() {
}

bool TLCacheNCache::lookup(const SimpleRequest &req) {
    bool ret;
    ++current_seq;
    auto it = key_map.find(req.id);
    if (it != key_map.end()) {
        auto list_idx = it->second.list_idx;
        auto list_pos = it->second.list_pos;
        Meta &meta = list_idx == 0 ?  in_cache_metas[list_pos]: out_cache_metas[uint32_t(list_pos - out_sidx)];
        if (is_sampling) 
            get_sample(meta, current_seq, true);
        if (prediction_map.find(req.id) != prediction_map.end())
            prediction_map.erase(req.id);
            
        if (list_idx) {
            uint32_t pos = out_cache_metas.size() - uint32_t(list_pos - out_sidx);
            if (pos <= eviction_counts[2]) {
                if (meta.status == 0) {
                    hit_distribution[1] ++;
                    hit_distribution[3] += pos;
                } else {
                    hit_distribution[0] ++;
                    hit_distribution[2] += pos;
                }
            }
            window_hit[0]++;
        } else {
            window_hit[1]++;
        }
        meta.update(current_seq);
        ret = !list_idx;
    } else {
        ret = false;
    }
    window_hit[2]++;
    if (ret)
        obj_hit += 1;
    else 
        obj_miss += 1;

    forget();
    return ret;
}

// Delete object metadata that exceeds the window
void TLCacheNCache::forget() {
    int MAX_SIZE = in_cache_metas.size() * hsw + 2;
    
    if (out_cache_metas.size() > MAX_SIZE) {
        auto &meta = out_cache_metas[0];
        if (meta.status != 2) {
            get_sample(meta, current_seq, false);
            key_map.erase(meta._key);
            meta.free();
        }
        out_cache_metas.pop_front();
        out_sidx++;
    }
}

// Cache new objects
void TLCacheNCache::admit(const SimpleRequest &req) {
    const uint64_t &size = req.size;
    if (size > _cacheSize) {
        LOG("L", _cacheSize, req.id, size);
        return;
    }
    auto it = key_map.find(req.id);
    uint32_t pos = in_cache_metas.size();
    if (it == key_map.end()){
        in_cache_metas.emplace_back(Meta(req.id, req.size, current_seq, req.extra_features));
        key_map[req.id] = {0, pos};
    } else {
        Meta &meta = out_cache_metas[uint32_t(it->second.list_pos - out_sidx)];
        in_cache_metas.emplace_back(meta);
        meta.status = 2;
        in_cache_metas[pos]._size = size;
        key_map.find(req.id)->second = {0, pos};
    }  
    if (booster) {
        QSize += size;
        Qkeys.push_back(req.id);
        in_cache_metas[pos].status = 0;
    }
    _currentSize += size;
    while (_currentSize > _cacheSize) { 
        evict();  
    }
}

// sample eviction candidates
uint32_t TLCacheNCache::rank() {
    vector<uint32_t> sampled_objects;
    uint32_t lenQ = in_cache_metas.size(), idx_row = 0;
    
    // CDT TRIGGER: Dynamic continuous tuning instead of hardcoded heuristics
    if (eviction_freq[0] > 10240) {
        double current_reward = 0.0;
        if (eviction_freq[0] > 0) {
            // Reward is higher when byte miss ratio is lower
            current_reward = 1.0 - (eviction_freq[1] * 1.0 / eviction_freq[0]); 
        }
        
        update_cdt_reward(current_reward);
        select_next_cdt_arm(); 
        
        eviction_freq[0] = 0, eviction_freq[1] = 0, eviction_freq[2] = 0;
    }
    
    uint32_t old_objs = 0;
    while (idx_row < sample_rate) {
        if (samplepointer >= lenQ) {
            if (TIME_BOUNDARY > current_seq)
                break;
            sample_rate = 1024;
            if (sample_rate >= lenQ * 0.01 + 2) 
                sample_rate = lenQ > 2 ? lenQ * 0.01 + 2 : 2;
            samplepointer = 0;
            if (f == 0)
                f = current_seq - TIME_BOUNDARY;
            TIME_BOUNDARY = TIME_BOUNDARY + (current_seq - TIME_BOUNDARY) * 0.1 + 1;
        }
        Meta meta = in_cache_metas[samplepointer];
        if (objective == byte_miss_ratio) {
            if (meta.status && (meta._past_timestamp <= TIME_BOUNDARY || (current_seq - meta._past_timestamp) / meta._freq > f)) {
                sampled_objects.emplace_back(samplepointer);
                idx_row++;
            }
        } else {
            if (meta.status && (meta._past_timestamp <= TIME_BOUNDARY || (current_seq - meta._past_timestamp) * 1.0 / meta._freq * meta._size > f)) {
                sampled_objects.emplace_back(samplepointer);
                idx_row++;
            }
        }

        samplepointer++;
    }
    old_objs = sampled_objects.size();
    quick_demotion(sampled_objects);

    prediction(sampled_objects, old_objs);
    return sampled_objects.size();
}

// Sample new objects.
void TLCacheNCache::quick_demotion(vector<uint32_t> &sampled_objects) {
    if (eviction_counts[2] > 0.01 * Qc * in_cache_metas.size() + 1) { 
        Qc+=1;
        if (hit_distribution[0] + hit_distribution[1] > 1024 && hit_distribution[0] > 0 && hit_distribution[1] > 0) {
            // CDT already handles Q dynamically
            hit_distribution[0] = 0, hit_distribution[1] = 0, hit_distribution[2] = 0, hit_distribution[3] = 0;
            if (eviction_counts[2] > in_cache_metas.size())
                eviction_counts[0] = 0, eviction_counts[1] = 0, eviction_counts[2] = 0, Qc = 1;
        }
    }
    int idx_row = 0;
    uint64_t key;
    uint64_t q = _cacheSize * Q * 0.01;
    while (QSize > q && idx_row < sample_rate * 10 && idx_row < Qkeys.size()) {
        key = Qkeys[idx_row];
        sampled_objects.push_back(key_map.find(key)->second.list_pos);
        QSize -= in_cache_metas[sampled_objects.back()]._size;
        in_cache_metas[sampled_objects.back()].status = 1;
        idx_row++;
    }
    Qkeys.erase(Qkeys.begin(), Qkeys.begin() + idx_row);
}

// evict an object at a time
void TLCacheNCache::evict() {
    auto epair = evict_with_distance();
    evict_with_candidate(epair);
}

void TLCacheNCache::evict_with_candidate(pair<uint64_t, uint32_t> &epair) {
    is_sampling = true;
    Ecounts--;
    uint64_t key = epair.first;
    uint32_t old_pos = epair.second;
    _currentSize -= in_cache_metas[old_pos]._size;
    if (prediction_map.find(key) != prediction_map.end())
        prediction_map.erase(key);
    key_map.find(key)->second = {1, uint32_t(out_cache_metas.size()) + out_sidx};
    out_cache_metas.emplace_back(in_cache_metas[old_pos]);
    uint32_t in_cache_tail_idx = in_cache_metas.size() - 1;
    if (old_pos != in_cache_tail_idx) {
        key_map.find(in_cache_metas[in_cache_tail_idx]._key)->second.list_pos = old_pos;
        in_cache_metas[old_pos] = in_cache_metas.back();
    }
    in_cache_metas.pop_back();
}

pair<uint64_t, uint32_t> TLCacheNCache::evict_with_distance(){
    {
        if (!booster) {
            uint32_t pos = _distribution(_generator) % in_cache_metas.size();
            auto &meta = in_cache_metas[pos];
            return {meta._key, pos};
        } 
    }
    double reuse_distance = -1;
    uint8_t idx = 0;
    if (Ecounts <= 0)
        Ecounts = rank() * eviction_rate;
    while (true) {
        reuse_distance = -1;
        idx = 0;
        for (int i = 0; i < GROUP_K; i++) {
            if (!prediction_results[i].empty() && prediction_results[i].back().first >= reuse_distance) {
                reuse_distance = prediction_results[i].back().first;
                idx = i;
            }
        }
        if (reuse_distance == -1) {
            Ecounts = rank() * eviction_rate;
            continue;
        }
        
        uint64_t key = prediction_results[idx].back().second;
        prediction_results[idx].pop_back();
        auto it = key_map.find(key);
        if (it != key_map.end() && it->second.list_idx == 0 && prediction_map.find(key) != prediction_map.end() && prediction_map.find(key)->second == reuse_distance) {
            auto &meta = in_cache_metas[it->second.list_pos];
            if (idx % 2 == 0) {
                if (objective == object_miss_ratio)
                    eviction_counts[0] += meta._size;
                else
                    eviction_counts[0]++;
                meta.status = 1;
            } else {
                if (objective == object_miss_ratio)
                    eviction_counts[1] += meta._size;
                else
                    eviction_counts[1]++;
                meta.status = 0;
            }
            eviction_counts[2]++;
            double freq;
            if (objective == byte_miss_ratio)
                freq = (current_seq - meta._past_timestamp) / meta._freq;
            else 
                freq = (current_seq - meta._past_timestamp) * 1.0 / meta._freq * meta._size;
            if (freq > f) {
                eviction_freq[1]++;
                if (freq > f * 2)
                    eviction_freq[2]++;
            }
            eviction_freq[0]++;
            return {key, it->second.list_pos};
        }
    }
    return {-1, -1};
}

// Predict the eviction candidates.
void TLCacheNCache::prediction(vector<uint32_t> sampled_objects, uint32_t old_objs) {
    auto timeBegin = chrono::system_clock::now();
    uint32_t sample_nums = sampled_objects.size();
    int32_t indptr[sample_nums + 1];
    indptr[0] = 0;
    int32_t indices[sample_nums * n_feature];
    double data[sample_nums * n_feature];
    uint32_t sizes[sample_nums];
    uint64_t keys[sample_nums];
    unsigned int idx_feature = 0;
    uint32_t pos;
    unsigned int idx_row = 0;
    for (; idx_row < sample_nums; idx_row++) {
        pos = sampled_objects[idx_row];
        auto &meta = in_cache_metas[pos];
        keys[idx_row] = meta._key;
        indices[idx_feature] = 0;
        data[idx_feature++] = current_seq - meta._past_timestamp;
        uint8_t j = 0;
        uint16_t n_within = meta._freq;
        if (meta._extra) {
            for (j = 0; j < meta._extra->_past_distance_idx && j < max_n_past_distances; ++j) {
                uint8_t past_distance_idx = (meta._extra->_past_distance_idx - 1 - j) % max_n_past_distances;
                uint32_t &past_distance = meta._extra->_past_distances[past_distance_idx];
                indices[idx_feature] = j + 1;
                data[idx_feature++] = past_distance;
            }
        }

        indices[idx_feature] = max_n_past_timestamps;
        data[idx_feature++] = meta._size;
        sizes[idx_row] = meta._size;

        indices[idx_feature] = max_n_past_timestamps + 1;
        data[idx_feature++] = n_within;
        indptr[idx_row + 1] = idx_feature;
    }
    int64_t len;
    double scores[sample_nums];
    LGBM_BoosterPredictForCSR(booster,
                              static_cast<void *>(indptr),
                              C_API_DTYPE_INT32,
                              indices,
                              static_cast<void *>(data),
                              C_API_DTYPE_FLOAT64,
                              idx_row + 1,
                              idx_feature,
                              n_feature,  //remove future t
                              C_API_PREDICT_NORMAL,
                              0,
                              inference_params,
                              &len,
                              scores);
    
    double _distance;
    vector<pair<double, uint64_t>> prediction_result;
    vector<pair<double, uint64_t>> new_prediction_result;
    if (objective == byte_miss_ratio) {
        for (int i = 0; i < sample_nums; ++i) {
            _distance = exp(scores[i]) + current_seq;
            prediction_map[keys[i]] = _distance;
            if (i < old_objs)
                prediction_result.push_back({_distance, keys[i]});
            else
                new_prediction_result.push_back({_distance, keys[i]});
        }
    } else {
        for (int i = 0; i < sample_nums; ++i) {
            _distance = double(sizes[i] * exp(scores[i]));
            prediction_map[keys[i]] = _distance;
            if (i < old_objs)
                prediction_result.push_back({_distance, keys[i]});
            else
                new_prediction_result.push_back({_distance, keys[i]});
        }
    }
    
    sort(prediction_result.begin(), prediction_result.end(), [](const auto& a, const auto& b) {
        return a.first < b.first; 
    });
    sort(new_prediction_result.begin(), new_prediction_result.end(), [](const auto& a, const auto& b) {
        return a.first < b.first; 
    });

    prediction_results[prediction_idx] = prediction_result;
    sample_times[prediction_idx++] = current_seq;
    prediction_results[prediction_idx] = new_prediction_result;
    sample_times[prediction_idx++] = current_seq;
    prediction_idx = prediction_idx % GROUP_K;
    inference_time += chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - timeBegin).count();
}