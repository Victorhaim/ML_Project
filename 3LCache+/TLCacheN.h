#ifndef WEBCACHESIM_TLCacheN_H
#define WEBCACHESIM_TLCacheN_H

#include "cache.h"
#include <unordered_map>
#include <unordered_set>
#include "../libCacheSim/dataStructure/sparsepp/spp.h"
#include <vector>
#include <random>
#include <cmath>
#include <LightGBM/c_api.h>
#include <assert.h>
#include <sstream>
#include <fstream>
#include <list>
#include <deque>
using namespace webcachesim;
using namespace std;

using spp::sparse_hash_map;

#define GROUP_K 16

namespace TLCacheN {
    static const uint8_t max_n_past_timestamps = 4;
    // Number of  inter-arrival times.
    static const uint8_t max_n_past_distances = 3;
    // Number of training samples.
    static const uint32_t batch_size = 131072 / 2;

// Metadata of object information
struct MetaExtra {
    uint32_t _past_distances[3];
    //the next index to put the distance
    uint8_t _past_distance_idx = 1;

    MetaExtra(const uint32_t &distance) {
        _past_distances[0] = distance;
    }

    void update(const uint32_t &distance) {
        uint8_t distance_idx = _past_distance_idx % max_n_past_distances;
        _past_distances[distance_idx] = distance;
        _past_distance_idx = _past_distance_idx + (uint8_t) 1;
        if (_past_distance_idx >= max_n_past_distances * 2)
            _past_distance_idx -= max_n_past_distances;
    }
};

class Meta {
public:
    uint64_t _key;
    uint32_t _size;
    uint64_t _past_timestamp;
    uint16_t _freq;
    MetaExtra *_extra = nullptr;
    uint8_t status;

    Meta(const uint64_t &key, const uint64_t &size, const uint64_t &past_timestamp,
            const vector<uint16_t> &extra_features) {
        _key = key;
        _size = size;
        _past_timestamp = past_timestamp;
        _freq = 1;
        status = 1;
    }

    virtual ~Meta() = default;

    void free() {
        delete _extra;
    }

    void update(const uint64_t &past_timestamp) {
        //distance
        if (max_n_past_distances > 0) {
            uint32_t _distance = past_timestamp - _past_timestamp;
            assert(_distance);
            if (!_extra) {
                _extra = new MetaExtra(_distance);
            } else
                _extra->update(_distance);
        }
        _past_timestamp = past_timestamp;
        if (_freq < 65535)
            _freq++;
    }
};

class TrainingData {
public:
    vector<float> labels;
    vector<int32_t> indptr;
    vector<int32_t> indices;
    vector<double> data;
    TrainingData(uint32_t n_feature) {
        labels.reserve(batch_size);
        indptr.reserve(batch_size + 1);
        indptr.emplace_back(0);
        indices.reserve(batch_size * n_feature);
        data.reserve(batch_size * n_feature);
    }

    void emplace_back(Meta &meta, uint32_t &age, uint64_t &future_interval) {
        int32_t counter = indptr.back();

        indices.emplace_back(0);
        // 等待时间
        data.emplace_back(age);
        ++counter;
        int j = 0;
        uint16_t n_within = meta._freq;
        if (meta._extra) {
            for (; j < meta._extra->_past_distance_idx && j < max_n_past_distances; ++j) {
                uint8_t past_distance_idx = (meta._extra->_past_distance_idx - 1 - j) % max_n_past_distances;
                const uint32_t &past_distance = meta._extra->_past_distances[past_distance_idx];
                indices.emplace_back(j + 1);
                data.emplace_back(past_distance);
            }
        }

        counter += j;

        indices.emplace_back(max_n_past_timestamps);
        data.push_back(meta._size);
        ++counter;
        indices.push_back(max_n_past_timestamps + 1);
        data.push_back(n_within);
        ++counter;
        labels.push_back(log1p(future_interval));
        indptr.push_back(counter);

    }

    void clear() {
        labels.clear();
        indptr.resize(1);
        indices.clear();
        data.clear();
    }
};

struct KeyMapEntryT {
    uint8_t list_idx;
    uint32_t list_pos;
};

class TLCacheNCache : public Cache {
public:
    uint64_t current_seq = -1;
    uint32_t n_feature;

       // 记录预测结果
    vector<vector<pair<double, uint64_t>>> prediction_results;
    // 记录预测时间
    vector<uint64_t> sample_times;
    uint16_t prediction_idx = 0;
    sparse_hash_map<uint64_t, double> prediction_map;
    uint8_t Q = 10;
    // 新对象
    deque<uint64_t> Qkeys;
    // 新对象占用地缓存空间
    uint64_t QSize = 0;
    uint8_t Qc = 1;
    // 驱逐对象的数量
    int Ecounts = 0;
    uint16_t sample_rate = 2;
    // 采样指针
    uint32_t samplepointer = -1;
    // 采样频率
    double f = 0;
    uint64_t eviction_freq[3] = {0, 0, 0};
    // 窗口大小
    float hsw = 1;
    uint64_t MAX_EVICTION_BOUNDARY[2] = {0, 0};
    uint32_t window_hit[3] = {0, 0, 0};
    // 两个数分别代表采样边界和时间区间大小
    uint64_t TIME_BOUNDARY = 0;
    uint64_t eviction_counts[3] = {0, 0, 0};
    uint64_t hit_distribution[4] = {0, 0, 0, 0};
    bool is_sampling = false;
    uint64_t obj_hit = 0;
    uint64_t obj_miss = 0;
    float eviction_rate = 0.5;
    uint32_t batch_size = 131072 / 4;

    sparse_hash_map<uint64_t, KeyMapEntryT> key_map;
    vector<Meta> in_cache_metas;
    deque<Meta> out_cache_metas;
    int out_sidx = 0;
    TrainingData *training_data;

    double training_loss = 0;
    int32_t n_force_eviction = 0;

    double training_time = 0;
    double inference_time = 0;

    // Determine whether the model has been trained.
    BoosterHandle booster = nullptr;

    // Model training parameters
    unordered_map<string, string> training_params = {
            {"boosting",         "gbdt"},
            {"objective",        "regression"},
            {"num_iterations",   "16"},
            {"num_leaves",       "32"},
            {"num_threads",      "1"},
            {"feature_fraction", "0.8"},
            {"bagging_freq",     "5"},
            {"bagging_fraction", "0.8"},
            {"learning_rate",    "0.1"},
            {"verbosity",        "-1"},
    };

    unordered_map<string, string> inference_params;

    enum ObjectiveT : uint8_t {
        byte_miss_ratio = 0, object_miss_ratio = 1
    };
    ObjectiveT objective = byte_miss_ratio;

    // random seed
    default_random_engine _generator = default_random_engine();
    uniform_int_distribution<std::size_t> _distribution = uniform_int_distribution<std::size_t>();

    uint64_t byte_million_req;
    void init_with_params(const map<string, string> &params) override {
        //set params
        for (auto &it: params) {
            if (it.first == "num_iterations") {
                training_params["num_iterations"] = it.second;
            } else if (it.first == "learning_rate") {
                training_params["learning_rate"] = it.second;
            } else if (it.first == "num_threads") {
                training_params["num_threads"] = it.second;
            } else if (it.first == "num_leaves") {
                training_params["num_leaves"] = it.second;
            } else if (it.first == "byte_million_req") {
                byte_million_req = stoull(it.second);
            } else if(it.first == "sample_rate") {
                sample_rate = stoull(it.second);
            } else if (it.first == "objective") {
                if (it.second == "byte-miss-ratio")
                    objective = byte_miss_ratio;
                else if (it.second == "object-miss-ratio")
                    objective = object_miss_ratio;
                else {
                    cerr << "error: unknown objective" << endl;
                    exit(-1);
                }
            } else {
                cerr << "3LCache unrecognized parameter: " << it.first << endl;
            }
        }
        n_feature = max_n_past_timestamps + 2;
        inference_params = training_params;
        training_data = new TrainingData(n_feature);
        prediction_results.resize(GROUP_K);
        sample_times.resize(GROUP_K);
    }

    bool lookup(const SimpleRequest &req) override;

    void admit(const SimpleRequest &req) override;

    void evict();

    void forget();

    uint32_t rank();

    void evict_with_candidate(pair<uint64_t, uint32_t> &epair);

    void quick_demotion(vector<uint32_t> &sampled_objects);

    void train();

    void prediction(vector<uint32_t> sampled_objects, uint32_t old_objs);

    void get_sample(Meta &meta, uint64_t cur_seq, bool isHit);

    void update_stat_periodic() override;

    pair<uint64_t, uint32_t> evict_with_distance();

    void remove_from_outcache_metas(Meta &meta, unsigned int &pos, const uint64_t &key);

    vector<int> get_object_distribution_n_past_timestamps() {
        vector<int> distribution(max_n_past_timestamps, 0);
        return distribution;
    }

};

}
#endif

