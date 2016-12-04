#include "program.h"

#include "cluster.hpp"

#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <cfloat>
#include <boost/filesystem.hpp>
#include <boost/random.hpp>


#ifdef __OPENMP__
#include <omp.h>
#endif

//#define SCORES
//#define VOTES

using namespace std;
using namespace cvtools;

Matrix samples;
Labels labels;
ZScore zscore;
PLSClassifier model;

#ifdef SCORES
ofstream fSCORES;
int iSCORES;
#endif

#ifdef VOTES
ofstream fVOTES;
#endif

/******************************************************************************************************
 * Input/Output
 *****************************************************************************************************/
void Save(const Set &set, const vector<Set> &extra,
        const string &outfile) {

    // Timer<chrono::seconds> timer;

    samples.release();
    labels.clear();

    /*
     *  Harvest features
     */
    // timer.tick();
    cout << "* Harvesting set [subjects: " << set.classes.size()
            << "]" << endl;
#pragma omp parallel for
    for (int s = 0; s < set.classes.size(); ++s) {
        for (int f = 0; f < set.classes[s].samples.size(); ++f) {

            Image image = read(set.classes[s].samples[f]);
            Matrix desc = descript(image);

#pragma omp critical
            {
                samples.rbind(desc);
                labels.push_back(set.classes[s].id);
            }
        }
    }
    // cout << "done in " << timer.tock() << " seconds" << endl;

    /*
     *  Extra features
     */
    Labels extra_labels;
    vector<string> extra_files;
    vector<int> indexes;
    for (const Set &e : extra) {

        cout << "* Counting extra samples [subjects: " << e.classes.size()
                << "]" << endl;
#pragma omp parallel for
        for (int s = 0; s < e.classes.size(); ++s) {
            for (int f = 0; f < e.classes[s].samples.size(); ++f) {

#pragma omp critical
                {
                    extra_files.push_back(e.classes[s].samples[f]);
                    extra_labels.push_back({e.classes[s].id, true});
                    indexes.push_back(indexes.size());
                }
            }
        }
        // cout << "done in " << timer.tock() << " seconds" << endl;
    }
    int extra_size = min(args["extra-size"].as<int>(), (int) indexes.size());
    cout << "* Harvesting " << extra_size << " extra samples from "
            << extra_labels.size() << " total" << endl;
    random_shuffle(indexes.begin(), indexes.end());
#pragma omp parallel for
    for (int i = 0; i < extra_size; ++i) {

        Image image = read(extra_files[i]);
        Matrix desc = descript(image);

#pragma omp critical
        {
            samples.rbind(desc);
            labels.push_back(extra_labels[i]);
        }
    }
    extra_files.clear();
    extra_labels.clear();
    indexes.clear();
    // cout << "done in " << timer.tock() << " seconds" << endl;

    /*
     *  PLSOAA
     */
    cout << "* Learning PLS OAA [samples: " << samples.rows
            << ", dim: " << samples.cols
            << "]" << endl;
    zscore.train_norm(samples);
    model = PLSClassifier(args["factors"].as<int>());
    model.train(samples, labels);

    /*
     * Save
     */
    string filename = args["features"].as<string>() + "-" + outfile + ".model";
    while (filename.find('/') != string::npos)
        filename.replace(filename.find('/'), 1, "-");
    ofstream output(filename);
    assert(output.is_open());

    samples.save(output);
    labels.save(output);
    zscore.save(output);
    model.save(output);

    /*
     * Display signature
     */
    cout << "* Signature" << endl;
    cout << samples(0, 0) << endl;
    cout << labels.back().id << endl;
    cout << model.models.back().regression(0, 0) << endl;
}

void Load(const string &inputfile) {

    string filename = args["features"].as<string>() + "-" + inputfile + ".model";
    while (filename.find('/') != string::npos)
        filename.replace(filename.find('/'), 1, "-");
    ifstream input(filename);
    assert(input.is_open());

    samples.load(input);
    labels.load(input);
    zscore.load(input);
    model.load(input);

    /*
     * Display signature
     */
    cout << "* Signature" << endl;
    cout << samples(0, 0) << endl;
    cout << labels.back().id << endl;
    cout << model.models.back().regression(0, 0) << endl;
}

/******************************************************************************************************
 * PLS Hash
 *****************************************************************************************************/

struct HashM {
    unique_ptr<Regression> hash_f;
    vector<int> subjects;
    vector<int> neg_subjects;
};
vector<vector<vector<HashM>>> hash_m;

void Learn() {

    // Timer<chrono::seconds> timer;

    vector<int> ndims = readvec(args["ndims"].as<string>());
    vector<int> nfactors = readvec(args["mfactors"].as<string>());
    assert(ndims.size() == nfactors.size());

    hash_m.resize(ndims.size());

    VSPLSRegression::TYPE type;
    if (args["type"].as<string>() != "full") {
        if (args["type"].as<string>() == "weights")
            type = VSPLSRegression::WEIGHTS;
        else if (args["type"].as<string>() == "beta")
            type = VSPLSRegression::BETA;
        else if (args["type"].as<string>() == "vip")
            type = VSPLSRegression::VIP;
        else if (args["type"].as<string>() == "random")
            type = VSPLSRegression::RANDOM;
        else
            assert(false);
    }

    unordered_set<string> u_extra;
    for (const Label &label : labels)
        if (label.extra)
            u_extra.insert(label.id);

    for (int e = 0; e < ndims.size(); ++e) {

        vector<int> nmodels = readvec(args["models"].as<string>());

        hash_m[e].resize(nmodels.back());

        cout << "* Learning Hash [samples: " << samples.rows
                << ", dim: " << samples.cols
                << ", models: " << hash_m[e].size()
                << ", ndims: " << ndims[e]
                << ", factors: " << nfactors[e]
                << "]" << endl;

        int count = 0;
#pragma omp parallel for
        for (int m = 0; m < hash_m[e].size(); ++m) {

#pragma omp critical
            {
                cout << "Hash " << ++count << "/" << hash_m[e].size()
                        << endl;
            }

            // resize to bits
            int base = args["base"].as<int>() == 2 ? 1 : args["base"].as<int>();
            hash_m[e][m].resize(base);

            // cluster division
            Groups group;
            group.resize(samples.rows);

            for (int u = 0; u < model.ulab.size(); ++u) {

                Group g = rand() % args["base"].as<int>();
                for (int row = 0; row < samples.rows; ++row)
                    if (!labels[row].extra)
                        if (labels[row].id == model.ulab[u].id)
                            group[row] = g;

                if (base == 1) {
                    if (g == 0)
                        hash_m[e][m][0].subjects.push_back(u);
                    else
                        hash_m[e][m][0].neg_subjects.push_back(u);
                } else {
                    hash_m[e][m][g].subjects.push_back(u);
                }
            }
            // deal with extra samples
            for (const Label &u : u_extra) {

                Group g = rand() % args["base"].as<int>();
                for (int row = 0; row < samples.rows; ++row)
                    if (labels[row].extra)
                        if (labels[row].id == u.id)
                            group[row] = g;
            }

            for (int b = 0; b < base; ++b) {

                Matrix responses(samples.rows, 1);
                responses = -1.0f;

                for (int row = 0; row < responses.rows; ++row)
                    if (group[row] == b)
                        responses(row) = 1.0f;

                Matrix s = samples.clone();
                if (args["type"].as<string>() == "full")
                    hash_m[e][m][b].hash_f = unique_ptr<Regression>(
                        new PLSRegression(nfactors[e]));
                else
                    hash_m[e][m][b].hash_f = unique_ptr<Regression>(
                        new VSPLSRegression(nfactors[e], ndims[e],
                        type, args["retrain"].as<bool>()));
                hash_m[e][m][b].hash_f->train(s, responses);
                
                if (type == VSPLSRegression::RANDOM) {

                    hash_m[e][m][b].subjects.clear();
                    Matrix r;

                    for (int u = 0; u < model.ulab.size(); ++u) {

                        for (int row = 0; row < samples.rows; ++row)
                            if (!labels[row].extra)
                                if (labels[row].id == model.ulab[u].id) {

                                    Matrix s = samples.row(row);
                                    hash_m[e][m][b].hash_f->predict(s, r);
                                    if (r() > 0)
                                        hash_m[e][m][b].subjects.push_back(u);
                                }
                    }
                }
            }
        }
        // cout << "done in " << timer.tock() << " seconds" << endl;
    }
    samples.release();
    //labels.clear();
}

typedef pair<int, float> Pif;

void Candidates_Sum(Matrix &desc, const int nmodels, const int e,
        vector<Pif> &list, Matrix &r, float &hash_sec) {

    // Timer<chrono::microseconds> timer;

    size_t hash_time = 0;
    for (int m = 0; m < nmodels; ++m) {

        // timer.tick();

        for (int b = 0; b < hash_m[e][m].size(); ++b) {

            hash_m[e][m][b].hash_f->predict(desc, r);

            float x = r();
            for (const int s : hash_m[e][m][b].subjects)
                list[s].second += x;

#ifdef SCORES
            string c = "negative";
            for (const int s : hash_m[e][m][b].subjects)
                if (s == iSCORES) {
                    c = "positive";
                    break;
                }
            fSCORES << x << '\t' << c << '\n';
#endif
        }

        // hash_time += timer.tock();
    }
#ifndef SCORES
#pragma omp critical
#endif
    {
        hash_sec += hash_time / (float) nmodels;
    }

    sort(list.begin(), list.end(),
            [](const Pif &a, const Pif & b) {
                return a.second > b.second;
            });
}

void Candidates_Prod(Matrix &desc, const int nmodels, const int e,
        vector<Pif> &list, Matrix &r, float &hash_sec) {

    // Timer<chrono::microseconds> timer;

    size_t hash_time = 0;
    for (int m = 0; m < nmodels; ++m) {

        // timer.tick();

        for (int b = 0; b < hash_m[e][m].size(); ++b) {

            hash_m[e][m][b].hash_f->predict(desc, r);

            float x = r();
            if (hash_m[e][m].size() > 1) {

                for (const int s : hash_m[e][m][b].subjects)
                    list[s].second += log(1 + x);

            } else {
                if (x > 0) {
                    for (const int s : hash_m[e][m][b].subjects)
                        list[s].second += log(1 + x);
                } else {
                    for (const int s : hash_m[e][m][b].neg_subjects)
                        list[s].second += log(1 - x);
                }
            }

#ifdef SCORES
            fSCORES << x << '\n';
#endif
        }

        // hash_time += timer.tock();
    }
#pragma omp critical
    {
        hash_sec += hash_time / (float) nmodels;
    }

    sort(list.begin(), list.end(),
            [](const Pif &a, const Pif & b) {
                return a.second > b.second;
            });
}

/******************************************************************************************************
 * Main
 *****************************************************************************************************/
int main(int argc, char** argv) {

    srand(time(NULL));
    // cv::initModule_features2d();

    // Timer<chrono::seconds> timer;

    options(argc, argv);

#ifdef __OPENMP__
    omp_set_num_threads(args["threads"].as<int>());
#endif

    /*
     * Parse sets
     */
    vector<Set> gallery = read_set(args["dataset"].as<string>(),
            args["learn"].as<string>());

    vector<Set> tests = read_set(args["dataset"].as<string>(),
            args["test"].as<string>());

    vector<Set> extra = read_set(args["extra-path"].as<string>(),
            args["extra-file"].as<string>());

    /*
     * Filter gallery
     */
    if (args["subjects"].as<int>() > 0)
        for (Set &set : gallery) {

            random_shuffle(set.classes.begin(), set.classes.end());
            set.classes.resize(args["subjects"].as<int>());
        }

    /*
     * Whether learn or test
     */
    if (args["exec"].as<string>() != "test") {
        for (Set &set : gallery)
            Save(set, extra, set.file);
        return EXIT_SUCCESS;
    }

    /*
     * If have one gallery for all tests
     */
    if (gallery.size() == 1) {
        Load(gallery[0].file);
        Learn();
    }

    /*
     * Test
     */
    ofstream output(args["output"].as<string>() + ".dat");
    output << "set\tpsubjects\tnmodels\tndims\tfactors\tcmc\tacc\ttime\tbfacc\tbftime\tspeedup" << endl;

#ifdef SCORES
    fSCORES.open(args["output"].as<string>() + ".scores");
#endif

#ifdef VOTES
    fVOTES.open(args["output"].as<string>() + ".votes");
#endif

    function<void(Matrix &desc, const int nmodels, const int e,
            vector<Pif> &list, Matrix &r, float &hash_sec) > Candidates;
    if (args["combination"].as<string>() == "sum") {
        Candidates = Candidates_Sum;
        cout << "* Using SUM to combine scores" << endl;
    } else if (args["combination"].as<string>() == "prod") {
        Candidates = Candidates_Prod;
        cout << "* Using PROD to combine scores" << endl;
    } else {
        assert(false);
    }

    for (int s = 0; s < tests.size(); ++s) {

        const Set &set = tests[s];

        cout << "* Testing [subjects: " << set.classes.size()
                << ", partition: " << set.file
                << "]" << endl;

        /*
         * if have one gallery per test
         */
        if (gallery.size() > 1) {
            Load(gallery[s].file);
            Learn();
        }

        vector<int> top = {
            //1,
            //(int) ceil(model.ulab.size()*0.001),
            //(int) ceil(model.ulab.size()*0.005),
            (int) ceil(model.ulab.size()*0.01),
            //(int) ceil(model.ulab.size()*0.03),
            //(int) ceil(model.ulab.size()*0.05),
            //(int) ceil(model.ulab.size()*0.07),
            //(int) ceil(model.ulab.size()*0.10),
            //(int) ceil(model.ulab.size()*0.13),
            //(int) ceil(model.ulab.size()*0.15),
            //(int) ceil(model.ulab.size()*0.20),
            //(int) ceil(model.ulab.size()*0.25),
            //(int) ceil(model.ulab.size()*0.30),
            (int) ceil(model.ulab.size()*1.00)
        };
        vector<int> nmodels = readvec(args["models"].as<string>());
        vector<int> ndims = readvec(args["ndims"].as<string>());
        vector<int> nfactors = readvec(args["mfactors"].as<string>());

        const int gsize = model.ulab.size();

        size_t bf_acc;
        size_t bf_sec;
        float hash_sec;
        float pls_sec;

        vector<vector<vector<int>>> e_m_t_cmc(ndims.size());
        vector<vector<vector<int>>> e_m_t_acc(ndims.size());
        vector<vector<vector < size_t >>> e_m_t_sec(ndims.size());

        int total;

        total = 0;
        bf_acc = 0;
        bf_sec = 0;
        hash_sec = 0;
        pls_sec = 0;

        vector<vector < vector<vector<float>>>> e_m_t_th_est(ndims.size());
        vector<vector<vector<float>>> e_m_t_th(ndims.size());

        for (int e = 0; e < ndims.size(); ++e) {

            e_m_t_cmc[e].resize(nmodels.size());
            e_m_t_acc[e].resize(nmodels.size());
            e_m_t_sec[e].resize(nmodels.size());

            e_m_t_th_est[e].resize(nmodels.size());
            e_m_t_th[e].resize(nmodels.size());

            for (int m = 0; m < nmodels.size(); ++m) {
                e_m_t_cmc[e][m].assign(top.size(), 0);
                e_m_t_acc[e][m].assign(top.size(), 0);
                e_m_t_sec[e][m].assign(top.size(), 0);

                e_m_t_th_est[e][m].resize(top.size());
                e_m_t_th[e][m].assign(top.size(), FLT_MAX);
            }
        }

#pragma omp parallel for
        for (int t = 0; t < set.classes.size(); ++t) {

            Matrix r;
            string trueID = set.classes[t].id;

            // check if we have this guy in the gallery
            bool cont = true;
            for (const Label &label : labels)
                if (!label.extra)
                    if (label.id == trueID) {
                        cont = false;
                        break;
                    }
            if (cont)
                continue;

            for (const string &filename : set.classes[t].samples) {

                // Timer<chrono::microseconds> perf;
                // Timer<chrono::microseconds> pls_timer;
#pragma omp critical
                {
                    total++;
                }

                Image image = read(filename);
                Matrix desc = descript(image);
                zscore.norm(desc);

                float max = -FLT_MAX;
                int maxIdx = -1;
                // perf.tick();
                size_t pls_time = 0;
                for (int i = 0; i < model.models.size(); ++i) {

                    // pls_timer.tick();
                    model.models[i].predict(desc, r);
                    // pls_time += pls_timer.tock();

                    model.zmean[i].unnorm(r);

                    if (r() > max) {
                        max = r();
                        maxIdx = i;
                    }
                }
                // int t = perf.tock();
#pragma omp critical
                {
                    bf_sec += t;
                    pls_sec += pls_time / (float) model.models.size();
                }

                if (model.ulab[maxIdx].id == trueID) {
#pragma omp critical
                    {
                        bf_acc++;
                    }
                }

                for (int e = 0; e < ndims.size(); ++e) {
                    for (int m = 0; m < nmodels.size(); ++m) {

                        vector<Pif> list(gsize);
                        for (int i = 0; i < list.size(); ++i) {
                            list[i].first = i;
                            list[i].second = 0;
                        }

#ifdef SCORES
                        size_t sec;
#pragma omp critical
                        {
                            for (int i = 0; i < model.ulab.size(); ++i)
                                if (model.ulab[i].id == trueID) {
                                    iSCORES = i;
                                    break;
                                }
                            // perf.tick();
                            Candidates(desc, nmodels[m], e, list, r, hash_sec);
                            // sec = perf.tock();
                        }
#else
                        // perf.tick();
                        Candidates(desc, nmodels[m], e, list, r, hash_sec);
                        // size_t sec = perf.tock();
#endif

#ifdef VOTES
#pragma omp critical
                        {
                            int index;
                            for (int i = 0; i < model.ulab.size(); ++i)
                                if (model.ulab[i].id == trueID) {
                                    index = i;
                                    break;
                                }

                            fVOTES << index;
                            for (int i = 0; i < list.size(); ++i)
                                fVOTES << '\t' << list[i].second;
                            fVOTES << '\n';
                        }
#endif

                        for (int t = 0; t < top.size(); ++t) {

                            for (int i = 0; i < min((int) list.size(), top[t]); ++i) {

                                int subIdx = list[i].first;
                                if (model.ulab[subIdx].id == trueID) {
#pragma omp critical
                                    {
                                        e_m_t_cmc[e][m][t]++;
                                    }
                                    break;
                                }
                            }

                            float max = -FLT_MAX;
                            int maxIdx = -1;
                            // perf.tick();
                            for (int i = 0; i < min((int) list.size(), top[t]); ++i) {

                                int subIdx = list[i].first;
                                model.models[subIdx].predict(desc, r);
                                model.zmean[subIdx].unnorm(r);

                                if (r() > max) {
                                    max = r();
                                    maxIdx = subIdx;
                                }

                                if (max > e_m_t_th[e][m][t])
                                    break;
                            }
#pragma omp critical
                            {
                                // e_m_t_sec[e][m][t] += perf.tock() + sec;
                            }

#pragma omp critical
                            {
                                if (e_m_t_th_est[e][m][t].size() < 15) {
                                    e_m_t_th_est[e][m][t].push_back(max);
                                } else if (e_m_t_th[e][m][t] == FLT_MAX) {
                                    sort(e_m_t_th_est[e][m][t].begin(), e_m_t_th_est[e][m][t].end());
                                    e_m_t_th[e][m][t] = e_m_t_th_est[e][m][t][8];
                                    cout << "* threshold ["
                                            << "value: " << e_m_t_th[e][m][t]
                                            << ", models: " << nmodels[m]
                                            << ", ndims: " << ndims[e]
                                            << ", nfactors: " << nfactors[e]
                                            << ", top: " << floor(top[t] * 100 / (float) model.models.size())
                                            << "]" << endl;
                                }
                            }

                            if (model.ulab[maxIdx].id == trueID) {
#pragma omp critical
                                {
                                    e_m_t_acc[e][m][t]++;
                                }
                            }
                        } // look at max top r of the list
                    } // n models
                } // experiment e
            } // subject samples
        } // test set
        float best_speedup = 0;
        int best_k = 0;
        float best_t = 0;
        float best_lsh = 0;
        float sp_acc = 0;
        float best_ndims = 0;
        int best_factors = 0;
        for (int e = 0; e < e_m_t_cmc.size(); ++e) {

            for (int m = 0; m < e_m_t_cmc[e].size(); ++m) {

                for (int t = 0; t < e_m_t_cmc[e][m].size(); ++t) {

                    float speedup = (bf_sec / (float) e_m_t_sec[e][m][t]);

                    output << set.file << '\t' // set
                            // %subjects
                            << ((int) (1000 * top[t] / (float) model.models.size())) / 10.0 << '\t'
                            // #models
                            << nmodels[m] << '\t'
                            // #ndims
                            << ndims[e] << '\t'
                            // factors
                            << nfactors[e] << '\t'
                            // cmc
                            << e_m_t_cmc[e][m][t] / (float) total << '\t'
                            // PLS accuracy
                            << e_m_t_acc[e][m][t] / (float) total << '\t'
                            // PLS microseconds
                            << e_m_t_sec[e][m][t] / (float) total << '\t'
                            // brute force accuracy
                            << bf_acc / (float) total << '\t'
                            // brute force microseconds
                            << bf_sec / (float) total << '\t'
                            // speedup
                            << speedup << '\n';

                    if (e_m_t_acc[e][m][t] > 0.95 * bf_acc)
                        if (best_speedup < speedup) {
                            best_speedup = speedup;
                            best_t = top[t] / (float) model.models.size();
                            best_k = nmodels[m];
                            best_factors = nfactors[e];
                            sp_acc = e_m_t_acc[e][m][t] / (float) total;
                            best_lsh = e_m_t_sec[e][m][t] / (float) total;
                            best_ndims = ndims[e];
                        }
                }
            }
        }

        output.flush();
        cout << "* Results [acc: " << bf_acc / (float) total
                << ", avg_time: " << bf_sec / (float) total
                << ", lsh_time: " << best_lsh
                << ", speedup: " << best_speedup
                << ", hash_time: " << hash_sec / (float) total
                << ", pls_time: " << pls_sec / (float) total
                << ", #models: " << best_k
                << ", #dims: " << best_ndims
                << ", factors: " << best_factors
                << ", max_search: " << best_t
                << ", sp_acc: " << sp_acc
                << "]" << endl;
        cout << "* Average time to compute features" << endl;
        for (const pair<string, size_t> &p : descriptor_time)
            cout << p.first << ":\t" << p.second / (float) total << endl;
        // cout << "done in " << timer.tock() << " seconds" << endl;
    }
    output.close();

    return EXIT_SUCCESS;
}
