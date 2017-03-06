#include "cluster.hpp"

#include <algorithm>
#include <unordered_set>
#include <fstream>

using namespace std;

namespace cvtools {

    /*
     * Groups
     */
    void Groups::save(std::ofstream& output) const {

        size_t s = size();

        output.write((char*) &s, sizeof (s));

        for (const Group &a : (*this))
            output << a << '\n';
    }

    void Groups::load(std::ifstream& input) {

        size_t s;
        input.read((char*) &s, sizeof (s));

        resize(s);

        for (Group &a : (*this))
            input >> a;

        char dump;
        input.read(&dump, sizeof (dump));
    }

    /*
     * Cluster
     */
    Cluster::Cluster(const int nclusters) : nclusters(nclusters) {

    }

    void Cluster::train(Matrix &samples, Groups &groups) {

        Labels labels;
        train_impl(samples, labels, groups);
    }

    void Cluster::train(Matrix &samples, const Labels &labels, Groups &groups) {

        train_impl(samples, labels, groups);
    }

    void Cluster::predict(Matrix &samples, Groups &groups) const {
        
        predict_impl(samples, groups);
    }

    void Cluster::save(std::ofstream &output) const {

        output.write((char*) &nclusters, sizeof (nclusters));

        save_impl(output);
    }

    void Cluster::load(std::ifstream &input) {

        input.read((char*) &nclusters, sizeof (nclusters));

        load_impl(input);
    }

    /*
     * K-Means
     */
    L2 KMeans::l2;
    
    KMeans::KMeans(const int nclusters, const int maxit, Distance &dist)
    : dist(dist), maxit(maxit), Cluster(nclusters) {

    }

    void KMeans::train_impl(Matrix &samples, const Labels &labels, Groups &groups) {

        assert(samples.rows >= nclusters);

        Groups old;
        groups.resize(samples.rows);

        vector<int> size;
        Matrix new_centroids;
        new_centroids.create(nclusters, samples.cols);

        // initialize centroids
        centroids.create(nclusters, samples.cols);
        sample(samples, centroids);

        // associate
        predict(samples, groups);

        for (int it = 0; it < maxit; ++it) {

            new_centroids = 0;
            size.assign(nclusters, 0);

            // assigment
            old.assign(groups.begin(), groups.end());
            predict(samples, groups);

            // update centroid
            int diff = 0;
            int max = 0;
            int min = INT_MAX;
            for (int row = 0; row < groups.size(); ++row) {

                Group group = groups[row];
                size[group]++;
                new_centroids.row(group) += samples.row(row);

                if (old[row] != group)
                    diff++;
            }
            for (int c = 0; c < nclusters; ++c) {

                new_centroids.row(c) /= size[c];

                if (size[c] > max)
                    max = size[c];
                if (size[c] < min)
                    min = size[c];
            }
            centroids = new_centroids.clone();

#pragma omp critical
            {
                cout << "it: " << it + 1 << '/' << maxit << " ["
                        << "diff: " << diff / (float) samples.rows
                        << ", min: " << min / (float) samples.rows
                        << ", max: " << max / (float) samples.rows
                        << "]" << endl;
            }
        }
    }

    void KMeans::predict_impl(Matrix &samples, Groups &groups) const {

        groups.resize(samples.rows);

#pragma omp parallel for schedule(dynamic)
        for (int row = 0; row < samples.rows; ++row) {

            // minimum distance
            float min = FLT_MAX;
            int imin = -1;
            for (int c = 0; c < nclusters; ++c) {

                float d = dist.compute(samples.row(row), centroids.row(c));
                if (d < min) {
                    min = d;
                    imin = c;
                }
            }
            groups[row] = imin;
        }
    }

    void KMeans::save_impl(std::ofstream &output) const {

        output.write((char*) &maxit, sizeof (maxit));

        dist.save(output);
        centroids.save(output);
    }

    void KMeans::load_impl(std::ifstream &input) {

        input.read((char*) &maxit, sizeof (maxit));

        dist.load(input);
        centroids.load(input);
    }
}

