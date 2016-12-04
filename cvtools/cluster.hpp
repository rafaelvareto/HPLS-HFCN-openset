#ifndef CLUSTER_H
#define	CLUSTER_H

#include "dist.hpp"

namespace cvtools {

    /*
     * Cluster
     */
    typedef int Group;

    class Groups : public std::vector<Group> {
    public:

        void save(std::ofstream &output) const;

        void load(std::ifstream &input);
    };
    
    class Cluster {
    public:

        Cluster(const int nclusters);

        void train(Matrix &samples, Groups &groups);

        void train(Matrix &samples, const Labels &labels, Groups &groups);

        void predict(Matrix &samples, Groups &groups) const;

        void save(std::ofstream &output) const;
        
        void load(std::ifstream &input);

        int nclusters;

    protected:

        virtual void train_impl(Matrix &samples, const Labels &labels,
                Groups &groups) = 0;

        virtual void predict_impl(Matrix &samples, Groups &groups) const = 0;

        virtual void save_impl(std::ofstream &output) const = 0;

        virtual void load_impl(std::ifstream &input) = 0;
    };

    /*
     * K-Means
     */
    class KMeans : public Cluster {
    public:

        KMeans(const int nclusters = 16, const int maxit = 10,
                Distance &dist = l2);

        static L2 l2;

    private:

        void train_impl(Matrix &samples, const Labels &labels, Groups &groups);

        void predict_impl(Matrix &samples, Groups &groups) const;

        void save_impl(std::ofstream &output) const;

        void load_impl(std::ifstream &input);

        int maxit;
        Distance &dist;
        Matrix centroids;
    };
}

#endif
