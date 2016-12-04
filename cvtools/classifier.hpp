#ifndef CLASSIFIER_H
#define	CLASSIFIER_H

#include "regression.hpp"

#include <vector>
#include <unordered_map>

namespace cvtools {

    /*
     * Classifier
     */
    class Classifier {
    public:

        void train(Matrix &samples, Labels &labels);

        void response(Matrix &samples, Matrix &resp) const;

        void save(std::ofstream &output) const;
        
        void load(std::ifstream &input);

    protected:

        virtual void train_impl(Matrix &samples, Labels &labels) = 0;

        virtual void response_impl(Matrix &samples, Matrix &resp) const = 0;

        virtual void save_impl(std::ofstream &output) const = 0;

        virtual void load_impl(std::ifstream &input) = 0;
    };
    
    /*
     * PLS Classifier
     */
    class PLSClassifier : public Classifier {
    public:

        static const std::string LAB_POS;
        static const std::string LAB_NEG;

        PLSClassifier(const int factors = 3);

        std::vector<PLSRegression> models;
        std::vector<ZMean> zmean;
        Labels ulab;
        int factors;

    private:

        virtual void train_impl(Matrix &samples, Labels &labels);

        virtual void response_impl(Matrix &samples, Matrix &resp) const;
        
        void save_impl(std::ofstream &output) const;

        void load_impl(std::ifstream &input);
    };
}

#endif