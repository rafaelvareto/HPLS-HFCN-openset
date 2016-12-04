#ifndef REGRESSION_H
#define	REGRESSION_H

#include "norm.hpp"

#include <vector>
#include <unordered_map>
#include <fstream>

namespace cvtools {

    /*
     * Regression
     */
    class Regression {
    public:

        void train(Matrix &samples, Matrix &resp);

        void predict(Matrix &samples, Matrix &resp) const;
        
        void save(std::ofstream &output) const;
        
        void load(std::ifstream &input);

    protected:

        virtual void train_impl(Matrix &samples, Matrix &resp) = 0;

        virtual void predict_impl(Matrix &samples, Matrix &resp) const = 0;
        
        virtual void save_impl(std::ofstream &output) const = 0;

        virtual void load_impl(std::ifstream &input) = 0;
    };

    /*
     PLSRegression
     */
    class PLSRegression : public Regression {
    public:

        PLSRegression(const int factors = 3);

        Matrix regression;
        int factors;

        void nipals(Matrix X, Matrix Y, Matrix &T, Matrix &P,
                Matrix &U, Matrix &Q, Matrix &W, Matrix &B)const;

    private:

        void train_impl(Matrix &samples, Matrix &resp);

        void predict_impl(Matrix &samples, Matrix &resp) const;
        
        void save_impl(std::ofstream &output) const;

        void load_impl(std::ifstream &input);
    };
    
    /*
     VSPLSRegression
     */
    class VSPLSRegression : public Regression {
    public:
        
        enum TYPE {
            WEIGHTS,
            BETA,
            VIP,
            RANDOM
        };

        VSPLSRegression(const int factors = 3, const int ndim = 100,
            const TYPE type = BETA, const bool retrain = true);

        PLSRegression model;
        std::vector<int> indexes;
        int ndim;
        TYPE type;
        bool retrain;

        void vs_weights(Matrix &samples, Matrix &resp);

        void vs_beta(Matrix &samples, Matrix &resp, Matrix &reg);

        void vs_vip(Matrix &samples, Matrix &resp);

    private:
        
        void train_impl(Matrix &samples, Matrix &resp);

        void predict_impl(Matrix &samples, Matrix &resp) const;
        
        void save_impl(std::ofstream &output) const;

        void load_impl(std::ifstream &input);
    };
}

#endif