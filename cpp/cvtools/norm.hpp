#ifndef NORM_H
#define	NORM_H

#include "util.hpp"

namespace cvtools {

    /*
     * Norm
     */
    class Norm {
    public:

        Norm& train(const Matrix &mat);

        Matrix& norm(Matrix &mat) const;

        void train_norm(Matrix &mat);

        Matrix& unnorm(Matrix &mat) const;
        
        void save(std::ofstream &output) const;
        
        void load(std::ifstream &input);

    protected:

        virtual void train_impl(const Matrix &mat) = 0;

        virtual void norm_impl(Matrix &mat) const = 0;

        virtual void unnorm_impl(Matrix &mat) const = 0;
        
        virtual void save_impl(std::ofstream &output) const = 0;

        virtual void load_impl(std::ifstream &input) = 0;
    };

    /*
     * Zero Mean
     */
    class ZMean : public Norm {
    public:

        Matrix mean;

    private:

        void train_impl(const Matrix &mat);

        void norm_impl(Matrix &mat) const;
        
        void unnorm_impl(Matrix &mat) const;
        
        void save_impl(std::ofstream &output) const;

        void load_impl(std::ifstream &input);
    };

    /*
     * Unitary Standard Deviation
     */
    class UStd : public Norm {
    public:

        Matrix std;

    private:

        void train_impl(const Matrix &mat);

        void norm_impl(Matrix &mat) const;

        void unnorm_impl(Matrix &mat) const;
        
        void save_impl(std::ofstream &output) const;

        void load_impl(std::ifstream &input);
    };

    /*
     * Z-Score
     */
    class ZScore : public Norm {
    public:

        ZMean zmean;
        UStd ustd;

    private:

        void train_impl(const Matrix &mat);

        void norm_impl(Matrix &mat) const;

        void unnorm_impl(Matrix &mat) const;
        
        void save_impl(std::ofstream &output) const;

        void load_impl(std::ifstream &input);
    };

    /*
     *  Logistic Norm
     */
    class LNorm : public Norm {
    public:

        ZMean zmean;
        UStd ustd;

    private:

        void train_impl(const Matrix &mat);

        void norm_impl(Matrix &mat) const;
        
        void save_impl(std::ofstream &output) const;

        void load_impl(std::ifstream &input);
    };
}

#endif