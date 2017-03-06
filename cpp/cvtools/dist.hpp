#ifndef DIST_H
#define	DIST_H

#include "util.hpp"

namespace cvtools {

    /*
     * Distance
     */
    class Distance {
    public:

        MType compute(const Matrix &a, const Matrix &b);

        void save(std::ofstream &output) const;
        
        void load(std::ifstream &input);

    protected:

        virtual MType compute_impl(const Matrix &a, const Matrix &b) = 0;
        
        virtual void save_impl(std::ofstream &output) const = 0;

        virtual void load_impl(std::ifstream &input) = 0;
    };

    /*
     * Euclidian
     */
    class L2 : public Distance {
    private:

        MType compute_impl(const Matrix &a, const Matrix &b);

        void save_impl(std::ofstream &output) const;

        void load_impl(std::ifstream &input);
    };
}

#endif
