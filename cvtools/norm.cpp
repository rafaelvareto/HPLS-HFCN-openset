#include "norm.hpp"

#include <fstream>
#include <boost/filesystem.hpp>

using namespace std;

namespace cvtools {

    /*
     * Normalization
     */
    Norm& Norm::train(const Matrix &mat) {
        train_impl(mat);
        return (*this);
    }

    Matrix& Norm::norm(Matrix &mat) const {
        norm_impl(mat);
        return mat;
    }

    void Norm::train_norm(Matrix &mat) {
        train(mat).norm(mat);
    }

    Matrix& Norm::unnorm(Matrix &mat) const {
        unnorm_impl(mat);
        return mat;
    }

    void Norm::save(std::ofstream &output) const {

        save_impl(output);
    }

    void Norm::load(std::ifstream &input) {

        load_impl(input);
    }

    /*
     * Zero Mean
     */
    void ZMean::train_impl(const Matrix &mat) {

        cv::reduce(mat, mean, 0, CV_REDUCE_AVG);
    }

    void ZMean::norm_impl(Matrix &mat) const {

        assert(mat.cols == mean.cols);
        for(int row = 0; row < mat.rows; ++row) {
            MType *p = mat.ptr<MType>(row, 0);
            MType *c = (MType*) mean.data;
            for(int col = 0; col < mat.cols; ++col, ++p, ++c)
                *p -= *c;
        }
    }

    void ZMean::unnorm_impl(Matrix &mat) const {

        assert(mat.cols == mean.cols);
        for(int row = 0; row < mat.rows; ++row) {
            MType *p = mat.ptr<MType>(row, 0);
            MType *c = (MType*) mean.data;
            for(int col = 0; col < mat.cols; ++col, ++p, ++c)
                *p += *c;
        }
    }

    void ZMean::save_impl(std::ofstream &output) const {

        mean.save(output);
    }

    void ZMean::load_impl(std::ifstream &input) {

        mean.load(input);
    }

    /*
     * Unitary Standard Deviation
     */
    void UStd::train_impl(const Matrix &mat) {

        ZMean zmean;
        zmean.train(mat);

        std.create(1, mat.cols);
        for (int col = 0; col < mat.cols; ++col) {

            Matrix c = mat.col(col).clone() - zmean.mean(col);
            c = c.t() * c / c.rows;

            std(0, col) = sqrt(c());
        }
    }

    void UStd::norm_impl(Matrix &mat) const {

        assert(mat.cols == std.cols);
        for(int row = 0; row < mat.rows; ++row) {
            MType *p = mat.ptr<MType>(row, 0);
            MType *c = (MType*) std.data;
            for(int col = 0; col < mat.cols; ++col, ++p, ++c)
                if (*c > 1e-6)
                    *p /= *c;
        }
    }

    void UStd::unnorm_impl(Matrix &mat) const {

        assert(mat.cols == std.cols);
        for(int row = 0; row < mat.rows; ++row) {
            MType *p = mat.ptr<MType>(row, 0);
            MType *c = (MType*) std.data;
            for(int col = 0; col < mat.cols; ++col, ++p, ++c)
                if (*c > 1e-6)
                    *p *= *c;
        }
    }

    void UStd::save_impl(std::ofstream &output) const {

        std.save(output);
    }

    void UStd::load_impl(std::ifstream &input) {

        std.load(input);
    }

    /*
     * Z-Score
     */
    void ZScore::train_impl(const Matrix &mat) {

        zmean.train(mat);
        ustd.train(mat);
    }

    void ZScore::norm_impl(Matrix &mat) const {

        zmean.norm(mat);
        ustd.norm(mat);
    }

    void ZScore::unnorm_impl(Matrix &mat) const {

        ustd.unnorm(mat);
        zmean.unnorm(mat);
    }

    void ZScore::save_impl(std::ofstream &output) const {

        zmean.save(output);
        ustd.save(output);
    }

    void ZScore::load_impl(std::ifstream &input) {

        zmean.load(input);
        ustd.load(input);
    }

    /*
     * Logistic Norm
     */
    void LNorm::train_impl(const Matrix &mat) {

        zmean.train(mat);
        ustd.train(mat);
    }

    void LNorm::norm_impl(Matrix &mat) const {

        if(!zmean.mean.empty())
            zmean.norm(mat);

        for (int col = 0; col < mat.cols; ++col) {

            if(!ustd.std.empty())
                if (ustd.std(col) > 1e-6)
                    mat.col(col) *= -1 / (2 * ustd.std(col));

            cv::exp(mat.col(col), mat.col(col));
            mat.col(col) = 1 / (1 + mat.col(col));
        }
    }

    void LNorm::save_impl(std::ofstream &output) const {

        zmean.save(output);
        ustd.save(output);
    }

    void LNorm::load_impl(std::ifstream &input) {

    }
}