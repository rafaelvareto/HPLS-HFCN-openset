#include "regression.hpp"

#include <unordered_set>
#include <fstream>
#include <numeric>
#include <random>

using namespace std;

std::default_random_engine rng;

namespace cvtools {

    /*
     * Regression
     */
    void Regression::train(Matrix &samples, Matrix &resp) {

        train_impl(samples, resp);
    }

    void Regression::predict(Matrix &samples, Matrix &resp) const {

        predict_impl(samples, resp);
    }

    void Regression::save(std::ofstream &output) const {

        save_impl(output);
    }

    void Regression::load(std::ifstream &input) {

        load_impl(input);
    }

    /*
     * PLS Regression
     */
    PLSRegression::PLSRegression(const int factors)
    : factors(factors) {

    }

    void PLSRegression::train_impl(Matrix& samples, Matrix& resp) {

        Matrix T, P, U, Q, W, B;

        nipals(samples, resp.clone(), T, P, U, Q, W, B);
        regression = ((W * (P.t() * W).inv(cv::DECOMP_SVD)) * (T.t() * T).inv(cv::DECOMP_SVD) * T.t() * resp).t();
        
        // print beta regression values and quit
        static ofstream output("regression.dat");
        sort(regression.begin(), regression.end());
        for(int row = 0; row < regression.rows; ++row)
            output << regression(row) << '\n';
    }

    void PLSRegression::predict_impl(Matrix &samples, Matrix &resp) const {

        assert(samples.cols == regression.cols);

        if (resp.cols != regression.rows || resp.rows != samples.rows)
            resp.create(samples.rows, regression.rows);

        for (int row = 0; row < samples.rows; ++row)
            for (int reg = 0; reg < regression.rows; ++reg) {

                const MType *s = samples.ptr<MType>(row, 0);
                const MType *r = regression.ptr<MType>(reg, 0);
                MType *t = resp.ptr<MType>(row, reg);
                *t = 0;

                for (int col = 0; col < samples.cols; ++col, ++s, ++r)
                    *t += (*s) * (*r);
            }
    }

    void PLSRegression::save_impl(std::ofstream &output) const {

        output.write((char*) &factors, sizeof (factors));

        // regression
        regression.save(output);
    }

    void PLSRegression::load_impl(std::ifstream &input) {

        input.read((char*) &factors, sizeof (factors));

        // regression
        regression.load(input);
    }

    void PLSRegression::nipals(Matrix X, Matrix Y, Matrix &T, Matrix &P,
            Matrix &U, Matrix &Q, Matrix &W, Matrix &B) const {

        //Setting the termination criteria
        int nMaxOuter = 10;
        double TermCrit = 10e-6, TempVal;
        Matrix tNorm;
        double MaxValX, MaxValY;
        int MaxIndexX, MaxIndexY;
        Matrix TempX, TempY;

        //Matrices for storing the intermediate values.
        Matrix tTemp, tNew, uTemp, wTemp, qTemp, pTemp, bTemp;

        //Allocating memory
        T.create(X.rows, factors);
        P.create(X.cols, factors);
        U.create(Y.rows, factors);
        Q.create(Y.cols, factors);
        W.create(X.cols, factors);
        B.create(factors, 1);
        tTemp.create(X.rows, 1);
        uTemp.create(Y.rows, 1);

        for (int index1 = 0; index1 < factors; index1++) {

            //Finding the column having the highest norm
            MaxValX = 0;
            MaxValY = 0;
            MaxIndexX = 0;
            MaxIndexY = 0;
            TempX.create(X.rows, 1);
            TempY.create(Y.rows, 1);
            for (int index3 = 0; index3 < X.cols; index3++) {
                for (int index2 = 0; index2 < X.rows; index2++) {
                    TempX.at<float>(index2, 0) = X.at<float>(index2, index3);
                }
                if (cv::norm(TempX) > MaxValX) {
                    MaxValX = cv::norm(TempX);
                    MaxIndexX = index3;
                }
            }
            for (int index3 = 0; index3 < Y.cols; index3++) {
                for (int index2 = 0; index2 < Y.rows; index2++) {
                    TempY.at<float>(index2, 0) = Y.at<float>(index2, index3);
                }
                if (cv::norm(TempY) > MaxValY) {
                    MaxValY = cv::norm(TempY);
                    MaxIndexY = index3;
                }
            }

            for (int index3 = 0; index3 < X.rows; index3++) {
                tTemp.at<float>(index3, 0) = X.at<float>(index3, MaxIndexX);
                uTemp.at<float>(index3, 0) = Y.at<float>(index3, MaxIndexY);
            }

            // Iteration for Outer Modelling
            for (int index2 = 0; index2 < nMaxOuter; index2++) {
                wTemp = X.t() * uTemp;
                wTemp = wTemp / cv::norm(wTemp);
                tNew = X * wTemp;
                qTemp = Y.t() * tNew;
                qTemp = qTemp / cv::norm(qTemp);
                uTemp = Y * qTemp;

                TempVal = cv::norm(tTemp - tNew);
                if (cv::norm(tTemp - tNew) < TermCrit) {
                    break;
                }
                tTemp = tNew.clone();
            }

            // Residual Deflation
            tNorm = tTemp.t() * tTemp;
            bTemp = uTemp.t() * tTemp / tNorm.at<float>(0, 0);
            pTemp = X.t() * tTemp / tNorm.at<float>(0, 0);
            X = X - tTemp * pTemp.t();
            Y = Y - bTemp.at<float>(0, 0) * (tTemp * qTemp.t());


            // Saving Results to Outputs.
            for (int index3 = 0; index3 != X.rows; index3++) {
                T.at<float>(index3, index1) = tTemp.at<float>(index3, 0);
                U.at<float>(index3, index1) = uTemp.at<float>(index3, 0);
            }
            for (int index3 = 0; index3 != X.cols; index3++) {
                P.at<float>(index3, index1) = pTemp.at<float>(index3, 0);
                W.at<float>(index3, index1) = wTemp.at<float>(index3, 0);
            }

            for (int index3 = 0; index3 != qTemp.rows; index3++) {
                Q.at<float>(index3, index1) = qTemp.at<float>(index3, 0);
            }
            B.at<float>(index1, 0) = bTemp.at<float>(0, 0);

            // Checking residual
            if ((cv::norm(X) == 0) || (cv::norm(Y) == 0)) {
                break;
            }
        }
    }

    /*
     VSPLSRegression
     */
    VSPLSRegression::VSPLSRegression(const int factors, const int ndim,
            const TYPE type, const bool retrain)
    : model(factors), ndim(ndim), type(type), retrain(retrain) {

    }

    void VSPLSRegression::train_impl(Matrix &samples, Matrix &resp) {

        assert(ndim <= samples.cols);

        if (ndim == -1)
            ndim = samples.cols;

        Matrix sampled(samples.rows, ndim);
        Matrix reg;

        switch (type) {
            case WEIGHTS: vs_weights(samples, resp);
                break;
            case BETA: vs_beta(samples, resp, reg);
                break;
            case VIP: vs_vip(samples, resp);
                break;
            case RANDOM:
            {
                normal_distribution<float> gen(0, 1);
                model.regression.create(1, samples.cols);
                for (int col = 0; col < samples.cols; ++col)
                    model.regression(col) = gen(rng);
                return;
            }
            default: assert(false);
        }

        sort(indexes.begin(), indexes.end());

        if (retrain) {
            for (int row = 0; row < samples.rows; ++row) {
                MType *t = sampled.ptr<MType>(row);
                MType *s = samples.ptr<MType>(row);
                for (int col = 0; col < indexes.size(); ++col)
                    t[col] = s[indexes[col]];
            }
            model.train(sampled, resp);
        } else {
            model.regression.create(1, indexes.size());
            for (int col = 0; col < indexes.size(); ++col)
                model.regression(col) = reg(indexes[col]);
        }
    }

    void VSPLSRegression::predict_impl(Matrix &samples, Matrix & resp) const {

        if (resp.cols != model.regression.rows || resp.rows != samples.rows)
            resp.create(samples.rows, model.regression.rows);

        for (int row = 0; row < samples.rows; ++row) {
            for (int reg = 0; reg < model.regression.rows; ++reg) {

                const MType *s = samples.ptr<MType>(row, 0);
                const MType *r = model.regression.ptr<MType>(reg, 0);
                MType *t = resp.ptr<MType>(row, reg);
                *t = 0;

                if (indexes.empty()) {
                    assert(samples.cols == model.regression.cols);
                    for (int col = 0; col < samples.cols; ++col, ++r)
                        *t += s[col] * (*r);
                } else {
                    int *i = (int*) indexes.data();
                    for (int col = 0; col < indexes.size(); ++col, ++r, ++i)
                        *t += s[*i] * (*r);
                }
            }
        }
    }

    void VSPLSRegression::save_impl(std::ofstream & output) const {

        output.write((char*) &ndim, sizeof (ndim));
        model.save(output);
        save_list_rw(output, indexes);
    }

    void VSPLSRegression::load_impl(std::ifstream & input) {

        input.read((char*) &ndim, sizeof (ndim));
        model.load(input);
        load_list_rw(input, indexes);
    }

    void VSPLSRegression::vs_weights(Matrix& samples, Matrix & resp) {

        PLSRegression prob(model.factors);
        Matrix T, P, U, Q, W, B;
        prob.nipals(samples.clone(), resp.clone(), T, P, U, Q, W, B);

        assert(W.cols == prob.factors && W.rows == samples.cols);
        W = cv::abs(W);

        Matrix w(samples.cols, 1);
        for (int f = 0; f < W.rows; ++f)
            w(f) = *max_element(W.ptr<float>(f), W.ptr<float>(f) + W.cols);

        vector<pair<int, float>> weights(w.rows);
        for (int col = 0; col < w.rows; ++col) {
            weights[col].first = col;
            weights[col].second = w(col);
        }

        sort(weights.begin(), weights.end(),
                [](const pair<int, float> &a, const pair<int, float> &b) {
                    return a.second > b.second;
                });

        indexes.clear();
        for (int col = 0; col < ndim; ++col)
            indexes.push_back(weights[col].first);
        weights.clear();
    }

    void VSPLSRegression::vs_beta(Matrix& samples, Matrix &resp, Matrix &reg) {

        PLSRegression prob(model.factors);
        Matrix c = samples.clone();
        prob.train(c, resp);
        c.release();
        reg = prob.regression.clone();
        prob.regression = abs(prob.regression);

        vector<pair<int, float>> weights(prob.regression.cols);
        for (int col = 0; col < prob.regression.cols; ++col) {
            weights[col].first = col;
            weights[col].second = prob.regression(col);
        }

        sort(weights.begin(), weights.end(),
                [](const pair<int, float> &a, const pair<int, float> &b) {
                    return a.second > b.second;
                });

        indexes.clear();
        for (int col = 0; col < ndim; ++col)
            indexes.push_back(weights[col].first);
        weights.clear();
    }

    void VSPLSRegression::vs_vip(Matrix& samples, Matrix & resp) {

        PLSRegression prob(model.factors);
        Matrix T, P, U, Q, W, B;
        prob.nipals(samples.clone(), resp.clone(), T, P, U, Q, W, B);

        // compute VIP
        Matrix vip;
        Matrix Wtemp, Btemp;

        cv::pow(W, 2, Wtemp); // W^2
        cv::pow(B, 2, Btemp); // B^2
        Matrix temp = Wtemp * Btemp; // W^2 * B^2
        cv::pow(temp, 0.5, temp); // (W^2 * B^2)^0.5

        vip = temp / cv::norm(B, cv::NORM_L2); // (W^2 * B^2)^0.5 / || B ||;
        vip *= sqrt(samples.rows); // sqrt(n) * ...

        vector<pair<int, float>> weights(vip.rows);

        for (int row = 0; row < vip.rows; ++row) {
            weights[row].first = row;
            weights[row].second = vip(row);
        }

        sort(weights.begin(), weights.end(),
                [](const pair<int, float> &a, const pair<int, float> &b) {
                    return a.second > b.second;
                });

        indexes.clear();
        for (int col = 0; col < ndim; ++col)
            indexes.push_back(weights[col].first);
        weights.clear();
    }
}
