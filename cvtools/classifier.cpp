#include "classifier.hpp"

#include <unordered_set>
#include <fstream>

using namespace std;

namespace cvtools {

    /*
     * Classifier
     */
    void Classifier::train(Matrix &samples, Labels &labels) {

        train_impl(samples, labels);
    }

    void Classifier::response(Matrix &samples, Matrix &resp) const {

        response_impl(samples, resp);
    }

    void Classifier::save(std::ofstream &output) const {

        save_impl(output);
    }

    void Classifier::load(std::ifstream &input) {

        load_impl(input);
    }

    /*
     * PLS Classifier
     */
    const std::string PLSClassifier::LAB_POS = "pos";
    const std::string PLSClassifier::LAB_NEG = "neg";

    PLSClassifier::PLSClassifier(const int factors)
    : factors(factors) {

    }

    void PLSClassifier::train_impl(Matrix &samples, Labels &labels) {

        ulab.clear();

        unordered_set<string> u;
        for (const Label &label : labels)
            if (!label.extra)
                u.insert(label.id);

        if (u.size() > 2) {
            for (const string &id : u)
                ulab.push_back(id);
        } else {
            assert(u.find(LAB_POS) != u.end());
            ulab.push_back(LAB_POS);
        }

        int count = 0;
        models.resize(ulab.size());
        zmean.resize(models.size());

#pragma omp parallel for
        for (int m = 0; m < models.size(); ++m) {

#pragma omp critical
            {
                //cout << "id: " << ++count << '/' << models.size() << endl;
            }

            assert(samples.rows > 0);
            Matrix r(samples.rows, 1);
            r = -1.0f;
            for (int row = 0; row < r.rows; ++row)
                if(labels[row].id == ulab[m].id)
                    r(row) = 1.0f;
            assert((*max_element(r.begin(), r.end())) != -1.0f);
            zmean[m].train_norm(r);

            Matrix s = samples.clone();
            models[m] = PLSRegression(factors);
            models[m].train(s, r);
            //models[m].regression = samples.row(0);
        }
    }

    void PLSClassifier::response_impl(Matrix &samples, Matrix &resp) const {

        if(resp.rows != samples.rows || resp.cols != models.size())
            ;
            resp.create(samples.rows, models.size());

#pragma omp parallel for
        for (int m = 0; m < models.size(); ++m) {
            Matrix r;
            models[m].predict(samples, r);
            zmean[m].unnorm(r);
            r.copyTo(resp.col(m));
        }
    }

    void PLSClassifier::save_impl(std::ofstream &output) const {

        output.write((char*) &factors, sizeof(factors));

        ulab.save(output);
        save_list_sl(output, zmean);
        save_list_sl(output, models);
    }

    void PLSClassifier::load_impl(std::ifstream &input) {

        input.read((char*) &factors, sizeof(factors));

        ulab.load(input);
        load_list_sl(input, zmean);
        load_list_sl(input, models);
    }
}
