#include "cluster.hpp"
#include "dist.hpp"

#include <algorithm>
#include <unordered_set>
#include <fstream>

using namespace std;

namespace cvtools {

    /*
     * Distance
     */
    MType Distance::compute(const Matrix &a, const Matrix &b) {
        
        return compute_impl(a, b);
    }
    
    void Distance::save(std::ofstream& output) const {
        
        save_impl(output);
    }
    
    void Distance::load(std::ifstream& input) {
        
        load_impl(input);
    }

    /*
     * L2
     */
    MType L2::compute_impl(const Matrix &a, const Matrix &b) {

        return cv::norm(a, b, cv::NORM_L2);
    }
    
    void L2::save_impl(std::ofstream& output) const {
        // nothing
    }
    
    void L2::load_impl(std::ifstream& input) {
        // nothing
    }
}
