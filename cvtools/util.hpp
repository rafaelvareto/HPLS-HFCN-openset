#ifndef UTIL_H
#define	UTIL_H

#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace cvtools {

    /*
     * Structures
     */
    typedef cv::Mat Image;

    class Label {
    public:
        
        Label();
        
        Label(const std::string &id, const bool extra = false);

        void save(std::ofstream &output) const;

        void load(std::ifstream &input);

        std::string id;
        bool extra;
    };

    class Labels : public std::vector<Label> {
    public:
        
        void save(std::ofstream &output) const;

        void load(std::ifstream &input);        
    };

    typedef float MType;

    class Matrix : public cv::Mat_<MType> {
    public:

        Matrix();

        Matrix(const int rows, const int cols);

        Matrix(cv::MatExpr mat);

        Matrix(cv::Mat_<MType> mat);

        Matrix(cv::Mat mat);

        Matrix& operator=(const float val);

        float& operator()();

        float& operator()(const int index);

        float& operator()(const int row, const int col);

        const float& operator()() const;

        const float& operator()(const int index) const;

        const float& operator()(const int row, const int col) const;

        void cbind(const Matrix &mat);

        void rbind(const Matrix &mat);

        void shuffle();

        void save(std::ofstream &output) const;

        void load(std::ifstream &input);
    };

    void sample(const Matrix &population, Matrix &samples);
    
    void gradient(const Image &image, Matrix &orientation, Matrix &magnitude);

    /*
     * Input/Output
     */
    template<typename T>
    void save_list_sl(std::ofstream &output, const std::vector<T> &list) {

        size_t size = list.size();

        output.write((char*) &size, sizeof (size));

        for (const auto &el : list)
            el.save(output);
    }

    template<typename T>
    void load_list_sl(std::ifstream &input, std::vector<T> &list) {

        size_t size = list.size();
        input.read((char*) &size, sizeof (size));

        list.resize(size);

        for (auto &el : list)
            el.load(input);
    }

    template<typename T>
    void save_list_rw(std::ofstream &output, const std::vector<T> &list) {

        size_t size = list.size();

        output.write((char*) &size, sizeof (size));

        for (const auto &el : list)
            output.write((char*) &el, sizeof (el));
    }

    template<typename T>
    void load_list_rw(std::ifstream &input, std::vector<T> &list) {

        size_t size = list.size();
        input.read((char*) &size, sizeof (size));

        list.resize(size);

        for (auto &el : list)
            input.read((char*) &el, sizeof (el));
    }

    /*
     * Files
     */
    typedef std::vector<std::string> FileList;

    FileList ls_files(const std::string &path);

    FileList ls_folders(const std::string &path);
    
    void save_image(const Image &image, const std::string &path);

    /*
     * Timer
     */
    // template<typename T = std::chrono::seconds>
    // class Timer {
    // public:

    //     Timer() {
    //         tick();
    //     }

    //     void tick() {
    //         ini = std::chrono::high_resolution_clock::now();
    //     }

    //     int tock() {
    //         end = std::chrono::high_resolution_clock::now();
    //         auto time = std::chrono::duration_cast<T>(end - ini);
    //         tick();

    //         return time.count();
    //     }

    // private:

    //     std::chrono::system_clock::time_point ini;
    //     std::chrono::system_clock::time_point end;
    // };
}

#endif
