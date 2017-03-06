#include "util.hpp"

#include <fstream>
#include <functional>
#include <boost/filesystem.hpp>

using namespace std;

namespace cvtools {

    /*
     * Label
     */
    Label::Label()
    : extra(false) {

    }

    Label::Label(const std::string &id, const bool extra)
    : id(id), extra(extra) {

    }

    void Label::save(std::ofstream &output) const {

        output.write((char*) &extra, sizeof (extra));
        output << id << '\n';
    }

    void Label::load(std::ifstream &input) {

        input.read((char*) &extra, sizeof (extra));
        input >> id;

        char dump;
        input.read(&dump, sizeof (dump));
    }

    /*
     * Labels
     */
    void Labels::save(std::ofstream& output) const {

        size_t s = size();

        output.write((char*) &s, sizeof (s));

        for (const Label &a : (*this))
            a.save(output);
    }

    void Labels::load(std::ifstream& input) {

        size_t s;
        input.read((char*) &s, sizeof (s));

        resize(s);

        for (Label &a : (*this))
            a.load(input);
    }

    /*
     * Matrix
     */
    Matrix::Matrix() {

    }

    Matrix::Matrix(const int rows, const int cols) : Mat_<MType>(rows, cols) {

    }

    Matrix::Matrix(cv::MatExpr mat) : Mat_<MType>(mat) {

    }

    Matrix::Matrix(cv::Mat_<MType> mat) : Mat_<MType>(mat) {

    }

    Matrix::Matrix(cv::Mat mat) : Mat_<MType>((Mat_<MType>) mat) {
        assert(mat.type() == CV_32F);
    }

    Matrix& Matrix::operator=(const MType val) {
        ((cv::Mat_<MType>) * this) = val;
    }

    MType& Matrix::operator()() {

        assert(rows == 1 && cols == 1);
        return at<MType>(0, 0);
    }

    MType& Matrix::operator()(const int index) {

        assert(rows == 1 || cols == 1);

        if (rows == 1)
            return at<MType>(0, index);
        return at<MType>(index, 0);
    }

    MType& Matrix::operator()(const int row, const int col) {
        return at<MType>(row, col);
    }

    const MType& Matrix::operator()() const {

        assert(rows == 1 && cols == 1);
        return at<MType>(0, 0);
    }

    const MType& Matrix::operator()(const int index) const {

        assert(rows == 1 || cols == 1);

        if (rows == 1)
            return at<MType>(0, index);
        return at<MType>(index, 0);
    }

    const MType& Matrix::operator()(const int row, const int col) const {
        return at<MType>(row, col);
    }

    void Matrix::cbind(const Matrix& mat) {

        if (empty())
            (*this) = mat.clone();
        else
            cv::hconcat(*this, mat, *this);
    }

    void Matrix::rbind(const Matrix& mat) {

        if (empty())
            (*this) = mat.clone();
        else
            cv::vconcat(*this, mat, *this);
    }

    void Matrix::shuffle() {

        vector<int> indexes;

        for (int c = 0; c < rows; ++c)
            indexes.push_back(c);

        random_shuffle(indexes.begin(), indexes.end());

        Matrix output;
        for (int c = 0; c < rows; c++)
            output.push_back(row(indexes[c]));

        (*this) = output;
    }

    void Matrix::save(std::ofstream &output) const {

        // number of columns
        output.write((char*) &rows, sizeof (rows));
        output.write((char*) &cols, sizeof (cols));
        // regression
        output.write((char*) data, rows * cols * sizeof (MType));
    }

    void Matrix::load(std::ifstream &input) {

        int cols, rows;
        input.read((char*) &rows, sizeof (rows));
        input.read((char*) &cols, sizeof (cols));

        // regression
        create(rows, cols);
        input.read((char*) data, rows * cols * sizeof (MType));
    }

    void sample(const Matrix &population, Matrix &samples) {

        assert(samples.rows <= population.rows);

        int row;
        for (row = 0; row < samples.rows; ++row)
            population.row(row).copyTo(samples.row(row));

        for (; row < population.rows; ++row) {
            int r = rand() % (row + 1);
            if (r < samples.rows)
                population.row(row).copyTo(samples.row(r));
        }
    }

    void gradient(const Image &image, Matrix &orientation, Matrix &magnitude) {

        Image gray;
        cv::cvtColor(image, gray, CV_RGB2BGR);

        static float kernel[] = {-1, 0, 1};
        static cv::Mat_<float> hkernel(1, 3, kernel), vkernel(3, 1, kernel);

        Image himage, vimage;
        cv::filter2D(gray, himage, -1, hkernel);
        cv::filter2D(gray, vimage, -1, vkernel);

        assert(gray.cols == himage.cols && gray.rows == himage.rows);
        assert(gray.cols == vimage.cols && gray.rows == vimage.rows);

        orientation.create(gray.rows, gray.cols);
        magnitude.create(gray.rows, gray.cols);

        for (int row = 0; row < gray.rows; ++row) {

            for (int col = 0; col < gray.cols; ++col) {

                float o;
                if (himage.at<uchar>(row, col) == 0 &&
                        vimage.at<uchar>(row, col) == 0)
                    o = 0;
                else
                    o = atan(
                        himage.at<uchar>(row, col) /
                        (float) vimage.at<uchar>(row, col));
                float m = sqrt(
                        himage.at<uchar>(row, col) * himage.at<uchar>(row, col) +
                        vimage.at<uchar>(row, col) * vimage.at<uchar>(row, col));

                orientation(row, col) = o;
                magnitude(row, col) = m;
            }
        }
    }

#if 0

    void shuffle(Matrix &matrix, ID &ids) {

        vector<int> indexes;
        assert(matrix.rows == ids.size());

        for (int c = 0; c < matrix.rows; ++c)
            indexes.push_back(c);

        random_shuffle(indexes.begin(), indexes.end());

        Matrix output;
        ID outlabels;
        for (int c = 0; c < matrix.rows; c++) {
            output.push_back(matrix.row(indexes[c]));
            outlabels.push_back(ids[indexes[c]]);
        }
        matrix = output.clone();
        ids = outlabels;
    }
#endif

    /*
     * Files
     */
    void ls(const std::string &path,
            function<void(const boost::filesystem::directory_iterator&) > f) {

        namespace fs = boost::filesystem;

        fs::path p(path);
        fs::directory_iterator end_iter;

        if (fs::exists(p) && fs::is_directory(p))
            for (fs::directory_iterator dir_iter(p); dir_iter != end_iter; ++dir_iter)
                f(dir_iter);
    }

    FileList ls_files(const std::string &path) {

        FileList output;

        namespace fs = boost::filesystem;

        ls(path, [&](const fs::directory_iterator it) {
            if (fs::is_regular_file(it->status()))
                output.push_back(it->path().string());
        });

        return output;
    }

    FileList ls_folders(const std::string &path) {

        FileList output;

        namespace fs = boost::filesystem;

        ls(path, [&](const fs::directory_iterator it) {
            if (fs::is_directory(it->status()))
                output.push_back(it->path().string());
        });

        return output;
    }

    void save_image(const Image &image, const string &path) {

        boost::filesystem::path dir(path);
        if (dir.has_parent_path())
            boost::filesystem::create_directories(dir.parent_path());

        cv::imwrite(path, image);
    }
}