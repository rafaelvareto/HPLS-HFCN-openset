#ifndef FEATS_H
#define	FEATS_H

#include <functional>
#include <unordered_map>
#include <memory>

#include "util.hpp"

namespace cvtools {

    /*
     * Regions
     */
    void sldwin(const Image &image, const cv::Size &size, const cv::Size &stride,
            std::function<void(const cv::Rect&) > callback);

    void sldwin(const cv::Size &image, const cv::Size &size, const cv::Size &stride,
            std::function<void(const cv::Rect&) > callback);

    void sldwin(const Image &image, const cv::Size &size, const cv::Size &stride,
            std::function<void(const Image&) > callback);

    void sldwin(const Image &image, const cv::Size &size, const cv::Size &stride,
            std::function<void(Image&) > callback);

    /*
     * Features
     */
    class Descriptor {
    public:

        Matrix compute(const Image &image) const;

    protected:

        virtual Matrix compute_impl(const Image &image) const = 0;
    };

    /*
     * Mean intensity
     */
    class Mean : public Descriptor {
    public:

        Mean(const cv::Size &block = cv::Size(4, 4),
                const cv::Size &stride = cv::Size(2, 2));

    private:

        Matrix compute_impl(const Image &image) const;

        const cv::Size block;
        const cv::Size stride;
    };

    /*
     * HOG
     */
    class Hog : public Descriptor {
    public:

        Hog(const cv::Size &block = cv::Size(16, 16),
                const cv::Size &stride = cv::Size(4, 4),
                const cv::Size &cell = cv::Size(4, 4));

    private:

        Matrix compute_impl(const Image &image) const;

        const cv::Size block;
        const cv::Size stride;
        const cv::Size cell;
    };

    /*
     * Color Histogram
     */
    class CHist : public Descriptor {
    public:

        CHist(const int hbin = 20, const int sbin = 32,
                const cv::Size &block = cv::Size(16, 16),
                const cv::Size &stride = cv::Size(8, 8));

    private:

        Matrix compute_impl(const Image &image) const;

        const int hbin;
        const int sbin;
        const cv::Size block;
        const cv::Size stride;
    };

    class Gabor : public Descriptor {
    public:

        Gabor(const int scales = 5, const int orientation = 8,
                const float downscale = .25,
                const cv::Size &kernel = cv::Size(16, 16));

    private:

        Matrix compute_impl(const Image &image) const;

        const int scales;
        const int orientation;
        const float downscale;
        const cv::Size kernel;
    };

    /*
     * LBP
     */
    class LBP : public Descriptor {
    public:

        LBP(const cv::Size &block = cv::Size(16, 16),
                const cv::Size &stride = cv::Size(8, 8));

    private:

        Matrix compute_impl(const Image &image) const;

        const cv::Size block;
        const cv::Size stride;
    };

    /*
     * CLBP
     */
    class CLBP : public Descriptor {
    public:

        CLBP(const int radius = 5,
                const cv::Size &block = cv::Size(16, 16),
                const cv::Size &stride = cv::Size(8, 8));

        template<typename Type>
        Matrix& structure(const cv::Mat &matrix, Matrix &descriptor,
                float tau = 3, const int max_transitions = 2) const {

            // compute code map
            std::unordered_map<uchar, uchar> codemap;
            uchar codeid = 1;
            for (int code = 0; code < 256; ++code) {

                // set bits to each transition
                char b = ((char) code) >> 1;
                char c = code ^ b;
                // count number of bits in c
                c = c - ((c >> 1) & 0x55555555);
                c = (c & 0x33333333) + ((c >> 2) & 0x33333333);
                int t = (((c + (c >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;

                if (t > max_transitions)
                    codemap[code] = 0;
                else
                    codemap[code] = codeid++;
            }

            descriptor.release();

            Image lbp_image(matrix.rows - 2 * radius,
                    matrix.cols - 2 * radius, CV_8UC1);

            // iterate through your data
            for (int i = radius; i < matrix.rows - radius; i++) {
                for (int j = radius; j < matrix.cols - radius; j++) {

                    char code = 0;
                    for (int n = 0; n < 8; n++) {

                        // sample points
                        float x = radius * cos(2.0 * M_PI * n / (float) 8);
                        float y = radius * -sin(2.0 * M_PI * n / (float) 8);

                        // relative indices
                        int fx = floor(x);
                        int fy = floor(y);
                        int cx = ceil(x);
                        int cy = ceil(y);

                        // fractional part
                        float ty = y - fy;
                        float tx = x - fx;

                        // set interpolation weights
                        float w1 = (1 - tx) * (1 - ty);
                        float w2 = tx * (1 - ty);
                        float w3 = (1 - tx) * ty;
                        float w4 = tx * ty;

                        float t =
                                w1 * matrix.at<Type>(i + fy, j + fx) +
                                w2 * matrix.at<Type>(i + fy, j + cx) +
                                w3 * matrix.at<Type>(i + cy, j + fx) +
                                w4 * matrix.at<Type>(i + cy, j + cx);

                        bool bit = ((t > matrix.at<Type>(i, j)) &&
                                (abs(t - matrix.at<Type>(i, j)) > tau));

                        code += bit << n;
                    }
                    lbp_image.at<uchar>(i - radius, j - radius) =
                            codemap[code];
                }
            }

            const int channels[] = {0};
            const int histSize[] = {static_cast<int> (codeid)};
            const float range[] = {0, static_cast<float> (codeid)};
            const float* ranges[] = {range};

            cv::Size _block = block;
            if (_block.width > lbp_image.cols)
                _block.width = lbp_image.cols;
            if (_block.height > lbp_image.rows)
                _block.height = lbp_image.rows;

            sldwin(lbp_image, _block, stride,
                    [&](Image & crop) {

                        cv::Mat hist;

                        cv::calcHist(&crop, 1, channels, cv::Mat(),
                                hist, 1, histSize, ranges, true, false);
                                hist.convertTo(hist, CV_32F);
                                hist = hist.reshape(1, 1);
                                hist /= cv::sum(hist)[0];
                                descriptor.cbind(hist);
                    });

            return descriptor;
        }

    private:

        Matrix compute_impl(const Image &image) const;

        const int radius;
        const cv::Size block;
        const cv::Size stride;
    };

    /*
     * Second Order
     */
    class SecondOrder : public Descriptor {
    public:

        SecondOrder(const Descriptor &descriptor,
                const cv::Size &block = cv::Size(32, 32),
                const cv::Size &stride = cv::Size(8, 8));

    private:

        Matrix compute_impl(const Image &image) const;

        const cv::Size block;
        const cv::Size stride;
        const Descriptor &descriptor;
    };

    /*
     * POEM
     */
    class POEM : public Descriptor {
    public:

        POEM(const int bins = 3, const int radius = 5,
                const cv::Size &cell = cv::Size(7, 7),
                const cv::Size &block = cv::Size(16, 16),
                const cv::Size &stride = cv::Size(8, 8));

    private:

        Matrix compute_impl(const Image &image) const;

        const int bins;
        const cv::Size cell;
        CLBP clbp;
    };

    /*
     * SIFT
     */
    class SIFT : public Descriptor {
    public:
        SIFT(const int numKeypoints = 16, const int Xstep = -1, const int Ystep = -1);

    private:
        Matrix compute_impl(const Image &image) const;

        cv::Ptr<cv::Feature2D> sift;
        int numKeypoints;
        int Xstep, Ystep;
    };

    /*
     * Dense
     */
//    class Dense : public Descriptor {
//    public:
//
//        Dense(cv::DescriptorExtractor* descriptor, const int step = 8);
//
//        ~Dense();
//
//    private:
//
//        Matrix compute_impl(const Image &image) const;
//
//        cv::DenseFeatureDetector detector;
//        cv::DescriptorExtractor *descriptor;
//    };
}
#endif
