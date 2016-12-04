
#include <opencv2/imgproc/types_c.h>

#include "feats.hpp"

using namespace std;

namespace cvtools {

    /*
     * Regions
     */
    void sldwin(const cv::Size &image, const cv::Size &size, const cv::Size &stride,
            std::function<void(const cv::Rect&) > callback) {

        assert(size.width <= image.width && size.height <= image.height);

        for (cv::Rect window(0, 0, size.width, size.height);
                window.y <= image.height - window.height;
                window.y += stride.height) {
            for (window.x = 0; window.x <= image.width - window.width;
                    window.x += stride.width) {
                callback(window);
            }
        }
    }

    void sldwin(const Image &image, const cv::Size &size, const cv::Size &stride,
            std::function<void(const cv::Rect&) > callback) {

        sldwin(cv::Size(image.cols, image.rows), size, stride, callback);
    }

    void sldwin(const Image &image, const cv::Size &size, const cv::Size &stride,
            std::function<void(const Image&) > callback) {

        sldwin(cv::Size(image.cols, image.rows), size, stride,
                [&](const cv::Rect & window) {
                    callback(image(window));
                });
    }

    void sldwin(const Image &image, const cv::Size &size, const cv::Size &stride,
            std::function<void(Image&) > callback) {

        sldwin(cv::Size(image.cols, image.rows), size, stride,
                [&](const cv::Rect & window) {
                    Image crop = image(window);
                    callback(crop);
                });
    }

    /*
     * Features
     */
    Matrix Descriptor::compute(const Image &image) const {

        return compute_impl(image).reshape(1, 1);
    }

    /*
     * Mean
     */
    Mean::Mean(const cv::Size& block, const cv::Size& stride) :
    block(block), stride(stride) {

    }

    Matrix Mean::compute_impl(const Image &image) const {

        Matrix features;
        sldwin(image, block, stride, [&](Image & crop) {

            Matrix val(1, 1);
            cv::cvtColor(crop, crop, CV_RGB2GRAY);
                    val = cv::mean(crop)[0];
                    features.cbind(val);
        });

        return features;
    }

    /*
     * HOG
     */
    Hog::Hog(const cv::Size &block, const cv::Size &stride,
            const cv::Size &cell)
    : block(block), stride(stride), cell(cell) {

    }

    Matrix Hog::compute_impl(const Image &image) const {

        Matrix features;

        // hog
        std::vector<float> hogDescriptor;

        cv::HOGDescriptor hog(
                cv::Size(image.cols, image.rows), // window
                block, // block
                stride, // stride
                cell, // cell
                9, // bins
                1,
                -1,
                cv::HOGDescriptor::L2Hys,
                0.4);
        hog.compute(image, hogDescriptor);
        cv::Mat(hogDescriptor, true).convertTo(features, CV_32F);

        return features;
    }

    /*
     * Color Histogram
     */
    CHist::CHist(const int hbin, const int sbin,
            const cv::Size &block, const cv::Size &stride)
    : hbin(hbin), sbin(sbin), block(block), stride(stride) {

    }

    Matrix CHist::compute_impl(const Image &image) const {

        Matrix features;

        cv::Mat hsv;
        cv::cvtColor(image, hsv, CV_BGR2HSV);

        int channels[] = {0, 1};

        int histSize[] = {hbin, sbin};

        float hranges[] = {0, 180};
        float sranges[] = {0, 256};
        const float* ranges[] = {hranges, sranges};

        sldwin(cv::Size(hsv.cols, hsv.rows), block, stride,
                [&](const cv::Rect & window) {

                    cv::Mat hist, fhist;
                    cv::Mat crop = hsv(window);

                            cv::calcHist(&crop, 1, channels, cv::Mat(),
                            hist, 2, histSize, ranges, true, false);
                            hist.convertTo(fhist, CV_32F);

                            fhist = fhist.reshape(1, 1);
                            features.cbind(fhist);
                });

        return features;
    }

    /*
     * Gabor
     */
    Gabor::Gabor(const int scales, const int orientation,
            const float downscale, const cv::Size &kernel)
    : scales(scales), orientation(orientation), downscale(downscale), kernel(kernel) {

    }

    Matrix Gabor::compute_impl(const Image &image) const {

        Image gray;
        Matrix features;

        // Gabor Filter
        cv::cvtColor(image, gray, CV_RGB2GRAY);

        // test for 5 scales
        for (int u = 0; u < scales; ++u) {

            float f = 1.414213562;
            float k = M_PI_2 / pow(f, u);

            // test for 8 orientations
            for (int v = 0; v < orientation; ++v) {

                float phi = v * M_PI / orientation;

                // build kernel
                Image rk(kernel.width * 2 + 1, kernel.height * 2 + 1, CV_32F); // real part
                Image ik(kernel.width * 2 + 1, kernel.height * 2 + 1, CV_32F); // imaginary part
                for (int x = -kernel.width; x <= kernel.width; ++x) {
                    for (int y = -kernel.height; y <= kernel.height; ++y) {

                        float _x = x * cos(phi) + y * sin(phi);
                        float _y = -x * sin(phi) + y * cos(phi);
                        float c = exp(-(_x * _x + _y * _y) / (4 * M_PI * M_PI));
                        rk.at<float>(x + kernel.width, y + kernel.height) = c * cos(k * _x);
                        ik.at<float>(x + kernel.width, y + kernel.height) = c * sin(k * _x);
                    }
                }

                // convolve with image
                Image i, r;
                cv::filter2D(gray, r, CV_32F, rk);
                cv::filter2D(gray, i, CV_32F, ik);

                // calc mag
                Image mag;
                cv::pow(i, 2, i);
                cv::pow(r, 2, r);
                cv::pow(i + r, 0.5, mag);

                // downsampling
                cv::resize(mag, mag, cv::Size(), downscale, downscale, CV_INTER_NN);

                // store
                features.rbind(mag);
            }
        }
        return features;
    }

    /*
     * LBP
     */
    LBP::LBP(const cv::Size &block, const cv::Size &stride)
    : block(block), stride(stride) {

    }

    Matrix LBP::compute_impl(const Image &image) const {

        Matrix descriptor;

        cv::Mat gray;
        cv::cvtColor(image, gray, CV_BGR2GRAY);

        Image lbp_image = Image::zeros(gray.rows - 2, gray.cols - 2, CV_32F);

        for (int i = 1; i < gray.rows - 1; i++) {
            for (int j = 1; j < gray.cols - 1; j++) {
                uchar center = gray.at<uchar>(i, j);
                unsigned char code = 0;
                code |= (gray.at<uchar>(i - 1, j - 1) > center) << 7;
                code |= (gray.at<uchar>(i - 1, j) > center) << 6;
                code |= (gray.at<uchar>(i - 1, j + 1) > center) << 5;
                code |= (gray.at<uchar>(i, j + 1) > center) << 4;
                code |= (gray.at<uchar>(i + 1, j + 1) > center) << 3;
                code |= (gray.at<uchar>(i + 1, j) > center) << 2;
                code |= (gray.at<uchar>(i + 1, j - 1) > center) << 1;
                code |= (gray.at<uchar>(i, j - 1) > center) << 0;
                lbp_image.at<float>(i - 1, j - 1) = code;
            }
        }

        const int channels[] = {0};
        const int histSize[] = {255};
        const float range[] = {0, 256};
        const float* ranges[] = {range};

        sldwin(lbp_image, block, stride,
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

    /*
     * CLBP
     */
    CLBP::CLBP(const int radius, const cv::Size &block, const cv::Size &stride)
    : radius(radius), block(block), stride(stride) {

    }

    Matrix CLBP::compute_impl(const Image &image) const {

        Matrix descriptor;

        Image gray;
        cv::cvtColor(image, gray, CV_BGR2GRAY);

        return structure<uchar>(gray, descriptor);
    }

    /*
     * SecondOrder
     */
    SecondOrder::SecondOrder(const Descriptor &descriptor,
            const cv::Size &block, const cv::Size &stride)
    : descriptor(descriptor), block(block), stride(stride) {

    }

    Matrix SecondOrder::compute_impl(const Image &image) const {

        Matrix SPD;
        int count = 0;
        sldwin(cv::Size(image.cols, image.rows), block, stride,
                [&](const cv::Rect & block) {

                    Matrix desc = descriptor.compute(image(block));

                    Matrix M;
                            cv::repeat(desc.t(), 1, desc.cols, M);

                    for (int row = 0; row < M.rows; ++row)
                            M.row(row) = M.row(row).mul(desc);

                        if (SPD.empty())
                                SPD = M;
                        else
                            SPD += M;
                                count++;
                        });
        SPD /= count;

        // recompute sigma as log(lambda)
        cv::Mat v, u, vt;
        cv::SVDecomp(SPD, v, u, vt, cv::SVD::MODIFY_A);
        cv::log(v, v);

        // reconstruct
        SPD = u * cv::Mat::diag(v) * vt;

        // compute descriptor
        Matrix output;
        for (int diag = 0; diag < SPD.cols; ++diag)
            output.rbind(SPD.diag(diag));

        return output;
    }

    /*
     * POEM
     */
    POEM::POEM(const int bins, const int radius, const cv::Size &cell,
            const cv::Size &block, const cv::Size &stride)
    : bins(bins), cell(cell), clbp(radius, block, stride) {

    }

    Matrix POEM::compute_impl(const Image &image) const {

        Matrix descriptor;

        // compute gradient image
        Matrix mag, ori;
        gradient(image, ori, mag);

        int hside = cell.width / 2;
        int vside = cell.height / 2;
        vector<Matrix> hist(bins);
        for (int h = 0; h < bins; ++h)
            hist[h] = Matrix::zeros(mag.rows - vside * 2, mag.cols - hside * 2);

        // compute histogram and replace mid pixel
        sldwin(mag, cell, cv::Size(1, 1), [&](const cv::Rect & window) {

            for (int row = 0; row < window.height; ++row) {
                for (int col = 0; col < window.width; ++col) {

                    float o = ori(window.y + row, window.x + col);
                    float m = mag(window.y + row, window.x + col);

                    o += M_PI_2;
                    int bin = o / ((M_PI + 0.001) / bins);

                    hist[bin](window.x, window.y) += m;
                }
            }
        });

        // compute LBP structure
        for (int o = 0; o < bins; ++o) {
            Matrix output;
            descriptor.cbind(clbp.structure<MType>(hist[o], output));
        }

        return descriptor;
    }

    /*
     * SIFT
     */
    SIFT::SIFT(const int numKeypoints, const int Xstep, const int Ystep) {
        this->sift = cv::xfeatures2d::SIFT::create();

        this->numKeypoints = numKeypoints;
        this->Xstep = Xstep;
        this->Ystep = Ystep;
    }

    Matrix SIFT::compute_impl(const Image& image) const {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;

        //this->sift->detect(image, keypoints);
        int stepX = this->Xstep;
        int stepY = this->Ystep;

        if(stepX == -1) {
            stepX = std::ceil(image.cols / (this->numKeypoints * 2));
        }
        if(stepY == -1) {
            stepY = std::ceil(image.rows / (this->numKeypoints * 2));
        }

        for (int y = stepY; y <= image.rows - stepY; y += (2 * stepY)) {
            for (int x = stepX; x <= image.cols - stepX; x += (2 * stepX)) {
                keypoints.push_back(cv::KeyPoint(float(x), float(y), float(std::min(stepX, stepY))));

            }
        }

        this->sift->compute(image, keypoints, descriptor);
        descriptor.convertTo(descriptor, CV_32F);

        return descriptor;
    }


    /*
     * Dense
     */
//    Dense::Dense(cv::DescriptorExtractor* descriptor, const int step)
//        : descriptor(descriptor), detector(1, 1, 0.1, step, 0, true, false) {
//    }
//
//    Dense::~Dense() {
//        //delete descriptor;
//    }
//
//    Matrix Dense::compute_impl(const Image &image) const {
//        cv::Mat output;
//
//        detector.detect(image, keypoints);
//        descriptor->compute(image, keypoints, output);
//        output.convertTo(output, CV_32F);
//
//        return output;
//    }
}
