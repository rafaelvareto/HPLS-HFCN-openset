#ifndef PROGRAM_H
#define	PROGRAM_H

#include "feats.hpp"
#include "classifier.hpp"
#include "tasks.hpp"

#include <boost/program_options.hpp>
#include <string>

/*
 * Program arguments
 */
static boost::program_options::variables_map args;

void options(int argc, char **argv) {

    boost::program_options::options_description program("Run PLS hash experiments on face identification datasets");
    program.add_options()
            ("help,h", "print this help")
            ("version,v", "version: 1.0")

            // input output
            ("output,o", boost::program_options::value<std::string>()
            ->default_value("output"), "output file")
            ("dataset,d", boost::program_options::value<std::string>()
            ->default_value("/Users/Vareto/Documents/PLSHface/frgcv1"), "path to dataset images")
            ("learn,l", boost::program_options::value<std::string>()
            ->default_value("/Users/Vareto/Documents/PLSHface/frgcv1/train_4.txt"), "one train set for all tests or one for each test set")
            ("test,e", boost::program_options::value<std::string>()
            ->default_value("/Users/Vareto/Documents/PLSHface/frgcv1/test_4.txt"), "one train set for all tests or one for each test set")
            ("subjects", boost::program_options::value<int>()
            ->default_value(-1), "limit number of subjects to train")

            ("extra-file", boost::program_options::value<std::string>()
            ->default_value(""), "file with extra imges")
            ("extra-path", boost::program_options::value<std::string>()
            ->default_value(""), "path to extra images")
            ("extra-size", boost::program_options::value<int>()
            ->default_value(300), "number of images randomly picked for extra")

            // description
            ("features,f", boost::program_options::value<std::string>()
            ->default_value("clbp,hog16,hog32,gabor,sift"), "features separated by comma (hog,gray,lbp,gabor,...)")

            // identification
            ("factors,s", boost::program_options::value<int>()
            ->default_value(20), "factors for PLS identification")

            // method
            ("models,m", boost::program_options::value<std::string>()
            ->default_value("1"), "one or more number of models to test separated by comma (10,50,100,150,200)")
            ("mfactors,r", boost::program_options::value<std::string>()
            ->default_value("10"), "factors for LSH models")
            ("ndims", boost::program_options::value<std::string>()
            ->default_value("20"), "num. dimensions on min. hash")
            ("type", boost::program_options::value<std::string>()
            ->default_value("full"), "type of feature selection (full,weights,beta,vip)")
            ("retrain", boost::program_options::value<bool>()
            ->default_value(false), "whether to retrain after feat. selection")
            ("combination", boost::program_options::value<std::string>()
            ->default_value("sum"), "whether to sum or product to combine scores")
            ("base", boost::program_options::value<int>()
            ->default_value(2), "number of values that each bit in the code assume")

            // execution
            ("threads,t", boost::program_options::value<int>()
            ->default_value(4), "number of threads")
            ("exec,c", boost::program_options::value<std::string>()
            ->default_value("train"), "execution type (train, test, view)")

            /*
             * PubFig: 80x60
             * LFW/Youtube: 110x70
             * LFW3d 90x90
             * FERET: 110x100
             * FRGC: 138x160
             */
            // image input
            ("crop-width", boost::program_options::value<int>()
            ->default_value(138), "crop width")
            ("crop-height", boost::program_options::value<int>()
            ->default_value(160), "crop height")
            ("resize-width", boost::program_options::value<int>()
            ->default_value(128), "resize crop to the given width")
            ("resize-height", boost::program_options::value<int>()
            ->default_value(128), "resize crop to the given height");

    boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv)
            .options(program)
            .run(), args);
    boost::program_options::notify(args);

    if (args.count("help") || args.count("version")) {
        std::cout << program << std::endl;
        exit(EXIT_FAILURE);
    }
}

cv::Mat read(const std::string &filename) {

    // read image
    cv::Mat image;
    image = cv::imread(filename, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cout << "image " << filename << " not found!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Rect crop;
    crop.width = args["crop-width"].as<int>();
    crop.height = args["crop-height"].as<int>();
    crop.x = image.cols / 2 - crop.width / 2;
    crop.y = image.rows / 2 - crop.height / 2;
    image = image(crop);

    cv::resize(image, image, cv::Size(args["resize-width"].as<int>(),
            args["resize-height"].as<int>()));

    return image;
}

std::unordered_map<std::string,
        std::function<cvtools::Matrix(const cvtools::Image &) >> descriptors = {
    {"smoke", [](const cvtools::Image & image) {
            static cvtools::Hog desc(cv::Size(128, 128), cv::Size(16, 16), cv::Size(64, 64));
            return desc.compute(image);
        }},
    {"hog", [](const cvtools::Image & image) {
            static cvtools::Hog desc(cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8));
            return desc.compute(image);
        }},
    {"hog16", [](const cvtools::Image & image) {
            static cvtools::Hog desc(cv::Size(16, 16), cv::Size(4, 4), cv::Size(8, 8));
            return desc.compute(image);
        }},
    {"hog32", [](const cvtools::Image & image) {
            static cvtools::Hog desc(cv::Size(32, 32), cv::Size(8, 8), cv::Size(16, 16));
            return desc.compute(image);
        }},
    {"hist", [](const cvtools::Image & image) {
            static cvtools::CHist desc;
            return desc.compute(image);
        }},
    {"gray", [](const cvtools::Image & image) {
            static cvtools::Mean desc;
            return desc.compute(image);
        }},
    {"lbp", [](const cvtools::Image & image) {
            static cvtools::LBP desc;
            return desc.compute(image);
        }},
    {"gabor", [](const cvtools::Image & image) {
            static cvtools::Gabor desc;
            return desc.compute(image);
        }},
    {"clbp", [](const cvtools::Image & image) {
            static cvtools::CLBP desc;
            return desc.compute(image);
        }},
    {"poem", [](const cvtools::Image & image) {
            static cvtools::POEM desc;
            return desc.compute(image);
        }},
    {"sift", [](const cvtools::Image & image) {
            static cvtools::SIFT desc;
            return desc.compute(image);
        }},
#if 0
    {"surf", [](const cvtools::Image & image) {
            static cvtools::Dense desc(new cv::SURF);
            return desc.compute(image);
        }},
    {"orb", [](const cvtools::Image & image) {
            static cvtools::Dense desc(new cv::ORB);
            return desc.compute(image);
        }},
    {"freak", [](const cvtools::Image & image) {
            cvtools::Dense desc(new cv::FREAK);
            return desc.compute(image);
        }},
    {"brisk", [](const cvtools::Image & image) {
            static cvtools::Dense desc(new cv::BRISK);
            return desc.compute(image);
        }},
#endif
    // second-order
    {"2nd", [](const cvtools::Image & image) {
            static cvtools::Hog desc(cv::Size(16, 16), cv::Size(8, 8), cv::Size(4, 4));
            static cvtools::SecondOrder snd(desc, cv::Size(16, 16), cv::Size(2, 2));
            return snd.compute(image);
        }}
};


std::unordered_map<std::string, size_t> descriptor_time;

cvtools::Matrix descript(const cvtools::Image &image) {

    static bool printed = false;

    // cvtools::Timer<std::chrono::microseconds> timer;
    cvtools::Matrix descriptor;

    std::string feature;
    std::istringstream parser(args["features"].as<std::string>());
    while (getline(parser, feature, ',')) {

        if (descriptor_time.find(feature) == descriptor_time.end())
            descriptor_time[feature] = 0;

        // timer.tick();
        cvtools::Matrix d = descriptors[feature](image);
        // descriptor_time[feature] += timer.tock();

        descriptor.cbind(d);
    }

#pragma omp critical
    {
        if (!printed) {
            std::cout << "* descriptor size: " << descriptor.cols << std::endl;
            printed = true;
        }
    }

    return descriptor;
}

std::vector<cvtools::Set> read_set(const std::string &dir, const std::string &sets) {

    std::vector<cvtools::Set> output;

    std::string set;
    std::istringstream parser(sets);
    while (getline(parser, set, ','))
        output.push_back(cvtools::Set(dir, set));

    return output;
}

std::vector<int> readvec(const std::string &vec) {

    std::vector<int> output;

    std::string i;
    std::istringstream parser(vec);
    while (getline(parser, i, ','))
        output.push_back(stoi(i));
    sort(output.begin(), output.end());

    return output;
}

#endif
