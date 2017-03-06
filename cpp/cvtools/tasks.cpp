#include "util.hpp"
#include "tasks.hpp"

#include <fstream>
#include <unordered_map>

using namespace std;

namespace cvtools {

    /*
     * Class
     */
    Class::Class() {

    }

    Class::Class(const std::string &id) : id(id) {

    }

    Class::Class(const std::string &id, const std::vector<std::string> &samples)
    : id(id), samples(samples) {

    }

    /*
     * Set
     */
    Set::Set() {

    }

    Set::Set(const std::string &path, const std::string &list) : file(list) {

        ifstream input(file);
        assert(input.is_open());

        unordered_map<string, vector < string>> set;

        while (input.good()) {

            string subject;
            string sample;

            input >> subject;
            input >> sample;

            if (subject.empty() && sample.empty())
                break;

            /* replace extension */
            size_t pos = sample.find(".png");
            assert(pos != string::npos);
            sample.replace(pos, pos + 3, ".cropped.png");

            set[subject].push_back(path + "/" + subject + "/cropped/" + sample);
            //set[subject].push_back(path + "/" + subject + "/" + sample);
        }
        for (const pair<string, vector < string>> &p : set)
            classes.push_back(Class(p.first, p.second));
    }

    void Set::save(const string &filename) {

        file = filename;

        ofstream output(file);

        for (const Class &c : classes)
            for (const string &s : c.samples) {
                string f = s.substr(s.find_last_of('/') + 1, string::npos);
                output << c.id << ' ' << f << '\n';
            }
    }

    void Set::gen(const std::string &path, const int plearn,
            Set &learn, Set &test) {

        for (const string &subject : ls_folders(path)) {

            vector<string> samples = ls_files(subject);

            random_shuffle(samples.begin(), samples.end());

            int i;
            int nlearn = ceil(plearn * samples.size());
            learn.classes.push_back(Class(subject));
            for (i = 0; i < nlearn; ++i)
                learn.classes.back().samples.push_back(samples[i]);

            test.classes.push_back(Class(subject));
            for (; i < samples.size(); ++i)
                test.classes.back().samples.push_back(samples[i]);
        }

        assert(learn.classes.size() == test.classes.size());
    }

    void Set::gen(const std::string &path, const int nlearn, const int ntest,
            Set &learn, Set &test) {

        for (const string &subject_dir : ls_folders(path)) {

            vector<string> samples = ls_files(subject_dir);

            string subject = subject_dir.substr(
                    subject_dir.find_last_of('/') + 1, string::npos);

            random_shuffle(samples.begin(), samples.end());

            int i;
            learn.classes.push_back(Class(subject));
            for (i = 0; i < nlearn; ++i)
                learn.classes.back().samples.push_back(samples[i]);

            test.classes.push_back(Class(subject));
            for (; i < samples.size(); ++i) {
                test.classes.back().samples.push_back(samples[i]);
                if (test.classes.back().samples.size() >= ntest)
                    break;
            }
        }

        assert(learn.classes.size() == test.classes.size());
    }
}
