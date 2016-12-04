#ifndef TASKS_H
#define	TASKS_H

#include <vector>
#include <string>

#include <boost/asio.hpp>

namespace cvtools {

    /*
     * Class
     */
    class Class {
    public:

        Class();

        Class(const std::string &id);

        Class(const std::string &id, const std::vector<std::string> &samples);

        std::string id;
        std::vector<std::string> samples;
    };

    /*
     * Set
     */
    class Set {
    public:

        static void gen(const std::string &path, const int plearn,
                Set &learn, Set &test);

        static void gen(const std::string &path, const int nlearn, const int ntest,
                Set &learn, Set &test);

        Set();

        Set(const std::string &path, const std::string &list);

        void save(const std::string &filename);

        std::string file;
        std::vector<Class> classes;
    };
}

#endif
