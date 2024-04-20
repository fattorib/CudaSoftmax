#include <random>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cmath>

#ifndef UTILS_H

#define UTILS_H

// returns 1D vector with values from U(-1,1)
void fill_matrix(std::vector<float> &vec)
{
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());

    float mean, stddev;
    mean = 0.0;
    stddev = 1.0;

    std::generate(vec.begin(), vec.end(), [&generator, &mean, &stddev]
                  { return std::normal_distribution<float>(mean, stddev)(generator); });
}

// computes (||vec1 - vec2||_2 / ||vec1||_2)
float relative_error(std::vector<float> &vec1, std::vector<float> &vec2)
{
    float denom = 0.0f;
    float num = 0.0f;

    int numel = vec1.size();

    for (int i = 0; i < numel; i++)
    {
        num += pow(vec1[i] - vec2[i], 2.0);
        denom += pow(vec1[i], 2.0);
    }

    return pow(num, 0.5) / pow(denom, 0.5);
}

// computes max abs diff
float max_abs_diff(std::vector<float> &vec1, std::vector<float> &vec2)
{

    float maxdiff = -1 * INFINITY;

    int numel = vec1.size();

    for (int i = 0; i < numel; i++)
    {
        maxdiff = std::max(std::fabs(vec1[i] - vec2[i]), maxdiff);

    }

    return maxdiff;
}

#endif