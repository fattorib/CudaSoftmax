#include <vector>
#include <cmath>
#include <algorithm>
#include <stdio.h>

#ifndef SOFTMAX_CPU_H

#define SOFTMAX_CPU_H

// standard CPU softmax on an row-major M x N vector
template <int rows, int cols>
void softmaxCpu(std::vector<float> &x, std::vector<float> &out)
{

    #pragma omp parallel for
    for (int r = 0; r < rows; r++)
    {

        float rowmax = -1* INFINITY;
        for (int c = 0; c < cols; c++){
            rowmax = std::max(x[r*cols + c], rowmax);
        }

        float denom = 0;

        // compute denom
        float tmp;
        for (int c = 0; c < cols; c++)
        {   
            tmp = std::exp(x[r*cols + c] - rowmax);
            denom += tmp;
            out[r*cols + c] = tmp;
        }

        for (int c = 0; c < cols; c++)
        {
            out[r*cols + c] /= denom;
        }
    }
}

#endif