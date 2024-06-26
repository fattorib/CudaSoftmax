from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="softmax_cuda",
    ext_modules=[
        CUDAExtension(
            "softmax_cuda",
            [
                "csrc/softmax.cpp",
                "csrc/softmax_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "-I csrc/include/kernels"
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
