import os

from distutils.sysconfig import get_config_vars
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = ' '.join(flag for flag in opt.split() if flag != '-Wstrict-prototypes')

setup(
    name='pointops',
    author='Hengshuang Zhao',
    install_requires=['torch', 'torchvision'],
    ext_modules=[
        CUDAExtension(
            'pointops_cuda',
            [
                'src/pointops_api.cpp',
                'src/ballquery/ballquery_cuda.cpp',
                'src/ballquery/ballquery_cuda_kernel.cu',
                'src/featuredistribute/featuredistribute_cuda.cpp',
                'src/featuredistribute/featuredistribute_cuda_kernel.cu',
                'src/grouping/grouping_cuda.cpp',
                'src/grouping/grouping_cuda_kernel.cu',
                'src/grouping_int/grouping_int_cuda.cpp',
                'src/grouping_int/grouping_int_cuda_kernel.cu',
                'src/interpolation/interpolation_cuda.cpp',
                'src/interpolation/interpolation_cuda_kernel.cu',
                'src/knnquery/knnquery_cuda.cpp',
                'src/knnquery/knnquery_cuda_kernel.cu',
                'src/knnquery_heap/knnquery_heap_cuda.cpp',
                'src/knnquery_heap/knnquery_heap_cuda_kernel.cu',
                'src/labelstat/labelstat_cuda.cpp',
                'src/labelstat/labelstat_cuda_kernel.cu',
                'src/sampling/sampling_cuda.cpp',
                'src/sampling/sampling_cuda_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']},
        )
    ],
    packages=['pointops'],
    cmdclass={'build_ext': BuildExtension},
)
