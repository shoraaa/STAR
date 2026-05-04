from setuptools import Extension, setup
import pybind11


ext_modules = [
    Extension(
        "_STAR",
        ["STAR/STAR.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++",
    )
]


setup(
    name="STAR",
    version="0.1.0",
    packages=["STAR"],
    ext_package="STAR",
    ext_modules=ext_modules,
)
