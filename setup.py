from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import sys, os, subprocess
from os import environ

## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext ):
    def build_extensions(self):
        is_msvc = self.compiler.compiler_type == "msvc"
        is_windows = sys.platform[:3] == "win"

        if is_msvc:
            for e in self.extensions:
                e.extra_compile_args = ['/openmp', '/O2', '/std:c++14']
        else:
            if not self.check_for_variable_dont_set_march() and not self.is_arch_in_cflags():
                self.add_march_native()
            self.set_cxxstd()
            self.add_openmp_linkage()
            self.add_restrict_qualifier()
            self.add_O3()
            self.add_no_math_errno()
            self.add_no_trapping_math()
            if not is_windows:
                self.add_link_time_optimization()

        build_ext.build_extensions(self)

    def check_for_variable_dont_set_march(self):
        return "DONT_SET_MARCH" in os.environ

    def add_march_native(self):
        arg_march_native = "-march=native"
        arg_mcpu_native = "-mcpu=native"
        if self.test_supports_compile_arg(arg_march_native):
            for e in self.extensions:
                e.extra_compile_args.append(arg_march_native)
        elif self.test_supports_compile_arg(arg_mcpu_native):
            for e in self.extensions:
                e.extra_compile_args.append(arg_mcpu_native)

    def add_link_time_optimization(self):
        arg_lto = "-flto"
        if self.test_supports_compile_arg(arg_lto):
            for e in self.extensions:
                e.extra_compile_args.append(arg_lto)
                e.extra_link_args.append(arg_lto)

    def add_no_math_errno(self):
        arg_fnme = "-fno-math-errno"
        if self.test_supports_compile_arg(arg_fnme):
            for e in self.extensions:
                e.extra_compile_args.append(arg_fnme)

    def add_no_trapping_math(self):
        arg_fntm = "-fno-trapping-math"
        if self.test_supports_compile_arg(arg_fntm):
            for e in self.extensions:
                e.extra_compile_args.append(arg_fntm)

    def set_cxxstd(self):
        arg_std17 = "-std=c++17"
        arg_std14 = "-std=gnu++14"
        if self.test_supports_compile_arg(arg_std17):
            for e in self.extensions:
                e.extra_compile_args.append(arg_std17)
        elif self.test_supports_compile_arg(arg_std14):
            for e in self.extensions:
                e.extra_compile_args.append(arg_std14)
        else:
            print("C++17 standard not supported. Package might fail to compile.")


    def add_openmp_linkage(self):
        arg_omp1 = "-fopenmp"
        arg_omp2 = "-qopenmp"
        arg_omp3 = "-xopenmp"
        arg_omp4 = "-fiopenmp"
        args_apple_omp = ["-Xclang", "-fopenmp", "-lomp"]
        args_apple_omp2 = ["-Xclang", "-fopenmp", "-L/usr/local/lib", "-lomp", "-I/usr/local/include"]
        if self.test_supports_compile_arg(arg_omp1, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp1)
                e.extra_link_args.append(arg_omp1)
        elif (sys.platform[:3].lower() == "dar") and self.test_supports_compile_arg(args_apple_omp, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-lomp"]
        elif (sys.platform[:3].lower() == "dar") and self.test_supports_compile_arg(args_apple_omp2, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-L/usr/local/lib", "-lomp"]
                e.include_dirs += ["/usr/local/include"]
        elif self.test_supports_compile_arg(arg_omp2, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp2)
                e.extra_link_args.append(arg_omp2)
        elif self.test_supports_compile_arg(arg_omp3, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp3)
                e.extra_link_args.append(arg_omp3)
        elif self.test_supports_compile_arg(arg_omp4, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp4)
                e.extra_link_args.append(arg_omp4)
        else:
            set_omp_false()

    def add_O3(self):
        O3 = "-O3"
        if self.test_supports_compile_arg(O3):
            for e in self.extensions:
                e.extra_compile_args.append(O3)

    def test_supports_compile_arg(self, comm, with_omp=False):
        is_supported = False
        try:
            if not hasattr(self.compiler, "compiler_cxx"):
                return False
            if not isinstance(comm, list):
                comm = [comm]
            print("--- Checking compiler support for option '%s'" % " ".join(comm))
            fname = "approxcdf_compiler_testing.cpp"
            with open(fname, "w") as ftest:
                ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
            try:
                if not isinstance(self.compiler.compiler_cxx, list):
                    cmd = list(self.compiler.compiler_cxx)
                else:
                    cmd = self.compiler.compiler_cxx
            except Exception:
                cmd = self.compiler.compiler_cxx
            val_good = subprocess.call(cmd + [fname])
            if with_omp:
                with open(fname, "w") as ftest:
                    ftest.write(u"#include <omp.h>\nint main(int argc, char**argv) {return 0;}\n")
            try:
                val = subprocess.call(cmd + comm + [fname])
                is_supported = (val == val_good)
            except Exception:
                is_supported = False
        except Exception:
            pass
        try:
            os.remove(fname)
        except Exception:
            pass
        return is_supported

    def is_arch_in_cflags(self):
        arch_flags = '-march -mtune -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2 -mcpu'.split()
        for env_var in ("CFLAGS", "CXXFLAGS"):
            if env_var in os.environ:
                for flag in arch_flags:
                    if flag in os.environ[env_var]:
                        return True

        return False

    def add_restrict_qualifier(self):
        supports_restrict = False
        try:
            if not hasattr(self.compiler, "compiler_cxx"):
                return None
            print("--- Checking compiler support for '__restrict' qualifier")
            fname = "approxcdf_compiler_testing.cpp"
            with open(fname, "w") as ftest:
                ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
            try:
                if not isinstance(self.compiler.compiler_cxx, list):
                    cmd = list(self.compiler.compiler_cxx)
                else:
                    cmd = self.compiler.compiler_cxx
            except Exception:
                cmd = self.compiler.compiler_cxx
            val_good = subprocess.call(cmd + [fname])
            try:
                with open(fname, "w") as ftest:
                    ftest.write(u"int main(int argc, char**argv) {double *__restrict x = 0; return 0;}\n")
                val = subprocess.call(cmd + [fname])
                supports_restrict = (val == val_good)
            except Exception:
                return None
        except Exception:
            pass
        try:
            os.remove(fname)
        except Exception:
            pass
        
        if supports_restrict:
            for e in self.extensions:
                e.define_macros += [("SUPPORTS_RESTRICT", "1")]

setup(
    name  = "approxcdf",
    packages = ["approxcdf"],
    version = '0.0.1',
    description = 'Approximations for fast CDF calculation of MVN distributions',
    author = 'David Cortes',
    author_email = 'david.cortes.rivera@gmail.com',
    url = 'https://github.com/david-cortes/approxcdf',
    keywords = ['cdf', 'tvbs', 'multivariate-normal', 'mvn'],
    cmdclass = {'build_ext': build_ext_subclass},
    ext_modules = [Extension(
                            "approxcdf._cpp_wrapper",
                            sources=["approxcdf/cpp_wrapper.pyx",
                                     "src/tvbs.cpp",
                                     "src/drezner.cpp",
                                     "src/genz.cpp",
                                     "src/plackett.cpp",
                                     "src/bhat.cpp",
                                     "src/bhat_lowdim.cpp",
                                     "src/gge.cpp",
                                     "src/other.cpp",
                                     "src/stdnorm.cpp",
                                     "src/preprocess_rho.cpp",
                                     "src/ldl.cpp"],
                            include_dirs=[np.get_include(), ".", "./src", "./approxcdf"],
                            language="c++",
                            install_requires = ["numpy", "cython", "scipy"],
                            define_macros = [("FOR_PYTHON", None)]
                            )]
)
