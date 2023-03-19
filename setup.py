from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import sys, os, subprocess
from os import environ

found_omp = True
def set_omp_false():
    global found_omp
    found_omp = False

## Modify this to make the output of the compilation tests more verbose
silent_tests = not (("verbose" in sys.argv)
                    or ("-verbose" in sys.argv)
                    or ("--verbose" in sys.argv))

## Workaround for python<=3.9 on windows
try:
    EXIT_SUCCESS = os.EX_OK
except AttributeError:
    EXIT_SUCCESS = 0

## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext ):
    def build_extensions(self):
        is_msvc = self.compiler.compiler_type == "msvc"
        is_windows = sys.platform[:3] == "win"

        if is_msvc:
            for e in self.extensions:
                e.extra_compile_args = ['/openmp', '/O2', '/GL', '/std:c++14']
        else:
            if not self.check_for_variable_dont_set_march() and not self.check_cflags_contain_arch():
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

    def check_cflags_contain_arch(self):
        if ("CFLAGS" in os.environ) or ("CXXFLAGS" in os.environ):
            has_cflags = "CFLAGS" in os.environ
            has_cxxflags = "CXXFLAGS" in os.environ
            arch_list = [
                "-march", "-mcpu", "-mtune", "-msse", "-msse2", "-msse3",
                "-mssse3", "-msse4", "-msse4a", "-msse4.1", "-msse4.2",
                "-mavx", "-mavx2", "-mavx512"
            ]
            for flag in arch_list:
                if has_cflags and flag in os.environ["CFLAGS"]:
                    return True
                if has_cxxflags and flag in os.environ["CXXFLAGS"]:
                    return True
        return False

    def check_for_variable_dont_set_march(self):
        return "DONT_SET_MARCH" in os.environ

    def add_march_native(self):
        args_march_native = ["-march=native", "-mcpu=native"]
        for arg_march_native in args_march_native:
            if self.test_supports_compile_arg(arg_march_native):
                for e in self.extensions:
                    e.extra_compile_args.append(arg_march_native)
                break

    def add_link_time_optimization(self):
        args_lto = ["-flto=auto", "-flto"]
        for arg_lto in args_lto:
            if self.test_supports_compile_arg(arg_lto):
                for e in self.extensions:
                    e.extra_compile_args.append(arg_lto)
                    e.extra_link_args.append(arg_lto)
                break

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

    def add_O3(self):
        O3 = "-O3"
        if self.test_supports_compile_arg(O3):
            for e in self.extensions:
                e.extra_compile_args.append(O3)

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
        arg_omp2 = "-fopenmp=libomp"
        args_omp3 = ["-fopenmp=libomp", "-lomp"]
        arg_omp4 = "-qopenmp"
        arg_omp5 = "-xopenmp"
        is_apple = sys.platform[:3].lower() == "dar"
        args_apple_omp = ["-Xclang", "-fopenmp", "-lomp"]
        args_apple_omp2 = ["-Xclang", "-fopenmp", "-L/usr/local/lib", "-lomp", "-I/usr/local/include"]
        has_brew_omp = False
        if is_apple:
            res_brew_pref = subprocess.run(["brew", "--prefix", "libomp"], capture_output=silent_tests)
            if res_brew_pref.returncode == EXIT_SUCCESS:
                has_brew_omp = True
                brew_omp_prefix = res_brew_pref.stdout.decode().strip()
                args_apple_omp3 = ["-Xclang", "-fopenmp", f"-L{brew_omp_prefix}/lib", "-lomp", f"-I{brew_omp_prefix}/include"]


        if self.test_supports_compile_arg(arg_omp1, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp1)
                e.extra_link_args.append(arg_omp1)
        elif is_apple and self.test_supports_compile_arg(args_apple_omp, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-lomp"]
        elif is_apple and self.test_supports_compile_arg(args_apple_omp2, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-L/usr/local/lib", "-lomp"]
                e.include_dirs += ["/usr/local/include"]
        elif is_apple and has_brew_omp and self.test_supports_compile_arg(args_apple_omp3, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += [f"-L{brew_omp_prefix}/lib", "-lomp"]
                e.include_dirs += [f"{brew_omp_prefix}/include"]
        elif self.test_supports_compile_arg(arg_omp2, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-fopenmp=libomp"]
                e.extra_link_args += ["-fopenmp"]
        elif self.test_supports_compile_arg(args_omp3, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-fopenmp=libomp"]
                e.extra_link_args += ["-fopenmp", "-lomp"]
        elif self.test_supports_compile_arg(arg_omp4, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp4)
                e.extra_link_args.append(arg_omp4)
        elif self.test_supports_compile_arg(arg_omp5, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp5)
                e.extra_link_args.append(arg_omp5)
        else:
            set_omp_false()

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
            if with_omp:
                with open(fname, "w") as ftest:
                    ftest.write(u"#include <omp.h>\nint main(int argc, char**argv) {return 0;}\n")
            try:
                val = subprocess.run(cmd + comm + [fname], capture_output=silent_tests).returncode
                is_supported = (val == EXIT_SUCCESS)
            except Exception:
                is_supported = False
        except Exception:
            pass
        try:
            os.remove(fname)
        except Exception:
            pass
        return is_supported

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
            try:
                with open(fname, "w") as ftest:
                    ftest.write(u"int main(int argc, char**argv) {double *__restrict x = 0; return 0;}\n")
                val = subprocess.run(cmd + comm + [fname], capture_output=silent_tests).returncode
                is_supported = (val == EXIT_SUCCESS)
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

if not found_omp:
    omp_msg  = "\n\n\nCould not detect OpenMP. Package will be built without multi-threading capabilities. "
    omp_msg += " To enable multi-threading, first install OpenMP"
    if (sys.platform[:3] == "dar"):
        omp_msg += " - for macOS: 'brew install libomp'\n"
    else:
        omp_msg += " modules for your compiler. "
    
    omp_msg += "Then reinstall this package from scratch: 'pip install --upgrade --no-deps --force-reinstall git+https://www.github.com/david-cortes/approxcdf.git'.\n"
    warnings.warn(omp_msg)
