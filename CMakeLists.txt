cmake_minimum_required (VERSION 3.12.4)
project (approxcdf VERSION 0.0.1)
set(CMAKE_BUILD_TYPE Release) 

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED False)

set(SRC_FILES ${PROJECT_SOURCE_DIR}/src/tvbs.cpp
              ${PROJECT_SOURCE_DIR}/src/drezner.cpp
              ${PROJECT_SOURCE_DIR}/src/genz.cpp
              ${PROJECT_SOURCE_DIR}/src/plackett.cpp
              ${PROJECT_SOURCE_DIR}/src/bhat.cpp
              ${PROJECT_SOURCE_DIR}/src/bhat_lowdim.cpp
              ${PROJECT_SOURCE_DIR}/src/gge.cpp
              ${PROJECT_SOURCE_DIR}/src/other.cpp
              ${PROJECT_SOURCE_DIR}/src/stdnorm.cpp
              ${PROJECT_SOURCE_DIR}/src/preprocess_rho.cpp
              ${PROJECT_SOURCE_DIR}/src/ldl.cpp
              ${PROJECT_SOURCE_DIR}/src/c_wrapper.cpp)
set(BUILD_SHARED_LIBS True)
add_library(approxcdf SHARED ${SRC_FILES})
target_include_directories(approxcdf PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_target_properties(approxcdf PROPERTIES PUBLIC_HEADER "${PROJECT_SOURCE_DIR}/include/approxcdf.h" SOVERSION 0 VERSION ${PROJECT_VERSION})
add_compile_definitions(APPROXCDF_COMPILE_TIME)

## set to OFF to export all symbols
include(CheckCXXSourceCompiles)
option(HIDE_INTERNAL_SYMBOLS "Set hidden visibility for non-exported symbols" ON)
if (HIDE_INTERNAL_SYMBOLS AND NOT WIN32)
    set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS " -fvisibility=hidden")
    check_cxx_source_compiles(
        "
        [[gnu::visibility(\"default\")]]
        int myfun() {
            return 0;
        }

        int main(int argc, char **argv)
        {
            return myfun();
        }
        "
        SUPPORTS_FVISIBILITY_HIDDEN
    )
    set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
    if (SUPPORTS_FVISIBILITY_HIDDEN)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fvisibility=hidden")
        add_compile_definitions(EXPLICITLTY_EXPORT_SYMBOLS)
    else()
        message(STATUS "Hidden symbols visibility not supported by the compiler, will export all symbols.")
    endif()
else()
    message(STATUS "Making all symbols in the library visible.")
endif()


find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(approxcdf PUBLIC OpenMP::OpenMP_CXX)
endif()

# Compiler optimizations
check_cxx_source_compiles(
    "
    #include <cstddef>
    int main(int argc, char **argv)
    {
        double *__restrict x = nullptr;
        return 0;
    }
    "
    SUPPORTS_RESTRICT
)
if (SUPPORTS_RESTRICT)
    add_compile_definitions(SUPPORTS_RESTRICT=1)
endif()

if (MSVC)
    if (NOT (${CMAKE_CXX_FLAGS_RELEASE} MATCHES "/O2"))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "/O2")
        check_cxx_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_O2
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_O2)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
        endif()
    endif()
    if (NOT (${CMAKE_CXX_FLAGS_RELEASE} MATCHES "/fp:except-"))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "/fp:except-")
        check_cxx_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_FPNOEX
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_FPNOEX)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /fp:except-")
        endif()
    endif()
else()
    if (NOT (${CMAKE_CXX_FLAGS_RELEASE} MATCHES "-O3 "))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "-O3")
        check_cxx_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_O3
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_O3)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
        else()
            set(CMAKE_REQUIRED_FLAGS "-O2")
            check_cxx_source_compiles(
                "
                int main(int argc, char **argv)
                {
                    return 0;
                }
                "
                SUPPORTS_O2
            )
            set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
            if (SUPPORTS_O2)
                set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
            endif()
        endif()
    endif()
    if (NOT (${CMAKE_CXX_FLAGS_RELEASE} MATCHES "-fno-trapping-math"))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "-fno-trapping-math")
        check_cxx_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_FNTM
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_FNTM)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-trapping-math")
        endif()
    endif()
    if (NOT (${CMAKE_CXX_FLAGS_RELEASE} MATCHES "-fno-math-errno"))
        set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
        set(CMAKE_REQUIRED_FLAGS "-fno-math-errno")
        check_cxx_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_FNE
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_FNE)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno")
        endif()
    endif()
endif()

option(USE_MARCH_NATIVE "Build with -march=native" OFF)
if (USE_MARCH_NATIVE AND NOT MSVC)
    set(OLD_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS "-march=native")
    check_cxx_source_compiles(
        "
        int main(int argc, char **argv)
        {
            return 0;
        }
        "
        SUPPORTS_MARCH_NATIVE
    )
    set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
    if (SUPPORTS_MARCH_NATIVE)
        message(STATUS "Adding flag -march=native.")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    else()
        set(CMAKE_REQUIRED_FLAGS "-mcpu=native")
        check_cxx_source_compiles(
            "
            int main(int argc, char **argv)
            {
                return 0;
            }
            "
            SUPPORTS_MCPU_NATIVE
        )
        set(CMAKE_REQUIRED_FLAGS ${OLD_FLAGS})
        if (SUPPORTS_MCPU_NATIVE)
            message(STATUS "Adding flag -mcpu=native.")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=native")
        else()
            message(WARNING "Flags -march=native and -mcpu=native not supported by the compiler.")
        endif()
    endif()
endif()

# Link-time optimization if supported
# https://stackoverflow.com/questions/31355692/how-do-i-enable-link-time-optimization-lto-with-cmake
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED)
if (LTO_SUPPORTED)
    set_property(TARGET approxcdf PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Linkage to BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

include_directories(${BLAS_INCLUDE_DIR})
include_directories(${LAPACK_INCLUDE_DIR})
target_link_libraries(approxcdf PUBLIC ${BLAS_LIBRARIES})
target_link_libraries(approxcdf PUBLIC ${LAPACK_LIBRARIES}) 


include(GNUInstallDirs)

install(
    TARGETS approxcdf
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

configure_file(approxcdf.pc.in approxcdf.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/approxcdf.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

# uninstall target
if (NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )
    add_custom_target(
        uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()
