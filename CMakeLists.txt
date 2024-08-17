cmake_minimum_required(VERSION 3.11)
project(particles)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Group number
set(GROUP_NAME "None" CACHE STRING "Your group name as it appears on bCourses (no spaces)")

# Packaging system
if (NOT ${GROUP_NAME} STREQUAL None)
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "particles")
    install(FILES gpu.cu DESTINATION .)
    install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
    include(CPack)
endif ()

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(gpu main.cu gpu.cu common.h)
target_compile_features(gpu PRIVATE cxx_std_14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
set_target_properties(gpu PROPERTIES CUDA_ARCHITECTURES "80")

# Copy the job scripts
configure_file(job-gpu job-gpu COPYONLY)
