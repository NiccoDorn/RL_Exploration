cmake_minimum_required(VERSION 3.10)
project(CityPlanningRL CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Source files
set(SOURCES
    main.cpp
    agent.cpp
    env.cpp
    exp_replay.cpp
    utils.cpp
)

# Headers
set(HEADERS
    agent.h
    common.h
    env.h
    exp_replay.h
    utils.h
)

# Create executable
add_executable(city_planning_rl ${SOURCES} ${HEADERS})

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Release build optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG")

    # Additional performance flags
    target_compile_options(city_planning_rl PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -ffast-math
        -funroll-loops
        -finline-functions
        -fomit-frame-pointer
        -flto
    )

    # Link-time optimization
    set_target_properties(city_planning_rl PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )

elseif(MSVC)
    # MSVC optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /Oi /Ot /GL /DNDEBUG")

    target_compile_options(city_planning_rl PRIVATE
        /W4
        /arch:AVX2
        /fp:fast
    )

    set_target_properties(city_planning_rl PROPERTIES
        LINK_FLAGS "/LTCG"
    )
endif()

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")
