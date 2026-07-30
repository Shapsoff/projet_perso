cmake_minimum_required(VERSION 3.15)

project(poker_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)

# ── Flags d'optimisation ──────────────────────────────────────────────────────
if(MSVC)
    set(RELEASE_FLAGS /O2 /Ob3 /arch:AVX2 /fp:fast /GL /DNDEBUG /EHsc /MP)
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE    "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
    message(STATUS "Compilateur : MSVC")
else()
    set(RELEASE_FLAGS -O3 -march=native -ffast-math -flto -DNDEBUG)
    message(STATUS "Compilateur : GCC/Clang")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── pybind11 ──────────────────────────────────────────────────────────────────
if(NOT pybind11_FOUND)
    find_package(pybind11 QUIET)
endif()

if(NOT pybind11_FOUND)
    foreach(PY_CMD python python3)
        execute_process(
            COMMAND ${PY_CMD} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(PYBIND11_CMAKE_DIR)
            message(STATUS "pybind11 trouve via ${PY_CMD} : ${PYBIND11_CMAKE_DIR}")
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
            find_package(pybind11 REQUIRED)
            break()
        endif()
    endforeach()
endif()

# ── OpenMP ────────────────────────────────────────────────────────────────────
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP active")
else()
    message(STATUS "OpenMP non trouve — single-thread")
endif()

# ── Sources communes phase 1 ──────────────────────────────────────────────────
set(PHASE1_SOURCES
    hand_eval.cpp
    monte_carlo.cpp
)

# ── Executable de test phase 1 ────────────────────────────────────────────────
add_executable(test_hand_eval
    ${PHASE1_SOURCES}
    test_hand_eval.cpp
)
target_compile_options(test_hand_eval PRIVATE ${RELEASE_FLAGS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_hand_eval PRIVATE OpenMP::OpenMP_CXX)
endif()

# ── Executable de test phase 2 (EHS standalone) ───────────────────────────────
add_executable(ehs_test
    ${PHASE1_SOURCES}
    ehs.cpp
    ehs_test.cpp
)
target_compile_options(ehs_test PRIVATE ${RELEASE_FLAGS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(ehs_test PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(ehs_test PRIVATE USE_OPENMP)
endif()

# ── Module Python poker_engine (phase 1 + phase 2) ───────────────────────────
if(pybind11_FOUND)
    pybind11_add_module(poker_engine
        ${PHASE1_SOURCES}
        ehs.cpp
        bindings.cpp
    )
    target_compile_options(poker_engine PRIVATE ${RELEASE_FLAGS})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(poker_engine PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(poker_engine PRIVATE USE_OPENMP)
    endif()
    message(STATUS "Module Python poker_engine : OK")
else()
    message(WARNING "pybind11 non trouve — module Python non compile. Lancer : pip install pybind11")
endif()

# ── Resume ────────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "Cibles disponibles :")
message(STATUS "  test_hand_eval — tests phase 1 (hand_eval + monte_carlo)")
message(STATUS "  ehs_test       — tests phase 2 (EHS Calculator)")
if(pybind11_FOUND)
    message(STATUS "  poker_engine   — module Python complet")
endif()
message(STATUS "")
