# ranshaw-ec-divisors: polynomial and EC-divisor arithmetic over Fp/Fq

set(EC_DIVISORS_SRC
    src/poly.cpp
    src/divisor.cpp
    src/divisor_eval.cpp
)

# Conditionally add AVX2 backend
if(ENABLE_AVX2)
    set(EC_DIVISORS_AVX2_SRC
        src/x64/avx2/divisor_eval_avx2.cpp
    )
    list(APPEND EC_DIVISORS_SRC ${EC_DIVISORS_AVX2_SRC})

    if(MSVC)
        set_source_files_properties(${EC_DIVISORS_AVX2_SRC} PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    else()
        set_source_files_properties(${EC_DIVISORS_AVX2_SRC} PROPERTIES
            COMPILE_FLAGS "-mavx2 ${_MINGW_SIMD_FIX}")
    endif()
endif()

# Conditionally add AVX-512 IFMA backend
if(ENABLE_AVX512)
    set(EC_DIVISORS_IFMA_SRC
        src/x64/ifma/divisor_eval_ifma.cpp
    )
    list(APPEND EC_DIVISORS_SRC ${EC_DIVISORS_IFMA_SRC})

    if(MSVC)
        set_source_files_properties(${EC_DIVISORS_IFMA_SRC} PROPERTIES COMPILE_FLAGS "/arch:AVX512")
    else()
        set_source_files_properties(${EC_DIVISORS_IFMA_SRC} PROPERTIES
            COMPILE_FLAGS "-mavx512f -mavx512ifma ${_MINGW_SIMD_FIX}")
    endif()
endif()

add_library(ranshaw-ec-divisors STATIC ${EC_DIVISORS_SRC})

target_include_directories(ranshaw-ec-divisors PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Add top-level include dir for ranshaw_cpuid.h (needed by SIMD dispatch)
if(ENABLE_AVX2 OR ENABLE_AVX512)
    target_include_directories(ranshaw-ec-divisors PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )
endif()

target_link_libraries(ranshaw-ec-divisors
    PUBLIC ranshaw-ran ranshaw-shaw ranshaw-fp ranshaw-fq ranshaw-common
    PRIVATE ranshaw-build-flags
)
