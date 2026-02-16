# ranshaw-fp: Fp field arithmetic (p = 2^255 - 19), all backends

if(NOT FORCE_PORTABLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^([xX]86_64|[aA][mM][dD]64|[aA][rR][mM]64|[aA][aA][rR][cC][hH]64)$")
    set(FP_SRC
        src/x64/fp_frombytes.cpp
        src/x64/fp_invert.cpp
        src/x64/fp_mul.cpp
        src/x64/fp_pow22523.cpp
        src/x64/fp_sq.cpp
        src/x64/fp_sq2.cpp
        src/x64/fp_sqn.cpp
        src/x64/fp_sqrt.cpp
        src/x64/fp_tobytes.cpp
    )
else()
    set(FP_SRC
        src/portable/fp_frombytes.cpp
        src/portable/fp_invert.cpp
        src/portable/fp_mul.cpp
        src/portable/fp_pow22523.cpp
        src/portable/fp_sq.cpp
        src/portable/fp_sq2.cpp
        src/portable/fp_sqn.cpp
        src/portable/fp_sqrt.cpp
        src/portable/fp_tobytes.cpp
    )
endif()

add_library(ranshaw-fp STATIC ${FP_SRC})

target_include_directories(ranshaw-fp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(ranshaw-fp
    PUBLIC ranshaw-common
    PRIVATE ranshaw-build-flags
)
