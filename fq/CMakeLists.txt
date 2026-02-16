# ranshaw-fq: Fq field arithmetic (q = 2^255 - gamma, Crandall prime), all backends

if(NOT FORCE_PORTABLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^([xX]86_64|[aA][mM][dD]64|[aA][rR][mM]64|[aA][aA][rR][cC][hH]64)$")
    set(FQ_SRC
        src/x64/fq_frombytes.cpp
        src/x64/fq_invert.cpp
        src/x64/fq_mul.cpp
        src/x64/fq_sq.cpp
        src/x64/fq_sq2.cpp
        src/x64/fq_sqn.cpp
        src/x64/fq_sqrt.cpp
        src/x64/fq_tobytes.cpp
    )
else()
    set(FQ_SRC
        src/portable/fq_frombytes.cpp
        src/portable/fq_invert.cpp
        src/portable/fq_mul.cpp
        src/portable/fq_sq.cpp
        src/portable/fq_sq2.cpp
        src/portable/fq_sqn.cpp
        src/portable/fq_sqrt.cpp
        src/portable/fq_tobytes.cpp
    )
endif()

add_library(ranshaw-fq STATIC ${FQ_SRC})

target_include_directories(ranshaw-fq PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(ranshaw-fq
    PUBLIC ranshaw-common
    PRIVATE ranshaw-build-flags
)
