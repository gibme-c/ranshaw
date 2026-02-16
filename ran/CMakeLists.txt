# ranshaw-ran: Ran curve operations (coordinates in Fp), all backends

if(NOT FORCE_PORTABLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^([xX]86_64|[aA][mM][dD]64|[aA][rR][mM]64|[aA][aA][rR][cC][hH]64)$")
    set(RAN_SRC
        src/x64/ran_dbl.cpp
        src/x64/ran_madd.cpp
        src/x64/ran_add.cpp
        src/x64/ran_tobytes.cpp
        src/x64/ran_frombytes.cpp
        src/x64/ran_scalarmult.cpp
        src/x64/ran_scalarmult_vartime.cpp
        src/x64/ran_msm_vartime.cpp
        src/x64/ran_map_to_curve.cpp
    )
else()
    set(RAN_SRC
        src/portable/ran_dbl.cpp
        src/portable/ran_madd.cpp
        src/portable/ran_add.cpp
        src/portable/ran_tobytes.cpp
        src/portable/ran_frombytes.cpp
        src/portable/ran_scalarmult.cpp
        src/portable/ran_scalarmult_vartime.cpp
        src/portable/ran_msm_vartime.cpp
        src/portable/ran_map_to_curve.cpp
    )
endif()

# AVX2 backend sources
if(ENABLE_AVX2)
    set(RAN_AVX2_SRC
        src/x64/avx2/ran_scalarmult.cpp
        src/x64/avx2/ran_scalarmult_vartime.cpp
        src/x64/avx2/ran_msm_vartime.cpp
    )
    list(APPEND RAN_SRC ${RAN_AVX2_SRC})

    if(MSVC)
        set_source_files_properties(${RAN_AVX2_SRC} PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    else()
        set_source_files_properties(${RAN_AVX2_SRC} PROPERTIES
            COMPILE_FLAGS "-mavx2 ${_MINGW_SIMD_FIX}")
    endif()
endif()

# IFMA backend sources
if(ENABLE_AVX512)
    set(RAN_IFMA_SRC
        src/x64/ifma/ran_scalarmult.cpp
        src/x64/ifma/ran_scalarmult_vartime.cpp
        src/x64/ifma/ran_msm_vartime.cpp
    )
    list(APPEND RAN_SRC ${RAN_IFMA_SRC})

    if(MSVC)
        set_source_files_properties(${RAN_IFMA_SRC} PROPERTIES COMPILE_FLAGS "/arch:AVX512")
    else()
        set_source_files_properties(${RAN_IFMA_SRC} PROPERTIES
            COMPILE_FLAGS "-mavx512f -mavx512ifma ${_MINGW_SIMD_FIX}")
    endif()
endif()

add_library(ranshaw-ran STATIC ${RAN_SRC})

target_include_directories(ranshaw-ran PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(ranshaw-ran
    PUBLIC ranshaw-fp ranshaw-common
    PRIVATE ranshaw-build-flags
)
