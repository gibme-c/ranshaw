# ranshaw-shaw: Shaw curve operations (coordinates in Fq), all backends

if(NOT FORCE_PORTABLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^([xX]86_64|[aA][mM][dD]64|[aA][rR][mM]64|[aA][aA][rR][cC][hH]64)$")
    set(SHAW_SRC
        src/x64/shaw_dbl.cpp
        src/x64/shaw_madd.cpp
        src/x64/shaw_add.cpp
        src/x64/shaw_tobytes.cpp
        src/x64/shaw_frombytes.cpp
        src/x64/shaw_scalarmult.cpp
        src/x64/shaw_scalarmult_vartime.cpp
        src/x64/shaw_msm_vartime.cpp
        src/x64/shaw_map_to_curve.cpp
    )
else()
    set(SHAW_SRC
        src/portable/shaw_dbl.cpp
        src/portable/shaw_madd.cpp
        src/portable/shaw_add.cpp
        src/portable/shaw_tobytes.cpp
        src/portable/shaw_frombytes.cpp
        src/portable/shaw_scalarmult.cpp
        src/portable/shaw_scalarmult_vartime.cpp
        src/portable/shaw_msm_vartime.cpp
        src/portable/shaw_map_to_curve.cpp
    )
endif()

# AVX2 backend sources
if(ENABLE_AVX2)
    set(SHAW_AVX2_SRC
        src/x64/avx2/shaw_scalarmult.cpp
        src/x64/avx2/shaw_scalarmult_vartime.cpp
        src/x64/avx2/shaw_msm_vartime.cpp
    )
    list(APPEND SHAW_SRC ${SHAW_AVX2_SRC})

    if(MSVC)
        set_source_files_properties(${SHAW_AVX2_SRC} PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    else()
        set_source_files_properties(${SHAW_AVX2_SRC} PROPERTIES
            COMPILE_FLAGS "-mavx2 ${_MINGW_SIMD_FIX}")
    endif()
endif()

# IFMA backend sources
if(ENABLE_AVX512)
    set(SHAW_IFMA_SRC
        src/x64/ifma/shaw_scalarmult.cpp
        src/x64/ifma/shaw_scalarmult_vartime.cpp
        src/x64/ifma/shaw_msm_vartime.cpp
    )
    list(APPEND SHAW_SRC ${SHAW_IFMA_SRC})

    if(MSVC)
        set_source_files_properties(${SHAW_IFMA_SRC} PROPERTIES COMPILE_FLAGS "/arch:AVX512")
    else()
        set_source_files_properties(${SHAW_IFMA_SRC} PROPERTIES
            COMPILE_FLAGS "-mavx512f -mavx512ifma ${_MINGW_SIMD_FIX}")
    endif()
endif()

add_library(ranshaw-shaw STATIC ${SHAW_SRC})

target_include_directories(ranshaw-shaw PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(ranshaw-shaw
    PUBLIC ranshaw-fq ranshaw-common
    PRIVATE ranshaw-build-flags
)
