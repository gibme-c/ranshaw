# ranshaw-ecfft: ECFFT polynomial multiplication over Fp and Fq

if(ENABLE_ECFFT)
    add_library(ranshaw-ecfft STATIC
        src/ecfft.cpp
    )

    target_include_directories(ranshaw-ecfft PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

    target_compile_definitions(ranshaw-ecfft PUBLIC RANSHAW_ECFFT=1)

    target_link_libraries(ranshaw-ecfft
        PUBLIC ranshaw-fp ranshaw-fq ranshaw-common
        PRIVATE ranshaw-build-flags
    )
else()
    # Header-only stub so downstream can unconditionally link
    add_library(ranshaw-ecfft INTERFACE)

    target_include_directories(ranshaw-ecfft INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
endif()
