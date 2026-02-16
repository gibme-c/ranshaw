# ranshaw-common: platform utilities, secure erase, ct_barrier, load helpers, mul128
add_library(ranshaw-common STATIC
    src/ranshaw_secure_erase.cpp
)

target_include_directories(ranshaw-common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(ranshaw-common PRIVATE ranshaw-build-flags)

if(FORCE_PORTABLE)
    target_compile_definitions(ranshaw-common PUBLIC FORCE_PORTABLE=1)
endif()
