cmake_minimum_required(VERSION 3.10)

if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: Debug, Release, RelWithDebInfo")
else()
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Debug, Release, RelWithDebInfo")
endif()

message(STATUS "Mode: ${CMAKE_BUILD_TYPE}")

set(LIB_MAJOR_VERSION "0")
set(LIB_MINOR_VERSION "1")
set(LIB_PATCH_VERSION "0")
set(LIB_VERSION_STRING "${LIB_MAJOR_VERSION}.${LIB_MINOR_VERSION}.${LIB_PATCH_VERSION}")

cmake_policy(SET CMP0048 NEW)
project(ranshaw VERSION "${LIB_VERSION_STRING}" LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SKIP_INSTALL_RULES OFF FORCE)
set(CMAKE_SKIP_PACKAGE_ALL_DEPENDENCY ON FORCE)
set(CMAKE_SUPPRESS_REGENERATION ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Internal build flags (PRIVATE to all ranshaw targets) ──
# Sub-libraries link PRIVATE to this target so compile flags do not leak
# to downstream consumers via add_subdirectory or FetchContent.
add_library(ranshaw-build-flags INTERFACE)
target_compile_features(ranshaw-build-flags INTERFACE cxx_std_17)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

if(MSVC)
    # Replace CMake's default /W3 with /W4 to avoid D9025 override warning
    string(REPLACE "/W3" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    target_compile_options(ranshaw-build-flags INTERFACE
        /W4 /GS /sdl /guard:cf /wd4267 /wd4804 /D_DLL
        # Debug: enable inlining + intrinsics even in debug for usable perf
        $<$<CONFIG:Debug>:/Ob2 /Oi>
    )

    # Spectre mitigation (opt-in; significantly increases MSVC compile time)
    option(ENABLE_SPECTRE_MITIGATIONS "Enable MSVC Spectre mitigations (/Qspectre)" OFF)
    if(ENABLE_SPECTRE_MITIGATIONS)
        check_cxx_compiler_flag("/Qspectre" HAS_QSPECTRE)
        if(HAS_QSPECTRE)
            target_compile_options(ranshaw-build-flags INTERFACE /Qspectre)
        endif()
    endif()

    # CET shadow stack (newer MSVC only)
    check_cxx_compiler_flag("/guard:ehcont" HAS_GUARD_EHCONT)
    if(HAS_GUARD_EHCONT)
        target_compile_options(ranshaw-build-flags INTERFACE /guard:ehcont)
    endif()
else()
    target_compile_options(ranshaw-build-flags INTERFACE
        -Wall -Wextra -Wuninitialized -Wformat-security -Wconversion -Wsign-conversion
        # Debug: full debug info + optimize for debugging
        $<$<CONFIG:Debug>:-g3 -Og>
    )

    # _FORTIFY_SOURCE requires optimization; not reliably supported on MinGW
    if(NOT WIN32)
        target_compile_options(ranshaw-build-flags INTERFACE
            $<$<NOT:$<CONFIG:Debug>>:-D_FORTIFY_SOURCE=2>
        )
    endif()

    # Stack protector: try-compile probe (catches any toolchain where
    # libssp / compiler-rt is missing, not just Clang+MinGW).
    set(CMAKE_REQUIRED_FLAGS "-fstack-protector-strong")
    check_cxx_source_compiles("int main() { char buf[64]; buf[0] = 0; return buf[0]; }" HAS_STACK_PROTECTOR_STRONG)
    unset(CMAKE_REQUIRED_FLAGS)
    if(HAS_STACK_PROTECTOR_STRONG)
        target_compile_options(ranshaw-build-flags INTERFACE -fstack-protector-strong)
    endif()

    # Stack clash protection (Linux only — not supported on macOS/Apple Clang,
    # and Windows Clang accepts the flag but emits "argument unused" warnings)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        check_cxx_compiler_flag("-fstack-clash-protection" HAS_STACK_CLASH_PROTECTION)
        if(HAS_STACK_CLASH_PROTECTION)
            target_compile_options(ranshaw-build-flags INTERFACE -fstack-clash-protection)
        endif()
    endif()

    # Disable auto-vectorization to prevent the compiler from introducing
    # SIMD instructions that could create data-dependent timing behavior,
    # which would undermine the constant-time guarantees of the crypto code.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(ranshaw-build-flags INTERFACE -fno-tree-vectorize)
        # GCC libstdc++ debug assertions (bounds checks on operator[], iterator validation)
        target_compile_options(ranshaw-build-flags INTERFACE
            $<$<CONFIG:Debug>:-D_GLIBCXX_ASSERTIONS>
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(ranshaw-build-flags INTERFACE -fno-vectorize -fno-slp-vectorize)
    endif()

    # CET control-flow integrity (Intel CET shadow stack + indirect branch tracking)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^([xX]86_64|[aA][mM][dD]64)$")
        check_cxx_compiler_flag("-fcf-protection=full" HAS_CF_PROTECTION)
        if(HAS_CF_PROTECTION)
            target_compile_options(ranshaw-build-flags INTERFACE -fcf-protection=full)
        endif()
    endif()
endif()

# ── Linker hardening helper ──
# Only use CMake's built-in MINGW (set for GCC with MinGW). Clang on Windows can
# target either MinGW or MSVC ABI, and the linker varies (GNU ld vs lld-link) —
# applying the wrong linker flags breaks the build. We let Clang on Windows rely
# on the OS defaults for DEP/ASLR (enabled by default on modern Windows) rather
# than guessing the linker style.
function(ranshaw_harden_executable target)
    if(MSVC)
        # PE linker hardening — DEP + ASLR + CFG
        set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS
            " /DYNAMICBASE /NXCOMPAT /HIGHENTROPYVA /guard:cf")
        # CET shadow stack compatibility — compile AND link flags must match
        if(HAS_GUARD_EHCONT)
            target_compile_options(${target} PRIVATE /guard:ehcont)
            set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS
                " /guard:ehcont /CETCOMPAT")
        endif()
    elseif(MINGW)
        # MinGW (GCC only) PE linker hardening — DEP + ASLR
        set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS
            " -Wl,--nxcompat -Wl,--dynamicbase -Wl,--high-entropy-va")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # ELF linker hardening
        set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS
            " -Wl,-z,relro,-z,now -Wl,-z,noexecstack")
    endif()

    # LTO link flags
    if(ENABLE_LTO)
        if(MSVC)
            set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS " /LTCG")
        else()
            set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS " -flto")
        endif()
    endif()

    # Sanitizer link flags
    if(NOT MSVC)
        if(ENABLE_ASAN)
            set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS
                " -fsanitize=address")
        endif()
        if(ENABLE_UBSAN)
            set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS
                " -fsanitize=undefined")
        endif()
    endif()
endfunction()

# ── Options ──
option(FORCE_PORTABLE "Force portable (32-bit) implementation even on 64-bit platforms" OFF)

# Detect x86_64 specifically (SIMD backends are x86_64-only)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^([xX]86_64|[aA][mM][dD]64)$")
    set(RANSHAW_IS_X86_64 TRUE)
else()
    set(RANSHAW_IS_X86_64 FALSE)
endif()

# SIMD options: default ON for x86_64, OFF otherwise
if(RANSHAW_IS_X86_64 AND NOT FORCE_PORTABLE)
    option(ENABLE_AVX512 "Enable AVX-512 IFMA backend" ON)
    option(ENABLE_AVX2 "Enable AVX2 backend" ON)
else()
    option(ENABLE_AVX512 "Enable AVX-512 IFMA backend" OFF)
    option(ENABLE_AVX2 "Enable AVX2 backend" OFF)
endif()

# Force SIMD off when portable is active
if(FORCE_PORTABLE)
    set(ENABLE_AVX512 OFF CACHE BOOL "" FORCE)
    set(ENABLE_AVX2 OFF CACHE BOOL "" FORCE)
    message(STATUS "RANSHAW: Forced portable (32-bit) implementation via FORCE_PORTABLE")
endif()

option(ENABLE_ECFFT "Enable ECFFT polynomial multiplication" OFF)

option(ENABLE_LTO "Enable link-time optimization" OFF)
if(ENABLE_LTO)
    if(MSVC)
        target_compile_options(ranshaw-build-flags INTERFACE /GL)
    else()
        target_compile_options(ranshaw-build-flags INTERFACE -flto)
    endif()
    message(STATUS "RANSHAW: Link-time optimization enabled")
endif()

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
if(NOT MSVC)
    if(ENABLE_ASAN)
        target_compile_options(ranshaw-build-flags INTERFACE
            -fsanitize=address -fno-omit-frame-pointer)
        message(STATUS "RANSHAW: AddressSanitizer enabled")
    endif()
    if(ENABLE_UBSAN)
        target_compile_options(ranshaw-build-flags INTERFACE
            -fsanitize=undefined -fno-omit-frame-pointer)
        message(STATUS "RANSHAW: UndefinedBehaviorSanitizer enabled")
    endif()
endif()

# ── MinGW GCC + Debug SIMD alignment fix ──
# Windows x64 ABI only guarantees 16-byte stack alignment, but YMM/ZMM
# registers need 32/64-byte alignment.  At -O0, GCC spills AVX registers
# to the stack with aligned moves (vmovdqa), which SIGSEGV on the
# misaligned stack.  Force -O1 for SIMD backends in Debug mode so GCC
# keeps values in registers.
set(_MINGW_SIMD_FIX "")
if(MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_MINGW_SIMD_FIX "$<$<CONFIG:Debug>:-O1>")
endif()

# ── Sub-libraries ──
add_subdirectory(common)
add_subdirectory(fp)
add_subdirectory(fq)
add_subdirectory(ran)
add_subdirectory(shaw)
add_subdirectory(ecfft)
add_subdirectory(ec-divisors)

# ── Main library: dispatch + master include + public C++ API ──
set(RANSHAW_API_SOURCES
    src/api_scalar.cpp
    src/api_point.cpp
    src/api_polynomial.cpp
    src/api_divisor.cpp
)

if(ENABLE_AVX2 OR ENABLE_AVX512)
    add_library(ranshaw STATIC
        src/cpuid.cpp
        src/dispatch.cpp
        ${RANSHAW_API_SOURCES}
    )
    message(STATUS "RANSHAW: SIMD dispatch enabled")

    if(ENABLE_AVX2)
        message(STATUS "RANSHAW: AVX2 backend enabled")
    endif()
    if(ENABLE_AVX512)
        message(STATUS "RANSHAW: AVX-512 IFMA backend enabled")
    endif()

    target_compile_definitions(ranshaw PUBLIC RANSHAW_SIMD=1)

    if(NOT ENABLE_AVX512)
        target_compile_definitions(ranshaw PUBLIC RANSHAW_NO_AVX512=1)
    endif()
    if(NOT ENABLE_AVX2)
        target_compile_definitions(ranshaw PUBLIC RANSHAW_NO_AVX2=1)
    endif()
else()
    add_library(ranshaw STATIC ${RANSHAW_API_SOURCES})
    target_compile_definitions(ranshaw PUBLIC RANSHAW_SIMD=0)
endif()

if(ENABLE_ECFFT)
    message(STATUS "RANSHAW: ECFFT polynomial multiplication enabled")
endif()

# Public interface for downstream consumers
target_compile_features(ranshaw PUBLIC cxx_std_17)
target_include_directories(ranshaw PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
# Downstream consumers get our headers as SYSTEM (suppresses their -Werror on our code)
target_include_directories(ranshaw SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(ranshaw PUBLIC
    ranshaw-ec-divisors
    ranshaw-ran
    ranshaw-shaw
    ranshaw-fp
    ranshaw-fq
    ranshaw-ecfft
    ranshaw-common
)

# Internal build flags (PRIVATE — not propagated to downstream)
target_link_libraries(ranshaw PRIVATE ranshaw-build-flags)
target_compile_definitions(ranshaw PRIVATE RANSHAW_INTERNAL_ACCESS)

# ── Tests ──
option(BUILD_TESTS "Build unit tests" OFF)
if(DEFINED ENV{BUILD_TESTS})
    set(BUILD_TESTS $ENV{BUILD_TESTS})
endif()

if(BUILD_TESTS)
    enable_testing()
    message(STATUS "RANSHAW: Unit tests enabled")

    add_executable(ranshaw-tests src/tests.cpp)
    target_link_libraries(ranshaw-tests PRIVATE ranshaw)
    target_compile_definitions(ranshaw-tests PRIVATE RANSHAW_INTERNAL_ACCESS)
    ranshaw_harden_executable(ranshaw-tests)
    add_test(NAME ranshaw-tests COMMAND ranshaw-tests)

    add_executable(ranshaw-fuzz-tests src/tests_fuzz.cpp)
    target_link_libraries(ranshaw-fuzz-tests PRIVATE ranshaw)
    target_compile_definitions(ranshaw-fuzz-tests PRIVATE RANSHAW_INTERNAL_ACCESS)
    ranshaw_harden_executable(ranshaw-fuzz-tests)
    add_test(NAME ranshaw-fuzz-tests COMMAND ranshaw-fuzz-tests)
endif()

# ── Benchmarks ──
option(BUILD_BENCHMARKS "Build benchmark executables" OFF)
if(DEFINED ENV{BUILD_BENCHMARKS})
    set(BUILD_BENCHMARKS $ENV{BUILD_BENCHMARKS})
endif()

if(BUILD_BENCHMARKS)
    message(STATUS "RANSHAW: Benchmarks enabled")

    add_executable(ranshaw-benchmark src/benchmark.cpp)
    target_link_libraries(ranshaw-benchmark PRIVATE ranshaw)
    target_compile_definitions(ranshaw-benchmark PRIVATE RANSHAW_INTERNAL_ACCESS)
    ranshaw_harden_executable(ranshaw-benchmark)

    add_executable(ranshaw-benchmark-contest src/benchmark_contest.cpp)
    target_link_libraries(ranshaw-benchmark-contest PRIVATE ranshaw)
    target_compile_definitions(ranshaw-benchmark-contest PRIVATE RANSHAW_INTERNAL_ACCESS)
    ranshaw_harden_executable(ranshaw-benchmark-contest)

    add_executable(ranshaw-benchmark-fcmpp src/benchmark_fcmpp.cpp)
    target_link_libraries(ranshaw-benchmark-fcmpp PRIVATE ranshaw)
    target_compile_definitions(ranshaw-benchmark-fcmpp PRIVATE RANSHAW_INTERNAL_ACCESS)
    ranshaw_harden_executable(ranshaw-benchmark-fcmpp)

    # Benchmarks are developer tools — relax sign-conversion warnings
    if(NOT MSVC)
        foreach(_bench ranshaw-benchmark ranshaw-benchmark-contest ranshaw-benchmark-fcmpp)
            target_compile_options(${_bench} PRIVATE -Wno-sign-conversion)
        endforeach()
    endif()
endif()

# ── Auxiliary tools ──
option(BUILD_TOOLS "Build auxiliary tools (ECFFT curve search)" OFF)
if(BUILD_TOOLS)
    # ECFFT tools use uint64_t field elements directly and require the x64 backend
    if(NOT FORCE_PORTABLE)
        add_executable(ranshaw-find-ecfft tools/find_ecfft_curve.cpp)
        target_link_libraries(ranshaw-find-ecfft PRIVATE ranshaw)
        ranshaw_harden_executable(ranshaw-find-ecfft)

        add_executable(ranshaw-gen-ecfft tools/gen_ecfft_data.cpp)
        target_link_libraries(ranshaw-gen-ecfft PRIVATE ranshaw)
        ranshaw_harden_executable(ranshaw-gen-ecfft)

        if(NOT MSVC)
            foreach(_tool ranshaw-find-ecfft ranshaw-gen-ecfft)
                target_compile_options(${_tool} PRIVATE -Wno-sign-conversion)
            endforeach()
        endif()
    endif()

    add_executable(ranshaw-gen-testvectors tools/gen_test_vectors.cpp)
    target_link_libraries(ranshaw-gen-testvectors PRIVATE ranshaw)
    target_compile_definitions(ranshaw-gen-testvectors PRIVATE RANSHAW_INTERNAL_ACCESS)
    ranshaw_harden_executable(ranshaw-gen-testvectors)

    if(NOT MSVC)
        target_compile_options(ranshaw-gen-testvectors PRIVATE -Wno-sign-conversion)
    endif()

    message(STATUS "RANSHAW: Auxiliary tools enabled")
endif()
