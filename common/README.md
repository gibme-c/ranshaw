# common — Platform Abstraction

This is the shared foundation that every other module in ranshaw depends on. It handles platform detection, constant-time primitives, and memory safety utilities — the kind of infrastructure you rarely call directly, but that determines whether the library runs correctly (and securely) on your machine.


## Platform Detection

The library needs to know what your CPU can do so it can pick the right code path at compile time. `ranshaw_platform.h` defines a set of macros based on the target architecture:

| Macro | Meaning |
|-------|---------|
| `RANSHAW_PLATFORM_X64` | Targeting x86-64 |
| `RANSHAW_PLATFORM_ARM64` | Targeting AArch64 |
| `RANSHAW_PLATFORM_64BIT` | 64-bit platform (unless `FORCE_PORTABLE` overrides) |
| `RANSHAW_HAVE_INT128` | GCC/Clang `unsigned __int128` is available |
| `RANSHAW_HAVE_UMUL128` | MSVC `_umul128` intrinsic is available |

The 64-bit flag controls which field representation gets compiled — 5×51-bit limbs on 64-bit, 10×25.5-bit limbs on 32-bit (or when forced portable). This single decision cascades through every module in the library.

A platform-abstracted 64×64→128-bit multiply (`mul128.h`) wraps the native `__int128` on GCC/Clang and `_umul128` on MSVC, giving field arithmetic a uniform interface regardless of compiler.


## Constant-Time Barriers

When you're doing cryptography, the compiler is too smart for its own good. You write branchless XOR-blend code to avoid leaking secrets through timing, and the optimizer helpfully "simplifies" it back into a conditional branch.

`ct_barrier_u32` and `ct_barrier_u64` are the countermeasure. On GCC/Clang, they use an inline `asm volatile` constraint that tells the compiler "this value might have changed — don't assume anything about it." On MSVC, a `volatile` round-trip achieves the same effect.

These barriers are used throughout the library anywhere a constant-time conditional move (`cmov`) or conditional negate (`cneg`) appears. They're small, but they're the reason the library's constant-time guarantees hold up under optimization.


## Secure Erase

When a secret scalar lives on the stack, you need to zero it before the function returns. The obvious approach — `memset(secret, 0, len)` — doesn't work. The compiler sees that nobody reads the zeroed memory afterward and removes the call entirely as a dead store.

`ranshaw_secure_erase` handles this per-platform:

- **Windows**: `SecureZeroMemory` (compiler intrinsic, guaranteed not to be optimized away)
- **Linux/macOS**: `explicit_bzero` (libc function with the same guarantee)
- **Fallback**: A volatile function pointer trick — the compiler can't prove the pointed-to function is `memset`, so it can't remove the call

This gets called at the end of every constant-time scalar multiplication path, and defensively in variable-time paths as well.


## Byte-Loading Helpers

`load_3` and `load_4` read 3 or 4 bytes from a buffer into a `uint64_t` in little-endian order. These are used during deserialization of field elements from their 32-byte wire format — small utilities, but they show up in every `frombytes` function across both fields.
