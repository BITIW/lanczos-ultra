#ifndef LANCZOS_ULTRA_H
#define LANCZOS_ULTRA_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Stable ABI v1. Windows uses the C calling convention (cdecl).
 * Link the generated import library on Windows or load the DLL dynamically. */
uint32_t lanczos_ultra_abi_version(void);
enum lanczos_ultra_status {
    LANCZOS_ULTRA_OK = 0,
    LANCZOS_ULTRA_INVALID_DIMENSIONS = 1,
    LANCZOS_ULTRA_INVALID_BUFFER = 2,
    LANCZOS_ULTRA_INVALID_LOBES = 3,
    LANCZOS_ULTRA_ALLOCATION_FAILED = 4,
    LANCZOS_ULTRA_PANIC = 5,
    LANCZOS_ULTRA_INVALID_RADIUS_PERCENT = 6,
    LANCZOS_ULTRA_INVALID_RADIUS_OPTIONS = 7
};
/* Straight (unassociated) RGBA8, tightly packed, in the supplied color space.
 * Lengths must be exactly width*height*4; dimensions must be nonzero.
 * lobes: 2..8, recommended 5. Buffers must be valid and nonoverlapping.
 * The caller owns both buffers; the library retains no pointers.
 * Do not mutate src or access dst concurrently during this synchronous call.
 * Returns a status above. Invalid pointers cannot be validated by the library.
 * Unrecoverable process failures (including allocator abort) cannot be caught. */
int32_t lanczos_ultra_resize_rgba8(
    const uint8_t *src, size_t src_len, uint32_t src_width, uint32_t src_height,
    uint8_t *dst, size_t dst_len, uint32_t dst_width, uint32_t dst_height,
    uint32_t lobes);
/* Same buffer contract as above; percent is 1..100 of the smaller input side.
 * Antialiased sinc, full percentage window and local ringing suppression.
 * Large radii use FFT when the dimension ratio has at most four phases. */
int32_t lanczos_ultra_resize_radius_rgba8(
    const uint8_t *src, size_t src_len, uint32_t src_width, uint32_t src_height,
    uint8_t *dst, size_t dst_len, uint32_t dst_width, uint32_t dst_height,
    uint32_t percent);
#ifdef __cplusplus
}
#endif
#endif
