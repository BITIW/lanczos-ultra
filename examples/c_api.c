#include "lanczos_ultra.h"
#include <assert.h>
int main(void) {
    const uint8_t src[4] = {40, 90, 150, 255};
    uint8_t dst[8 * 8 * 4];
    assert(lanczos_ultra_abi_version() == 1);
    assert(lanczos_ultra_resize_rgba8(src, sizeof src, 1, 1,
        dst, sizeof dst, 8, 8, 5) == LANCZOS_ULTRA_OK);
    for (size_t i = 0; i < sizeof dst; ++i) assert(dst[i] == src[i % 4]);
    return 0;
}
