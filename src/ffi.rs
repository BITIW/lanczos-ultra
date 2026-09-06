use crate::{ResizeError, Resizer, buffer_len};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// C ABI version, independent of the crate version.
#[unsafe(no_mangle)]
pub extern "C" fn lanczos_ultra_abi_version() -> u32 {
    1
}

/// Resize caller-owned RGBA8 buffers. See `include/lanczos_ultra.h`.
///
/// # Safety
/// Non-null pointers must refer to live allocations of the specified lengths.
/// Source must be readable and destination writable, exclusively for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanczos_ultra_resize_rgba8(
    src: *const u8,
    src_len: usize,
    sw: u32,
    sh: u32,
    dst: *mut u8,
    dst_len: usize,
    dw: u32,
    dh: u32,
    lobes: u32,
) -> i32 {
    // SAFETY: forwarded unchanged from this function's caller contract.
    unsafe { resize_impl(src, src_len, sw, sh, dst, dst_len, dw, dh, lobes, false) }
}

/// Resize with a radius expressed as a percentage of the smaller input side.
///
/// # Safety
/// Same pointer and exclusive access contract as `lanczos_ultra_resize_rgba8`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanczos_ultra_resize_radius_rgba8(
    src: *const u8,
    src_len: usize,
    sw: u32,
    sh: u32,
    dst: *mut u8,
    dst_len: usize,
    dw: u32,
    dh: u32,
    percent: u32,
) -> i32 {
    // SAFETY: forwarded unchanged from this function's caller contract.
    unsafe { resize_impl(src, src_len, sw, sh, dst, dst_len, dw, dh, percent, true) }
}

#[allow(clippy::too_many_arguments)]
unsafe fn resize_impl(
    src: *const u8,
    src_len: usize,
    sw: u32,
    sh: u32,
    dst: *mut u8,
    dst_len: usize,
    dw: u32,
    dh: u32,
    parameter: u32,
    radius: bool,
) -> i32 {
    let run = || -> Result<(), ResizeError> {
        let expected_src = buffer_len(sw, sh)?;
        let expected_dst = buffer_len(dw, dh)?;
        if src.is_null() || dst.is_null() || src_len != expected_src || dst_len != expected_dst {
            return Err(ResizeError::InvalidBuffer);
        }
        let a = src as usize;
        let b = dst as usize;
        let a_end = a.checked_add(src_len).ok_or(ResizeError::InvalidBuffer)?;
        let b_end = b.checked_add(dst_len).ok_or(ResizeError::InvalidBuffer)?;
        if a < b_end && b < a_end {
            return Err(ResizeError::InvalidBuffer);
        }
        let plan = if radius {
            Resizer::with_radius_percent(sw, sh, dw, dh, parameter)?
        } else {
            Resizer::with_lobes(sw, sh, dw, dh, parameter)?
        };
        // SAFETY: the caller supplies live buffers; sizes and overlap were checked above.
        let source = unsafe { std::slice::from_raw_parts(src, src_len) };
        // SAFETY: same contract, with exclusive writable destination access.
        let destination = unsafe { std::slice::from_raw_parts_mut(dst, dst_len) };
        plan.resize_into(source, destination)
    };
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(())) => 0,
        Ok(Err(ResizeError::InvalidDimensions)) => 1,
        Ok(Err(ResizeError::InvalidBuffer)) => 2,
        Ok(Err(ResizeError::InvalidLobes)) => 3,
        Ok(Err(ResizeError::AllocationFailed)) => 4,
        Err(_) => 5,
        Ok(Err(ResizeError::InvalidRadiusPercent)) => 6,
        Ok(Err(ResizeError::InvalidRadiusOptions)) => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ffi_checks_and_copy() {
        let src = [12, 34, 56, 255];
        let mut dst = [0; 4];
        unsafe {
            assert_eq!(
                lanczos_ultra_resize_rgba8(src.as_ptr(), 4, 1, 1, dst.as_mut_ptr(), 4, 1, 1, 5),
                0
            );
            assert_eq!(dst, src);
            assert_eq!(
                lanczos_ultra_resize_radius_rgba8(
                    src.as_ptr(),
                    4,
                    1,
                    1,
                    dst.as_mut_ptr(),
                    4,
                    1,
                    1,
                    50
                ),
                0
            );
            assert_eq!(dst, src);
            assert_eq!(
                lanczos_ultra_resize_radius_rgba8(
                    src.as_ptr(),
                    4,
                    1,
                    1,
                    dst.as_mut_ptr(),
                    4,
                    1,
                    1,
                    0
                ),
                6
            );
            assert_eq!(
                lanczos_ultra_resize_radius_rgba8(
                    std::ptr::null(),
                    4,
                    1,
                    1,
                    dst.as_mut_ptr(),
                    4,
                    1,
                    1,
                    50
                ),
                2
            );
            assert_eq!(
                lanczos_ultra_resize_radius_rgba8(
                    dst.as_ptr(),
                    4,
                    1,
                    1,
                    dst.as_mut_ptr(),
                    4,
                    1,
                    1,
                    75
                ),
                2
            );
            assert_eq!(
                lanczos_ultra_resize_rgba8(std::ptr::null(), 4, 1, 1, dst.as_mut_ptr(), 4, 1, 1, 5),
                2
            );
            assert_eq!(
                lanczos_ultra_resize_rgba8(dst.as_ptr(), 4, 1, 1, dst.as_mut_ptr(), 4, 1, 1, 5),
                2
            );
            assert_eq!(
                lanczos_ultra_resize_rgba8(src.as_ptr(), 4, 1, 1, dst.as_mut_ptr(), 4, 1, 1, 0),
                3
            );
            assert_eq!(
                lanczos_ultra_resize_rgba8(src.as_ptr(), 4, 0, 1, dst.as_mut_ptr(), 4, 1, 1, 5),
                1
            );
        }
    }
}
