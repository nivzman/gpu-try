#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

use cuda_std::prelude::*;
use cuda_std::GpuFloat;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn add(a: &[f32], b: &[f32], c: *mut f32) {
    let idx = thread::index_1d() as usize;
    if idx < a.len() {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[idx] * b[i];
        }
        let elem = &mut *c.add(idx);
        *elem = sum;
    }
}
