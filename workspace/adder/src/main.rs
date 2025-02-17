use cust::prelude::*;
use std::error::Error;
use nanorand::{Rng, WyRand};

static PTX: &str = include_str!("../../../resources/gpu-funcs.ptx");

const NUMBERS_LEN: usize = 4_000_000;

fn test_gpu() {
    let mut wyrand = WyRand::new();
    let mut a = vec![0.0f32; NUMBERS_LEN];
    wyrand.fill(&mut a);
    let mut b = vec![0.0f32; NUMBERS_LEN];
    wyrand.fill(&mut b);

    let _ctx = cust::quick_init().unwrap();

    let module = Module::from_ptx(PTX, &[]).unwrap();

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let a_gpu = a.as_slice().as_dbuf().unwrap();
    let b_gpu = b.as_slice().as_dbuf().unwrap();

    let mut out = vec![0.0f32; NUMBERS_LEN];
    let out_gpu = out.as_slice().as_dbuf().unwrap();

    let func = module.get_function("add").unwrap();

    let (_, block_size) = func.suggested_launch_configuration(0, 0.into()).unwrap();

    let grid_size = (NUMBERS_LEN as u32 + block_size - 1) / block_size;

    println!(
        "using {} blocks and {} threads per block",
        grid_size, block_size
    );

    let start = std::time::Instant::now();

    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            func<<<grid_size, block_size, 0, stream>>>(
                a_gpu.as_device_ptr(),
                a_gpu.len(),
                b_gpu.as_device_ptr(),
                b_gpu.len(),
                out_gpu.as_device_ptr(),
            )
        ).unwrap();
    }

    stream.synchronize().unwrap();
    out_gpu.copy_to(&mut out).unwrap();

    println!("GPU output first sum: {:?}", out[0]);
    println!("elapsed GPU: {} seconds", start.elapsed().as_secs_f64());
}


fn test_cpu() {
    let mut wyrand = WyRand::new();
    let mut a = vec![0.0f32; NUMBERS_LEN];
    wyrand.fill(&mut a);
    let mut b = vec![0.0f32; NUMBERS_LEN];
    wyrand.fill(&mut b);
    let mut out = vec![0.0f32; NUMBERS_LEN];

    let start = std::time::Instant::now();
    for i in 0..NUMBERS_LEN {
        let mut s = 0.0f32;
        for j in 0..NUMBERS_LEN {
            s += a[i] * b[j];
        }
        out[i] = s;
    }

    println!("CPU output first sum: {:?}", out[0]);
    println!("elapsed CPU: {} seconds", start.elapsed().as_secs_f64());
}

fn main() -> Result<(), Box<dyn Error>> {
    test_gpu();
    //test_cpu();
    Ok(())
}
