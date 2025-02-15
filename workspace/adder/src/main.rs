use cust::prelude::*;
use std::error::Error;

static PTX: &str = include_str!("../../../resources/gpu-funcs.ptx");

fn main() -> Result<(), Box<dyn Error>> {
    let a = vec![1.0f32,2.,3.];
    let b = vec![4.0f32,5.,6.];

    let _ctx = cust::quick_init()?;

    let module = Module::from_ptx(PTX, &[])?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let a_gpu = a.as_slice().as_dbuf()?;
    let b_gpu = b.as_slice().as_dbuf()?;

    let mut out = vec![0.0f32,0.,0.];
    let out_gpu = out.as_slice().as_dbuf()?;

    let func = module.get_function("add")?;

    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            func<<<1, 3, 0, stream>>>(
                a_gpu.as_device_ptr(),
                a_gpu.len(),
                b_gpu.as_device_ptr(),
                b_gpu.len(),
                out_gpu.as_device_ptr(),
            )
        )?;
    }

    stream.synchronize()?;
    out_gpu.copy_to(&mut out)?;

    println!("out: {:?}", out);

    println!("Success!");

    Ok(())
}
