use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../gpu-funcs")
        .copy_to("../../resources/gpu-funcs.ptx")
        .build()
        .unwrap();
}
