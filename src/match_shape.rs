use ocl::core;
use ocl::{Buffer, ProQue, Device};
use std::{i32, cmp};
use rayon::prelude::*;

enum Processor<'a> {
    SingleCore,
    MultiCore,
    GPU(&'a Device, usize),
}

static OCL_MATCH_KERNEL: &'static str = r#"
    __kernel void match(__constant int* s_xs, __constant int* s_ys,
                        const __global int* dt, __global int* cm, int grid_width, int s_len) {
        int idx = get_global_id(0) * grid_width + get_global_id(1);
        int acc = 0;
        for (int k = 0; k < s_len; k++) {
            acc += dt[idx + s_xs[k] * grid_width + s_ys[k]];
        }
        cm[idx] = acc;
    }
"#;

#[inline]
fn match_kernel(i: usize, j: usize, grid_width: usize, sp: &[(i32, i32)], dt: &[i32]) -> i32 {
    let idx = i * grid_width + j;
    sp.iter().map(|&(x, y)| dt[idx + x as usize * grid_width + y as usize]).sum()
}

impl<'a> Processor<'a> {
    fn process(self,
               sm: &[(i32, i32)],
               s_dim: (usize, usize),
               dt: &[i32],
               dim: (usize, usize))
               -> Vec<i32> {
        let (m, n) = dim;
        let (s, t) = s_dim;
        match self {
            Processor::SingleCore => {
                let mut cm = vec![i32::MAX; n*m];
                for i in 0..m - s {
                    for j in 0..n - t {
                        cm[i * n + j] = match_kernel(i, j, n, sm, dt);
                    }
                }
                cm
            }
            Processor::MultiCore => {
                let cm = vec![i32::MAX; n*m];
                (0..m - s).collect::<Vec<_>>().par_iter().for_each(|&i| {
                    // Get pointer to start of the i-th row.
                    unsafe {
                        let ptr = cm.as_ptr().offset((i * n) as isize) as *mut i32;
                        for j in 0..n - t {
                            *(ptr).offset(j as isize) = match_kernel(i, j, n, sm, dt);
                        }
                    }
                });
                cm
            }
            Processor::GPU(device, step_size) => {
                // TODO: unwrap less
                let mut cm = vec![i32::MAX; n*m];
                let (xs, ys): (Vec<i32>, Vec<i32>) = sm.iter().cloned().unzip();
                let pro_que =
                    ProQue::builder().device(device).src(OCL_MATCH_KERNEL).build().unwrap();
                let queue = pro_que.queue();
                let d_dt = Buffer::new(queue.clone(),
                                       Some(core::MEM_READ_ONLY | core::MEM_COPY_HOST_PTR),
                                       (m * n,),
                                       Some(dt))
                    .unwrap();
                let d_cm = Buffer::new(queue.clone(),
                                       Some(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR),
                                       (m * n,),
                                       Some(&cm))
                    .unwrap();
                let d_xs = Buffer::new(queue.clone(),
                                       Some(core::MEM_READ_ONLY | core::MEM_COPY_HOST_PTR),
                                       (sm.len(),),
                                       Some(&xs))
                    .unwrap();
                let d_ys = Buffer::new(queue.clone(),
                                       Some(core::MEM_READ_ONLY | core::MEM_COPY_HOST_PTR),
                                       (sm.len(),),
                                       Some(&ys))
                    .unwrap();
                let kernel = pro_que.create_kernel("match")
                    .unwrap()
                    .arg_buf_named("s_xs", Some(&d_xs))
                    .arg_buf_named("s_ys", Some(&d_ys))
                    .arg_buf_named("dt", Some(&d_dt))
                    .arg_buf_named("cm", Some(&d_cm))
                    .arg_scl_named("grid_width", Some(n))
                    .arg_scl_named("s_len", Some(sm.len()));
                for offset in (0..m).step_by(step_size) {
                    kernel.cmd()
                        .gws((cmp::min(step_size, m - offset - s), n - t))
                        .gwo((offset, 0))
                        .enq()
                        .unwrap();
                }
                d_cm.read(&mut cm).enq().unwrap();
                queue.finish();
                cm
            }
        }
    }
}

#[inline]
fn shape_points<BMatrix: AsRef<[BRow]>, BRow: AsRef<[bool]>>(mask: BMatrix) -> Vec<(i32, i32)> {
    mask.as_ref()
        .iter()
        .enumerate()
        .flat_map(|(i, m)| {
            m.as_ref()
                .iter()
                .enumerate()
                .filter(|&(_, v)| *v)
                .map(move |(j, _)| (i as i32, j as i32))
        })
        .collect()
}


/// Set to large value any cell that is not the minimum of its 8-neighborhood.
fn non_min_suppress(mat: &mut [i32], dim: (usize, usize)) {
    // At each pixel, suppress any neighbor pixels greater than it.
    let (m, n) = dim;
    for i in 0..m {
        for j in 0..n {
            for x in cmp::max(0, i - 1)..cmp::min(m, i + 2) {
                for y in cmp::max(0, j - 1)..cmp::min(n, j + 2) {
                    if mat[i * n + j] < mat[x * n + y] {
                        mat[x * n + y] = i32::MAX;
                    }
                }
            }
        }
    }
}


// TODO: benchmark match shape single core/parallel/opencl
/// Match provided mask matrix on provided distance transform matrix.
/// dt and mask matrices are read from in parallel.
/// Return score of match at each square.
pub fn match_shape<BMatrix: AsRef<[BRow]>, BRow: AsRef<[bool]>>(dt: &[i32],
                                                                dim: (usize, usize),
                                                                mask: BMatrix)
                                                                -> Vec<i32> {
    match_shape_generic(dt, dim, mask, Processor::MultiCore)
}

/// Sequentially match provided mask matrix on provided distance transform matrix.
/// Return score of match at each square.
pub fn match_shape_slow<BMatrix: AsRef<[BRow]>, BRow: AsRef<[bool]>>(dt: &[i32],
                                                                     dim: (usize, usize),
                                                                     mask: BMatrix)
                                                                     -> Vec<i32> {
    match_shape_generic(dt, dim, mask, Processor::SingleCore)
}

/// Sequentially match provided mask matrix on provided distance transform matrix.
/// Copies dt and mask matrices to specified GPU device.
/// Return score of match at each square.
pub fn match_shape_ocl<BMatrix: AsRef<[BRow]>, BRow: AsRef<[bool]>>(dt: &[i32],
                                                                    dim: (usize, usize),
                                                                    mask: BMatrix,
                                                                    device: &Device,
                                                                    step_size: Option<usize>)
                                                                    -> Vec<i32> {
    match_shape_generic(dt,
                        dim,
                        mask,
                        Processor::GPU(device, step_size.unwrap_or(256)))
}


/// Match provided mask matrix on provided distance transform matrix.
/// dt and mask matrices are read from in parallel.
/// Return score of match at each square.
fn match_shape_generic<BMatrix: AsRef<[BRow]>, BRow: AsRef<[bool]>>(dt: &[i32],
                                                                    dim: (usize, usize),
                                                                    mask: BMatrix,
                                                                    processor: Processor)
                                                                    -> Vec<i32> {
    let sp_dim = (mask.as_ref().len(), mask.as_ref()[0].as_ref().len());
    let sp = shape_points(mask);
    let mut cm = processor.process(&sp, sp_dim, dt, dim);
    non_min_suppress(&mut cm, dim);
    cm
}
