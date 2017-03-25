use ocl::core;
use ocl::{Buffer, ProQue, Device};
use std::{i32, cmp};
use rayon::prelude::*;

enum Processor {
    SingleCore,
    MultiCore,
    GPU(Device),
}

static OCL_MATCH_KERNEL: &'static str = r#"
    __kernel void match(__global int* sm, __global int* dt, __global int* cm,
                        int grid_width, int sm_height, int sm_width) {
        size_t i = get_global_id(0);
        size_t j = get_global_id(1);
        int acc = 0;
        for (int x = 0; x < sm_height; x++) {
            int y = 0;
            while (y < sm_width) {
                int val = sm[x * sm_width + y];
                if (val == 0) {
                    acc += dt[(i + x) * grid_width + j + y];
                    y += 1;
                } else {
                    y += val;
                }
            }
        }
        if (acc < 0) {
            // set to max int
            acc = 2147483647;
        }
        cm[i * grid_width + j] = acc;
    }
"#;

#[inline]
fn match_kernel(i: usize, j: usize, grid_width: usize, sm: &[Vec<i32>], dt: &[i32]) -> i32 {
    let t = sm[0].len();
    let mut acc = 0;
    for (x, s_row) in sm.iter().enumerate() {
        let mut y = 0;
        while y < t {
            let val = s_row[y];
            if val == 0 {
                acc += dt[(i + x) * grid_width + j + y];
                y += 1;
            } else {
                y += val as usize;
            }
        }
    }
    if acc < 0 {
        // Then I suppose it overflowed, flip it back around.
        acc = i32::MAX;
    }
    acc
}

impl Processor {
    fn process(self, sm: &[Vec<i32>], dt: &[i32], dim: (usize, usize)) -> Vec<i32> {
        let (m, n) = dim;
        let s = sm.len();
        let t = sm[0].len();
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
            Processor::GPU(specifier) => {
                // TODO: unwrap less
                let mut cm = vec![i32::MAX; n*m];
                let pro_que = ProQue::builder()
                    .device(specifier)
                    .src(OCL_MATCH_KERNEL)
                    .dims((m, n))
                    .build()
                    .unwrap();
                let flat_sm: Vec<_> = sm.iter().flat_map(|x| x.iter()).cloned().collect();
                let queue = pro_que.queue();
                let sm_buffer = Buffer::new(queue.clone(),
                                            Some(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR),
                                            (s * t,),
                                            Some(&flat_sm))
                    .unwrap();
                let dt_buffer = Buffer::new(queue.clone(),
                                            Some(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR),
                                            (m * n,),
                                            Some(dt))
                    .unwrap();
                let cm_buffer: Buffer<i32> =
                    Buffer::new(queue.clone(), Some(core::MEM_READ_WRITE), (m * n,), None).unwrap();
                let kernel = pro_que.create_kernel("match")
                    .unwrap()
                    .arg_buf_named("sm", Some(&sm_buffer))
                    .arg_buf_named("dt", Some(&dt_buffer))
                    .arg_buf_named("cm", Some(&cm_buffer))
                    .arg_scl_named("grid_width", Some(n))
                    .arg_scl_named("sm_height", Some(s))
                    .arg_scl_named("sm_width", Some(t));
                kernel.enq().unwrap();
                cm_buffer.read(&mut cm);
                queue.finish();
                cm
            }
        }
    }
}

fn stride_mask<BMatrix: AsRef<[BRow]>, BRow: AsRef<[bool]>>(mask: BMatrix) -> Vec<Vec<i32>> {
    let mask = mask.as_ref();
    let m = mask.len();
    if m == 0 {
        return vec![vec![]];
    }
    let n = mask[0].as_ref().len();
    let mut stride = vec![vec![0; n]; m];
    for (r, row) in mask.iter().enumerate() {
        let row = row.as_ref();
        let mut last_set = row.len();
        for (c, val) in row.iter().enumerate().rev() {
            if *val {
                last_set = c;
            } else {
                stride[r][c] = (last_set - c) as i32;
            }
        }
    }
    stride
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
                                                                    device: Device)
                                                                    -> Vec<i32> {
    match_shape_generic(dt, dim, mask, Processor::GPU(device))
}


/// Match provided mask matrix on provided distance transform matrix.
/// dt and mask matrices are read from in parallel.
/// Return score of match at each square.
fn match_shape_generic<BMatrix: AsRef<[BRow]>, BRow: AsRef<[bool]>>(dt: &[i32],
                                                                    dim: (usize, usize),
                                                                    mask: BMatrix,
                                                                    processor: Processor)
                                                                    -> Vec<i32> {
    let sm = stride_mask(mask);
    let mut cm = processor.process(&sm, dt, dim);
    non_min_suppress(&mut cm, dim);
    cm
}
