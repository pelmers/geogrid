use std::{i32, cmp};
use rayon::prelude::*;

#[cfg(feature="opencl")]
use ocl::{core, Buffer, ProQue, Device, Program};

#[cfg(not(feature="opencl"))]
/// Dummy unused class to compile when ocl is not available.
pub struct Device {}

pub enum Processor<'a> {
    SingleCore,
    MultiCore,
    GPU(&'a Device, usize),
}

#[cfg(feature="opencl")]
static OCL_MATCH_KERNEL: &'static str = r#"
    __kernel void match(__constant int* s_xs, __constant int* s_ys,
                        const __global int* dt, __global int* cm,
                        const int grid_width, const int s_len) {
        int idx = get_global_id(0) * grid_width + get_global_id(1);
        int acc = 0;
        for (int k = 0; k < s_len; k++) {
            acc += dt[idx + s_xs[k] * grid_width + s_ys[k]];
        }
        cm[idx] = acc;
    }
"#;

#[cfg(feature="opencl")]
static OCL_SUPPRESS_KERNEL: &'static str = r#"
    __kernel void suppress(const __global int* cm_in, __global int* cm_out,
                           const int grid_height, const int grid_width, const int radius) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        int idx = i * grid_width + j;
        int val = cm_in[idx];
        cm_out[idx] = val;
        for (int x = max(0, i - radius); x < min(grid_height, i + radius + 1); x++) {
            for (int y = max(0, j - radius); y < min(grid_width, j + radius + 1); y++) {
                if (cm_in[x * grid_width + y] < val) {
                    cm_out[idx] = INT_MAX;
                }
            }
        }
    }
"#;


#[inline]
fn match_kernel(i: usize, j: usize, grid_width: usize, sp: &[(i32, i32)], dt: &[i32]) -> i32 {
    let idx = i * grid_width + j;
    sp.iter().map(|&(x, y)| dt[idx + x as usize * grid_width + y as usize]).sum()
}

#[inline]
/// Return `i32::MAX` if mat[i][j] is no the minimum of its radius-neighborhood, otherwise return
/// mat[i][j].
fn suppress_kernel(mat: &[i32], dim: (usize, usize), i: i32, j: i32, radius: i32) -> i32 {
    let (m, n) = dim;
    let val = mat[i as usize * n + j as usize];
    for x in cmp::max(0, i - radius)..cmp::min(m as i32, i + radius + 1) {
        for y in cmp::max(0, j - radius)..cmp::min(n as i32, j + radius + 1) {
            if mat[x as usize * n + y as usize] < val {
                return i32::MAX;
            }
        }
    }
    val
}

impl<'a> Processor<'a> {
    fn process(self,
               sm: &[(i32, i32)],
               s_dim: (usize, usize),
               dt: &[i32],
               dim: (usize, usize),
               s_radius: usize)
               -> Vec<i32> {
        let (m, n) = dim;
        let (s, t) = s_dim;
        let mut cm = vec![i32::MAX; n*m];
        match self {
            Processor::SingleCore => {
                for i in 0..m - s {
                    for j in 0..n - t {
                        cm[i * n + j] = match_kernel(i, j, n, sm, dt);
                    }
                }
                let mut cm_suppressed = vec![i32::MAX; n*m];
                for i in 0..m - s {
                    for j in 0..n - t {
                        cm_suppressed[i * n + j] = suppress_kernel(&cm, dim, i as i32, j as i32, s_radius as i32);
                    }
                }
                cm_suppressed
            }
            Processor::MultiCore => {
                (0..m - s).collect::<Vec<_>>().par_iter().for_each(|&i| {
                    // Get pointer to start of the i-th row.
                    unsafe {
                        let ptr = cm.as_ptr().offset((i * n) as isize) as *mut i32;
                        for j in 0..n - t {
                            *(ptr).offset(j as isize) = match_kernel(i, j, n, sm, dt);
                        }
                    }
                });
                let cm_suppressed = vec![i32::MAX; n*m];
                // Do it again, but with suppression.
                (0..m - s).collect::<Vec<_>>().par_iter().for_each(|&i| {
                    unsafe {
                        let ptr = cm_suppressed.as_ptr().offset((i * n) as isize) as *mut i32;
                        for j in 0..n - t {
                            *(ptr).offset(j as isize) = suppress_kernel(&cm, dim, i as i32, j as i32, s_radius as i32);
                        }
                    }
                });
                cm_suppressed
            }
            Processor::GPU(device, step_size) => {
                Processor::process_opencl(device, step_size, dt, m, n, sm, s, t, s_radius, cm)
            }
        }
    }

    #[cfg(not(feature="opencl"))]
    fn process_opencl(_: &'a Device,
                      _: usize,
                      _: &[i32],
                      _: usize,
                      _: usize,
                      _: &[(i32, i32)],
                      _: usize,
                      _: usize,
                      _: usize,
                      cm: Vec<i32>)
                      -> Vec<i32> {
        cm
    }

    #[cfg(feature="opencl")]
    fn process_opencl(device: &'a Device,
                      step_size: usize,
                      dt: &[i32],
                      m: usize,
                      n: usize,
                      sm: &[(i32, i32)],
                      s: usize,
                      t: usize,
                      s_radius: usize,
                      mut cm: Vec<i32>)
                      -> Vec<i32> {
        // TODO: unwrap less
        let (xs, ys): (Vec<i32>, Vec<i32>) = sm.iter().cloned().unzip();
        let pro_que = ProQue::builder()
            .device(device)
            .prog_bldr(Program::builder().src(OCL_MATCH_KERNEL).src(OCL_SUPPRESS_KERNEL))
            .build()
            .unwrap();
        let queue = pro_que.queue();
        let d_dt = Buffer::new(queue.clone(),
                               Some(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR),
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
            .arg_scl_named("grid_width", Some(n as i32))
            .arg_scl_named("s_len", Some(sm.len() as i32));
        for i in 0..((m - s) / step_size) + 1 {
            let offset = i * step_size;
            if offset < m - s {
                kernel.cmd()
                    .gws((cmp::min(step_size, m - offset - s), n - t))
                    .gwo((offset, 0))
                    .enq()
                    .unwrap();
            }
            queue.finish();
        }
        // Here I'm re-using the distance transform input buffer as the output buffer for the
        // suppression operation since they have the same size.
        let kernel = pro_que.create_kernel("suppress")
            .unwrap()
            .arg_buf_named("cm_in", Some(&d_cm))
            .arg_buf_named("cm_out", Some(&d_dt))
            .arg_scl_named("grid_height", Some(m as i32))
            .arg_scl_named("grid_width", Some(n as i32))
            .arg_scl_named("radius", Some(s_radius as i32));
        for i in 0..((m - s) / step_size) + 1 {
            let offset = i * step_size;
            if offset < m - s {
                kernel.cmd()
                    .gws((cmp::min(step_size, m - offset - s), n - t))
                    .gwo((offset, 0))
                    .enq()
                    .unwrap();
            }
            queue.finish();
        }
        d_dt.read(&mut cm).enq().unwrap();
        queue.finish();
        cm
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


/// Match provided mask matrix on provided distance transform matrix.
/// dt and mask matrices are read from in parallel.
/// Return score of match at each square.
pub fn match_shape<BMatrix: AsRef<[BRow]>, BRow: AsRef<[bool]>>(dt: &[i32],
                                                                dim: (usize, usize),
                                                                mask: BMatrix,
                                                                suppress_radius: usize,
                                                                processor: Processor)
                                                                -> Vec<i32> {
    let sp_dim = (mask.as_ref().len(), mask.as_ref()[0].as_ref().len());
    let sp = shape_points(mask);
    processor.process(&sp, sp_dim, dt, dim, suppress_radius)
}
