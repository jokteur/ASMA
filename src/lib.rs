// Source adopted from
// https://github.com/tildeio/helix-website/blob/master/crates/word_count/src/lib.rs

// use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Mutex;

#[allow(non_snake_case)]
fn f_SRM(x: f64, c: f64, Delta: f64, theta: f64) -> f64 {
    c * f64::exp((x - theta) / Delta)
}

#[pyfunction]
fn particle_population_py<'py>(
    py: Python<'py>,
    time_end: f64,
    dt: f64,
    gamma: PyReadonlyArrayDyn<f64>,
    lambda: PyReadonlyArrayDyn<f64>,
    c: f64,
    Delta: f64,
    theta: f64,
    j: f64,
    lambda_kappa: f64,
    i_ext_time: f64,
    i_ext: f64,
    n: i32,
    m0: PyReadonlyArrayDyn<f64>,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let gamma = gamma.as_array();
    let lambda = lambda.as_array();
    let m0 = m0.as_array();

    let steps = (time_end / dt) as usize;
    let dim = gamma.shape()[0];

    let mut m_t = Array::<f64, _>::zeros((steps, dim));
    // let mut n_t = Array::<f64, _>::zeros((steps, dim, dim));
    let mut activity = Array::<f64, _>::zeros(steps);
    let mut x_t = Array::<f64, _>::zeros(steps);

    let ts = Array::linspace(0., time_end, steps);
    let mut m = Array::<f64, _>::zeros((n as usize, dim)); // TODO: copy M0
    let uniform_distr = Uniform::new(0., 1.);
    let mut rng = rand::thread_rng();

    for s in 1..steps {
        let x_fixed = if i_ext_time < dt * s as f64 { i_ext } else { 0. };

        let mut m_t_av = Mutex::new(Array::<f64, _>::zeros(dim));

        let noise = Array::random_using(n as usize, uniform_distr, &mut rng);
        let mut num_activations = AtomicI32::new(0);
        m.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(index, mut val)| {
                if 1. - f_SRM(-dt * f64::exp(val.sum() + x_t[s - 1]), c, Delta, theta) > noise[index] {
                    val += &gamma;
                    let mut av = m_t_av.lock().unwrap();
                    *av += &val;
                    num_activations.fetch_add(1, Ordering::Relaxed);
                } else {
                    val += &(&lambda * &val * (-dt));
                }
            });
        let num_activations = *num_activations.get_mut();
        let mut m_t_av = m_t_av.lock().unwrap();
        if num_activations > 0 {
            m_t.slice_mut(s![s, ..])
                .assign(&m_t_av.mapv(|a| a / num_activations as f64));
        } else {
            for d in 0..dim {
                m_t[[s, d]] = m_t[[s - 1, d]];
            }
        }
        activity[s] = 1. / n as f64 * num_activations as f64 / dt;
        x_t[s] = x_t[s - 1] + dt * (-lambda_kappa * x_t[s - 1] + lambda_kappa * (j * activity[s] + x_fixed));
    }
    let m_t = m_t.into_pyarray(py);
    Ok(m_t)
    // axpy(a, x, y).into_pyarray(py)
}

#[pymodule]
fn accelerated(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    // Particle simulation
    module.add("population", wrap_pyfunction!(particle_population_py, module)?)?;
    Ok(())
}
