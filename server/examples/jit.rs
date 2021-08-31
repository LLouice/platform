use anyhow::Result;
use platform::common::*;

pub fn main() -> Result<()> {
    let model = tch::CModule::load("models/model_fb.pt")?;
    let num_ent = 14541;
    let num_rel = 237 * 2;
    let batch = 128;

    let device = Device::Cpu;

    let x_e = Tensor::randint1(0, num_ent, &[batch], (Kind::Int64, device));
    let x_rel = Tensor::randint1(0, num_rel, &[batch], (Kind::Int64, device));

    let logit = model.forward_ts(&[x_e, x_rel])?;
    let score = logit.sigmoid();
    let mx = score.max2(1, false);
    mx.0.print();
    mx.1.print();
    Ok(())
}
