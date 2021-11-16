use ndarray::prelude::*;
use ndarray::{concatenate, stack, Array1, Array2, Axis};
fn main() {
    let arr = Array1::from(vec![1, 2, 3]);
    // let tmp = arr.slice(s![NewAxis]);
    println!("{:?}", arr.shape());
    let arr = arr.into_shape((1, 3)).unwrap();
    let arr2 = Array1::from(vec![4, 5, 6]).into_shape((1, 3)).unwrap();
    println!("{:?}", arr);
    println!("{:?}", arr2);
    let vs = vec![arr, arr2];
    println!("=====");
    let tmp = vs.iter().map(|x| x.view()).collect::<Vec<_>>();
    println!("{:?}", tmp.len());
    println!("=====");
    let res = concatenate(Axis(0), &vs).unwrap();
    println!("{:?}", res);
    println!("{:?}", res.shape());
}
