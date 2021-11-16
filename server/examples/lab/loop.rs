use paste::paste;

const FOO_1: i32 = 1;
const FOO_2: i32 = 2;
const FOO_3: i32 = 3;
const FOO_4: i32 = 4;

// macro_rules! foo_i{
//   ($i:expr) => {
//     // 拼接 FOO_ 前缀和 expr i 成 FOO_1, FOO_2 ???
//     println!("{}", foo_$i);
//   };
// }

macro_rules! foo_i2 {
    ($i:expr) => {
        println!("{}", paste! {[<FOO_ $i>]});
    };
}

fn main() {
    // work
    foo_i2!(1);
    // 使用循环 i, not work
    for i in 1..=4 {
        // foo_i!(i);
        foo_i2!(i);
    }
}
