use seq_macro::seq;

// fn main() {
//     let tuple = (1000, 100, 10);
//     let mut sum = 0;

//     // Expands to:
//     //
//     //     sum += tuple.0;
//     //     sum += tuple.1;
//     //     sum += tuple.2;
//     //
//     // This cannot be written using an ordinary for-loop because elements of
//     // a tuple can only be accessed by their integer literal index, not by a
//     // variable.
//     seq!(N in 0..=2 {
//         sum += tuple.N;
//     });

//     assert_eq!(sum, 1110);

// }

const FOO_1: i32 = 1;
const FOO_2: i32 = 2;
const FOO_3: i32 = 3;
const FOO_4: i32 = 4;

const FOO_a: char = 'a';
const FOO_b: char = 'b';
const FOO_c: char = 'c';
const FOO_d: char = 'd';

fn main() {
    seq!(N in 1..=4 {
        println!("{}", FOO_#N);
    });

    seq!(C in 'a'..='d' {
        println!("{}", FOO_#C);
    });
}
