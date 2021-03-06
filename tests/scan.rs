use rio_api::parser::*;
use rio_turtle::*;
use std::io::Cursor;

#[test]
fn ntriples_error_recovery() {
    let data = "<http://foo.com> <http://bar.com> <http://baz.com> .\n<http://foo.com> <http://bar.com> < .\n<http://foo.com> <http://bar.com> <http://bat.com> .\n<http://foo.com> <http://bar.com> <bat> .\n<http://foo.com> <http://bar.com> <http://bat.com> .";

    let mut line = 0;
    let mut count = 0;
    let mut count_err = 0;
    let mut parser = NTriplesParser::new(Cursor::new(&data));
    while !parser.is_end() {
        line += 1;
        let step = parser.parse_step(&mut |_| {
            count += 1;
            Ok(()) as Result<(), TurtleError>
        });
        if step.is_err() {
            println!("err, {}", line);
            count_err += 1;
        }
    }

    assert_eq!(count, 3);
    assert_eq!(count_err, 2);
}
