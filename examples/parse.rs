#[macro_use]
extern crate log;
use anyhow::{anyhow, bail, Result};
use env_logger;
use platform::triple::Triple;
use std::fs::{File, OpenOptions};
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;

fn main() -> Result<()> {
    println!("RUST_LOG: {:?}", std::env::var("RUST_LOG"));
    env_logger::init();
    let st = std::time::Instant::now();
    let mut args = std::env::args();
    args.next();
    let file = args.next().expect("must give a ttl file name");
    let filename = format!("data/chinese_symptom/{}.ttl", file);
    let outfile: String = args
        .next()
        .or(Some("data/chinese_symptom/triples.json".to_string()))
        .unwrap();

    let mut err_not_triple_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open("data/chinese_symptom/not_triple.txt")?;

    let mut err_not_valid_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open("data/chinese_symptom/not_valid.txt")?;

    let mut err_line_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open("data/chinese_symptom/err_line.txt")?;

    // 默认truncate is true
    let mut truncate = if args.next().is_none() { true } else { false };

    let mut triples = vec![];
    let buf = BufReader::new(File::open(filename)?);
    let mut linum = 0;
    for line in buf.lines() {
        linum += 1;
        debug!("line: [{}]", linum);
        if let Ok(line) = line {
            let res = parse_line(&line, &mut err_not_valid_file, &mut err_not_triple_file);
            if let Ok(res) = res {
                if res.len() == 3 {
                    let triple = Triple::from(res);
                    info!("{}\t{}\t{}", triple.head, triple.rel, triple.tail);
                    triples.push(triple);
                } else {
                    writeln!(err_line_file, "{}", &line)?;
                }
            } else {
                writeln!(err_line_file, "{}", &line)?;
            }
        }
        if triples.len() == 10000 {
            Triple::write2file(&outfile, &triples, truncate)?;

            if truncate {
                truncate = !truncate;
            }
            println!("==> {}", linum);
            triples.clear();
        }
    }

    // write the last triples
    Triple::write2file(&outfile, &triples, truncate)?;

    println!("time cost: {:?} seconds", st.elapsed().as_secs());
    Ok(())
}

fn parse_line<S: AsRef<str>>(
    line: S,
    err_not_valid_file: &mut File,
    err_not_triple_file: &mut File,
) -> Result<Vec<String>> {
    debug!("parse line...");
    let line = line.as_ref();
    if line.starts_with("<http://") {
        debug!("{:?}", line);
        let stuff: Vec<&str> = line.split("\t").collect();
        let mut res = vec![];
        for seg in stuff {
            let seg = seg.trim_end_matches(" .");
            let pos = seg.rfind('/');
            if let Some(pos) = pos {
                debug!("found pos");
                debug!("cur seg is {:?}", seg);
                let seg_b = seg.as_bytes();
                let length = seg_b.len();
                let targ;
                if seg_b[0] == b'#' {
                    debug!("start with #");
                    targ = &seg_b[pos + 2..length - 1];
                } else {
                    targ = &seg_b[pos + 1..length - 1];
                }
                let targ2: String = String::from_utf8_lossy(&targ).into();
                res.push(targ2);
            } else {
                let seg_b = seg.as_bytes();
                let pos = seg.rfind("\"");
                if let Some(pos) = pos {
                    let targ = &seg_b[1..pos];
                    let targ2: String = String::from_utf8_lossy(&targ).into();
                    // write to err record
                    if let (Some(head), Some(rel)) = (res.get(0), res.get(1)) {
                        writeln!(err_not_triple_file, "{}\t{}\t{}", head, rel, targ2)?;
                        error!("{}\t{}\t{}", head, rel, targ2);
                    }
                    bail!("not valid triples");
                }
            }
        }

        if !line.contains("#") {
            return Ok(res);
        } else {
            if let (Some(head), Some(rel), Some(tail)) = (res.get(0), res.get(1), res.get(2)) {
                writeln!(err_not_triple_file, "{}\t{}\t{}", head, rel, tail)?;
                error!("{}\t{}\t{}", head, rel, tail);
            }
        }
    }
    bail!("not valid triples");
}
