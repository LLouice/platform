use anyhow::{anyhow, Result};
use rio_api::model::{NamedNode, NamedOrBlankNode, Term};
use rio_api::parser::TriplesParser;
use sophia_api::triple::stream::TripleSource;

use rio_turtle::TurtleParser;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::BufReader;
use std::io::Write;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
struct Triple {
    head: String,
    rel: String,
    tail: String,
}

impl Triple {
    fn parse(subject: NamedOrBlankNode, predicate: NamedNode, object: Term) -> Result<Self> {
        let head = parse_head(subject)?;
        let rel = predicate.iri.to_owned();
        let tail = parse_tail(object)?;
        let head = parse_word(head).ok_or_else(|| anyhow!("head is not a valid word"))?;

        let rel = parse_word(rel).ok_or_else(|| anyhow!("rel is not a valid word"))?;
        let tail = parse_word(tail).ok_or_else(|| anyhow!("tail is not a valid word"))?;
        Ok(Self { head, rel, tail })
    }
}

fn main() -> Result<()> {
    let st = std::time::Instant::now();
    let mut args = std::env::args();
    args.next();
    let file = args.next().expect("must give a ttl file name");
    let filename = format!("data/chinese_symptom/{}.ttl", file);
    let outfile: String = args
        .next()
        .or(Some("data/chinese_symptom/triples.json".to_string()))
        .unwrap();

    let mut err_file1 = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open("data/chinese_symptom/not_triple.txt")?;
    let mut err_file2 = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open("data/chinese_symptom/not_valid.txt")?;

    // 默认truncate is true
    let mut truncate = if args.next().is_none() { true } else { false };

    let mut triples = vec![];
    let mut count = 0;

    let mut parser = TurtleParser::new(BufReader::new(File::open(filename)?), None);
    // let mut parser = TurtleParser::new(Cursor::new(File::open(filename)?), None);

    while !parser.is_end() {
        let step = parser.for_each_triple(&mut |t| {
            count += 1;
            println!("line: {}", count);
            let subject = t.subject;
            let predicate = t.predicate;
            let object = t.object;
            let triple = Triple::parse(subject, predicate, object);
            if triple.is_ok() {
                println!("{:#?}", triple);
                triples.push(triple.unwrap());
            } else {
                // eprintln!("line:[{:?}], triple: {:#?}\n", count, t);
                writeln!(err_file1, "line:[{:?}], triple: {:#?}\n", count, t);
            }
            if triples.len() == 10000 {
                write2file(&outfile, &triples, truncate)?;

                if truncate {
                    truncate = !truncate;
                }
                triples.clear();
            }

            Ok(()) as Result<()>
        });

        if let Err(e) = step {
            println!("line: {} over", count);
            println!("{}", parser.is_end());
            eprintln!("{:?}", e);
            // writeln!(err_file2, "{:?}", e);
        }
    }

    // write the last triples
    write2file(&outfile, &triples, truncate)?;

    println!("triples count: {:?}", triples.len());
    println!("time cost: {:?} seconds", st.elapsed().as_secs());
    Ok(())
}

fn parse_head(subject: NamedOrBlankNode) -> Result<String> {
    let head: Option<String> = if let NamedOrBlankNode::NamedNode(nn) = subject {
        Some(nn.iri.to_owned())
    } else {
        None
    };
    head.ok_or_else(|| anyhow!("no head!"))
}

fn parse_tail(object: Term) -> Result<String> {
    let tail: Option<String> = if let Term::NamedNode(nn) = object {
        Some(nn.iri.to_owned())
    } else {
        None
    };
    tail.ok_or_else(|| anyhow!("no tail!"))
}

fn parse_word<P: AsRef<str>>(string: P) -> Option<String> {
    let string = string.as_ref();
    if !string.contains("#") {
        string.split("/").last().map(|x| x.to_string())
    } else {
        None
    }
}

fn write2file<P: AsRef<Path>>(outfile: P, triples: &[Triple], truncate: bool) -> Result<()> {
    let mut opt = OpenOptions::new();
    opt.write(true).create(true);
    if truncate {
        opt.truncate(true);
    } else {
        opt.append(true);
    }

    let mut outfile = opt.open(outfile)?;

    for triple in triples.into_iter() {
        let json = serde_json::to_string(&triple)?;
        writeln!(outfile, "{}", json)?;
    }

    Ok(())
}
