use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::convert::From;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
pub struct Triple {
    pub head: String,
    pub rel: String,
    pub tail: String,
}

impl Triple {
    pub fn write2file<P: AsRef<Path>>(
        outfile: P,
        triples: &[Triple],
        truncate: bool,
    ) -> Result<()> {
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
}

impl From<Vec<String>> for Triple {
    fn from(mut vs: Vec<String>) -> Self {
        let (tail, rel, head) = (vs.pop().unwrap(), vs.pop().unwrap(), vs.pop().unwrap());
        Self { head, rel, tail }
    }
}
