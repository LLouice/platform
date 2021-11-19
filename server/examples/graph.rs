///! graph.rs is cli that transform neo4j raw triple txt to tfrecord dataset
///! the processing is:
///! (neo4j) graph.txt: 22028 5198 SYMPTOM_RELATE_DISEASE
///! graph.txt - load_graph -> split_dataset -> graph_trn/val/test.txt
///! graph_trn/val/test.txt - add_rev -> graph_trn/val/test_with_rev.txt
///! gen_tfrecord:  symptom_trn/val/test.tfrecord <- map_to_tfrecord  <- get_hr_ts_maps - graph_trn/val/test_with_rev.txt
use anyhow::{anyhow, Result};
use clap::{App, Arg};
use graph::prelude::*;
use num_enum::FromPrimitive;
use platform::init_env_logger;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::ops::Deref;
use std::path::PathBuf;
use std::str::FromStr;
use strum::{Display, EnumIter, EnumString};
// use delegate::delegate;
use rand::distributions::Distribution;
use statrs::distribution::Multinomial;
use tfrecord::{Example, ExampleWriter, Feature, RecordWriterInit};

fn main() -> Result<()> {
    init_env_logger!();

    let default_cat = RelationShip::SymptomRelateDisease.to_string();
    let mut app = App::new("kg")
        .version("1.0")
        .author("LLouice")
        .about("knowledge graph CLI")
        .license("MIT OR Apache-2.0")
        .arg(
            Arg::new("load")
                .short('l')
                .long("load")
                .about("load graph from text file"),
        )
        .arg(
            Arg::new("file")
                .short('f')
                .long("file")
                .takes_value(true)
                .about("export graph to text file"),
        )
        .arg(
            Arg::new("split")
                .short('s')
                .long("split")
                .requires("file")
                .requires("cat")
                .about("split dataset"),
        )
        .arg(
            Arg::new("suffix")
                .long("suffix")
                .takes_value(true)
                .default_value("")
                .about("suffix, play with cat"),
        )
        .arg(
            Arg::new("add_rev")
                .short('r')
                .long("add_rev")
                .requires("suffix")
                .about("add rev to graph text file"),
        )
        .arg(
            Arg::new("record")
                .short('R')
                .long("record")
                .requires("cat")
                .about("generate tfrecord"),
        )
        .arg(Arg::new("record_outfile").long("RO").about("tfrecord name"))
        .arg(
            Arg::new("full")
                .long("full")
                .requires("record")
                .about("generate full label"),
        )
        .arg(
            Arg::new("debug")
                .short('d')
                .long("debug")
                .about("debug / inspect the graph"),
        )
        .arg(
            Arg::new("cat")
                .short('c')
                .long("cat")
                .requires("suffix")
                .takes_value(true)
                .about("select category"),
        )
        .arg(Arg::new("size").long("size").about("dataset size"))
        .arg(Arg::new("ex").short('e').long("ex").about("do experiment"))
        .subcommand(
            App::new("filter").about("filter graph").arg(
                Arg::new("cat")
                    .short('c')
                    .long("cat")
                    .takes_value(true)
                    .default_value(default_cat.as_str())
                    .about("select category"),
            ),
        );

    let mut help = Vec::new();
    app.write_help(&mut help).expect("error on write app help");
    let help = String::from_utf8(help).expect("failed to convert help Vec<u8> to String");
    let matches = app.get_matches();

    let mut res = None;

    let cats = matches.values_of("cat");
    let suffix = matches.value_of("suffix").unwrap();

    let subset_relationship = match cats {
        Some(cats) => cats
            .map(|x| RelationShip::from_str(x))
            .collect::<Result<Vec<_>, _>>()
            .ok(),
        None => None,
    };

    if matches.is_present("load") {
        res = Some(Kg::load_from_file("graph_with_rev.txt"));
    } else if let Some(file) = matches.value_of("file") {
        res = Some(Kg::load_from_file(file));
    } else if matches.is_present("record") {
        Kg::gen_tfrecord(subset_relationship.clone(), suffix)?;
    } else if matches.is_present("add_rev") {
        log::info!("add_rev");
        graph_text::add_rev(&format!("graph{}_trn", suffix))?;
        graph_text::add_rev(&format!("graph{}_val", suffix))?;
        graph_text::add_rev(&format!("graph{}_test", suffix))?;
    } else if matches.is_present("size") {
        Kg::dataset_size(subset_relationship.clone(), suffix)?;
    } else if matches.is_present("ex") {
        lab::ex();
    } else if matches.subcommand_name().is_some() {
    } else {
        println!("{}", help);
    }

    if let Some(res) = res {
        match res {
            Err(e) => {
                log::error!("{:?}", e);
            }
            Ok(mut graph) => {
                graph.set_subset_relationship(subset_relationship);
                log::info!(
                    "graph:: nodes: {}, edges: {}",
                    graph.node_count(),
                    graph.edge_count()
                );
                log::info!("get graph successfully!");

                if matches.is_present("split") {
                    graph.split_dataset(suffix)?;
                }

                // if matches.is_present("record") {
                //     let full = matches.is_present("full");
                //     let outfile = matches.value_of("RO").unwrap_or_else(|| "symptom.tfrecord");
                //     graph.to_tfrecord(outfile, full)?;
                // }

                if matches.is_present("debug") {
                    graph.inspect();
                }
            }
        }
    }
    if let Some(matches) = matches.subcommand_matches("filter") {
        let cats: Result<Vec<_>, _> = matches
            .values_of("cat")
            .unwrap()
            .map(|x| RelationShip::from_str(x))
            .collect();
        filter::filter_graph(cats?.as_slice(), "dis")?;
    }
    eprintln!("main end");
    Ok(())
}

type KgDGraph = DirectedCsrGraph<usize, RelationShip>;
type Triple = (usize, usize, u8);
type HRTsM = HashMap<(i64, u8), Vec<i64>>;

struct Kg {
    inner: KgDGraph,
    subset_relationship: Option<Vec<RelationShip>>,
}

// TODO: make this generic
impl Kg {
    pub fn load_from_file(file: &str) -> Result<Self> {
        Ok(Self {
            inner: load_graph(file)?,
            subset_relationship: None,
        })
    }

    pub fn set_subset_relationship(&mut self, subset_relationship: Option<Vec<RelationShip>>) {
        self.subset_relationship = subset_relationship;
    }

    /// print hr_ts_map_trn(val, test) size
    fn dataset_size(subset_relationship: Option<Vec<RelationShip>>, suffix: &str) -> Result<()> {
        let [hr_ts_map_trn, hr_ts_map_val, hr_ts_map_test, hr_ts_map_all] =
            Self::get_hr_ts_maps(subset_relationship, suffix)?;

        println!("train_size: {}", hr_ts_map_trn.len());
        println!("val_size: {}", hr_ts_map_val.len());
        println!("test_size: {}", hr_ts_map_test.len());
        println!("all_size: {}", hr_ts_map_all.len());

        // num_ent
        let get_ent_set = |map: &HRTsM, name: &str| {
            let mut ent_set: HashSet<i64> = HashSet::new();
            for (&(h, r), ts) in map {
                ent_set.insert(h);
                println!("[{}], h: {}, r: {}, ts: {:?}", name, h, r, ts);
                for t in ts.clone() {
                    ent_set.insert(t);
                }
            }
            ent_set
        };
        let ent_set_trn = get_ent_set(&hr_ts_map_trn, "trn");
        let ent_set_val = get_ent_set(&hr_ts_map_val, "val");
        let ent_set_test = get_ent_set(&hr_ts_map_test, "test");
        let ent_set_all = get_ent_set(&hr_ts_map_all, "all");
        assert!(ent_set_trn.is_superset(&ent_set_val));
        assert!(ent_set_trn.is_superset(&ent_set_test));
        assert!(ent_set_trn.is_superset(&ent_set_all));
        assert!(ent_set_trn.is_subset(&ent_set_all));
        let num_ent = ent_set_trn.len();
        println!("num_ent: {}", num_ent);
        Ok(())
    }

    /// separate all edges into trn/val/test_set(triple set) and write these set to graph_trn/val/test.txt
    /// for splitting into corresponding ratio, use multinomial to generate [1., 0., 0.] / [0., 1., 0.1] / [0., 1., 0.1] to
    /// decide trn/test/val dataset util reach to size
    /// This function also check the constraint:
    ///          1. trn/val/test_set not all disjoint
    ///          2. trn_ent_set is the superset of val/test_ent_set
    pub fn split_dataset(&self, suffix: &str) -> Result<()> {
        log::info!("split_dataset...");
        let nodes_num = self.inner.node_count();
        // node degree map
        let mut degrees: Vec<usize> = (0..nodes_num).collect();
        self.inner
            .for_each_node_par(&mut degrees, move |g, node, node_state| {
                let degree = g.out_degree(node) + g.in_degree(node);
                *node_state = degree;
            })
            .unwrap();
        let mut degree_map: HashMap<usize, usize> = degrees.into_iter().enumerate().collect();

        // all edges
        let all_sz = self.inner.edge_count();
        const TRN_RATIO: f64 = 0.7;
        const TEST_RATIO: f64 = 0.1;
        let trn_sz: usize = (all_sz as f64 * TRN_RATIO) as usize;
        let test_sz: usize = (all_sz as f64 * TEST_RATIO) as usize;
        let val_sz = all_sz - trn_sz - test_sz;

        // random select <- random number
        // Multinomial Distribution
        let d = Multinomial::new(&[TRN_RATIO, TEST_RATIO, 1. - TRN_RATIO - TEST_RATIO], 1)
            .expect("failed to build Multinomial distribution");
        let mut rng = rand::thread_rng();

        // iterate all nodes to separate edge
        // send all edges
        let (tx, rx) = std::sync::mpsc::channel();
        let mut txs: Vec<_> = vec![Some(tx); nodes_num];
        self.inner
            .for_each_node_par(&mut txs, move |g, node, tx| {
                let tx = tx.take().unwrap();
                let targets = g.out_neighbors_with_values(node);
                for x in targets {
                    let rel = x
                        .value
                        .to_u8(self.subset_relationship.as_ref().map(|x| x.as_slice()));
                    let tail = x.target;
                    tx.send((node, tail, rel)).expect("send fail!");
                }
            })
            .unwrap();

        // cut the edge
        fn cut_edge(h: usize, t: usize, degree_map: &mut HashMap<usize, usize>) -> bool {
            let d_h = *degree_map.get(&h).unwrap();
            let d_t = *degree_map.get(&t).unwrap();

            if d_h > 1 && d_t > 1 {
                *degree_map.get_mut(&h).unwrap() -= 1;
                *degree_map.get_mut(&t).unwrap() -= 1;
                return true;
            }

            return false;
        }

        // select
        let mut trn_set = HashSet::new();
        let mut val_set = HashSet::new();
        let mut test_set = HashSet::new();

        let mut filter = |triple: Triple,
                          val_set: &mut HashSet<Triple>,
                          test_set: &mut HashSet<Triple>|
         -> bool {
            let (h, t, r) = triple;
            let v = d.sample(&mut rng);
            // filter
            match v.as_slice() {
                [0., 1., 0.] => {
                    if test_set.len() < test_sz {
                        if cut_edge(h, t, &mut degree_map) {
                            test_set.insert((h, t, r));
                            return true;
                        }
                    }
                }
                [0., 0., 1.] => {
                    if val_set.len() < val_sz {
                        if cut_edge(h, t, &mut degree_map) {
                            val_set.insert((h, t, r));
                            return true;
                        }
                    }
                }
                _ => return false,
            }
            return false;
        };

        for triple in rx.iter() {
            if !filter(triple, &mut val_set, &mut test_set) {
                trn_set.insert(triple);
            }
        }
        while val_set.len() < val_sz || test_set.len() < test_sz {
            let mut cache = Vec::new();
            for triple in trn_set.iter() {
                if filter(*triple, &mut val_set, &mut test_set) {
                    cache.push(*triple);
                }
            }
            for x in &cache {
                trn_set.remove(x);
            }
        }

        assert_eq!(val_set.len(), val_sz);
        assert_eq!(test_set.len(), test_sz);
        assert_eq!(trn_set.len() + val_sz + test_sz, all_sz);
        assert!(trn_set.is_disjoint(&val_set));
        assert!(trn_set.is_disjoint(&test_set));
        assert!(val_set.is_disjoint(&test_set));

        // check all val / test entity in trn_set
        let get_ent_set = |set| {
            let mut ent_set = HashSet::new();
            for &(h, t, _) in set {
                ent_set.insert(h);
                ent_set.insert(t);
            }
            ent_set
        };
        let trn_ent_set: HashSet<usize> = get_ent_set(&trn_set);
        let val_ent_set = get_ent_set(&val_set);
        let test_ent_set = get_ent_set(&test_set);
        assert!(trn_ent_set.is_superset(&val_ent_set));
        assert!(trn_ent_set.is_superset(&test_ent_set));

        // write to file
        graph_text::write_set_to_file(trn_set, &format!("graph{}_trn.txt", suffix))?;
        graph_text::write_set_to_file(test_set, &format!("graph{}_test.txt", suffix))?;
        graph_text::write_set_to_file(val_set, &format!("graph{}_val.txt", suffix))?;

        Ok(())
    }

    ///  get hr_ts_trn/val/test (<-graph_trn/val/test_with_rev.txt);
    ///  and hr_ts_all (combine three);
    pub fn get_hr_ts_maps(
        subset_relationship: Option<Vec<RelationShip>>,
        suffix: &str,
    ) -> Result<[HRTsM; 4]> {
        let hr_ts_map_trn = Self::get_hr_ts_map(
            &format!("graph{}_trn_with_rev.txt", suffix),
            subset_relationship.clone(),
        )?;
        let hr_ts_map_val = Self::get_hr_ts_map(
            &format!("graph{}_val_with_rev.txt", suffix),
            subset_relationship.clone(),
        )?;
        let hr_ts_map_test = Self::get_hr_ts_map(
            &format!("graph{}_test_with_rev.txt", suffix),
            subset_relationship,
        )?;

        let mut hr_ts_map_all = HashMap::with_capacity(
            hr_ts_map_trn.len() + hr_ts_map_val.len() + hr_ts_map_test.len(),
        );

        let mut ext = |m: &HRTsM| {
            for (k, v) in m.iter() {
                let v2 = hr_ts_map_all.entry(k.to_owned()).or_insert(vec![]);
                (*v2).extend_from_slice(v);
            }
        };
        ext(&hr_ts_map_trn);
        ext(&hr_ts_map_val);
        ext(&hr_ts_map_test);

        Ok([hr_ts_map_trn, hr_ts_map_val, hr_ts_map_test, hr_ts_map_all])
    }

    /// iterate all nodes, for each node, get all it's out_neighbors => (h,r)->(ts:t1,t2,..,tn)
    fn get_hr_ts_map(file: &str, subset_relationship: Option<Vec<RelationShip>>) -> Result<HRTsM> {
        let graph = Self::load_from_file(file)?;

        let nodes_num = graph.inner.node_count();
        let (tx, rx) = std::sync::mpsc::channel();
        let mut txs: Vec<_> = vec![Some(tx); nodes_num];
        graph
            .inner
            .for_each_node_par(&mut txs, move |g, node, tx| {
                let mut hr_ts_map = HashMap::new();
                let targets = g.out_neighbors_with_values(node);
                targets.iter().for_each(|x| {
                    let rel = x
                        .value
                        .to_u8(subset_relationship.as_ref().map(|x| x.as_slice()));
                    let tail = x.target;
                    hr_ts_map
                        .entry((node as i64, rel))
                        .or_insert(Vec::with_capacity(1))
                        .push(tail as i64);
                });

                // important: drop the tx
                let tx = tx.take();
                tx.unwrap()
                    .send(hr_ts_map)
                    .expect("failed to send hr_t_map");
            })?;

        let mut hr_ts_map = HashMap::new();
        while let Ok(map) = rx.recv_timeout(std::time::Duration::from_secs(5)) {
            for (key, values) in map {
                hr_ts_map.insert(key, values);
            }
        }
        Ok(hr_ts_map)
    }

    /// HRTsM -> tfrecord
    /// This function check the no.2(trn_ent_set is all ent_set) constraint again
    pub fn gen_tfrecord(
        subset_relationship: Option<Vec<RelationShip>>,
        suffix: &str,
    ) -> Result<()> {
        log::info!("gen_tfrecord...");
        let [hr_ts_map_trn, hr_ts_map_val, hr_ts_map_test, hr_ts_map_all] =
            Self::get_hr_ts_maps(subset_relationship, suffix)?;

        // num_ent
        let get_ent_set = |map: &HRTsM| {
            let mut ent_set: HashSet<i64> = HashSet::new();
            for (&(h, _), ts) in map {
                ent_set.insert(h);
                for t in ts.clone() {
                    ent_set.insert(t);
                }
            }
            ent_set
        };
        let ent_set_trn = get_ent_set(&hr_ts_map_trn);
        let ent_set_val = get_ent_set(&hr_ts_map_val);
        let ent_set_test = get_ent_set(&hr_ts_map_test);
        let ent_set_all = get_ent_set(&hr_ts_map_all);
        assert!(ent_set_trn.is_superset(&ent_set_val));
        assert!(ent_set_trn.is_superset(&ent_set_test));
        assert!(ent_set_trn.is_superset(&ent_set_all));
        assert!(ent_set_trn.is_subset(&ent_set_all));
        let num_ent = ent_set_trn.len();

        Self::map_to_tfrecord(
            hr_ts_map_trn,
            None,
            &format!("symptom{}_trn.tfrecord", suffix),
            num_ent,
        )?;
        Self::map_to_tfrecord(
            hr_ts_map_val,
            Some(&hr_ts_map_all),
            &format!("symptom{}_val.tfrecord", suffix),
            num_ent,
        )?;
        Self::map_to_tfrecord(
            hr_ts_map_test,
            Some(&hr_ts_map_all),
            &format!("symptom{}_test.tfrecord", suffix),
            num_ent,
        )?;
        Ok(())
    }

    /// trn/(val, test) HRTsM((h,r)->(ts: t1, t2,..,tn) -> tfrecord
    /// Record feature field:
    ///   trn: input; label; num_ent
    ///   dev: input; label; num_ent; aux_label
    fn map_to_tfrecord(
        map: HRTsM,
        map_all: Option<&HRTsM>,
        outfile: &str,
        num_ent: usize, // full: Option<usize>,
    ) -> Result<()> {
        let mut writer: ExampleWriter<_> = RecordWriterInit::create(outfile)?;
        let mut count = 0;

        for (key, values) in map {
            count += 1;
            if count % 500 == 0 {
                log::debug!("COUNT: {}", count);
            }
            let input_feature = Feature::Int64List(vec![key.0 as i64, key.1 as i64]);

            // let labels = if let Some(nodes_num) = full {
            //     let mut labels = vec![0_i64; nodes_num];
            //     values.into_iter().for_each(|v| {
            //         labels[v as usize] = v;
            //     });
            //     labels
            // } else {
            //     values
            // };

            let num_ent_feature = Feature::Int64List(vec![num_ent as i64]);
            let labels = values;
            if let Some(map_all) = map_all {
                let label_feature = Feature::Int64List(labels);
                let aux_feature = Feature::Int64List(map_all[&key].clone());

                let example = vec![
                    ("input".into(), input_feature),
                    ("label".into(), label_feature),
                    ("num_ent".into(), num_ent_feature),
                    ("aux_label".into(), aux_feature),
                ]
                .into_iter()
                .collect::<Example>();
                writer
                    .send(example)
                    .expect("failed to write example into tfrecord");
            } else {
                let label_feature = Feature::Int64List(labels);

                let example = vec![
                    ("input".into(), input_feature),
                    ("label".into(), label_feature),
                    ("num_ent".into(), num_ent_feature),
                ]
                .into_iter()
                .collect::<Example>();
                writer
                    .send(example)
                    .expect("failed to write example into tfrecord");
            }
        }

        log::info!("write done!");
        Ok(())
    }

    #[deprecated = "use gen_tfrecord"]
    pub fn to_tfrecord(&self, outfile: &str, full: bool) -> Result<()> {
        let nodes_num = self.inner.node_count();

        let (tx, rx) = std::sync::mpsc::channel();
        let mut txs: Vec<_> = vec![Some(tx); nodes_num];
        self.inner.for_each_node_par(&mut txs, move |g, node, tx| {
            let mut hr_ts_map = HashMap::new();
            let targets = g.out_neighbors_with_values(node);
            targets.iter().for_each(|x| {
                let rel = x
                    .value
                    .to_u8(self.subset_relationship.as_ref().map(|x| x.as_slice()));
                let tail = x.target;
                hr_ts_map
                    .entry((node, rel))
                    .or_insert(Vec::with_capacity(1))
                    .push(tail as i64);
            });

            // important: drop the tx
            let tx = tx.take();
            tx.unwrap()
                .send(hr_ts_map)
                .expect("failed to send hr_t_map");
        })?;

        let mut writer: ExampleWriter<_> = RecordWriterInit::create(outfile)?;
        let mut count = 0;
        // write to record
        while let Ok(hr_t_map) = rx.recv_timeout(std::time::Duration::from_secs(5)) {
            count += 1;
            if count % 500 == 0 {
                log::debug!("COUNT: {}", count);
            }
            for (key, values) in hr_t_map {
                let input_feature = Feature::Int64List(vec![key.0 as i64, key.1 as i64]);

                let labels = if full {
                    let mut labels = vec![0_i64; nodes_num];
                    values.into_iter().for_each(|v| {
                        labels[v as usize] = v;
                    });
                    labels
                } else {
                    values
                };
                let label_feature = Feature::Int64List(labels);

                let example = vec![
                    ("input".into(), input_feature),
                    ("label".into(), label_feature),
                ]
                .into_iter()
                .collect::<Example>();
                writer
                    .send(example)
                    .expect("failed to write example into tfrecord");
            }
        }
        log::info!("write done!");
        Ok(())
    }

    pub fn inspect(&self) {
        let mut stat_map = BTreeMap::new();
        let nodes_num = self.inner.node_count();

        let (tx, rx) = std::sync::mpsc::channel();
        let mut txs: Vec<_> = vec![Some(tx); nodes_num];
        self.inner
            .for_each_node_par(&mut txs, move |g, node, tx| {
                let tx = tx.take();
                // let degree = g.out_degree(node)+g.in_degree(node);
                let degree = g.out_degree(node);
                tx.unwrap().send(degree).expect("send fail!");
            })
            .unwrap();
        for degree in rx.iter() {
            *stat_map.entry(degree).or_insert(0) += 1;
        }
        // log::info!("{:#?}", stat_map);
    }

    // TODO: define Neighbors
    // use indexmap?
    // return Vec<Neighbors>
    // pub fn find_all_neighbors(&self, nodes: Vec<usize>) -> Vec<&Target<usize, RelationShip>> {
    //     // parallel
    //     let mut in_neighbors = self.inner.in_neighbors_with_values(0);
    //     let out_neighbors = self.inner.out_neighbors_with_values(0);
    //     in_neighbors.iter().chain(out_neighbors).collect()
    // }
}

impl Deref for Kg {
    type Target = KgDGraph;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

fn load_graph(file: &str) -> Result<KgDGraph> {
    let file = [env!("CARGO_MANIFEST_DIR"), file]
        .iter()
        .collect::<PathBuf>();

    let graph: KgDGraph = GraphBuilder::new()
        .csr_layout(CsrLayout::Sorted)
        .file_format(EdgeListInput::default())
        .path(file)
        .build()?;
    // .expect("loading failed");

    Ok(graph)
}

/// only SYMPTOM_RELATE_* current
// #[rustfmt::skip]
#[non_exhaustive]
#[derive(
    Debug,
    Display,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    EnumIter,
    EnumString,
    FromPrimitive,
)]
#[repr(u8)]
enum RelationShip {
    #[strum(
        serialize = "SYMPTOM_RELATE_DISEASE",
        serialize = "SymptomRelateDisease"
    )]
    SymptomRelateDisease = 0,
    #[strum(serialize = "SYMPTOM_RELATE_DRUG", serialize = "SymptomRelateDrug")]
    SymptomRelateDrug = 1,
    #[strum(
        serialize = "SYMPTOM_RELATE_DEPARTMENT",
        serialize = "SymptomRelateDepartment"
    )]
    SymptomRelateDepartment = 2,
    #[strum(serialize = "SYMPTOM_RELATE_CHECK", serialize = "SymptomRelateCheck")]
    SymptomRelateCheck = 3,
    #[strum(serialize = "SYMPTOM_RELATE_AREA", serialize = "SymptomRelateArea")]
    SymptomRelateArea = 4,

    #[strum(
        serialize = "SYMPTOM_RELATE_DISEASE_REV",
        serialize = "SymptomRelateDiseaseRev"
    )]
    SymptomRelateDiseaseRev = 5,
    #[strum(
        serialize = "SYMPTOM_RELATE_DRUG_REV",
        serialize = "SymptomRelateDrugRev"
    )]
    SymptomRelateDrugRev = 6,
    #[strum(
        serialize = "SYMPTOM_RELATE_DEPARTMENT_REV",
        serialize = "SymptomRelateDepartmentRev"
    )]
    SymptomRelateDepartmentRev = 7,
    #[strum(
        serialize = "SYMPTOM_RELATE_CHECK_REV",
        serialize = "SymptomRelateCheckRev"
    )]
    SymptomRelateCheckRev = 8,
    #[strum(
        serialize = "SYMPTOM_RELATE_AREA_REV",
        serialize = "SymptomRelateAreaRev"
    )]
    SymptomRelateAreaRev = 9,

    #[num_enum(default)]
    #[strum(disabled)]
    Nonsense,
}

impl RelationShip {
    /// convert RelationShip variant to u8, the set can be subset of the whole RelationShip
    pub fn to_u8(self, subset: Option<&[RelationShip]>) -> u8 {
        let rel_idx = self as u8;
        if let Some(subset) = subset {
            let sz = subset.len();
            let rel_idx_not_rev = if rel_idx < 5 { rel_idx } else { rel_idx - 5 };

            let idx = subset
                .iter()
                .position(|x| *x as u8 == rel_idx_not_rev)
                .expect(&format!(
                    "the rel: {:?} not in subset: {:?}",
                    RelationShip::from(rel_idx_not_rev),
                    subset
                ));
            if sz == 1 {
                assert_eq!(idx, 0);
            }
            if rel_idx < 5 {
                idx as u8
            } else {
                (idx + sz) as u8
            }
        } else {
            rel_idx
        }
    }
}

impl Default for RelationShip {
    fn default() -> Self {
        Self::Nonsense
    }
}

/// implement `ParseValue` for load graph form graph.txt
impl ParseValue for RelationShip {
    fn parse(bytes: &[u8]) -> (Self, usize) {
        // find '\n'
        let len = bytes.iter().take_while(|&&b| b != b'\n').count();

        (
            RelationShip::from_str(
                std::str::from_utf8(&bytes[..len]).expect("failed to convert str from utf8"),
            )
            .expect("failed to convert str to RelationShip variant"),
            len,
        )
    }
}

mod graph_text {
    use super::*;

    use std::fs::{File, OpenOptions};
    use std::io::{BufRead, BufReader, BufWriter, Write};

    /// dump triple set to file
    pub(crate) fn write_set_to_file(set: HashSet<Triple>, outfile: &str) -> Result<()> {
        let outfile = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(outfile)?;
        let mut stream = BufWriter::new(outfile);
        for (h, t, r) in set {
            let line = format!("{} {} {}", h, t, RelationShip::from(r));
            writeln!(stream, "{}", &line)?;
            // writeln!(stream, "{}REV", line);
        }
        Ok(())
    }

    /// graph_trn(val, test).txt -> grah_trn(val, test)_with_rev.txt
    pub(crate) fn add_rev(infile: &str) -> Result<()> {
        let outfile = format!("{}_with_rev.txt", infile);
        let infile = format!("{}.txt", infile);

        let outfile = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(outfile)?;
        let mut stream = BufWriter::new(outfile);

        let mut count = 0;
        for line in BufReader::new(File::open(infile)?).lines() {
            if let Ok(line) = line {
                if count == 0 {
                    dbg!(&line);
                    count += 1;
                }
                writeln!(stream, "{}", line)?;
                let splits: Vec<_> = line.split_whitespace().collect();
                writeln!(stream, "{} {} {}_REV", splits[1], splits[0], splits[2])?;
            }
        }
        Ok(())
    }
}

/// graph.txt id_name_map.json -> graph_dis.txt id_name_dis_map.json name_id_dis_map.json
mod filter {
    use std::fs::OpenOptions;

    use super::*;
    pub(super) fn filter_graph(cats: &[RelationShip], suffix: &str) -> Result<()> {
        let id_name_map: HashMap<usize, String> = {
            let reader = BufReader::new(File::open("id_name_map.json")?);
            serde_json::from_reader(reader)?
        };

        let mut filtered_triples: HashSet<(String, String, u8)> = HashSet::new();
        let mut filtered_name_id_map: HashMap<String, usize> = HashMap::new();

        let filename = "graph.txt";
        for (idx, line) in BufReader::new(File::open(filename)?).lines().enumerate() {
            if let Ok(line) = line {
                let triple = line.trim().split_whitespace().collect::<Vec<_>>();
                let h = id_name_map
                    .get(
                        &triple
                            .get(0)
                            .ok_or_else(|| anyhow!("no head in triple: {:?}", triple))?
                            .parse::<usize>()?,
                    )
                    .ok_or_else(|| anyhow!("triple: {:?} head not in id_name_map"))?;
                let t = id_name_map
                    .get(
                        &triple
                            .get(1)
                            .ok_or_else(|| anyhow!("no tail in triple: {:?}", triple))?
                            .parse::<usize>()?,
                    )
                    .ok_or_else(|| anyhow!("triple: {:?} tail not in id_name_map"))?;

                let rel = triple
                    .get(2)
                    .ok_or_else(|| anyhow!("no rel in triple: {:?}", triple))?;
                let rel = RelationShip::from_str(rel)?;
                let mut is_selected = false;
                // let mut rel_idx = 0;
                let rel_idx = rel as u8;
                for (idx, cat) in cats.into_iter().enumerate() {
                    if rel == *cat {
                        is_selected = true;
                        // rel_idx = idx;
                        break;
                    }
                    is_selected = false;
                }
                if is_selected {
                    // println!("{} {} {}", h, t, rel);
                    filtered_triples.insert((h.clone(), t.clone(), rel_idx as u8));
                    let sz = filtered_name_id_map.len();
                    filtered_name_id_map.entry(h.clone()).or_insert(sz);
                    let sz = filtered_name_id_map.len();
                    filtered_name_id_map.entry(t.clone()).or_insert(sz);
                }
                // break;
            } else {
                eprintln!("error in line {}: {:?}", idx, line);
            }
        }

        let outfile = format!("graph_{}.txt", suffix); // e.g. graph_dis.txt
        let outfile = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(outfile)?;
        let mut stream = BufWriter::new(outfile);
        for (h, t, rel) in filtered_triples {
            let h = *filtered_name_id_map
                .get(&h)
                .ok_or_else(|| anyhow!("head {} not in filtered_name_id_map", h))?;
            let t = *filtered_name_id_map
                .get(&t)
                .ok_or_else(|| anyhow!("tail {} not in filtered_name_id_map", h))?;
            writeln!(stream, "{} {} {}", h, t, RelationShip::from(rel))?;
        }

        println!("num_ent: {:?}", filtered_name_id_map.len());

        let filtered_id_name_map: HashMap<usize, String> = filtered_name_id_map
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect();
        // dump the map
        let map_string = serde_json::to_string(&filtered_name_id_map)?;
        std::fs::write(format!("name_id{}_map.json", suffix), map_string)?;

        let map_string = serde_json::to_string(&filtered_id_name_map)?;
        std::fs::write(format!("id_name{}_map.json", suffix), map_string)?;
        Ok(())
    }
}

fn demo() {
    let graph: DirectedCsrGraph<usize> = GraphBuilder::new()
        .edges(vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
        .build();

    assert_eq!(graph.node_count(), 4);
    assert_eq!(graph.edge_count(), 5);

    assert_eq!(graph.out_degree(1), 2);
    assert_eq!(graph.in_degree(1), 1);

    assert_eq!(graph.out_neighbors(1), &[2, 3]);
    assert_eq!(graph.in_neighbors(1), &[0]);
}

mod dist {
    fn get_multinomial() {
        todo!()
    }
}

mod lab {
    use super::*;

    const TRN_RATIO: f64 = 0.7;
    const TEST_RATIO: f64 = 0.1;

    pub fn ex() {
        let d = Multinomial::new(&[TRN_RATIO, TEST_RATIO, 1. - TRN_RATIO - TEST_RATIO], 1)
            .expect("failed to build Multinomial distribution");
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let v = d.sample(&mut rng);
            println!("{:?}", v);
        }
    }
}
