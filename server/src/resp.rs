use anyhow::Result;
use nebula_fbthrift_common_v2::types::Row;
use nebula_fbthrift_common_v2::Value;
use nebula_fbthrift_graph_v2::{types::ErrorCode, ExecutionResponse};

pub fn get_row_value(row: Row) -> Result<Vec<String>> {
    let values = row.values;
    let mut res = vec![];
    for value in values {
        match value {
            Value::sVal(inner) => {
                res.push(String::from_utf8(inner)?);
            }
            _ => {}
        }
    }
    Ok(res)
}

/*
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionResponse {
    pub error_code: crate::types::ErrorCode,
    pub latency_in_us: ::std::primitive::i32,
    pub data: ::std::option::Option<common::types::DataSet>,
    pub space_name: ::std::option::Option<::std::vec::Vec<::std::primitive::u8>>,
    pub error_msg: ::std::option::Option<::std::vec::Vec<::std::primitive::u8>>,
    pub plan_desc: ::std::option::Option<crate::types::PlanDescription>,
    pub comment: ::std::option::Option<::std::vec::Vec<::std::primitive::u8>>,
}
*/

#[derive(Debug)]
pub struct Resp {
    error_code: ErrorCode,
    latency_in_us: i32,
    space_name: Option<String>,
    error_msg: Option<String>,
    comment: Option<String>,
    cols: Option<Vec<String>>,
    rows: Option<Vec<Vec<String>>>,
}

impl Resp {
    pub fn parse(resp: ExecutionResponse) -> Result<Self> {
        let ExecutionResponse {
            error_code,
            latency_in_us,
            data,
            space_name,
            error_msg,
            plan_desc,
            comment,
        } = resp;

        let byte2string = |x: Option<Vec<u8>>| x.map(|y| String::from_utf8(y).ok()).flatten();
        let space_name = byte2string(space_name);
        let error_msg = byte2string(error_msg);
        let comment = byte2string(comment);

        let mut cols = None;
        let mut rows = None;

        if let Some(data) = data {
            let cols_: Result<Vec<String>, _> = data
                .column_names
                .into_iter()
                .map(|x| String::from_utf8(x))
                .collect();
            cols = Some(cols_?);
            let rows_: Result<Vec<Vec<String>>> =
                data.rows.into_iter().map(|x| get_row_value(x)).collect();
            rows = Some(rows_?);
        }
        // bail!("unvalid resp!");
        let resp = Resp {
            error_code,
            latency_in_us,
            space_name,
            error_msg,
            comment,
            cols,
            rows,
        };

        Ok(resp)
    }
}
