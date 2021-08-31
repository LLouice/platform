use anyhow::Result;
use rio_api::model::NamedNode;
use rio_api::parser::TriplesParser;
use rio_turtle::{TurtleError, TurtleParser};

fn main() -> Result<()> {
    let file = b"@prefix schema: <http://schema.org/> .
<http://example.com/foo> a schema:Person ;
    schema:name  \"Foo\" .
<http://example.com/bar> a schema:Person ;
    schema:name  \"Bar\" .";

    let rdf_type = NamedNode {
        iri: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    };
    let schema_person = NamedNode {
        iri: "http://schema.org/Person",
    };
    let mut count = 0;
    TurtleParser::new(file.as_ref(), None).parse_all(&mut |t| {
        if t.predicate == rdf_type && t.object == schema_person.into() {
            count += 1;
        }
        Ok(()) as Result<(), TurtleError>
    })?;
    assert_eq!(2, count);
    Ok(())
}
