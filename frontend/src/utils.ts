import * as d3 from "d3";

export function drag(simulation: any) {
  function dragstarted(event: any, d: any) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(event: any, d: any) {
    d.fx = event.x;
    d.fy = event.y;
  }

  function dragended(event: any, d: any) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }

  return d3
    .drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);
}

export function mycolor(d: any): any {
  // return d3.scaleOrdinal(d3.schemeCategory10)(d.category);

  // let scale = d3.scaleOrdinal(d3.schemeCategory10);
  // console.log(d.category);

  // const scales = d3.schemeCategory10;
  const scales = d3.schemeAccent;

  // console.log(scale(d.category + 1));
  return scales[d.category + 1];
}

// ------ zoom ------

function _handleZoom(e: any, selector: string) {
  d3.select(selector).attr("transform", e.transform);
}

function getZoom(selector: string): any {
  return d3.zoom().on("zoom", (e) => _handleZoom(e, selector));
}

export function initZoom(selector: string) {
  let zoom = getZoom(selector);
  d3.select("svg").call(zoom);
}
