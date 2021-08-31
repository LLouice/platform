import * as d3 from "d3";
import { Data, DatumForce, DLinks } from "./types";
import { drag, mycolor, initZoom } from "./utils";

// const data = require("./dumpy.json");

const categories = ["Symptom", "Disease", "Drug", "Department", "Check"];

// colors function
const colors = d3.scaleOrdinal().domain(categories).range(d3.schemeAccent);

// ----- filter state ------
type FilterState = {
  symphtom: boolean;
  disease: boolean;
  drug: boolean;
  department: boolean;
  check: boolean;
};

var filterState = {
  symphtom: true,
  disease: true,
  drug: true,
  department: true,
  check: true,
};

console.log(filterState);

async function fetchData() {
  const reps = await d3.json(
    "http://localhost:9090/get_out_links_d3?src_type=Symptom&name=肩背痛"
  );

  var nodes = reps["data"];
  var links = reps["links"];

  return { nodes: nodes, links: links };
}

async function main() {
  var { nodes, links } = await fetchData();
  // the all data
  const nodes_all = Object.assign(nodes);
  const links_all = Object.assign(links);

  console.log("&&&", nodes.length, links.length);
  const w: number = 800;
  const h: number = 600;

  // side effect! change data.nodes and data.links

  const simulation = d3
    .forceSimulation()
    // .force("link", forceLink.links(links))
    .force(
      "link",
      d3.forceLink().id(function (d: Data) {
        return d.id;
      })
    )
    .force("charge", d3.forceManyBody().strength(-1000))
    .force("center", d3.forceCenter(w / 2, h / 2))
    .on("tick", ticked);

  // ------ filter ------

  function _filter(cat: String) {
    console.log("=== in _filter ===");
    console.log("cat", cat); // Disease

    _toggleFilterState(cat);

    let nodes = Object.assign(nodes_all);
    let links = Object.assign(links_all);

    console.log(
      "[global]",
      nodes_all.length,
      links_all.length,
      nodes.length,
      links.length
    );

    console.log("filterState", filterState);

    let filteredNodes = nodes.filter((x) => {
      let c = x.category;
      console.log("filter nodes", c);
      if (!filterState.symphtom && c === 0) {
        return false;
      } else if (!filterState.disease && c === 1) {
        return false;
      } else if (!filterState.drug && c === 2) {
        return false;
      } else if (!filterState.department && c === 3) {
        return false;
      } else if (!filterState.check && c === 4) {
        return false;
      }
      return true;
    });

    let filteredEdges = links.filter((x) => {
      let c = x.target.category;
      console.log("filter edges", c);
      if (!filterState.symphtom && c === 0) {
        return false;
      } else if (!filterState.disease && c === 1) {
        return false;
      } else if (!filterState.drug && c === 2) {
        return false;
      } else if (!filterState.department && c === 3) {
        return false;
      } else if (!filterState.check && c === 4) {
        return false;
      }
      return true;
    });

    console.log("length: ", filteredNodes.length, filteredEdges.length);

    // simulation.stop();
    // simulation.nodes(filteredNodes);
    // // forceLink.links(filteredEdges);
    // //@ts-ignore
    // simulation.force("link").links(filteredEdges);

    refresh(filteredNodes, filteredEdges);
  }

  function _toggleFilterState(cat: String) {
    if (cat === "Disease") {
      filterState.disease = !filterState.disease;
    } else if (cat === "Drug") {
      filterState.drug = !filterState.drug;
    } else if (cat === "Department") {
      filterState.department = !filterState.department;
    } else if (cat === "Check") {
      filterState.check = !filterState.check;
    } else {
      console.log("Error: unreachable");
    }
  }

  function initNetwork() {
    // @types/d3 ValueFn::Result has bug!
    // @ts-ignore
    var svg = d3.select("#main").append("svg").attr("viewBox", [0, 0, w, h]);

    // ------ legend ------
    let legend_size = 20;

    // legend bind with categories data
    let legend = svg
      .append("g")
      .attr("class", "legend")
      .attr("transform", "translate(-80,0)")
      .selectAll("g.mydots")
      .data(categories)
      .join("g")
      // .attr("onclick", (d, i) => "filter" + categories[i] + "()")
      .attr("class", "mydots");

    legend
      .append("rect")
      .attr("class", "_filter")
      .attr("x", 100)
      .attr("y", (d, i) => 100 + i * (legend_size + 5))
      .attr("width", legend_size)
      .attr("height", legend_size)
      .attr("fill", (d) => colors(d));

    // Add one dot in the legend for each name.
    legend
      .append("text")
      .attr("x", 100 + legend_size * 1.2)
      .attr("y", function (d, i) {
        return 100 + i * (legend_size + 5) + legend_size / 2;
      }) // 100 is where the first dot appears. 25 is the distance between dots
      .text((d) => d)
      // .attr("fill", (d) => colors(d))
      .attr("text-anchor", "left")
      .style("alignment-baseline", "middle");

    // ------ legend end ------

    // ------ element ------
    // network in one group
    var g_network = svg.append("g").attr("class", "network");

    g_network
      .append("g")
      .attr("class", "links")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6);

    g_network
      .append("g")
      .attr("class", "nodes")
      .selectAll("g.node")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5);

    // ------ zoom ------
    initZoom("g.network");

    // filters
    d3.selectAll("._filter").on("click", (e, d) => {
      if (d !== "Symptom") {
        _filter(d);
      }
    });

    // return { g_nodes: g_nodes, g_links: g_links };
  }

  function refresh(nodes, links) {
    console.log("enter refresh");
    console.log("refresh::length", nodes.length, links.length);
    simulation.stop();
    simulation.nodes(nodes);
    simulation.force("link").links(links);
    // ----- join nodes & links ------
    let g_nodes = d3.select("g.nodes");
    g_nodes
      .selectAll("g.node") // one node group
      .data(nodes, (d: DatumForce) => d.id) // !!! key function, default base on index
      .join(
        function (enter) {
          console.log("node enter");
          console.log(enter);
          console.log(this);

          let g_node = enter.append("g").attr("class", "node");

          let node = g_node
            .append("circle")
            .attr("r", 20)
            .attr("fill", (d) => colors(categories[d.category]));

          let text = g_node
            .append("text")
            .style("text-anchor", "middle")
            .attr("y", 3)
            // .style("stroke-width", "2px")
            // .style("stroke-opacity", 0.75)
            // .style("stroke", "white")
            .style("font-size", "10px")
            .text(function (d: Data) {
              return d.name;
            })
            .style("pointer-events", "none");
          return g_node;
        },
        (update) => update,

        function (exit) {
          console.log("node exit");
          console.log(exit);
          console.log(this);
          return exit.transition().duration(300).style("opacity", 0).remove();
        }
      )
      .call(drag(simulation));

    let g_links = d3.select("g.links");
    g_links
      .selectAll("line.link")
      .data(links)
      .join(
        function (enter) {
          console.log("_filter");
          console.log("link enter");
          return enter.append("line").attr("class", "link");
        },
        (update) => update,
        function (exit) {
          console.log("link exit");
          return exit.transition().duration(300).style("opacity", 0).remove();
        }
      );

    // ticked (start?)
    // simulation.tick();
    simulation.alpha(1).restart();
  }

  // This function is run at each iteration of the force algorithm, updating the nodes position.
  function ticked() {
    // console.log("in ticked");
    // change element
    d3.selectAll("line.link")
      .attr("x1", function (d: DLinks) {
        return d.source.x;
      })
      .attr("y1", function (d: DLinks) {
        return d.source.y;
      })
      .attr("x2", function (d: DLinks) {
        return d.target.x;
      })
      .attr("y2", function (d: DLinks) {
        return d.target.y;
      });

    // not explicitly each, default all item use same function logic
    d3.selectAll("g.node").attr("transform", function (d: DatumForce) {
      // console.log("=== ticked end ===");
      return "translate(" + d.x + "," + d.y + ")";
    });

    // node
    //   .attr("cx", function (d) {
    //     return d.x;
    //   })
    //   .attr("cy", function (d) {
    //     return d.y;
    //   });
  }

  initNetwork();
  refresh(nodes, links);
}

main();
