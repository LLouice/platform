
    let link = svg
      .append("g")
      .attr("class", "links")
      .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6);
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("class", "link");

node
  .append("circle")
  .attr("r", 20)
  .attr("fill", (d) => colors(categories[d.category]));
// .attr("fill", (d) => colors(4));

node
  .append("text")
  .style("text-anchor", "middle")
  .attr("y", 3)
  // .style("stroke-width", "2px")
  // .style("stroke-opacity", 0.75)
  // .style("stroke", "white")
  .style("font-size", "10px")
  // .style
  .text(function (d: Data) {
    return d.name;
  })
  .style("pointer-events", "none");

/*

  function _filter(cat: String) {
    console.log("=== in _filter ===");
    let currentNodes = d3.selectAll("g.node").data();
    let currentEdges = d3.selectAll("line.link").data();

    console.log("click", cat);
    console.log(currentNodes);
    console.log("---");
    console.log(currentEdges);

    let c = categories.indexOf(cat);
    let filteredNodes = currentNodes.filter(function (x) {
      return x.category !== c;
    });

    let filteredEdges = currentEdges.filter(function (x) {
      console.log(x.target.name, "=>", c);
      return x.target.category !== c;
    });

    simulation.stop();
    console.log(currentNodes.length, currentEdges.length);
    console.log(filteredNodes.length, filteredEdges.length);
    simulation.nodes(filteredNodes);
    // forceLink.links(filteredEdges);
    simulation.force("link").links(filteredEdges);

    d3.selectAll("g.node")
      .data(filteredNodes, function (d) {
        return d.id;
      })
      .exit()
      .transition()
      .duration(500)
      .style("opacity", 0)
      .remove();

    console.log("***", filteredEdges.length);

    d3.selectAll("line.link")
      .data(filteredEdges)
      .exit()
      .transition()
      .duration(500)
      .style("opacity", 0)
      .remove();

    // simulation.tick();
    simulation.alpha(1).restart();
    // updateNetwork(filteredNodes, filteredEdges);
    // simulation.alpha(1).restart();

    console.log("=== end _filter ===");
  }
      */




    // ------ simulation ------

    /*
    d3.select("g.nodes")
      .selectAll("g.node")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
      .data(filteredNodes, function (d: DatumForce) {
        return d.id;
      })
      .join(
        function (enter) {
          return enter
            .append("g.node")
            .append("circle")
            .attr("r", 20)
            .attr("fill", (d) => colors(categories[d.category]));
        },
        function (exit) {
          return exit.transition().duration(300).style("opacity", 0).remove();
        }
      );
      */
    d3.selectAll("g.node")
      .data(filteredNodes, function (d: DatumForce) {
        return d.id;
      })
      .exit()
      .transition()
      .duration(300)
      .style("opacity", 0)
      .remove();

    d3.selectAll("line.link")
      .data(filteredEdges)
      .exit()
      .transition()
      .duration(300)
      .style("opacity", 0)
      .remove();

    // simulation.tick();
    simulation.alpha(1).restart();
    // updateNetwork(filteredNodes, filteredEdges);
    // simulation.alpha(1).restart();

    console.log("=== end _filter ===");
