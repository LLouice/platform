"use strict";
exports.__esModule = true;
exports.mycolor = exports.drag = void 0;
var d3 = require("d3");
function drag(simulation) {
    function dragstarted(event, d) {
        if (!event.active)
            simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    function dragended(event, d) {
        if (!event.active)
            simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    return d3
        .drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
}
exports.drag = drag;
function mycolor(d) {
    // return d3.scaleOrdinal(d3.schemeCategory10)(d.category);
    // let scale = d3.scaleOrdinal(d3.schemeCategory10);
    // console.log(d.category);
    // const scales = d3.schemeCategory10;
    var scales = d3.schemeAccent;
    // console.log(scale(d.category + 1));
    return scales[d.category + 1];
}
exports.mycolor = mycolor;
