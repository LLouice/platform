// common

async function fetchJsonData(url, data = {}) {
    const base_url = "http://localhost:9090/";

    var final_url = base_url + url;

    console.log("*** final_url", final_url);


    console.log("ajax get data:\n", data);
    var reps = {};
    try {

        reps = await $.ajax({
            type: "GET",
            url: final_url,
            crossDomain: true,
            // for send cookies
            xhrFields: {
                withCredentials: true
            },
            data: data,
        });

    } catch(err) {
        console.error(err);
    }
    return reps;
}

function getEchartInstance(selector = "network", renderer = "svg") {
    var chartDom = document.getElementById(selector);
    var myChart = echarts.init(chartDom, null, {
        renderer: renderer
    });
    return myChart;
}

// network
async function displayNetwork(src_type = "Symptom", name = "肩背痛") {
    let networkChart = getEchartInstance("network");
    networkChart.showLoading();
    let networkOption = await renderNetwork(src_type, name);
    networkChart.hideLoading();
    networkChart.setOption(networkOption);

    // drill
    // networkChart.on("click", (params) => {
    //   console.log(params);
    //   // get src_type and name
    //   let data = params.data;
    //   let name = data.name;
    //   let src_type = data.des.split("::")[0];
    //   renderNetwork(src_type, name).then((option) =>
    //     networkChart.setOption(option)
    //   );
    // });

    // dbclick delete current node
    networkChart.on("click", {
            dataType: 'node'
        },
        (params) => {
            // TODO refeactor it as function
            // 
            console.log("click node, the params: ", params);

            // get current graph data
            var {
                chart,
                opt
            } = getOption();
            var series = opt.series[0];
            var nodes = series.data;
            var links = series.links;

            console.log(nodes);
            console.log(links);

            let currentNodeIdx = params.dataIndex;
            let currentNodeId = params.data.id;

            // var {nodes, newLinks } = deleteNode(currentNodeIdx, currentNodeId, nodes/* &mut */, links /* & */);
            // series.data = nodes;
            // series.links = newLinks;
            // // set new option
            // chart.setOption(opt);

            let [src_type, name] = params.data.des.split("::");
            // increateUpdate(src_type, name, nodes, links).then((data)=> console.log(data));
            increateUpdate(src_type, name, nodes, links);
        });

    function deleteNode(currentNodeIdx, currentNodeId, nodes, links) {
        // --- delete the node and it's links --
        nodes.splice(currentNodeIdx, 1);

        let s = currentNodeId.toString();
        console.log("s: ", s);
        let newLinks = links.filter((x) => {
            if (x.source === s || x.target === s) {
                return false;
            }
            return true;
        });
        console.log(nodes);
        console.log(newLinks);
        return {
            nodes: nodes,
            newLinks: newLinks
        };
    }

    // click node to increasing update
    function increateUpdate(src_type, name, nodes /* & */ , links /* & */ ) {
        let current_nodes_id = nodes.map((x) => x.id);
        let current_links_pair = links.map((x) => [parseInt(x.source), parseInt(x.target)]);

        // query database

        const base_url = "http://localhost:9090/";
        let url = "increase_update";
        let final_url = base_url + url;
        let payload = {
            node_info: {
                src_type: src_type,
                name: name,
            },
            current_nodes_id: current_nodes_id,
            current_links_pair: current_links_pair,
        };
        // post
        console.log("post payload:\n", payload);
        $.ajax({
            type: "POST",
            url: final_url,
            crossDomain: true,
            // for send cookies
            xhrFields: {
                withCredentials: true
            },
            data: JSON.stringify(payload),
            contentType: "application/json; charset=utf-8",
            dataType: "json",
            success: function(data) {
                console.log(data);
            },
            failure: function(errMsg) {
                console.log(errMsg);
            }
        });
    }
}

async function renderNetwork(src_type = "Symptom", name = "肩背痛") {
    let url = "get_out_links";
    let data = {
        src_type: src_type,
        name: name,
    };
    console.log("renderNetwork: url", url);
    var graphData = await fetchJsonData(url, data);

    let categories = graphData.categories;

    // graphData.data.forEach(function (node) {
    //   node.symbolSize = 20;
    // });

    var option = {
        animation: false,
        animationDuration: 200,
        darkMode: true,
        // 图的标题
        title: {
            left: "center",
            top: "5%",
            text: "ECharts 关系图",
            left: "center",
        },
        // 提示框的配置
        tooltip: {
            formatter: function(x) {
                return x.data.des;
            },
        },
        // 工具箱
        toolbox: {
            // 显示工具箱
            show: true,
            feature: {
                // 还原
                restore: {
                    show: true,
                },
                // 保存为图片
                saveAsImage: {
                    show: true,
                },
            },
            top: "5%",
            right: "5%",
        },
        legend: {
            left: "center",
            top: "15%",
            orient: "horizontal",
            selectedMode: "multiple",
            data: categories.map(function(x) {
                return x.name;
            }),
        },
        series: [{
            type: "graph", // 类型:关系图
            layout: "force", //图的布局，类型为力导图
            roam: true, // 是否开启鼠标缩放和平移漫游。默认不开启。如果只想要开启缩放或者平移,可以设置成 'scale' 或者 'move'。设置成 true 为都开启
            // edgeSymbol: ["circle", "arrow"],
            edgeSymbolSize: [2, 10],

            zoom: 0.67,
            left: "5%",
            top: "20%",

            force: {
                repulsion: 700,
                // edgeLength: [10, 30],
                layoutAnimation: false,
                // friction: 0.9,
            },

            // lineStyle: {
            //   normal: {
            //     width: 2,
            //     color: "#4b565b",
            //     curveness: 0.1,
            //   },
            // },

            edgeLabel: {
                show: false,
                rich: {
                    textStyle: {
                        fontSize: 20,
                    },
                },
                formatter: function(x) {
                    return x.data.name;
                },
            },

            label: {
                show: true,
            },

            draggable: true,

            data: graphData.data,
            links: graphData.links,
            categories: graphData.categories,
        }, ],
    };

    return option;
}

// stats
async function displayStats() {
    var pieData = await fetchJsonData("get_stats");
    // let data_nodes = pieData.nodes_pie;
    // let data_three = [pieData.sym_pie, pieData.dis_pie, pieData.check_pie];
    let data_rels = pieData.rels_pie;

    // displayNodesStats(data_nodes);
    // displayThreeStats(data_three);
    displayRelsStats(data_rels);
}

function displayNodesStats(data) {
    let statChart = getEchartInstance("stats_node");

    const option = {
        title: {
            top: "15%",
            text: "知识图谱结点类型分布",
            left: "center",
        },
        tooltip: {
            trigger: "item",
        },
        legend: {
            /* orient: 'vertical', */
            left: "center",
            bottom: "15%",
        },

        series: [{
            name: "类型",
            type: "pie",
            radius: "50%",
            label: {
                alignTo: "edge",
                formatter: "{name|{b}}\n{value|{c}|{d}% }",
                minMargin: 5,
                edgeDistance: 10,
                lineHeight: 15,
                rich: {
                    time: {
                        fontSize: 10,
                        color: "#999",
                    },
                },
            },
            data: data,
            emphasis: {
                itemStyle: {
                    shadowBlur: 10,
                    shadowOffsetX: 0,
                    shadowColor: "rgba(0, 0, 0, 0.5)",
                },
            },
        }, ],
    }; // option end

    statChart.setOption(option);
}

function displayThreeStats(data) {
    let statChart = getEchartInstance("stats_three");
    console.log(data);

    var option = {
        title: {
            top: 20,
            text: "知识图谱类型关系分布",
            subtext: "症状|疾病|检查",
            left: "center",
        },
        tooltip: {
            trigger: "item",
        },
        legend: {
            /* orient: 'vertical', */
            left: "center",
            bottom: 10,
        },

        series: data.map(function(data, idx) {
            var top = idx * 33.3;
            return {
                type: "pie",
                radius: [20, 60],
                top: top + "%",
                height: "33.33%",
                /* left: 'center', */
                itemStyle: {
                    borderColor: "#fff",
                    borderWidth: 1,
                },
                label: {
                    alignTo: "edge",
                    formatter: "{name|{b}}\n{value|{c}|{d}% }",
                    minMargin: 5,
                    edgeDistance: 10,
                    lineHeight: 15,
                    rich: {
                        time: {
                            fontSize: 10,
                            color: "#999",
                        },
                    },
                },
                labelLine: {
                    length: 15,
                    length2: 0,
                    maxSurfaceAngle: 80,
                },
                labelLayout: function(params) {
                    var isLeft = params.labelRect.x < statChart.getWidth() / 2;
                    var points = params.labelLinePoints;
                    // Update the end point.
                    points[2][0] = isLeft ?
                        params.labelRect.x :
                        params.labelRect.x + params.labelRect.width;

                    return {
                        labelLinePoints: points,
                    };
                },
                data: data,
            };
        }), //series end
    }; // option end
    statChart.setOption(option);
}

function displayRelsStats(data) {
    let statChart = getEchartInstance("stats_rels");
    console.log(data);

    var option = {
        title: {
            text: "知识图谱关系分布",
            subtext: "",
            left: "center",
            top: "5%",
        },
        tooltip: {
            trigger: "item",
        },
        // 工具箱
        toolbox: {
            // 显示工具箱
            show: true,
            feature: {
                // 保存为图片
                saveAsImage: {
                    show: true,
                },
            },
            top: "5%",
            right: "5%",
        },
        legend: {
            type: "scroll",
            orient: "vertical",
            left: "5%",
            top: "10%",
            // right: 10,
            // top: 20,
            // bottom: 20,
        },

        series: [{
            name: "类型",
            type: "pie",
            radius: "50%",
            label: {
                /* alignTo: 'edge', */
                formatter: "{name|{b}}\n{value|{c}|{d}% }",
                minMargin: 5,
                edgeDistance: 10,
                lineHeight: 15,
                rich: {
                    value: {
                        fontSize: 12,
                        color: "#000",
                        fontWeight: "bolder",
                    },
                },
            },
            labelLine: {
                length: 15,
                length2: 0,
                maxSurfaceAngle: 80,
            },
            data: data,
            emphasis: {
                itemStyle: {
                    shadowBlur: 10,
                    shadowOffsetX: 0,
                    shadowColor: "rgba(0, 0, 0, 0.5)",
                },
            },
        }, ],
    }; // option end
    statChart.setOption(option);
}

// main

function main() {
    $(document).ready(checkContainer);

    function checkContainer() {
        if ($("#network").is(":visible")) {
            _main();
        } else {
            setTimeout(checkContainer, 50);
        }
    }
}

async function _main() {
    displayNetwork();
    displayStats();

    // responsive
    $(window).on('resize', resizeChart);

    // Resize function
    function resizeChart() {
        setTimeout(function () {
            // Resize chart
            getEchartInstance("network").resize();
            getEchartInstance("stats_rels").resize();
        }, 200);
    }
}

main();

// slider
function insert_slider() {
    let network_svg = $("#network svg:first")[0];
    console.log(network_svg);
    var svgns = "http://www.w3.org/2000/svg"; // svg namespace

    // var shape = document.createElementNS(svgns, "rect");
    // shape.setAttributeNS(null,"width",50);
    // shape.setAttributeNS(null,"height",80);
    // shape.setAttributeNS(null,"fill","#f00");
    // network_svg.appendChild(shape);
    // network_svg.appendChild(shape);

    var slider = document.createElementNS(svgns, "g");
    var line = document.createElementNS(svgns, "line");

    // let w = parseInt(network_svg.getAttribute("width"));
    // let h = parseInt(network_svg.getAttribute("height"));
    let w = window.innerWidth;
    let h = window.innerHeight;
    let x1 = w * 0.75;
    let y1 = h * 0.15;
    let line_len = w * 0.1;
    let x2 = x1 + line_len;
    line.setAttributeNS(null, "x1", x1);
    line.setAttributeNS(null, "y1", y1);
    line.setAttributeNS(null, "x2", x2);
    line.setAttributeNS(null, "y2", y1);
    line.setAttributeNS(null, "stroke", "#45c589");
    line.setAttributeNS(null, "stroke-width", 5);
    // line.setAttributeNS(null,"stroke-dasharray","1 28");
    // '<line x1="4" y1="0" x2="480" y2="0" stroke="#444" stroke-width="12" stroke-dasharray="1 28"></line>'
    slider.appendChild(line);

    var dot = document.createElementNS(svgns, "circle");
    let r = 5;
    let c = x1 + r + line_len / 2;
    dot.setAttributeNS(null, "r", r);
    dot.setAttributeNS(null, "transform", "translate(" + c + " " + y1 + ")");
    dot.setAttributeNS(null, "id", "dot");


    $(window).on("resize", function (e) {
        // let w = parseInt(network_svg.getAttribute("width"));
        // let h = parseInt(network_svg.getAttribute("height"));

        let w = window.innerWidth;
        let h = window.innerHeight;
        console.log(w, h);
        let x1 = w * 0.75;
        let y1 = h * 0.15;
        let line_len = w * 0.1;
        let x2 = x1 + line_len;
        let r = 5;
        let c = x1 + r + line_len / 2;
        line.setAttributeNS(null, "x1", x1);
        line.setAttributeNS(null, "y1", y1);
        line.setAttributeNS(null, "x2", x2);
        line.setAttributeNS(null, "y2", y1);
        dot.setAttributeNS(null, "transform", "translate(" + c + " " + y1 + ")");
        // dot.transform.baseVal[0].matrix.e = c;
    });


    // drag
    dot.onmousedown = function (event) {
        event.preventDefault();
        // let pageX = event.pageX; // all document
        // let pageY = event.pageY; // visible document
        // let clientX = event.clientX;
        // let clientY = event.clientY;

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        window._D = dot.transform;

        let initX = event.clientX;
        let startX = parseInt(dot.getBoundingClientRect().left);
        let x1 = parseInt(line.getAttribute("x1"));
        let x2 = parseInt(line.getAttribute("x2"));

        function onMouseMove(event) {
            var moveDistance = event.clientX - initX;
            var newX = startX + moveDistance;
            if (newX < x1) {
                newX = x1;
            }
            if (newX > (x2 - 10)) {
                newX = (x2 - 10);
            }
            dot.transform.baseVal[0].matrix.e = (newX + 5);
        }

        function onMouseUp() {
            document.removeEventListener('mouseup', onMouseUp);
            document.removeEventListener('mousemove', onMouseMove);
        }
    }

    slider.ondblclick = function (_e) {
        dot.transform.baseVal[0].matrix.e = c;
    }
    dot.ondrag = (_) => false;

    slider.appendChild(dot);
    network_svg.appendChild(slider);
}

window.insert_slider = insert_slider;
setTimeout(function () {
    insert_slider()
}, 2000);

function getOption(selector = "network") {
    let chartDom = document.getElementById(selector);
    var chart = echarts.getInstanceByDom(chartDom);
    console.log("chart: ", chart);
    var opt = chart.getOption();
    return {
        chart: chart,
        opt: opt
    };
}


function getZoom(selector = "network") {
    var {
        chart,
        opt
    } = getOption(selector);
    return opt.series[0].zoom;
}


function setZoom(zoom = 1, selector = "network") {
    var {
        chart,
        opt
    } = getOption(selector);
    opt.series[0].zoom = zoom;
    console.log("zoom: " + opt.series[0].zoom + "->" + zoom)
    chart.setOption(opt);
};

function _d(selector = "network") {
    var {
        chart,
        opt
    } = getOption(selector);
    let series = opt.series[0];
    return {
        zoom: series.zoom,
        center: series.center
    }
};



window.getOption = getOption;
window.getZoom = getZoom;
window.setZoom = setZoom;
window._d = _d;

