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

  } catch (err) {
    console.error(err);
  }
  return reps;
}

export function getEchartInstance(selector = "network", renderer = "svg") {
  var chartDom = document.getElementById(selector);
  var myChart = echarts.init(chartDom, null, { renderer: renderer });
  return myChart;
}

// network
export async function displayNetwork(src_type = "Symptom", name = "肩背痛") {
  let networkChart = getEchartInstance("network");
  networkChart.showLoading();
  let networkOption = await renderNetwork(src_type, name);
  networkChart.hideLoading();
  networkChart.setOption(networkOption);

  // drill
  networkChart.on("click", (params) => {
    console.log(params);
    // get src_type and name
    let data = params.data;
    let name = data.name;
    let src_type = data.des.split("::")[0];
    renderNetwork(src_type, name).then((option) =>
      networkChart.setOption(option)
    );
  });
}

export async function renderNetwork(src_type = "Symptom", name = "肩背痛") {
  // let url = "get_out_links";
  let url = "query_links";
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
    },
    // 提示框的配置
    tooltip: {
      formatter: function (x) {
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
      data: categories.map(function (x) {
        return x.name;
      }),
    },
    series: [
      {
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
          formatter: function (x) {
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
      },
    ],
  };

  return option;
}

// stats
export async function displayStats() {
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

    series: [
      {
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
      },
    ],
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

    series: data.map(function (data, idx) {
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
        labelLayout: function (params) {
          var isLeft = params.labelRect.x < statChart.getWidth() / 2;
          var points = params.labelLinePoints;
          // Update the end point.
          points[2][0] = isLeft
            ? params.labelRect.x
            : params.labelRect.x + params.labelRect.width;

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
    },

    series: [
      {
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
      },
    ],
  }; // option end
  statChart.setOption(option);
}


export function getOption(selector = "network") {
  let chartDom = document.getElementById(selector);
  var chart = echarts.getInstanceByDom(chartDom);
  console.log("chart: ", chart);
  var opt = chart.getOption();
  return {
    chart: chart,
    opt: opt
  };
}

// export function setOption(chart, opt) {
//   chart.setOption(opt);
// }


// main for rust
export async function main() {
  await displayNetwork();
  await displayStats();
}


export function main_with_cb(cb) {
  main().then(cb);
}


// main in js
/*
export function main() {
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
  function resizeChart() {
    setTimeout(function () {
    // Resize chart
    getEchartInstance("network").resize();
    getEchartInstance("stats_rels").resize();
    }, 200);
  }
}
*/