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
  return echarts.init(chartDom, null, {renderer: renderer});
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
  const graphData = await fetchJsonData(url, data);

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


export function displaySymptomWordCloud(data = []) {
  console.log("recevie data: ", data);
  if (data.length === 0) {
    data = [
      {name: "龙头镇", value: "111"},
      {name: "大埔镇", value: "222"},
      {name: "太平镇", value: "458"},
      {name: "沙埔镇", value: "445"},
      {name: "东泉镇", value: "456"},
      {name: "凤山镇", value: "647"},
      {name: "六塘镇", value: "189"},
      {name: "冲脉镇", value: "864"},
      {name: "寨隆镇", value: "652"},
      {name: "样例数据", value: "1000"},
    ];
  }
  console.log(data);

  // let wcChart= getEchartInstance("symptom_word_cloud", "canvas");
  let wcChart = getEchartInstance("symptom_word_cloud");

  var option = {
    series: [{
      type: 'wordCloud',

      // The shape of the "cloud" to draw. Can be any polar equation represented as a
      // callback function, or a keyword present. Available presents are circle (default),
      // cardioid (apple or heart shape curve, the most known polar equation), diamond (
      // alias of square), triangle-forward, triangle, (alias of triangle-upright, pentagon, and star.

      shape: 'circle',

      // A silhouette image which the white area will be excluded from drawing texts.
      // The shape option will continue to apply as the shape of the cloud to grow.

      // maskImage: maskImage,

      // Folllowing left/top/width/height/right/bottom are used for positioning the word cloud
      // Default to be put in the center and has 75% x 80% size.

      left: 'center',
      top: 'center',
      width: '70%',
      height: '80%',
      right: null,
      bottom: null,

      // Text size range which the value in data will be mapped to.
      // Default to have minimum 12px and maximum 60px size.

      sizeRange: [12, 60],

      // Text rotation range and step in degree. Text will be rotated randomly in range [-90, 90] by rotationStep 45

      rotationRange: [-90, 90],
      rotationStep: 45,

      // size of the grid in pixels for marking the availability of the canvas
      // the larger the grid size, the bigger the gap between words.

      gridSize: 8,

      // set to true to allow word being draw partly outside of the canvas.
      // Allow word bigger than the size of the canvas to be drawn
      drawOutOfBound: false,

      // If perform layout animation.
      // NOTE disable it will lead to UI blocking when there is lots of words.
      layoutAnimation: true,

      // Global text style
      textStyle: {
        fontFamily: 'sans-serif',
        fontWeight: 'bold',
        // Color can be a callback function or a color string
        color: function () {
          // Random color
          return 'rgb(' + [
            Math.round(Math.random() * 160),
            Math.round(Math.random() * 160),
            Math.round(Math.random() * 160)
          ].join(',') + ')';
        }
      },
      emphasis: {
        focus: 'self',

        textStyle: {
          textShadowBlur: 10,
          textShadowColor: '#333'
        }
      },

      // Data is an array. Each array item must have name and value property.
      // data: [{
      //     name: 'Farrah Abraham',
      //     value: 366,
      //     // Style of single text
      //     textStyle: {}
      // }]
      data: data,
    }]
  }
  wcChart.setOption(option);
}


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