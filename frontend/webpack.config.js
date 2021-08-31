// global dependencies
const path = require("path");
// const HTMLWebpackPlugin = require("html-webpack-plugin");

module.exports = {
  entry: "./src/index.ts",
  devtool: "inline-source-map",
  mode: "development",
  output: {
    path: path.resolve(__dirname, "./dist_pack"),
    filename: "bundle.js",
  },
  devServer: {
    static: {
      directory: path.join(__dirname, "dist_pack"),
    },
    compress: true,
    port: 9093,
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        exclude: /(node_modules|bower_components)/,
        use: {
          // `.swcrc` in the root can be used to configure swc
          loader: "swc-loader",
        },
      },
      {
        test: /\.html$/,
        use: [
          {
            loader: "html-loader",
            options: { minimize: true },
          },
        ],
      },

      // {
      //   test: /\.scss/i,
      //   use: ["style-loader", "css-loader", "sass-loader"],
      // },
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
  },
  // plugins: [
  //   new HTMLWebpackPlugin({
  //     filename: "./index.html",
  //     template: path.join(__dirname, "public/index.html"),
  //   }),
  // ],
};
