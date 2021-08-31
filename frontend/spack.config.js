// importing config creator
const { config } = require("@swc/core/spack");

// import path module
const path = require("path");

// export config
module.exports = config({
  // start file
  entry: {
    // build: path.join(__dirname, "/src/index.ts"),
    build: path.join(__dirname, "/src/ex.js"),
    // build: path.join(__dirname, "/src/foo.ts"),
  },

  // output file
  output: {
    path: path.join(__dirname + "/dist_spack"),
  },
});
