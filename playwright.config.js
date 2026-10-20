const {defineConfig} = require("@playwright/test");
module.exports = defineConfig({
  testDir: "./tests/ui",
  timeout: 60000,
  workers: 1,
  use: {baseURL: "http://localhost:8000", headless: true},
});
