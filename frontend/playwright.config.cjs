const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
 testDir: './tests', use: {baseURL:'http://127.0.0.1:5173',headless:true},
 webServer: {command:'node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 --strictPort',url:'http://127.0.0.1:5173'},
});
