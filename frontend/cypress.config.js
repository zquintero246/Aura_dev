const { defineConfig } = require('cypress');

module.exports = defineConfig({
  e2e: {
    specPattern: 'cypress/e2e/**/*.cy.ts',
    supportFile: 'cypress/support/e2e.ts',
    baseUrl: 'http://127.0.0.1:4028',
    viewportWidth: 1280,
    viewportHeight: 720,
    defaultCommandTimeout: 8000,
    requestTimeout: 10000,
    responseTimeout: 15000,
    watchForFileChanges: false,
    chromeWebSecurity: false,
    retries: {
      runMode: 1,
      openMode: 0,
    },
    env: {
      backendUrl: 'http://127.0.0.1:8000',
      chatApiBase: 'http://127.0.0.1:5080',
      microservicesUrl: 'http://20.246.73.238:5050',
      defaultUserEmail: 'qa@aura.dev',
    },
    setupNodeEvents(on, config) {
      on('task', {
        log(message) {
          console.log(message);
          return null;
        },
      });
      return config;
    },
  },
});
