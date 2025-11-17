/// <reference types="cypress" />

import './commands';

Cypress.on('uncaught:exception', (error) => {
  const suppressed = ['ResizeObserver loop limit exceeded', 'ResizeObserver loop limit exceeded'];
  if (error?.message && suppressed.some((fragment) => error.message.includes(fragment))) {
    return false;
  }
  return true;
});

beforeEach(() => {
  cy.viewport(1280, 720);
  cy.clearCookies();
  cy.clearLocalStorage();
  cy.intercept('GET', 'https://countriesnow.space/api/v0.1/**', {
    statusCode: 200,
    body: { error: false, msg: 'mocked', data: [] },
  });
  cy.intercept('POST', 'https://countriesnow.space/api/v0.1/**', {
    statusCode: 200,
    body: { error: false, msg: 'mocked', data: [] },
  });
  cy.intercept('GET', 'https://api.open-meteo.com/**', {
    statusCode: 200,
    body: { hourly: { time: [], temperature_2m: [], relative_humidity_2m: [] } },
  });
});
