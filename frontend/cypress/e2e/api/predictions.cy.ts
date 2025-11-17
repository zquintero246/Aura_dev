describe('Microservices – Predicciones y dispositivos', () => {
  const base = (Cypress.env('microservicesUrl') as string) || 'http://20.246.73.238:5050';

  it('registra respuestas exitosas del dashboard', () => {
    cy.intercept(`${base}/api/dashboard**`, { statusCode: 200, body: { temperature: 21, humidity: 48 } }).as('dashboard');
    cy.window()
      .then((win) => win.fetch(`${base}/api/dashboard?lat=4.7&lon=-74.0`))
      .then(() => cy.wait('@dashboard'))
      .its('response.body')
      .should('include', { humidity: 48 });
  });

  it('simula error 500 al activar un dispositivo', () => {
    cy.intercept(`${base}/api/devices/light-1/power`, { statusCode: 500, body: { error: 'timeout' } }).as('devicePower');
    cy.window()
      .then((win) =>
        win.fetch(`${base}/api/devices/light-1/power`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'toggle' }),
        }),
      )
      .then(() => cy.wait('@devicePower'))
      .its('response.statusCode')
      .should('equal', 500);
  });

  it('valida campos faltantes y responde 422', () => {
    cy.intercept(`${base}/api/devices/thermostat-1/update`, (req) => {
      const payload = req.body ? JSON.parse(req.body as string) : {};
      if (typeof payload.temperature !== 'number') {
        req.reply({ statusCode: 422, body: { message: 'temperature_required' } });
        return;
      }
      req.reply({ statusCode: 200, body: { ...payload, id: 'thermostat-1' } });
    }).as('updateDevice');

    cy.window()
      .then((win) =>
        win.fetch(`${base}/api/devices/thermostat-1/update`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        }),
      )
      .then(() => cy.wait('@updateDevice'))
      .its('response.body.message')
      .should('equal', 'temperature_required');
  });

  it('presenta respuesta retrasada y sin datos', () => {
    cy.intercept(`${base}/api/devices`, (req) => {
      req.on('response', (res) => {
        res.setDelay(600);
        res.send({ statusCode: 200, body: { devices: [] } });
      });
    }).as('devicesDelay');
    cy.window().then((win) => win.fetch(`${base}/api/devices`));
    cy.wait('@devicesDelay');
  });
});
