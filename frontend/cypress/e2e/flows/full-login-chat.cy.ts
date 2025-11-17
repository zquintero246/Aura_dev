describe('Flujo completo – Login y chat protegido', () => {
  it('ejecuta login, navega al chat y envía un mensaje end to end', () => {
    cy.interceptApi();
    cy.visit('/login');
    cy.get('input[type="email"]').first().type('qa@aura.dev');
    cy.get('input[type="password"]').first().type('Password123!');
    cy.get('button[type="submit"]').click({ force: true });
    cy.wait('@login');
    cy.wait('@me');
    cy.window().then((win) => win.localStorage.setItem('aura:pat', '1|mocked-token'));

    cy.visit('/chat');
    cy.wait('@chatHistory');
    cy.get('textarea').first().type('Genera un reporte del estado actual de los dispositivos y la telemetría.');
    cy.contains(/enviar/i).click({ force: true });
    cy.wait('@apiChat');
    cy.wait('@chatMessage');
    cy.contains(/telemetría/i).should('exist');
    cy.contains(/dispositivos/i).should('exist');
  });

  it('permite cerrar sesión correctamente después del chat', () => {
    cy.interceptApi();
    cy.login();
    cy.visit('/chat');
    cy.contains(/cerrar sesión/i).first().click({ force: true });
    cy.wait('@logout');
    cy.url().should('include', '/login');
  });
});
