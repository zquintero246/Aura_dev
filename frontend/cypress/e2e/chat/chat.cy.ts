describe('Chat Panel – Flujos de mensajes', () => {
  beforeEach(() => {
    cy.interceptApi();
    cy.login();
    cy.visit('/chat');
    cy.wait('@chatHistory');
  });

  it('renderiza el panel principal y permite enviar un mensaje al backend', () => {
    cy.intercept('POST', '**/api/chat', { statusCode: 200, body: { content: 'Respuesta generada' } }).as('apiChat');
    cy.intercept('POST', `${Cypress.env('chatApiBase') ?? 'http://127.0.0.1:5080'}/chat/message`, {
      statusCode: 201,
      body: { ok: true },
    }).as('persistMessage');

    cy.get('textarea').first().should('exist').type('Simula un resumen de energía');
    cy.contains(/enviar/i).click({ force: true });
    cy.wait('@apiChat').its('response.body.content').should('equal', 'Respuesta generada');
    cy.wait('@persistMessage');
  });

  it('maneja errores del endpoint del chat y muestra feedback al usuario', () => {
    cy.intercept('POST', '**/api/chat', { statusCode: 500, body: { message: 'Timeout' } }).as('apiChatError');
    cy.get('textarea').first().type('Fallo intencional');
    cy.contains(/enviar/i).click({ force: true });
    cy.wait('@apiChatError');
    cy.contains(/no pude responder/i).should('exist');
  });
});
