describe('Authentication – Verify Email', () => {
  beforeEach(() => {
    cy.interceptApi();
    cy.visit('/verify-email');
  });

  it('permite reenviar el correo y muestra el mensaje de confirmación', () => {
    cy.contains(/reenviar/i).click({ force: true });
    cy.wait('@resendEmail');
    cy.contains(/correo reenviado/i).should('exist');
  });

  it('redirecciona al chat cuando el backend actualiza el estado del usuario', () => {
    let count = 0;
    cy.fixture('user').then((user) => {
      cy.intercept('GET', '**/api/auth/me', (req) => {
        count += 1;
        req.reply({
          statusCode: 200,
          body: { user: { ...user, email_verified_at: count > 1 ? new Date().toISOString() : null } },
        });
      }).as('mePolling');
    });

    cy.wait('@mePolling');
    cy.wait('@mePolling');
    cy.url().should('include', '/chat');
  });
});
