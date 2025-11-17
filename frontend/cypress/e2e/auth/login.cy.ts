describe('Authentication – Login', () => {
  beforeEach(() => {
    cy.interceptApi();
    cy.visit('/login');
  });

  it('muestra el formulario y permite iniciar sesión con credenciales válidas', () => {
    cy.get('input[type="email"]').should('exist');
    cy.get('input[type="password"]').should('exist');
    cy.get('button[type="submit"]').first().should('be.enabled');

    cy.login();
    cy.window()
      .its('localStorage')
      .invoke('getItem', 'aura:pat')
      .should('match', /^1\|mocked-token$/);
  });

  it('muestra error cuando el backend rechaza las credenciales', () => {
    cy.intercept('POST', '**/api/auth/login', { statusCode: 422, body: { message: 'Credenciales inválidas' } }).as(
      'loginInvalid',
    );
    cy.get('button[type="submit"]').first().click({ force: true });
    cy.wait('@loginInvalid');
    cy.contains(/credenciales inválidas/i).should('exist');
  });
});
