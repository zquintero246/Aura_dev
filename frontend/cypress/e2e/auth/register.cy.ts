describe('Authentication – Register', () => {
  beforeEach(() => {
    cy.interceptApi();
    cy.visit('/register');
  });

  it('valida el formulario y avanza al flujo de verificación', () => {
    cy.get('input[name="name"]').clear().type('Aura QA');
    cy.get('input[name="email"]').clear().type('qa.new@aura.dev');
    cy.get('input[name="password"]').clear().type('Password123!');
    cy.get('input[name="password_confirmation"]').clear().type('Password123!');
    cy.get('button[type="submit"]').first().click({ force: true });
    cy.wait('@register');
    cy.url().should('include', '/verify-email');
  });

  it('muestra el error de validación cuando las contraseñas no coinciden', () => {
    cy.get('input[name="password"]').clear().type('Password123!');
    cy.get('input[name="password_confirmation"]').clear().type('Password1234!');
    cy.get('button[type="submit"]').first().click({ force: true });
    cy.contains(/contraseñ/i).should('exist');
  });
});
