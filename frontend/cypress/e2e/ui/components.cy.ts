describe('UI – Componentes clave', () => {
  beforeEach(() => {
    cy.visit('/');
  });

  it('renderiza la cabecera con el logo y botones de acción', () => {
    cy.get('header').should('exist');
    cy.contains(/Aura/i).should('exist');
    cy.get('header').within(() => {
      cy.get('button, a').should('have.length.greaterThan', 0);
    });
  });

  it('presenta enlaces a registro, login y secciones destacadas', () => {
    cy.get('nav').within(() => {
      cy.get('a').then((links) => {
        expect(links.length).to.be.greaterThan(0);
      });
    });
    cy.contains(/registro/i).should('exist');
    cy.contains(/iniciar sesión|login/i).should('exist');
  });

  it('muestra mensajes clave de marketing relacionados con IA e IoT', () => {
    cy.contains(/inteligencia artificial/i).should('exist');
    cy.contains(/dispositivos/i).should('exist');
  });
});
