describe('Chat + Home Assistant Dashboard', () => {
  beforeEach(() => {
    cy.interceptApi();
    cy.login();
    cy.visit('/chat');
    cy.wait('@chatHistory');
    cy.wait('@dashboard');
    cy.wait('@devices');
  });

  it('crea una conversación nueva con mensaje predictivo y utiliza la telemetría', () => {
    cy.contains(/nueva conversación/i).click({ force: true });
    cy.wait('@chatStart');
    cy.get('textarea').first().type('Describe el estado de los dispositivos conectados');
    cy.contains(/enviar/i).click({ force: true });
    cy.wait('@chatMessage');
    cy.contains(/telemetría/i).should('exist');
    cy.contains(/dispositivos/i).should('exist');
  });

  it('genera alertas de dispositivos y dashboard simulados', () => {
    cy.mockPrediction({ condition: 'Soleado', temperature: 28 });
    cy.visit('/chat');
    cy.wait('@dashboard');
    cy.contains(/Soleado/i).should('exist');
    cy.wait('@devices');
    cy.contains(/Lámpara Sala/i).should('exist');
  });
});
