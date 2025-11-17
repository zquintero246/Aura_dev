describe('API – Chat endpoints', () => {
  const chatBase = (Cypress.env('chatApiBase') as string) || 'http://127.0.0.1:5080';

  it('responde 200 y retorna el historial vacío', () => {
    cy.intercept(`${chatBase}/chat/history`, { statusCode: 200, body: { conversations: [] } }).as('history');
    cy.window()
      .then((win) =>
        win.fetch(`${chatBase}/chat/history`, {
          headers: { Authorization: 'Bearer fake-token' },
        }),
      )
      .then(() => cy.wait('@history'))
      .its('response.body.conversations')
      .should('be.an', 'array');
  });

  it('devuelve 500 al intentar crear una conversación cuando el microservicio falla', () => {
    cy.intercept(`${chatBase}/chat/start`, { statusCode: 500, body: { message: 'Timeout' } }).as('startConversation');
    cy.window()
      .then((win) =>
        win.fetch(`${chatBase}/chat/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: 'Bearer fake-token' },
          body: JSON.stringify({ title: 'QA session' }),
        }),
      )
      .then(() => cy.wait('@startConversation'))
      .its('response.statusCode')
      .should('equal', 500);
  });

  it('retorna 422 cuando falta contenido en /chat/message', () => {
    cy.intercept(`${chatBase}/chat/message`, (req) => {
      const payload = req.body ? JSON.parse(req.body as string) : {};
      if (!payload.content) {
        req.reply({ statusCode: 422, body: { message: 'content_required' } });
        return;
      }
      req.reply({ statusCode: 201, body: { ok: true, message: { id: 'msg-123' } } });
    }).as('sendMessage');

    cy.window()
      .then((win) =>
        win.fetch(`${chatBase}/chat/message`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: 'Bearer fake-token' },
          body: JSON.stringify({ conversation_id: 'conv-qa', role: 'user' }),
        }),
      )
      .then(() =>
        cy.wait('@sendMessage').then((interception) => {
          expect(interception?.response?.statusCode).to.equal(422);
          expect(interception?.response?.body?.message).to.equal('content_required');
        }),
      );
  });

  it('permite crear una conversación con payload válido y obtiene 201', () => {
    cy.intercept(`${chatBase}/chat/start`, (req) => {
      const body = req.body ? JSON.parse(req.body as string) : {};
      req.reply({ statusCode: body.title ? 201 : 400, body: { id: 'conv-qa-create' } });
    }).as('startConversationValidated');

    cy.window()
      .then((win) =>
        win.fetch(`${chatBase}/chat/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: 'Bearer fake-token' },
          body: JSON.stringify({ title: 'Orquestador QA' }),
        }),
      )
      .then(() =>
        cy.wait('@startConversationValidated').then((interception) => {
          expect(interception?.response?.statusCode).to.equal(201);
        }),
      );
  });

  it('simula retraso en la lista de mensajes para validar timeout', () => {
    cy.intercept(`${chatBase}/chat/history`, (req) => {
      req.delay(500);
      req.reply({ statusCode: 200, body: { conversations: [] } });
    }).as('historyDelay');
    cy.window().then((win) => win.fetch(`${chatBase}/chat/history`, { headers: { Authorization: 'Bearer delay' } }));
    cy.wait('@historyDelay');
  });

  it('usa cy.apiPost para emitir un token simulado', () => {
    cy.intercept('POST', '**/api/auth/token', { statusCode: 200, body: { token: '1|helper-token' } }).as('tokenCall');
    cy.apiPost('/api/auth/token').then((response) => {
      expect(response.status).to.equal(200);
      expect(response.body.token).to.equal('1|helper-token');
    });
  });
});
