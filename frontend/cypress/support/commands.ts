/// <reference types="cypress" />

export type FixtureUser = {
  id: number;
  name: string;
  email: string;
  email_verified_at: string | null;
  avatar_url?: string | null;
};

export type LocationPayload = {
  country: string;
  city: string;
  latitude: number;
  longitude: number;
  timezone?: string;
};

export type DeviceFixture = {
  id: string;
  name: string;
  type: string;
  protocol?: string;
  is_on?: boolean;
  [key: string]: any;
};

export type DashboardFixture = {
  temperature: number;
  humidity: number;
  condition: string;
  air_quality_index: number;
  source: string;
  [key: string]: any;
};

type InterceptOptions = {
  user?: FixtureUser;
};

declare global {
  namespace Cypress {
    interface Chainable {
      login(email?: string, password?: string): Chainable<Window>;
      logout(): Chainable<Window>;
      interceptApi(options?: InterceptOptions): Chainable<null>;
      mockPrediction(overrides?: Partial<DashboardFixture>): Chainable<null>;
      createHome(overrides?: Partial<LocationPayload>): Chainable<null>;
      apiPost(url: string, body?: Record<string, any>, opts?: Partial<Cypress.RequestOptions>): Chainable<Cypress.Response<any>>;
    }
  }
}

const backendUrl = (Cypress.env('backendUrl') as string) || 'http://127.0.0.1:8000';
const chatBase = (Cypress.env('chatApiBase') as string) || 'http://127.0.0.1:5080';
const microservicesUrl = (Cypress.env('microservicesUrl') as string) || 'http://20.246.73.238:5050';

function aliasAuthRequests(user: FixtureUser) {
  cy.intercept(
    { method: 'POST', url: '**/api/auth/login' },
    { statusCode: 200, body: { message: 'Sesión iniciada', user } },
  ).as('login');
  cy.intercept(
    { method: 'POST', url: '**/api/auth/register' },
    { statusCode: 201, body: { message: 'Usuario registrado', user } },
  ).as('register');
  cy.intercept({ method: 'GET', url: '**/api/auth/me' }, { statusCode: 200, body: { user } }).as('me');
  cy.intercept({ method: 'POST', url: '**/api/auth/token' }, { statusCode: 200, body: { token: '1|mocked-token' } }).as(
    'token',
  );
  cy.intercept(
    { method: 'POST', url: '**/api/auth/email/resend' },
    { statusCode: 200, body: { message: 'Correo reenviado' } },
  ).as('resendEmail');
  cy.intercept({ method: 'POST', url: '**/api/auth/logout' }, { statusCode: 200, body: { message: 'Sesión cerrada' } }).as(
    'logout',
  );
}

function aliasProfileRequests(user: FixtureUser) {
  cy.intercept(
    { method: 'POST', url: '**/api/profile' },
    { statusCode: 200, body: { user: { ...user, name: 'QA Tester Aura' } } },
  ).as('profileUpdate');
}

function aliasLocationRequests(home: LocationPayload) {
  cy.intercept({ method: 'POST', url: '**/api/location' }, { statusCode: 201, body: { data: home } }).as('locationStore');
  cy.intercept({ method: 'GET', url: '**/api/location/me' }, { statusCode: 200, body: { data: home } }).as('locationGet');
}

function aliasDevicesRequests(devices: DeviceFixture[]) {
  cy.intercept({ method: 'GET', url: '**/api/devices' }, { statusCode: 200, body: { devices } }).as('devices');
  cy.intercept(
    { method: 'POST', url: '**/api/devices/*/power' },
    (req) => req.reply({ statusCode: 200, body: { ...devices[0], is_on: !devices[0].is_on } }),
  ).as('devicePower');
  cy.intercept(
    { method: 'POST', url: '**/api/devices/*/update' },
    (req) => req.reply({ statusCode: 200, body: devices[0] }),
  ).as('deviceUpdate');
}

function aliasDashboardRequests(dashboard: DashboardFixture) {
  cy.intercept({ method: 'GET', url: '**/api/dashboard**' }, { statusCode: 200, body: dashboard }).as('dashboard');
}

function aliasChatRequests() {
  cy.intercept(`${chatBase}/chat/history`, { statusCode: 200, body: { conversations: [] } }).as('chatHistory');
  cy.intercept(`${chatBase}/chat/start`, { statusCode: 201, body: { id: 'conv-qa-1', title: 'QA Flow' } }).as(
    'chatStart',
  );
  cy.intercept(`${chatBase}/chat/message`, { statusCode: 201, body: { ok: true, message: { id: 'msg-qa-1' } } }).as(
    'chatMessage',
  );
  cy.intercept({ method: 'POST', url: '**/api/chat' }, { statusCode: 200, body: { content: 'Respuesta mock' } }).as(
    'apiChat',
  );
}

Cypress.Commands.add('login', (email?: string, password?: string) => {
  cy.interceptApi();
  const normalizedEmail = email ?? (Cypress.env('defaultUserEmail') as string) ?? 'qa@aura.dev';
  const normalizedPassword = password ?? 'Password123!';

  cy.visit('/login');
  cy.get('input[type="email"]').first().clear().type(normalizedEmail, { log: false });
  cy.get('input[type="password"]').first().clear().type(normalizedPassword, { log: false });
  cy.get('button[type="submit"]').first().click({ force: true });

  cy.wait('@login');
  cy.wait('@me');
  cy.window().then((win) => win.localStorage.setItem('aura:pat', '1|mocked-token'));
  return cy.wrap(null);
});

Cypress.Commands.add('logout', () => {
  cy.intercept({ method: 'POST', url: '**/api/auth/logout' }, { statusCode: 200, body: { message: 'Sesión cerrada' } }).as(
    'logout',
  );

  cy.window()
    .then((win) =>
      win.fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include',
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
      }),
    )
    .then(() => cy.wait('@logout'));

  cy.window().then((win) => win.localStorage.removeItem('aura:pat'));
  return cy.visit('/login');
});

Cypress.Commands.add('mockPrediction', (overrides: Partial<DashboardFixture> = {}) => {
  cy.fixture<DashboardFixture>('dashboard').then((dashboard) => {
    aliasDashboardRequests({ ...dashboard, ...overrides });
  });
  return cy.wrap(null);
});

Cypress.Commands.add('createHome', (overrides: Partial<LocationPayload> = {}) => {
  cy.fixture<LocationPayload>('home').then((home) => {
    aliasLocationRequests({ ...home, ...overrides });
  });
  return cy.wrap(null);
});

Cypress.Commands.add('interceptApi', (options: InterceptOptions = {}) => {
  cy.fixture<FixtureUser>('user').then((user) => {
    aliasAuthRequests(options.user ?? user);
    aliasProfileRequests(options.user ?? user);
  });

  cy.fixture<LocationPayload>('home').then((home) => aliasLocationRequests(home));
  cy.fixture<DeviceFixture[]>('device').then((devices) => aliasDevicesRequests(devices));
  cy.fixture<DashboardFixture>('dashboard').then((dashboard) => aliasDashboardRequests(dashboard));
  aliasChatRequests();
  return cy.wrap(null);
});

Cypress.Commands.add('apiPost', (url, body = {}, opts = {}) => {
  return cy.request<Cypress.Response<any>>({
    method: 'POST',
    url: `${backendUrl}${url}`,
    body,
    failOnStatusCode: false,
    ...opts,
  });
});
