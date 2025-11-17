# <p align="center"><img src="https://i.ibb.co/TBfnv0RL/Logo-debajo-aura.png" alt="Aura logo" width="200" /></p>

<p align="center"><strong><span style="color:#CA5CF5">Autores: </span></strong>Equipo Aura – Ingeniería de Software II</p>

## <span style="color:#CA5CF5">Descripción</span>

Aura es una **plataforma inteligente de automatización y asistencia personal** que une **Inteligencia Artificial (IA)** con **Internet de las Cosas (IoT)**. Combina modelos de lenguaje generativos y aprendizaje adaptativo para entender hábitos, anticipar necesidades y orquestar dispositivos del hogar mediante reglas y telemetría en tiempo real. El ecosistema está diseñado para ofrecer recomendadores contextuales, asistentes conversacionales y flujos automatizados que conectan la IA con la vida cotidiana.

El sistema integra un **backend en Flask/FastAPI**, un **frontend en React + TypeScript** y una **base de datos híbrida (PostgreSQL + MongoDB)**. Todos los servicios corren en **contenedores Docker**, orquestados desde `docker-compose.yml`, lo que permite pruebas locales rápidas y despliegues repetibles.

## <span style="color:#CA5CF5">Instalación</span>

1. Clonar el repositorio:

```bash
git clone https://github.com/usuario/aura.git
cd aura
```

2. Instalar dependencias compartidas:

```bash
pip install -r requirements.txt
npm install
```

La lista en `requirements.txt` cubre los servicios Python y los microservicios (Flask/FastAPI, persistencia, tests y componentes de IA) para que ejecutar el stack desde la raíz sea suficiente antes de lanzar backend/FR.

3. Ejecutar el backend (Flask/FastAPI):

```bash
python app.py
```

4. Ejecutar el frontend (Vite + React):

```bash
npm run dev
```

5. Alternativamente, iniciar todo el stack con Docker Compose:

```bash
docker compose up --build
```

## <span style="color:#CA5CF5">Tecnologías usadas</span>

- Python 3.10 / FastAPI + Flask (APIs REST, WebSocket y agentes programados)
- React + TypeScript + Vite (SPA con rutas protegidas y tema dinámico)
- PostgreSQL + MongoDB (persistencia relacional + chats históricos)
- Docker & Docker Compose (contenedores backend, frontend y bases)
- GitHub Actions (CI/CD para lint, tests y despliegue)
- Tailwind CSS, lucide-react, framer-motion (UI/animaciones)

## <span style="color:#CA5CF5">Pruebas</span>

- Ejecuta la suite principal desde la carpeta raíz para cubrir backend y scripts auxiliares:

```bash
pytest -v
```

- Para los microservicios (p. ej. `polyglot_persistence/chat_service`), también se provee testeo aislado con `mongomock`:

```bash
cd polyglot_persistence/chat_service
pytest
```

Esto comprueba los creadores de conversaciones, título automático y validaciones de propiedad sin requerir una base Mongo real. Los requisitos listan `pytest` y `mongomock` para que pueda ejecutarse en entornos locales o CI.

## <span style="color:#CA5CF5">Cómo ejecutar los tests</span>

La ejecución de los tests centrales debe hacerse desde la raíz del repositorio (`AURA_dev/`). Usa el siguiente comando para correr únicamente los casos del microservicio:

```bash
python -m pytest polyglot_persistence/chat_service/tests -v
```

Python resuelve `polyglot_persistence` porque estás ubicado en la raíz del módulo y, al especificar la ruta relativa, pytest carga los tests del chat service sin necesidad de moverte a esa carpeta.

## <span style="color:#CA5CF5">Manual Técnico</span>

Toda la documentación técnica está organizada dentro de la carpeta `docs/`. Allí encontrarás manuales detallados, diagramas, guías de integración y modelos de datos que describen cómo interactúan los componentes backend, frontend y microservicios del proyecto.

## <span style="color:#CA5CF5">Arquitectura y Diagramas</span>

Aura tiene una arquitectura modular basada en microservicios. El backend principal gestiona usuarios, autenticación, perfiles y telemetría, mientras que el microservicio de chat conversa con OpenRouter/Mongo y ofrece títulos dinámicos para cada conversación. El frontend consume las APIs protegidas, mantiene tokens Sanctum en `localStorage:aura:pat` y activa paneles especiales (perfil, telemetría, dispositivos) desde un layout reactivo. La capa de datos combina PostgreSQL para los modelos relacionales y MongoDB para los historiales y métricas.

Cada componente se conecta mediante una API Gateway interna y se despliega en contenedores Docker para asegurar aislamiento y reproducibilidad. Los archivos `docs/` describen rutas, controladores y servicios clave que conviene revisar antes de extender la plataforma. (Se recomienda incluir un diagrama de arquitectura o flujo para documentar mejor los límites entre microservicios y datos.)

## <span style="color:#CA5CF5">Despliegue</span>

- Versión estable: **v1.0 – 2025**
- Repositorio central: GitHub – Proyecto Aura
- Entorno de referencia: Docker Compose (backend + frontend + PostgreSQL + MongoDB)
- Variables críticas: `APP_URL`, `VITE_BACKEND_URL`, `VITE_CHAT_API_BASE`, `OPENROUTER_API_KEY`, `DB_HOST_MONGO` y credenciales de PostgreSQL.

## <span style="color:#CA5CF5">Licencia</span>

Este proyecto, denominado **Aura**, ha sido desarrollado como parte de un trabajo académico universitario.  
Todo el código, documentación, diagramas, manuales, modelos, interfaces y demás materiales incluidos en este repositorio se publican bajo los términos de la **Licencia MIT**, con las siguientes consideraciones adicionales relacionadas con el uso del contenido en entornos académicos.

---

## <span style="color:#CA5CF5">Clonación del repositorio</span>

Puedes clonar o descargar este repositorio para fines educativos, de estudio, revisión técnica o referencia.  
Sin embargo, el repositorio clonado **debe conservar su estructura, contenido y créditos originales**.

No está permitido presentar una copia —con o sin modificaciones— como si fuera el desarrollo original del equipo creador del proyecto.

---

## <span style="color:#CA5CF5">Modificaciones y trabajos derivados</span>

Están permitidas las modificaciones al código, documentación o estructura del proyecto, siempre que se cumpla con:

- Mantener íntegra la licencia original.
- Incluir una atribución explícita al equipo desarrollador original.
- Identificar claramente qué partes han sido añadidas, modificadas o eliminadas.
- No presentar versiones modificadas como proyectos académicos propios sin autorización del equipo creador.

---

## <span style="color:#CA5CF5">Redistribución</span>

Cualquier redistribución del código, los recursos o una versión derivada del proyecto deberá incluir:

- Una copia completa de esta licencia.
- Atribución visible al equipo creador de Aura.
- Enlace al repositorio original, cuando aplique.

---

## <span style="color:#CA5CF5">Uso académico y ético</span>

Dado que este proyecto es parte de un trabajo universitario, **queda prohibido** utilizarlo —o crear derivados del mismo— para presentarlo como autoría propia en actividades académicas evaluadas, salvo que exista consentimiento explícito del equipo desarrollador original.

---

## Licencia MIT (texto oficial)

MIT License

Copyright (c) 2025 Equipo desarrollador de Aura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
