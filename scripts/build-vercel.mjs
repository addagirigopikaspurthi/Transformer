import { cpSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const source = resolve(root, 'web_app', 'static');
const output = resolve(root, 'dist');
const apiUrl = (process.env.PUBLIC_API_URL || process.argv[2] || '').replace(/\/$/, '');

if (!apiUrl) {
  throw new Error('Set PUBLIC_API_URL to the public HTTPS address of the persistent Python backend.');
}
const parsed = new URL(apiUrl);
const local = ['localhost', '127.0.0.1'].includes(parsed.hostname);
if (!['http:', 'https:'].includes(parsed.protocol) || (!local && parsed.protocol !== 'https:') || (process.env.VERCEL && local)) {
  throw new Error('PUBLIC_API_URL must be a public HTTPS URL for a Vercel deployment.');
}
if (parsed.username || parsed.password || parsed.search || parsed.hash || parsed.pathname !== '/') {
  throw new Error('PUBLIC_API_URL must be an origin without credentials, a path, or query parameters.');
}

mkdirSync(resolve(output, 'static'), { recursive: true });
cpSync(resolve(source, 'index.html'), resolve(output, 'index.html'));
for (const asset of ['app.js', 'styles.css', 'favicon.svg']) {
  cpSync(resolve(source, asset), resolve(output, 'static', asset));
}
writeFileSync(resolve(output, 'static', 'config.js'), `window.TRANSFORMER_API_URL = ${JSON.stringify(apiUrl)};\n`);
console.log(`Built Vercel frontend for ${apiUrl}`);
