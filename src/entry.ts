const engine = new URLSearchParams(location.search).get('engine');

if (engine === 'v6') {
  await import('./v6/app');
} else {
  await import('./app');
}
