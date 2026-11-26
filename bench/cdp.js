const WebSocket = require(require('path').join(__dirname, '../web/node_modules/ws'));
const URL = process.argv[2], PORT = process.argv[3] || 9222;
const http = require('http');
const get = (p) => new Promise((res, rej) => http.get({host:'127.0.0.1',port:PORT,path:p}, r => {
  let b=''; r.on('data',d=>b+=d); r.on('end',()=>res(JSON.parse(b))); }).on('error',rej));
(async () => {
  for (let i=0;i<60;i++){ try { await get('/json/version'); break } catch { await new Promise(r=>setTimeout(r,500)) } }
  const tabs = await get('/json/list');
  const tab = tabs.find(t=>t.type==='page');
  const ws = new WebSocket(tab.webSocketDebuggerUrl, {maxPayload: 1e8});
  let id = 0; const pending = new Map();
  const send = (method, params={}) => new Promise(r => { const i=++id; pending.set(i,r); ws.send(JSON.stringify({id:i,method,params})); });
  ws.on('message', m => { const o = JSON.parse(m); if (o.id && pending.has(o.id)) { pending.get(o.id)(o.result); pending.delete(o.id); } });
  await new Promise(r => ws.on('open', r));
  await send('Page.enable'); await send('Runtime.enable');
  await send('Page.navigate', {url: URL});
  const t0 = Date.now();
  while (Date.now() - t0 < 300000) {
    await new Promise(r=>setTimeout(r,1000));
    const r = await send('Runtime.evaluate', {expression: 'window.DONE === true', returnByValue: true});
    if (r.result && r.result.value === true) break;
  }
  const out = await send('Runtime.evaluate', {expression: "document.getElementById('log').textContent", returnByValue: true});
  console.log(out.result.value);
  const ua = await send('Runtime.evaluate', {expression: 'navigator.userAgent', returnByValue: true});
  console.log('UA:', ua.result.value);
  ws.close(); process.exit(0);
})();
