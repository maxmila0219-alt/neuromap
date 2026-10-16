/* =====================================================================
   NeuroMap GL — tiny WebGL2 engine for glowing neural worlds. MIT.
   - additive point-sprite renderer with HDR bloom + filmic tonemap
   - orbit / fly-to camera with inertia, pinch & wheel zoom
   - procedural generators: particle brain, white-matter tracts,
     cortical column of real-shaped neurons (dendrites, spines, axon,
     myelin with nodes of Ranvier, synaptic boutons)
   - leaky integrate-and-fire spiking network driving the visuals
   ===================================================================== */
(function (root) {
'use strict';
const clamp = (v, a, b) => Math.max(a, Math.min(b, v)), lerp = (a, b, k) => a + (b - a) * k;
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296 } }
const hex = h => { const n = parseInt(h.slice(1), 16); return [(n >> 16) / 255, (n >> 8 & 255) / 255, (n & 255) / 255] };

/* ---------- math ---------- */
const M4 = {
  persp(fov, asp, n, f) { const t = 1 / Math.tan(fov / 2), nf = 1 / (n - f); return new Float32Array([t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) * nf, -1, 0, 0, 2 * f * n * nf, 0]) },
  look(e, c, u) {
    let zx = e[0] - c[0], zy = e[1] - c[1], zz = e[2] - c[2]; let l = Math.hypot(zx, zy, zz) || 1; zx /= l; zy /= l; zz /= l;
    let xx = u[1] * zz - u[2] * zy, xy = u[2] * zx - u[0] * zz, xz = u[0] * zy - u[1] * zx; l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0, -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1])
  },
  mul(a, b) { const o = new Float32Array(16); for (let c = 0; c < 4; c++)for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++)s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s } return o }
};

/* ---------- shaders ---------- */
const VS_PTS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aCol;   // rgb, kind
layout(location=2) in vec4 aA;     // size, group, path, phase
uniform mat4 uPV; uniform vec3 uCam;
uniform float uTime, uPx, uFade, uFocus, uVel, uMy, uIN, uSat, uDist, uFocusD, uDof;
uniform float uGain[16]; uniform float uTr[16];
uniform vec4 uWave[6]; uniform float uWaveA[6];
uniform vec3 uBase;
uniform highp sampler2D uSpk;
out vec3 vCol; out float vA;
float flash(vec4 t, float k){ vec4 x=uTime-t; vec4 e=exp(-max(x,0.)*k)*step(0.,x); return e.x+e.y+e.z+e.w; }
float pulse(vec4 t, float d, float w){ vec4 x=uTime-t-d; vec4 e=exp(-(x*x)/w); return e.x+e.y+e.z+e.w; }
void main(){
  vec3 p=aPos; float kind=aCol.w, g=aA.y, ph=aA.w, sz=aA.x, b=1.; vec3 col=aCol.rgb;
  if(kind<.5){                         /* brain surface */
    int gi=int(g+.5); float gain=uGain[gi];
    float w=0.; for(int i=0;i<6;i++){ if(uWaveA[i]<=0.)continue; float d=length(p-uWave[i].xyz)-uWave[i].w; w+=exp(-d*d*60.)*uWaveA[i]; }
    b=(.18+.95*gain)*(.82+.18*sin(uTime*1.7+ph))+w*1.2;
    col=mix(uBase,col,clamp(gain*uSat+w,0.,1.));
    if(uFocus>-.5){ if(abs(g-uFocus)<.5){ b=b*1.25+.35+.2*sin(uTime*3.+ph); col=aCol.rgb; } else b*=.3; }
    vec3 dir=normalize(uCam); float facing=dot(normalize(p),dir); b*=.45+.55*clamp(facing*.5+.6,0.,1.);
  } else if(kind<1.5){                 /* white-matter tract stream */
    float tg=uTr[int(g+.5)];
    float s=pow(.5+.5*sin(aA.z*22.-uTime*2.6+ph*.2),8.);
    b=tg*(.12+1.5*s); sz*=.7+.6*s;
  } else if(kind<2.5){                 /* drifting dust */
    p+=.03*vec3(sin(uTime*.31+ph),cos(uTime*.23+ph*1.3),sin(uTime*.19+ph*.7));
    b=.22+.12*sin(uTime*.8+ph*3.);
  } else {                             /* neuron parts: 3 soma 4 dendrite 5 axon 6 myelin 7 bouton 8 spine */
    int n=int(g+.5); vec4 sp=texelFetch(uSpk,ivec2(n,0),0), inp=texelFetch(uSpk,ivec2(n,1),0);
    float pt=aA.z, e=0.;
    if(kind<3.5){ e=flash(sp,6.)*1.8+flash(inp,9.)*.3; b=.32+e; }
    else if(kind<4.5||kind>7.5){ e=flash(inp,4.)*exp(-pt*2.2)*.9+flash(sp,8.)*exp(-pt*4.)*.6; b=(kind>7.5?.7:.6)+e*(kind>7.5?1.4:1.1); }
    else {
      float v=uVel; float d=pt/v;
      if(uMy>.5){ d=floor(pt/uIN)*uIN/v; }                  /* saltatory conduction: jumps node to node */
      float pw=kind>6.5?.012:.0016;
      e=pulse(sp,d,pw)*(kind>6.5?2.4:1.9)+(kind>6.5?flash(sp+d,3.)*.7:0.);
      b=(kind>5.5&&kind<6.5?.5:kind>6.5?1.1:.5)+e;
      sz*=1.+min(e,2.)*.55;
    }
    col=mix(col,vec3(1.,.98,.92),clamp(e*.55,0.,.85));
    b*=.85+.15*sin(uTime*1.3+ph);
  }
  vec4 c=uPV*vec4(p,1.); gl_Position=c;
  float coc=clamp(abs(c.w-uFocusD)*uDof,0.,1.8);              /* depth of field: out-of-focus points become soft bokeh discs */
  gl_PointSize=clamp(sz*uPx/max(c.w,.05)*(1.+coc*1.3),1.,80.);
  float fog=clamp(1.1-(c.w-uDist)*.32,.18,1.);
  vCol=col*b; vA=uFade*fog/(1.+coc*coc*3.);
}`;
const FS_PTS = `#version 300 es
precision mediump float;
in vec3 vCol; in float vA; out vec4 o;
void main(){ vec2 d=gl_PointCoord*2.-1.; float r=dot(d,d); if(r>1.)discard;
  float a=exp(-r*6.)+.28*exp(-r*1.8)*(1.-r); o=vec4(vCol*a*vA,0.); }`;
const VS_FS = `#version 300 es
const vec2 P[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3)); out vec2 vUv;
void main(){ vec2 p=P[gl_VertexID]; vUv=p*.5+.5; gl_Position=vec4(p,0,1); }`;
const FS_BRIGHT = `#version 300 es
precision mediump float; in vec2 vUv; uniform sampler2D uT; uniform vec2 uTx; uniform float uTh; out vec4 o;
void main(){ vec3 c=(texture(uT,vUv+uTx*vec2(-1,-1)).rgb+texture(uT,vUv+uTx*vec2(1,-1)).rgb+texture(uT,vUv+uTx*vec2(-1,1)).rgb+texture(uT,vUv+uTx*vec2(1,1)).rgb)*.25;
  float l=max(c.r,max(c.g,c.b)); o=vec4(c*max(l-uTh,0.)/max(l,1e-4),1.); }`;
const FS_BLUR = `#version 300 es
precision mediump float; in vec2 vUv; uniform sampler2D uT; uniform vec2 uDir; out vec4 o;
void main(){ vec3 c=texture(uT,vUv).rgb*.227027;
  c+=(texture(uT,vUv+uDir*1.3846).rgb+texture(uT,vUv-uDir*1.3846).rgb)*.316216;
  c+=(texture(uT,vUv+uDir*3.2308).rgb+texture(uT,vUv-uDir*3.2308).rgb)*.070270; o=vec4(c,1.); }`;
const VS_MESH = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aN;
uniform mat4 uPV; uniform vec3 uEye; out vec3 vN; out vec3 vV; out vec3 vP;
void main(){ vP=aPos; vN=aN; vV=uEye-aPos; gl_Position=uPV*vec4(aPos,1.); }`;
const FS_MESH = `#version 300 es
precision highp float;
in vec3 vN; in vec3 vV; in vec3 vP; uniform float uTime, uFade, uScan; uniform vec3 uTint; out vec4 o;
vec3 film(float x){ return .5+.5*cos(6.2831*(vec3(0.,.33,.67)+x)); }      /* thin-film iridescence */
void main(){ vec3 N=normalize(vN), V=normalize(vV); float ndv=abs(dot(N,V)); float fr=pow(1.-ndv,2.6);
  vec3 c=mix(uTint,film(ndv*1.4+vP.y*.35+uTime*.025),.5)*fr*.5+uTint*.018;
  float sp=fract(uTime*.075)*2.8-1.4, band=exp(-pow((vP.z-sp)*30.,2.));     /* MRI-like scan plane */
  c+=vec3(.45,.85,1.)*band*(.35+fr)*uScan;
  o=vec4(c*uFade,0.); }`;
const VS_LINE = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos; layout(location=1) in vec4 aC; layout(location=2) in float aS;
uniform mat4 uPV; uniform float uTime, uFade, uFlow, uDist; out vec3 vC;
void main(){ vec4 c=uPV*vec4(aPos,1.); gl_Position=c;
  float pulse=pow(.5+.5*sin(aC.w*26.-uTime*2.1+aS*6.2831),14.);
  float fog=clamp(1.15-(c.w-uDist)*.35,.2,1.);
  vC=aC.rgb*(.16+pulse*uFlow*1.4)*uFade*fog; }`;
const FS_LINE = `#version 300 es
precision mediump float; in vec3 vC; out vec4 o; void main(){ o=vec4(vC,0.); }`;
const FS_COMP = `#version 300 es
precision mediump float; in vec2 vUv; uniform sampler2D uS, uB, uB2; uniform float uBloom, uExp, uOpaque, uTime, uCA; uniform vec3 uBg; out vec4 o;
float hash(vec2 p){ return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453); }
vec3 aces(vec3 x){ return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.); }   /* ACES filmic tonemapping */
void main(){ vec2 dc=vUv-.5; vec2 off=dc*dot(dc,dc)*uCA;                         /* lens chromatic aberration */
  vec3 s=vec3(texture(uS,vUv+off).r,texture(uS,vUv).g,texture(uS,vUv-off).b);
  vec3 b=texture(uB,vUv).rgb*.6+texture(uB2,vUv).rgb*.9;
  vec3 c=aces((s+b*uBloom)*uExp);
  c=pow(c,vec3(.97,.98,1.)); float vg=smoothstep(1.25,.35,length(vUv-.5)*1.4);
  c*=mix(1.,vg,uOpaque*.6); c+=(hash(vUv*913.+uTime)-.5)*.012;
  if(uOpaque>.5){ o=vec4(uBg*vg+c,1.); } else { float a=clamp(max(c.r,max(c.g,c.b))*1.05,0.,1.); o=vec4(c,a); } }`;

function compile(gl, vs, fs) {
  const mk = (t, s) => { const sh = gl.createShader(t); gl.shaderSource(sh, s); gl.compileShader(sh); if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh)); return sh };
  const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i), name = info.name.replace(/\[0\]$/, ''); u[name] = gl.getUniformLocation(p, info.name) }
  return { p, u };
}

let SUPPORT = null;
function supported() {
  if (SUPPORT != null) return SUPPORT;
  try { const c = document.createElement('canvas'), gl = c.getContext('webgl2'); SUPPORT = !!gl; if (gl) { const e = gl.getExtension('WEBGL_lose_context'); e && e.loseContext() } } catch (e) { SUPPORT = false }
  if (/[?&]nogl\b/.test(location.search)) SUPPORT = false;
  try { if (/[?&]gl\b/.test(location.search)) localStorage.removeItem('nm_nogl'); else if (localStorage.getItem('nm_nogl') === '2') SUPPORT = false } catch (e) { }
  return SUPPORT;
}

/* =====================================================================
   RENDERER
   ===================================================================== */
function create(canvas, opt) {
  opt = Object.assign({ opaque: false, bg: [0.016, 0.02, 0.045], bloom: 1.1, exposure: 1.05, maxDpr: 1.75, controls: true, dof: 0, ca: .9 }, opt || {});
  const gl = canvas.getContext('webgl2', { alpha: true, antialias: false, premultipliedAlpha: true, powerPreference: 'high-performance' });
  if (!gl) return null;
  /* half-float targets give real HDR bloom, but many phone GPUs can't render to them → verify, else fall back to RGBA8 */
  let hdr = !/[?&]ldr\b/.test(location.search) && (!!gl.getExtension('EXT_color_buffer_float') || !!gl.getExtension('EXT_color_buffer_half_float'));
  const P = compile(gl, VS_PTS, FS_PTS), PB = compile(gl, VS_FS, FS_BRIGHT), PL = compile(gl, VS_FS, FS_BLUR), PC = compile(gl, VS_FS, FS_COMP);
  let PM = null, PLN = null; try { PM = compile(gl, VS_MESH, FS_MESH) } catch (e) { console.warn('mesh shader', e) } try { PLN = compile(gl, VS_LINE, FS_LINE) } catch (e) { console.warn('line shader', e) }
  const dbgInfo = gl.getExtension('WEBGL_debug_renderer_info');
  const info = { renderer: dbgInfo ? gl.getParameter(dbgInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER), hdr, fbo: '', errors: [] };
  const emptyVao = gl.createVertexArray();
  /* spike texture: 64 neurons × 2 rows × RGBA32F (last 4 spike / input times) */
  const SPK_W = 64, spk = new Float32Array(SPK_W * 2 * 4).fill(-100);
  const spkTex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, spkTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, SPK_W, 2, 0, gl.RGBA, gl.FLOAT, spk);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  let fb = null, W = 0, H = 0, scale = 1, dpr = 1;
  function rt(w, h) {
    const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, hdr ? gl.RGBA16F : gl.RGBA8, w, h, 0, gl.RGBA, hdr ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    return { t, f, w, h, ok };
  }
  function rt8(w, h) { const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0); return { t, f, w, h } }
  let probed = false;
  /* draw a known colour through the whole HDR → composite chain and read it back: proves this GPU really renders it */
  function probe() {
    const pt = rt8(4, 4), px = new Uint8Array(4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb.s.f); gl.viewport(0, 0, W, H); gl.clearColor(.6, .3, .1, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    for (const k of ['c', 'e']) { gl.bindFramebuffer(gl.FRAMEBUFFER, fb[k].f); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT) }
    composite(pt, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, pt.f); gl.readPixels(1, 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.deleteTexture(pt.t); gl.deleteFramebuffer(pt.f);
    gl.getError(); info.probe = Array.from(px).join(',');
    return px[0] > 20 && !/[?&]glfail\b/.test(location.search);
  }
  function freeFB() { if (!fb) return; for (const k in fb) { gl.deleteTexture(fb[k].t); gl.deleteFramebuffer(fb[k].f) } fb = null }
  function resize() {
    const cw = canvas.clientWidth, ch = canvas.clientHeight; if (!cw || !ch) return false;
    dpr = Math.min(window.devicePixelRatio || 1, opt.maxDpr) * scale;
    const w = Math.max(2, Math.round(cw * dpr)), h = Math.max(2, Math.round(ch * dpr));
    if (w === W && h === H && fb) return true;
    W = w; H = h; canvas.width = w; canvas.height = h;
    for (let attempt = 0; attempt < 2; attempt++) {
      freeFB(); const m = k => Math.max(2, k);
      fb = { s: rt(w, h), a: rt(m(w >> 1), m(h >> 1)), b: rt(m(w >> 2), m(h >> 2)), c: rt(m(w >> 2), m(h >> 2)), d: rt(m(w >> 3), m(h >> 3)), e: rt(m(w >> 3), m(h >> 3)) };
      if (Object.values(fb).every(x => x.ok)) break;
      if (!hdr) { fail('framebuffer incomplete'); return false }
      hdr = false; info.hdr = false;                                    /* retry with plain 8-bit targets */
    }
    info.fbo = hdr ? 'RGBA16F' : 'RGBA8'; gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return true;
  }
  /* ---------- camera ---------- */
  const cam = { target: [0, 0, 0], dist: 3.4, yaw: .6, pitch: .25, fov: .8, roll: 0 };
  const goal = { target: [0, 0, 0], dist: 3.4, yaw: .6, pitch: .25 };
  let fly = null, auto = opt.auto || 0, drag = null, vel = 0, lastInput = 0;
  function flyTo(g, dur) { fly = { from: { target: [...cam.target], dist: cam.dist, yaw: cam.yaw, pitch: cam.pitch }, to: Object.assign({ target: [...goal.target], dist: goal.dist, yaw: goal.yaw, pitch: goal.pitch }, g), t0: now(), dur: (dur || 1.6) * 1000 };
    let dy = fly.to.yaw - fly.from.yaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy)); fly.to.yaw = fly.from.yaw + dy;
    return new Promise(r => fly.done = r) }
  function setCam(g) { Object.assign(cam, g); Object.assign(goal, JSON.parse(JSON.stringify(g))); if (g.target) { cam.target = [...g.target]; goal.target = [...g.target] } fly = null }
  const t0 = performance.now(), now = () => performance.now() - t0;
  const pointers = new Map(); let pinch0 = 0;
  function onDown(e) { if (!opt.controls) return; pointers.set(e.pointerId, [e.clientX, e.clientY]); if (pointers.size === 1) drag = { x: e.clientX, y: e.clientY, moved: 0, t: performance.now() }; if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch0 = Math.hypot(a[0] - b[0], a[1] - b[1]) } try { canvas.setPointerCapture(e.pointerId) } catch (_) { } lastInput = performance.now() }
  function onMove(e) {
    if (!pointers.has(e.pointerId)) return; const prev = pointers.get(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]);
    if (pointers.size === 2) { const [a, b] = [...pointers.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]); if (pinch0) { goal.dist = clamp(goal.dist * pinch0 / d, opt.minDist || 1.2, opt.maxDist || 8); fly = null } pinch0 = d; return }
    if (!drag) return; const dx = e.clientX - prev[0], dy = e.clientY - prev[1]; drag.moved += Math.abs(dx) + Math.abs(dy);
    if (drag.moved > 4) { fly = null; goal.yaw -= dx * .0065; goal.pitch = clamp(goal.pitch + dy * .005, -1.25, 1.25); vel = -dx * .0065 / Math.max(.008, 1 / 60) } lastInput = performance.now();
  }
  function onUp(e) { const d = drag; pointers.delete(e.pointerId); if (pointers.size < 2) pinch0 = 0; if (pointers.size === 0) { drag = null; if (d && d.moved < 6 && R.onTap) { const r = canvas.getBoundingClientRect(); R.onTap(e.clientX - r.left, e.clientY - r.top) } } }
  function onWheel(e) { if (!opt.controls || !opt.wheel) return; e.preventDefault(); goal.dist = clamp(goal.dist * Math.exp(e.deltaY * .0012), opt.minDist || 1.2, opt.maxDist || 8); fly = null; lastInput = performance.now() }
  canvas.addEventListener('pointerdown', onDown); canvas.addEventListener('pointermove', onMove); canvas.addEventListener('pointerup', onUp); canvas.addEventListener('pointercancel', onUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  /* ---------- layers ---------- */
  const layers = [];
  function addMesh(d, uni) {
    if (!PM) return null; const vao = gl.createVertexArray(); gl.bindVertexArray(vao); const bufs = [];
    const attr = (loc, arr, size) => { const b = gl.createBuffer(); bufs.push(b); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0) };
    attr(0, d.pos, 3); attr(1, d.nrm, 3); const ib = gl.createBuffer(); bufs.push(ib); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, d.idx, gl.STATIC_DRAW); gl.bindVertexArray(null);
    const L = Object.assign({ type: 'mesh', vao, bufs, n: d.idx.length, fade: 1, tint: [.45, .6, 1], scan: 1, visible: true }, uni || {}); layers.push(L); return L;
  }
  function addLines(d, uni) {
    if (!PLN) return null; const vao = gl.createVertexArray(); gl.bindVertexArray(vao); const bufs = [];
    const attr = (loc, arr, size) => { const b = gl.createBuffer(); bufs.push(b); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0) };
    attr(0, d.pos, 3); attr(1, d.col, 4); attr(2, d.ph, 1); gl.bindVertexArray(null);
    const L = Object.assign({ type: 'lines', vao, bufs, n: d.n, fade: 1, flow: 1, visible: true }, uni || {}); layers.push(L); return L;
  }
  function addLayer(d, uni) {
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao); const bufs = [];
    const attr = (loc, arr, size) => { const b = gl.createBuffer(); bufs.push(b); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0) };
    attr(0, d.pos, 3); attr(1, d.col, 4); attr(2, d.a, 4); gl.bindVertexArray(null);
    const L = Object.assign({ type: 'points', vao, bufs, n: d.n, fade: 1, focus: -1, gain: new Float32Array(16).fill(.6), tr: new Float32Array(16).fill(0), sat: 1.2, base: [.42, .5, 1], vel: 1, my: 0, inn: .1, visible: true, waves: [] }, uni || {});
    layers.push(L); return L;
  }
  function removeLayer(L) { const i = layers.indexOf(L); if (i < 0) return; layers.splice(i, 1); gl.deleteVertexArray(L.vao); L.bufs.forEach(b => gl.deleteBuffer(b)) }
  /* ---------- frame ---------- */
  let raf = 0, running = false, ft = [], visible = true, lastT = 0, frames = 0, failed = false, destroyed = false;
  function fail(why) { if (failed || destroyed) return; failed = true; info.errors.push(why); console.warn('NeuroGL failed:', why); R.stop(); R.failed = why; try { R.onFail && R.onFail(why) } catch (e) { } }
  let dbg = null; if (/[?&]gldebug\b/.test(location.search)) { dbg = document.createElement('pre'); dbg.style.cssText = 'position:fixed;left:6px;bottom:90px;z-index:99;font:11px monospace;color:#9f9;background:rgba(0,0,0,.75);padding:6px 8px;border-radius:8px;pointer-events:none;max-width:90vw;white-space:pre-wrap'; document.body.appendChild(dbg) }
  const io = 'IntersectionObserver' in window ? new IntersectionObserver(es => { visible = es[0].isIntersecting }) : null; io && io.observe(canvas);
  const waveA = new Float32Array(6), waveV = new Float32Array(24);
  function frame() {
    raf = requestAnimationFrame(frame);
    if (!visible || document.hidden) return;
    const tms = now(), t = tms / 1000, dt = Math.min(.05, Math.max(0, t - lastT)); lastT = t;
    if (failed || !resize()) return;
    if (!probed) { probed = true; if (!probe()) { if (hdr) { hdr = false; info.hdr = false; W = 0; probed = false; return } fail('probe ' + info.probe); return } }
    frames++;
    /* adaptive resolution: keep weak phones smooth */
    ft.push(dt); if (ft.length > 40) { const avg = ft.reduce((a, b) => a + b) / ft.length; ft = []; if (avg > .028 && scale > .55) { scale = Math.max(.55, scale - .15); W = 0 } else if (avg < .017 && scale < 1) { scale = Math.min(1, scale + .1); W = 0 } }
    /* camera */
    if (fly) { const k = clamp((tms - fly.t0) / fly.dur, 0, 1), e = k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      cam.dist = lerp(fly.from.dist, fly.to.dist, e) * (1 + Math.sin(k * Math.PI) * (fly.to.arc || 0)); cam.yaw = lerp(fly.from.yaw, fly.to.yaw, e); cam.pitch = lerp(fly.from.pitch, fly.to.pitch, e);
      for (let i = 0; i < 3; i++)cam.target[i] = lerp(fly.from.target[i], fly.to.target[i], e);
      if (k >= 1) { goal.dist = fly.to.dist; goal.yaw = fly.to.yaw; goal.pitch = fly.to.pitch; goal.target = [...fly.to.target]; const d = fly.done; fly = null; d && d() } }
    else {
      if (!drag) { goal.yaw += vel * dt; vel *= Math.pow(.04, dt); if (auto && performance.now() - lastInput > 2500) goal.yaw += auto * dt }
      const k = 1 - Math.pow(.0005, dt); cam.yaw = lerp(cam.yaw, goal.yaw, k); cam.pitch = lerp(cam.pitch, goal.pitch, k); cam.dist = lerp(cam.dist, goal.dist, k * .7);
      for (let i = 0; i < 3; i++)cam.target[i] = lerp(cam.target[i], goal.target[i], k * .7);
    }
    R.onFrame && R.onFrame(t, dt);
    const cp = Math.cos(cam.pitch), eye = [cam.target[0] + Math.sin(cam.yaw) * cp * cam.dist, cam.target[1] + Math.sin(cam.pitch) * cam.dist, cam.target[2] + Math.cos(cam.yaw) * cp * cam.dist];
    const PV = M4.mul(M4.persp(cam.fov, W / H, .02, 60), M4.look(eye, cam.target, [0, 1, 0])); R.PV = PV; R.eye = eye;
    if (R.spkDirty) { gl.bindTexture(gl.TEXTURE_2D, spkTex); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, SPK_W, 2, gl.RGBA, gl.FLOAT, spk); R.spkDirty = false }
    /* scene pass */
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb.s.f); gl.viewport(0, 0, W, H); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); gl.disable(gl.DEPTH_TEST);
    gl.useProgram(P.p); const u = P.u;
    gl.uniformMatrix4fv(u.uPV, false, PV); gl.uniform3fv(u.uCam, [eye[0] - cam.target[0], eye[1] - cam.target[1], eye[2] - cam.target[2]]);
    gl.uniform1f(u.uDist, cam.dist); gl.uniform1f(u.uFocusD, R.focusD || cam.dist); gl.uniform1f(u.uDof, R.dof); gl.uniform1f(u.uTime, t); gl.uniform1f(u.uPx, H * .5 / Math.tan(cam.fov / 2) * .01);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, spkTex); gl.uniform1i(u.uSpk, 0);
    for (const L of layers) {
      if (L.type !== 'points' || !L.visible || L.fade <= .003) continue;
      waveA.fill(0); L.waves.forEach((w, i) => { if (i < 6) { waveV.set([w.x, w.y, w.z, w.r], i * 4); waveA[i] = w.a } });
      gl.uniform4fv(u.uWave, waveV); gl.uniform1fv(u.uWaveA, waveA);
      gl.uniform1f(u.uFade, L.fade); gl.uniform1f(u.uFocus, L.focus); gl.uniform1fv(u.uGain, L.gain); gl.uniform1fv(u.uTr, L.tr);
      gl.uniform1f(u.uSat, L.sat); gl.uniform3fv(u.uBase, L.base); gl.uniform1f(u.uVel, L.vel); gl.uniform1f(u.uMy, L.my); gl.uniform1f(u.uIN, L.inn);
      gl.bindVertexArray(L.vao); gl.drawArrays(gl.POINTS, 0, L.n);
    }
    const eyeRel = [eye[0], eye[1], eye[2]];
    for (const L of layers) {
      if (L.type === 'points' || !L.visible || L.fade <= .003) continue;
      if (L.type === 'mesh') { gl.useProgram(PM.p); gl.uniformMatrix4fv(PM.u.uPV, false, PV); gl.uniform3fv(PM.u.uEye, eyeRel); gl.uniform1f(PM.u.uTime, t); gl.uniform1f(PM.u.uFade, L.fade); gl.uniform1f(PM.u.uScan, L.scan); gl.uniform3fv(PM.u.uTint, L.tint);
        gl.bindVertexArray(L.vao); gl.drawElements(gl.TRIANGLES, L.n, gl.UNSIGNED_INT, 0) }
      else { gl.useProgram(PLN.p); gl.uniformMatrix4fv(PLN.u.uPV, false, PV); gl.uniform1f(PLN.u.uTime, t); gl.uniform1f(PLN.u.uFade, L.fade); gl.uniform1f(PLN.u.uFlow, L.flow); gl.uniform1f(PLN.u.uDist, cam.dist);
        gl.bindVertexArray(L.vao); gl.drawArrays(gl.LINES, 0, L.n) }
    }
    gl.bindVertexArray(emptyVao); gl.disable(gl.BLEND);
    /* bloom: bright-pass ½ → blur ¼ → blur ⅛ */
    const pass = (prog, target, tex, set) => { gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.f : null); gl.viewport(0, 0, target ? target.w : W, target ? target.h : H); gl.useProgram(prog.p); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(prog.u.uT, 0); set && set(prog.u); gl.drawArrays(gl.TRIANGLES, 0, 3) };
    pass(PB, fb.a, fb.s.t, u => { gl.uniform2f(u.uTx, .5 / W, .5 / H); gl.uniform1f(u.uTh, .35) });
    pass(PL, fb.b, fb.a.t, u => gl.uniform2f(u.uDir, 1 / fb.a.w, 0)); pass(PL, fb.c, fb.b.t, u => gl.uniform2f(u.uDir, 0, 1 / fb.b.h));
    pass(PL, fb.d, fb.c.t, u => gl.uniform2f(u.uDir, 1 / fb.c.w, 0)); pass(PL, fb.e, fb.d.t, u => gl.uniform2f(u.uDir, 0, 1 / fb.d.h));
    composite(null, t);
    /* self-check: any GL error in the first frames → hand over to the 2D renderer */
    if (frames <= 4 || frames % 300 === 0) { const e = gl.getError(); if (e !== gl.NO_ERROR && e !== gl.CONTEXT_LOST_WEBGL) fail('gl error 0x' + e.toString(16)) }
    if (dbg && frames % 15 === 0) dbg.textContent = `GL ${info.renderer}\nfbo ${info.fbo} · ${W}×${H} · scale ${scale.toFixed(2)} · ${(1 / Math.max(dt, .001)).toFixed(0)} fps\nlayers ${layers.length} · ${info.errors.join(', ') || 'ok'}`;
  }
  function composite(target, t) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.f : null); gl.viewport(0, 0, target ? target.w : W, target ? target.h : H); gl.useProgram(PC.p); gl.disable(gl.BLEND); gl.bindVertexArray(emptyVao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, fb.s.t); gl.uniform1i(PC.u.uS, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, fb.c.t); gl.uniform1i(PC.u.uB, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, fb.e.t); gl.uniform1i(PC.u.uB2, 2);
    gl.uniform1f(PC.u.uCA, R.ca * .02); gl.uniform1f(PC.u.uBloom, R.bloom); gl.uniform1f(PC.u.uExp, R.exposure); gl.uniform1f(PC.u.uOpaque, opt.opaque ? 1 : 0); gl.uniform1f(PC.u.uTime, t % 100); gl.uniform3fv(PC.u.uBg, opt.bg);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.drawArrays(gl.TRIANGLES, 0, 3); gl.activeTexture(gl.TEXTURE0);
  }
  function project(p) { const m = R.PV; if (!m) return null; const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    if (w <= .01) return null; return [(x / w * .5 + .5) * canvas.clientWidth, (1 - (y / w * .5 + .5)) * canvas.clientHeight, w] }
  const R = {
    gl, cam, goal, layers, addLayer, addMesh, addLines, removeLayer, info, dof: opt.dof, ca: opt.ca, focusD: 0, flyTo, setCam, project, now: () => now() / 1000, spk, SPK_W, spkDirty: true, bloom: opt.bloom, exposure: opt.exposure,
    set auto(v) { auto = v }, get auto() { return auto }, set controls(v) { opt.controls = v },
    start() { if (!running) { running = true; lastT = now() / 1000; raf = requestAnimationFrame(frame) } return R },
    stop() { running = false; cancelAnimationFrame(raf) },
    destroy() { destroyed = true; R.stop(); io && io.disconnect(); canvas.removeEventListener('pointerdown', onDown); canvas.removeEventListener('pointermove', onMove); canvas.removeEventListener('pointerup', onUp); canvas.removeEventListener('pointercancel', onUp); canvas.removeEventListener('wheel', onWheel);
      [...layers].forEach(removeLayer); freeFB(); dbg && dbg.remove(); const e = gl.getExtension('WEBGL_lose_context'); e && e.loseContext() },
    get size() { return [canvas.clientWidth, canvas.clientHeight] },
    get frames() { return frames }, get running() { return running }
  };
  canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); if (!destroyed) fail('context lost') });
  return R;
}

/* =====================================================================
   GENERATORS
   ===================================================================== */
function packer(cap) {
  const pos = new Float32Array(cap * 3), col = new Float32Array(cap * 4), a = new Float32Array(cap * 4); let n = 0;
  return { push(x, y, z, c, kind, size, group, path, phase) { if (n >= cap) return; pos.set([x, y, z], n * 3); col.set([c[0], c[1], c[2], kind], n * 4); a.set([size, group, path, phase], n * 4); n++ },
    get n() { return n }, done() { return { pos: pos.subarray(0, n * 3), col: col.subarray(0, n * 4), a: a.subarray(0, n * 4), n } } };
}
const DOMAINS = ['logic', 'numbers', 'spatial', 'memory', 'attention', 'speed', 'control', 'reflex'];
const LOBES = ['frontal', 'motor', 'parietal', 'temporal', 'occipital', 'cereb', 'stem'];
function domainOf(x, y, z) { if (z < -.6) return 'speed'; if (z > .48) return 'logic'; if (z > .12) return y > .34 ? 'attention' : 'control'; if (z > -.16) return 'reflex'; return y > .38 ? 'spatial' : 'numbers' }
/* particle brain; mode 'domain' → groups = DOMAINS index (8 = none), 'lobe' → LOBES index */
function brain(o) {
  o = Object.assign({ n: 24000, mode: 'domain', colors: {}, seed: 20261016 }, o);
  const r = mulberry32(o.seed), pk = packer(o.n), base = [.45, .55, 1];
  const names = o.mode === 'lobe' ? LOBES : DOMAINS;
  const gyri = (a, b, c) => Math.sin(12 * a + 5 * Math.sin(6 * c)) * Math.cos(11 * b + 4 * Math.sin(5 * a)) + .55 * Math.sin(14 * c + 7 * b + 3 * Math.sin(4 * a));
  const sph = () => { const u = r() * 2 - 1, th = r() * 6.2832, s = Math.sqrt(1 - u * u); return [s * Math.cos(th), u, s * Math.sin(th)] };
  const cen = {}; names.forEach(k => cen[k] = [0, 0, 0, 0]);
  let guard = 0;
  while (pk.n < o.n && guard++ < o.n * 40) {
    const k = r(); let x, y, z, dom = null, lobe, br = 1;
    if (k < .13) { const [dx, dy, dz] = sph(); if (dy > .4) continue; x = dx * .52; y = -.5 + dy * .2; z = -.6 + dz * .3; dom = 'reflex'; lobe = 'cereb'; br = .45 + .55 * Math.abs(Math.sin(dy * 34 + dz * 6)) }
    else if (k < .16) { const a = r() * 6.2832, h = r(); x = Math.cos(a) * .11 * (1 - h * .25); z = -.22 - h * .12 + Math.sin(a) * .11; y = -.38 - h * .6; br = .7; lobe = 'stem' }
    else if (k < .37) { const [dx, dy, dz] = sph(), side = r() < .5 ? -1 : 1; if (side * dx < -.3 && r() < .85) continue; const g = gyri(dx, dy, dz); if (g < -.45 && r() < .75) continue;
      const bump = 1 + .05 * g; x = side * (.47 + dx * .27 * bump); y = -.36 + dy * .19 * bump; z = .08 + dz * .5 * bump; if (z > .38) y += (z - .38) * .25; dom = 'memory'; lobe = 'temporal'; br = .55 + .45 * clamp((g + 1.2) / 2, 0, 1) }
    else { const [dx, dy, dz] = sph(), side = r() < .5 ? -1 : 1; if (side * dx < -.5 && r() < .8) continue; const g = gyri(dx * 1.1, dy * 1.1, dz * 1.1); if (g < -.45 && r() < .72) continue;
      const bump = 1 + .05 * g; x = side * .37 + dx * .43 * bump; y = .02 + dy * .68 * bump; z = dz * bump;
      if (y < -.16) y = -.16 + (y + .16) * .45; if (z < -.55) { x *= 1 - (-.55 - z) * .5; y *= 1 - (-.55 - z) * .3 } if (z > .6) y *= 1 - (z - .6) * .35;
      if (Math.abs(x) > .5 && y < -.04 && z > -.3 && z < .62 && r() < .9) continue;
      const cs = .02 - .28 * y; if (Math.abs(z - cs) < .03 && y > .05 && r() < .85) continue; if (Math.abs(x) < .06 && y > .25 && r() < .85) continue;
      dom = domainOf(x, y, z); lobe = z < -.55 ? 'occipital' : z > cs ? (z - cs < .15 && y > -.02 ? 'motor' : 'frontal') : 'parietal'; br = .5 + .5 * clamp((g + 1.2) / 2, 0, 1) }
    y += .1;
    const key = o.mode === 'lobe' ? lobe : dom, gi = key ? names.indexOf(key) : 8, c = key && o.colors[key] ? hex(o.colors[key]) : base;
    if (key && cen[key]) { const q = cen[key]; q[0] += x; q[1] += y; q[2] += z; q[3]++ }
    pk.push(x, y, z, c, 0, (1.1 + r() * .9) * br, gi < 0 ? 8 : gi, 0, r() * 6.28);
  }
  for (const k in cen) { const q = cen[k]; cen[k] = q[3] ? [q[0] / q[3], q[1] / q[3], q[2] / q[3]] : [0, 0, 0] }
  return Object.assign(pk.done(), { cen, names });
}
/* curved white-matter tracts between region centroids, streaming light */
function tracts(cen, pairs, colors, per) {
  per = per || 150; const pk = packer(pairs.length * per * 3), r = mulberry32(77);
  pairs.forEach(([a, b], ti) => {
    const A = cen[a], B = cen[b]; if (!A || !B) return;
    const m = [(A[0] + B[0]) * .25, (A[1] + B[1]) * .3 + .05, (A[2] + B[2]) * .25];
    const ca = hex(colors[a] || '#7d9bff'), cb = hex(colors[b] || '#7d9bff');
    for (let s = 0; s < 3; s++) {
      const off = [(r() - .5) * .05, (r() - .5) * .05, (r() - .5) * .05];
      for (let i = 0; i < per; i++) { const t = i / (per - 1), u = 1 - t;
        const P = [0, 1, 2].map(k => u * u * A[k] * .92 + 2 * u * t * (m[k] + off[k] * 2) + t * t * B[k] * .92 + off[k] * Math.sin(t * Math.PI));
        const c = [0, 1, 2].map(k => lerp(ca[k], cb[k], t));
        pk.push(P[0], P[1], P[2], c, 1, .7 + r() * .4, ti, t, r() * 6.28) }
    }
  });
  return pk.done();
}
function dust(n, R, seed) { const r = mulberry32(seed || 5), pk = packer(n); for (let i = 0; i < n; i++) { const u = r() * 2 - 1, th = r() * 6.28, s = Math.sqrt(1 - u * u), d = R * (.35 + Math.pow(r(), .5) * .65); pk.push(s * Math.cos(th) * d, u * d * .7, s * Math.sin(th) * d, [.55, .65, 1], 2, .35 + r() * .5, 0, 0, r() * 6.28) } return pk.done() }

/* ---------- cortical column of neurons ---------- */
function column(o) {
  o = Object.assign({ n: 24, seed: 1, color: '#5ef2ff', myelin: 0, density: .6, inter: .22, detail: 1, R: 1.05, H: 1.35 }, o);
  let rnd = mulberry32(o.seed); const N = Math.min(64, o.n), C = hex(o.color), step = .0078 / o.detail;
  const cellRng = i => mulberry32((o.seed * 7919 + i * 104729) >>> 0);
  const hash2 = (a, b) => { let h = Math.imul(a ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + o.seed, 0xc2b2ae35); h ^= h >>> 13; h = Math.imul(h, 0x27d4eb2f); return ((h ^ h >>> 16) >>> 0) / 4294967296 };
  const white = [.95, .97, 1], axc = [lerp(C[0], .75, .55), lerp(C[1], .85, .55), lerp(C[2], 1, .55)], my = [.86, .84, 1], bout = [1, .92, .62];
  const cells = [], dend = [], axon = [];
  /* somata: pyramidal cells in layers II/III and V, interneurons scattered */
  for (let i = 0; i < N; i++) {
    rnd = cellRng(i);
    const inter = rnd() < o.inter && i > 2, layer = rnd() < .55 ? .4 : -.3;
    let p; for (let g = 0; g < 40; g++) { const a = rnd() * 6.28, d = Math.sqrt(rnd()) * o.R * .85; p = [Math.cos(a) * d, layer + (rnd() - .5) * .62, Math.sin(a) * d]; if (cells.every(c => Math.hypot(c.p[0] - p[0], c.p[1] - p[1], c.p[2] - p[2]) > .17)) break }
    cells.push({ p, inter, axLen: 0 });
  }
  const growth = [];  /* branches as polylines with cumulative path length */
  function grow(i, kind, start, dir, len, path0, curl) {
    const pts = [[...start]], d = [...dir]; let L = path0; const n = Math.max(3, Math.round(len / .03));
    for (let s = 0; s < n; s++) { d[0] += (rnd() - .5) * curl; d[1] += (rnd() - .5) * curl * .6; d[2] += (rnd() - .5) * curl; const l = Math.hypot(...d); d[0] /= l; d[1] /= l; d[2] /= l;
      const q = pts[pts.length - 1], seg = len / n; if (Math.abs(q[1]) > o.H * .92) d[1] *= .4; const nq = [q[0] + d[0] * seg, q[1] + d[1] * seg, q[2] + d[2] * seg];
      const rr = Math.hypot(nq[0], nq[2]); if (rr > o.R * 1.3) { d[0] -= nq[0] * .3; d[2] -= nq[2] * .3 }
      pts.push(nq) }
    const b = { i, kind, pts, path0, len, dir: d }; growth.push(b); return b;
  }
  const at = (b, f) => { const k = Math.min(b.pts.length - 1, Math.round(f * (b.pts.length - 1))); return [b.pts[k], b.path0 + b.len * k / (b.pts.length - 1)] };
  const rdir = (yb) => { const a = rnd() * 6.28, y = yb + (rnd() - .5) * .5; const h = Math.sqrt(Math.max(0, 1 - y * y)); return [Math.cos(a) * h, y, Math.sin(a) * h] };
  cells.forEach((c, i) => {
    rnd = cellRng(i + 5000);
    if (!c.inter) {
      const ap = grow(i, 'd', c.p, [0, 1, 0], clamp(o.H - c.p[1], .5, 1.3) * .95, 0, .35);
      for (let k = 0; k < 4; k++) { const [q, pp] = at(ap, .2 + k * .17); grow(i, 'd', q, rdir(.25), .18 + rnd() * .16, pp, .5) }
      const [top, tp] = at(ap, 1); for (let k = 0; k < 4; k++)grow(i, 'd', top, rdir(.45), .2 + rnd() * .2, tp, .55);
      for (let k = 0; k < 5; k++) { const b = grow(i, 'd', c.p, rdir(-.25), .22 + rnd() * .16, 0, .5); const [q, pp] = at(b, .6); grow(i, 'd', q, rdir(-.2), .12 + rnd() * .1, pp, .6) }
      const ax = grow(i, 'a', c.p, [rnd() * .2 - .1, -1, rnd() * .2 - .1], clamp(c.p[1] + o.H, .5, 1.5) * .92, 0, .22); ax.main = true; c.axLen = ax.len;
      for (let k = 0; k < 3; k++) { const [q, pp] = at(ax, .18 + k * .25); const cb = grow(i, 'a', q, rdir(.35), .35 + rnd() * .35, pp, .45); const [q2, p2] = at(cb, .6); grow(i, 'a', q2, rdir(.2), .2 + rnd() * .2, p2, .5) }
    } else {
      for (let k = 0; k < 6; k++)grow(i, 'd', c.p, rdir(0), .14 + rnd() * .14, 0, .7);
      for (let k = 0; k < 5; k++) { const b = grow(i, 'a', c.p, rdir(0), .22 + rnd() * .2, 0, .6); const [q, pp] = at(b, .5); grow(i, 'a', q, rdir(0), .15 + rnd() * .15, pp, .7) }
    }
  });
  /* sample points */
  const pk = packer(N * 2600 * o.detail + 4000), grid = new Map(), cell = .06, key = (x, y, z) => ((x / cell | 0) + 500) * 1e6 + ((y / cell | 0) + 500) * 1e3 + ((z / cell | 0) + 500);
  cells.forEach((c, i) => { rnd = cellRng(i + 9000); const n = c.inter ? 34 : 60; for (let k = 0; k < n; k++) { const v = rdir(0), d = Math.pow(rnd(), .6) * (c.inter ? .028 : .04); const elong = c.inter ? 1 : 1.35; pk.push(c.p[0] + v[0] * d, c.p[1] + v[1] * d * elong + (c.inter ? 0 : d * .4), c.p[2] + v[2] * d, white, 3, 1.05 + rnd() * .6, i, 0, rnd() * 6.28) } });
  const myelinOn = o.myelin > 0, myFrac = clamp(o.myelin / 5, 0, 1), IN = .1;
  for (const b of growth) {
    const cc = cells[b.i]; rnd = cellRng(b.i + 13000 + b.pts.length * 31);
    for (let s = 0; s < b.pts.length - 1; s++) {
      const A = b.pts[s], B = b.pts[s + 1], seg = Math.hypot(B[0] - A[0], B[1] - A[1], B[2] - A[2]), m = Math.max(1, Math.round(seg / step));
      for (let k = 0; k < m; k++) {
        const f = k / m, x = lerp(A[0], B[0], f), y = lerp(A[1], B[1], f), z = lerp(A[2], B[2], f), path = b.path0 + (s + f) * b.len / (b.pts.length - 1);
        if (b.kind === 'd') {
          const taper = 1 - clamp((path) / 1.3, 0, .6);
          pk.push(x, y, z, cc.inter ? [lerp(C[0], 1, .25), lerp(C[1], .7, .3), lerp(C[2], .6, .3)] : C, 4, (.62 + .3 * taper) * (1 + rnd() * .2), b.i, path, rnd() * 6.28);
          const g = key(x, y, z); let arr = grid.get(g); if (!arr) grid.set(g, arr = []); arr.push([x, y, z, b.i]);
          if (rnd() < .15 * o.detail && !cc.inter) { const v = rdir(0), l = .012 + rnd() * .01; pk.push(x + v[0] * l, y + v[1] * l, z + v[2] * l, [lerp(C[0], 1, .4), lerp(C[1], 1, .4), lerp(C[2], 1, .4)], 8, .55, b.i, path, rnd() * 6.28) }
        } else {
          const sheath = b.main && myelinOn && path > .05 && path < .05 + cc.axLen * myFrac && (path % IN) > .014;
          if (sheath) { for (let q = 0; q < 2; q++) { const v = rdir(0), l = .007; pk.push(x + v[0] * l, y + v[1] * l, z + v[2] * l, my, 6, 1.0, b.i, path, rnd() * 6.28) } }
          else pk.push(x, y, z, cc.inter ? [1, .62, .55] : axc, 5, .55, b.i, path, rnd() * 6.28);
          axon.push([x, y, z, b.i, path, b.main && myelinOn && path < .05 + cc.axLen * myFrac]);
        }
      }
    }
  }
  /* synapses: an axon passing close to another cell's dendrite forms a bouton */
  const syn = [], perPair = new Map();
  for (let q = 0; q < axon.length; q += 2) {
    const [x, y, z, i, path] = axon[q]; if (path < .08) continue;
    const gx = x / cell | 0, gy = y / cell | 0, gz = z / cell | 0; let hit = null;
    for (let a = -1; a <= 1 && !hit; a++)for (let b = -1; b <= 1 && !hit; b++)for (let c = -1; c <= 1 && !hit; c++) {
      const arr = grid.get((gx + a + 500) * 1e6 + (gy + b + 500) * 1e3 + (gz + c + 500)); if (!arr) continue;
      for (const d of arr) { if (d[3] === i) continue; if (Math.hypot(d[0] - x, d[1] - y, d[2] - z) < .058) { hit = d; break } } }
    if (!hit) continue; const pairK = i * 100 + hit[3], cnt = perPair.get(pairK) || 0;
    if (cnt >= 3 || hash2(i, Math.round(path * 1000)) > o.density) continue; if (syn.some(s => s.pre === i && Math.abs(s.path - path) < .07)) continue;
    perPair.set(pairK, cnt + 1);
    syn.push({ pre: i, post: hit[3], path, my: axon[q][5], w: cells[i].inter ? -.75 : .5 + rnd() * .25 });
    pk.push(x, y, z, bout, 7, 2.6, i, path, rnd() * 6.28);
    pk.push(hit[0], hit[1], hit[2], bout, 8, 1.2, hit[3], 0, rnd() * 6.28);
  }
  return Object.assign(pk.done(), { cells, syn, N });
}

/* =====================================================================
   SPIKING NETWORK — leaky integrate-and-fire with conduction delays
   ===================================================================== */
function Net(col, R, o) {
  o = Object.assign({ tau: .09, th: 1, ref: .12, ext: .62, rate: 5, act: 1, vel: 1, myVel: 3 }, o || {});
  const N = col.N, v = new Float32Array(N), refU = new Float32Array(N).fill(-1), q = [], ring = new Uint8Array(N), ringI = new Uint8Array(N), out = Array.from({ length: N }, () => []);
  col.syn.forEach(s => out[s.pre].push(s));
  let last = R.now(), spikes = 0, rateEMA = 0;
  const S = R.spk, W = R.SPK_W;
  function fire(i, t) { v[i] = 0; refU[i] = t + o.ref; S[i * 4 + ring[i]] = t; ring[i] = (ring[i] + 1) & 3; R.spkDirty = true; spikes++;
    for (const s of out[i]) { const vel = s.my ? o.myVel : o.vel; q.push({ t: t + s.path / vel, post: s.post, w: s.w }) } net.onSpike && net.onSpike(i, t) }
  function input(i, t, w) { if (t < refU[i]) return; v[i] += w; S[(W + i) * 4 + ringI[i]] = t; ringI[i] = (ringI[i] + 1) & 3; R.spkDirty = true; if (v[i] >= o.th) fire(i, t) }
  const net = {
    o, N,
    step() {
      const t = R.now(), dt = Math.min(.1, t - last); last = t; if (dt <= 0) return;
      const dec = Math.exp(-dt / o.tau); for (let i = 0; i < N; i++)v[i] *= dec;
      const lam = o.rate * o.act * dt; for (let i = 0; i < N; i++)if (Math.random() < lam) input(i, t - Math.random() * dt, o.ext * (.8 + Math.random() * .5));
      q.sort((a, b) => a.t - b.t); while (q.length && q[0].t <= t) { const e = q.shift(); input(e.post, e.t, e.w) }
      rateEMA = lerp(rateEMA, spikes / dt, .05); spikes = 0;
    },
    stim(i, t) { t = t || R.now(); if (i == null) i = Math.random() * N | 0; fire(i, t) },
    get rate() { return rateEMA }
  };
  return net;
}


/* ---------- glass shell: density field from the particle brain → naive surface nets ---------- */
function shell(b, o) {
  o = Object.assign({ res: 56, iso: .08, blur: 4 }, o || {});
  let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
  for (let i = 0; i < b.n; i++)for (let k = 0; k < 3; k++) { const v = b.pos[i * 3 + k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v }
  const pad = .1; for (let k = 0; k < 3; k++) { mn[k] -= pad; mx[k] += pad }
  const span = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]), cs = span / o.res;
  const nx = Math.ceil((mx[0] - mn[0]) / cs) + 1, ny = Math.ceil((mx[1] - mn[1]) / cs) + 1, nz = Math.ceil((mx[2] - mn[2]) / cs) + 1;
  const I = (x, y, z) => x + nx * (y + ny * z); let F = new Float32Array(nx * ny * nz);
  for (let i = 0; i < b.n; i++) { const x = Math.round((b.pos[i * 3] - mn[0]) / cs), y = Math.round((b.pos[i * 3 + 1] - mn[1]) / cs), z = Math.round((b.pos[i * 3 + 2] - mn[2]) / cs); F[I(x, y, z)] = 1 }  /* occupancy, so dense parts don't dominate */
  /* separable box blur ×blur ≈ gaussian */
  const tmp = new Float32Array(F.length);
  for (let pass = 0; pass < o.blur; pass++)for (let ax = 0; ax < 3; ax++) {
    for (let z = 0; z < nz; z++)for (let y = 0; y < ny; y++)for (let x = 0; x < nx; x++) {
      let sum = 0, c = 0; for (let d = -1; d <= 1; d++) { const X = ax === 0 ? x + d : x, Y = ax === 1 ? y + d : y, Z = ax === 2 ? z + d : z; if (X < 0 || Y < 0 || Z < 0 || X >= nx || Y >= ny || Z >= nz) continue; sum += F[I(X, Y, Z)]; c++ }
      tmp[I(x, y, z)] = sum / 3 }
    F.set(tmp) }
  const iso = o.iso, vid = new Int32Array(F.length).fill(-1), pos = [], nrm = [], idx = [];
  const val = (x, y, z) => F[I(x, y, z)] - iso;
  const E = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const C = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  for (let z = 0; z < nz - 1; z++)for (let y = 0; y < ny - 1; y++)for (let x = 0; x < nx - 1; x++) {
    const v = C.map(c => val(x + c[0], y + c[1], z + c[2])); let m = 0; v.forEach((q, i) => { if (q > 0) m |= 1 << i }); if (m === 0 || m === 255) continue;
    let px = 0, py = 0, pz = 0, n = 0;
    for (const [a, bb] of E) { if ((v[a] > 0) === (v[bb] > 0)) continue; const t = v[a] / (v[a] - v[bb]); px += C[a][0] + (C[bb][0] - C[a][0]) * t; py += C[a][1] + (C[bb][1] - C[a][1]) * t; pz += C[a][2] + (C[bb][2] - C[a][2]) * t; n++ }
    px /= n; py /= n; pz /= n;
    const gx = (val(Math.min(nx - 1, x + 1), y, z) - val(Math.max(0, x - 1), y, z)), gy = (val(x, Math.min(ny - 1, y + 1), z) - val(x, Math.max(0, y - 1), z)), gz = (val(x, y, Math.min(nz - 1, z + 1)) - val(x, y, Math.max(0, z - 1)));
    const gl = Math.hypot(gx, gy, gz) || 1;
    vid[I(x, y, z)] = pos.length / 3; pos.push(mn[0] + (x + px) * cs, mn[1] + (y + py) * cs, mn[2] + (z + pz) * cs); nrm.push(-gx / gl, -gy / gl, -gz / gl);
  }
  /* one quad per sign-changing grid edge, joining the 4 cells around it */
  for (let z = 1; z < nz - 1; z++)for (let y = 1; y < ny - 1; y++)for (let x = 1; x < nx - 1; x++) {
    const s0 = val(x, y, z) > 0;
    const q = (a, b2, c, d) => { if (a < 0 || b2 < 0 || c < 0 || d < 0) return; idx.push(a, b2, c, a, c, d) };
    if (s0 !== (val(x + 1, y, z) > 0)) q(vid[I(x, y - 1, z - 1)], vid[I(x, y, z - 1)], vid[I(x, y, z)], vid[I(x, y - 1, z)]);
    if (s0 !== (val(x, y + 1, z) > 0)) q(vid[I(x - 1, y, z - 1)], vid[I(x, y, z - 1)], vid[I(x, y, z)], vid[I(x - 1, y, z)]);
    if (s0 !== (val(x, y, z + 1) > 0)) q(vid[I(x - 1, y - 1, z)], vid[I(x, y - 1, z)], vid[I(x, y, z)], vid[I(x - 1, y, z)]);
  }
  /* smooth normals a bit by averaging the vertex positions once (Laplacian), keeps it glassy */
  return { pos: new Float32Array(pos), nrm: new Float32Array(nrm), idx: new Uint32Array(idx), verts: pos.length / 3 };
}

/* ---------- tractography: streamlines through a white-matter direction field, DTI colour code ---------- */
/* convention used in diffusion MRI: red = left↔right, green = front↔back, blue = up↔down */
function fibers(o) {
  o = Object.assign({ n: 1200, steps: 70, h: .022, seed: 4242 }, o || {});
  const r = mulberry32(o.seed);
  const inside = (x, y, z) => {
    const hemi = s => { const a = (x - s * .33) / .44, b = (y - .14) / .62, c = z / .95; return a * a + b * b + c * c < 1 };
    const temp = s => { const a = (x - s * .45) / .26, b = (y + .24) / .2, c = (z - .08) / .48; return a * a + b * b + c * c < 1 };
    return (hemi(1) || hemi(-1) || temp(1) || temp(-1)) && Math.abs(x) < .78;
  };
  const noise = (x, y, z) => [Math.sin(y * 5.1 + z * 2.3) * .25, Math.sin(z * 4.3 + x * 3.7) * .25, Math.sin(x * 4.7 + y * 3.1) * .25];
  function field(x, y, z) {
    const cc = 1.3 * Math.exp(-Math.pow((y - .22) / .24, 2)) * Math.exp(-Math.pow(x / .55, 2)) * (Math.abs(z) < .7 ? 1 : .25);   /* corpus callosum: L↔R arch */
    const cst = 1.2 * Math.exp(-Math.pow((Math.abs(x) - .26) / .15, 2)) * Math.exp(-Math.pow((z + .02) / .38, 2));             /* corticospinal: up↔down fan */
    const asc = Math.max(0, 1 - 1.3 * (cc + cst));                                                                                      /* association: front↔back */
    let v = [cc * 1 + cst * Math.sign(x) * .25 + asc * x * .15, cc * (-(x) * .7) + cst * 1 + asc * y * .2, cc * 0 + cst * (-z * .4) + asc * 1];
    const nn = noise(x, y, z); v = [v[0] + nn[0], v[1] + nn[1], v[2] + nn[2]];
    const l = Math.hypot(...v) || 1; return [v[0] / l, v[1] / l, v[2] / l];
  }
  const pos = [], col = [], ph = []; let lines = 0, guard = 0;
  while (lines < o.n && guard++ < o.n * 30) {
    const s = [(r() - .5) * 1.5, (r() - .5) * 1.2 + .05, (r() - .5) * 1.9]; if (!inside(...s)) continue;
    const path = [[...s]];
    for (const dir of [1, -1]) { let p = [...s], prev = null;
      for (let k = 0; k < o.steps / 2; k++) { let v = field(...p); if (prev && v[0] * prev[0] + v[1] * prev[1] + v[2] * prev[2] < 0) v = v.map(c => -c); v = v.map(c => c * dir);
        if (prev) v = v.map((c, i) => c * .6 + prev[i] * .4); const n = [p[0] + v[0] * o.h, p[1] + v[1] * o.h, p[2] + v[2] * o.h]; if (!inside(...n)) break; prev = v.map(c => c * dir); p = n;
        dir > 0 ? path.push(n) : path.unshift(n) } }
    if (path.length < 12) continue;
    const phase = r(), L = path.length;
    for (let i = 0; i < L - 1; i++) { const A = path[i], B = path[i + 1], d = [B[0] - A[0], B[1] - A[1], B[2] - A[2]], l = Math.hypot(...d) || 1;
      const c = [Math.abs(d[0]) / l, Math.abs(d[2]) / l, Math.abs(d[1]) / l], m = Math.max(...c), sat = c.map(v => .15 + .85 * Math.pow(v / m, 1.6));
      pos.push(...A, ...B); col.push(...sat, i / L, ...sat, (i + 1) / L); ph.push(phase, phase) }
    lines++;
  }
  return { pos: new Float32Array(pos), col: new Float32Array(col), ph: new Float32Array(ph), n: pos.length / 3, lines };
}

root.NGL = { supported, create, brain, tracts, dust, column, Net, shell, fibers, DOMAINS, LOBES, hex, mulberry32 };
})(window);
