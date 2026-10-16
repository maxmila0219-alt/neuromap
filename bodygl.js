/* =====================================================================
   NeuroMap BODY-GL — embedded, real-time living-body scenes (WebGL2, rasterised = fast)
   · 'blood'   : flight down a pulsing artery, ~300 instanced biconcave red cells
   · 'neurons' : a forest of tube-mesh neurons with action potentials running down the axons
   Everything is animated on the GPU (instancing + vertex shaders): no per-frame CPU geometry.
   Pauses itself when off-screen; resolution adapts to keep the page smooth.
   ===================================================================== */
(function (root) {
'use strict';
const TAU = Math.PI * 2, clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const RM = !!(root.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296 } }
const M4 = {
  persp(fov, asp, n, f) { const t = 1 / Math.tan(fov / 2), nf = 1 / (n - f); return new Float32Array([t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) * nf, -1, 0, 0, 2 * f * n * nf, 0]) },
  look(e, c, u) { let zx = e[0] - c[0], zy = e[1] - c[1], zz = e[2] - c[2], l = Math.hypot(zx, zy, zz) || 1; zx /= l; zy /= l; zz /= l;
    let xx = u[1] * zz - u[2] * zy, xy = u[2] * zx - u[0] * zz, xz = u[0] * zy - u[1] * zx; l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0, -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1]) },
  mul(a, b) { const o = new Float32Array(16); for (let i = 0; i < 4; i++)for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++)s += a[k * 4 + j] * b[i * 4 + k]; o[i * 4 + j] = s } return o }
};
function compile(gl, vs, fs) {
  const mk = (t, s) => { const sh = gl.createShader(t); gl.shaderSource(sh, s); gl.compileShader(sh); if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh)); return sh };
  const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, a.name) }
  return { p, u };
}
function vao(gl, attrs, index, inst) {
  const v = gl.createVertexArray(); gl.bindVertexArray(v); const bufs = [];
  attrs.forEach(([loc, data, size, div]) => { const b = gl.createBuffer(); bufs.push(b); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0); if (div) gl.vertexAttribDivisor(loc, div) });
  if (index) { const b = gl.createBuffer(); bufs.push(b); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, index, gl.STATIC_DRAW) }
  gl.bindVertexArray(null); return { v, bufs };
}
const NOISE = `
float h21(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float vn(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(h21(i),h21(i+vec2(1,0)),f.x),mix(h21(i+vec2(0,1)),h21(i+vec2(1,1)),f.x),f.y);}
`;
const PATH = `vec3 C(float s){return vec3(sin(s*.11)*1.6+sin(s*.047)*1.2, cos(s*.083)*1.1, s);}
void frame(float s,out vec3 c,out vec3 N,out vec3 B){c=C(s);vec3 T=normalize(C(s+.05)-C(s-.05));N=normalize(cross(T,vec3(0,1,0)));B=cross(N,T);}`;
const TONE = `vec3 aces(vec3 x){return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);}`;

/* ======================= BLOOD ======================= */
const BL_TUBE_VS = `#version 300 es
layout(location=0) in vec2 aUV;
uniform mat4 uPV; uniform float uS0, uLen, uBeat;
out vec3 vW; out vec3 vN; out vec2 vC;
${PATH}
void main(){ float s=uS0+aUV.y*uLen; vec3 c,N,B; frame(s,c,N,B);
  float a=aUV.x*6.2831853; vec3 d=cos(a)*N+sin(a)*B;
  float R=2.35+.07*uBeat+.12*sin(a*3.+s*.7)+.07*sin(a*7.-s*1.3)+.04*sin(s*2.1+a*11.);
  vec3 p=c+d*R; vW=p; vN=-d; vC=vec2(a,s); gl_Position=uPV*vec4(p,1);}`;
const BL_TUBE_FS = `#version 300 es
precision highp float;
in vec3 vW; in vec3 vN; in vec2 vC; out vec4 o;
uniform vec3 uEye; uniform float uBeat, uFade;
${NOISE}${TONE}
void main(){
  vec2 q=vec2(vC.x*2.2,vC.y*1.3);
  float h=vn(q*1.6)*.6+vn(q*4.1)*.3+vn(q*9.3)*.1;
  float vein=smoothstep(.04,.0,abs(vn(q*.9+vec2(vn(q*2.)*1.5,0.))-.5))*.8;
  vec3 P=vW+vN*h*.12; vec3 n=normalize(cross(dFdx(P),dFdy(P))); if(dot(n,uEye-vW)<0.)n=-n;
  vec3 L=uEye-vW; float d=length(L); L/=d;
  float dif=max(dot(n,L),0.), wrap=max((dot(n,L)+.5)/1.5,0.);
  float sp=pow(max(dot(n,normalize(L+normalize(uEye-vW))),0.),60.);
  vec3 alb=mix(vec3(.42,.05,.07),vec3(.7,.2,.22),h)*(1.-vein*.5);
  vec3 col=(alb*(dif*.85+.1)+vec3(.9,.1,.06)*pow(wrap,2.)*.35)*(1.5/(1.+d*d*.09))*(1.+.25*uBeat)+sp*vec3(1.,.85,.8)*.45/(1.+d*d*.06);
  col=mix(vec3(.09,.008,.01),col,exp(-d*.075));
  o=vec4(aces(col*1.2)*uFade,1.);}`;
const BL_CELL_VS = `#version 300 es
layout(location=0) in vec3 aP; layout(location=1) in vec3 aN; layout(location=2) in float aRho;
layout(location=3) in vec4 aI; layout(location=4) in vec4 aJ;
uniform mat4 uPV; uniform float uS0, uLen, uT, uFlow;
out vec3 vW; out vec3 vN; out float vRho;
${PATH}
mat3 rot(vec3 ax,float a){ax=normalize(ax);float s=sin(a),c=cos(a),oc=1.-c;
  return mat3(oc*ax.x*ax.x+c,oc*ax.x*ax.y+ax.z*s,oc*ax.z*ax.x-ax.y*s, oc*ax.x*ax.y-ax.z*s,oc*ax.y*ax.y+c,oc*ax.y*ax.z+ax.x*s, oc*ax.z*ax.x+ax.y*s,oc*ax.y*ax.z-ax.x*s,oc*ax.z*ax.z+c);}
void main(){ float s=uS0+mod(aI.z*uLen+uFlow*(.85+aJ.w*.3)-uS0,uLen); vec3 c,N,B; frame(s,c,N,B);
  float a=aI.x+sin(uT*.3+aI.w)*.1; vec3 ctr=c+(cos(a)*N+sin(a)*B)*aI.y;
  mat3 R=rot(aJ.xyz,aI.w*6.+uT*(.5+aJ.w*1.2));
  vec3 p=ctr+R*(aP*.5); vW=p; vN=R*aN; vRho=aRho; gl_Position=uPV*vec4(p,1);}`;
const BL_CELL_FS = `#version 300 es
precision highp float;
in vec3 vW; in vec3 vN; in float vRho; out vec4 o;
uniform vec3 uEye; uniform float uFade, uBeat;
${TONE}
void main(){ vec3 n=normalize(vN), V=normalize(uEye-vW); if(dot(n,V)<0.)n=-n;
  vec3 L=V; float d=length(uEye-vW);
  float dif=max(dot(n,L),0.), wrap=max((dot(n,L)+.6)/1.6,0.);
  float sp=pow(max(dot(n,normalize(L+V+vec3(0,.3,0))),0.),70.);
  float fr=pow(1.-max(dot(n,V),0.),3.);
  float thin=pow(1.-vRho,2.);
  vec3 alb=vec3(.62,.03,.05);
  vec3 col=alb*(wrap*1.0+.05)+vec3(.9,.16,.06)*thin*(.25+.45*(1.-dif)) + fr*vec3(.8,.15,.18)*.3;
  col=col*(1.6/(1.+d*d*.08))*(1.+.2*uBeat)+sp*vec3(1.,.9,.88)*1.1/(1.+d*d*.05);
  col=mix(vec3(.09,.008,.01),col,exp(-d*.075));
  o=vec4(aces(col*1.05)*uFade,1.);}`;
function rbcMesh() {   /* lathe of the Evans–Fung biconcave profile */
  const NR = 18, NA = 36, P = [], Nn = [], Rh = [], I = [];
  const prof = r => { const rr = Math.min(r * r, 1); return .5 * Math.sqrt(Math.max(1 - rr, 0)) * (.207 + 2.003 * rr - 1.123 * rr * rr) * 1.1 + .012 * Math.sqrt(Math.max(1 - rr, 0)) };
  for (let side = 0; side < 2; side++) { const sg = side ? -1 : 1, base = P.length / 3;
    for (let i = 0; i <= NR; i++) { const r = Math.sin(i / NR * Math.PI / 2);                  /* denser near the rim */
      for (let j = 0; j <= NA; j++) { const a = j / NA * TAU, y = prof(r) * sg; P.push(Math.cos(a) * r, y, Math.sin(a) * r);
        const e = .002, dy = (prof(Math.min(1, r + e)) - prof(Math.max(0, r - e))) / (2 * e) * sg; let nx = -dy * Math.cos(a), nz = -dy * Math.sin(a), ny = sg; if (i === NR) { nx = Math.cos(a); ny = 0; nz = Math.sin(a) }
        const l = Math.hypot(nx, ny, nz); Nn.push(nx / l, ny / l, nz / l); Rh.push(r) } }
    for (let i = 0; i < NR; i++)for (let j = 0; j < NA; j++) { const a = base + i * (NA + 1) + j, b = a + NA + 1; side ? I.push(a, a + 1, b, b, a + 1, b + 1) : I.push(a, b, a + 1, b, b + 1, a + 1) } }
  /* smooth normals from faces, welded across the rim seam */
  const acc = new Float32Array(P.length), key = new Map();
  for (let i = 0; i < I.length; i += 3) { const [a, b, c] = [I[i], I[i + 1], I[i + 2]], ax = P[b * 3] - P[a * 3], ay = P[b * 3 + 1] - P[a * 3 + 1], az = P[b * 3 + 2] - P[a * 3 + 2], bx = P[c * 3] - P[a * 3], by = P[c * 3 + 1] - P[a * 3 + 1], bz = P[c * 3 + 2] - P[a * 3 + 2];
    const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx; for (const v of [a, b, c]) { acc[v * 3] += nx; acc[v * 3 + 1] += ny; acc[v * 3 + 2] += nz } }
  for (let v = 0; v < P.length / 3; v++) { const k = P.slice(v * 3, v * 3 + 3).map(x => x.toFixed(4)).join(); if (!key.has(k)) key.set(k, []); key.get(k).push(v) }
  for (const vs of key.values()) { let x = 0, y = 0, z = 0; vs.forEach(v => { x += acc[v * 3]; y += acc[v * 3 + 1]; z += acc[v * 3 + 2] }); const l = Math.hypot(x, y, z) || 1; vs.forEach(v => { Nn[v * 3] = x / l; Nn[v * 3 + 1] = y / l; Nn[v * 3 + 2] = z / l }) }
  return { P: new Float32Array(P), N: new Float32Array(Nn), R: new Float32Array(Rh), I: new Uint16Array(I) };
}
function sceneBlood(gl, mob) {
  const TU = compile(gl, BL_TUBE_VS, BL_TUBE_FS), CE = compile(gl, BL_CELL_VS, BL_CELL_FS);
  const RA = 56, LA = mob ? 150 : 220, uv = [], idx = [];
  for (let j = 0; j <= LA; j++)for (let i = 0; i <= RA; i++)uv.push(i / RA, j / LA);
  for (let j = 0; j < LA; j++)for (let i = 0; i < RA; i++) { const a = j * (RA + 1) + i, b = a + RA + 1; idx.push(a, b, a + 1, a + 1, b, b + 1) }
  const tube = vao(gl, [[0, new Float32Array(uv), 2]], new Uint32Array(idx)); const tubeN = idx.length;
  const M = rbcMesh(), NI = mob ? 170 : 320, R = rng(12), I1 = [], I2 = [];
  for (let k = 0; k < NI; k++) { I1.push(R() * TAU, .55 + Math.sqrt(R()) * 1.2, R(), R() * TAU); I2.push(R() - .5, R() - .5, R() - .5, R()) }
  const cells = vao(gl, [[0, M.P, 3], [1, M.N, 3], [2, M.R, 1], [3, new Float32Array(I1), 4, 1], [4, new Float32Array(I2), 4, 1]], M.I);
  const LEN = 46; let fd = 0, lt = null;
  return {
    bg: [.09, .008, .01],
    draw(t, beat, look, aspect, fade) {
      const dt = lt == null ? 0 : Math.max(0, Math.min(.1, t - lt)); lt = t; fd += (2.2 + 1.8 * beat) * dt;     /* pulsatile flow, integrated */
      const sc = t * 1.05, C = s => [Math.sin(s * .11) * 1.6 + Math.sin(s * .047) * 1.2, Math.cos(s * .083) * 1.1, s];
      const e = C(sc), f = C(sc + 3); e[0] += Math.sin(t * .3) * .25 + look[0] * .5; e[1] += Math.cos(t * .23) * .2 + look[1] * .4;
      const PV = M4.mul(M4.persp(1.25, aspect, .05, 60), M4.look(e, [f[0] + look[0] * 1.6, f[1] + look[1] * 1.2, f[2]], [Math.sin(t * .1) * .15, 1, 0]));
      const S0 = sc - 2;
      gl.useProgram(TU.p); gl.uniformMatrix4fv(TU.u.uPV, false, PV); gl.uniform1f(TU.u.uS0, S0); gl.uniform1f(TU.u.uLen, LEN); gl.uniform1f(TU.u.uBeat, beat); gl.uniform3fv(TU.u.uEye, e); gl.uniform1f(TU.u.uFade, fade);
      gl.bindVertexArray(tube.v); gl.drawElements(gl.TRIANGLES, tubeN, gl.UNSIGNED_INT, 0);
      gl.useProgram(CE.p); gl.uniformMatrix4fv(CE.u.uPV, false, PV); gl.uniform1f(CE.u.uS0, S0); gl.uniform1f(CE.u.uLen, LEN); gl.uniform1f(CE.u.uT, t); gl.uniform1f(CE.u.uFlow, fd);
      gl.uniform3fv(CE.u.uEye, e); gl.uniform1f(CE.u.uFade, fade); gl.uniform1f(CE.u.uBeat, beat);
      gl.bindVertexArray(cells.v); gl.drawElementsInstanced(gl.TRIANGLES, M.I.length, gl.UNSIGNED_SHORT, 0, NI);
    }
  };
}

/* ======================= NEURONS ======================= */
const NE_VS = `#version 300 es
layout(location=0) in vec3 aP; layout(location=1) in vec3 aN; layout(location=2) in vec3 aA;   /* arc, neuron id, kind */
uniform mat4 uPV;
out vec3 vW; out vec3 vN; out vec3 vA;
void main(){ vW=aP; vN=aN; vA=aA; gl_Position=uPV*vec4(aP,1);}`;
const NE_FS = `#version 300 es
precision highp float;
in vec3 vW; in vec3 vN; in vec3 vA; out vec4 o;
uniform vec3 uEye; uniform float uT, uFade, uKick;
${TONE}
vec3 pal(float i){ return mix(mix(vec3(.30,.55,1.),vec3(.62,.38,1.),smoothstep(.2,.6,i)),vec3(.25,.95,.95),smoothstep(.75,1.,i)); }
void main(){ vec3 n=normalize(vN), V=uEye-vW; float d=length(V); V/=d; if(dot(n,V)<0.)n=-n;
  vec3 L=normalize(vec3(.4,.8,.3));
  float dif=max(dot(n,L),0.)*.6+max(dot(n,V),0.)*.4, fr=pow(1.-max(dot(n,V),0.),2.5);
  vec3 base=pal(vA.y);
  float rate=.16+vA.y*.22, ph=fract(uT*rate+vA.y*7.3), head=ph*34.;
  float pulse=exp(-pow(vA.x-head,2.)*(vA.z>.5?1.2:2.5))*(vA.z>.5?1.:.55)*smoothstep(1.,.85,ph);
  float soma=step(vA.x,.01)*(exp(-ph*10.)*2.5+.35);
  vec3 col=base*(.04+dif*.28)+fr*mix(base,vec3(.8,.9,1.),.5)*1.4;
  col+=(vec3(.75,.95,1.)*pulse*2.6+base*soma)*(1.+uKick*1.5);
  col=mix(vec3(.03,.05,.13),col,exp(-d*.075));
  o=vec4(aces(col*1.3)*uFade,1.);}`;
const GL_VS = `#version 300 es
layout(location=0) in vec4 aG; uniform mat4 uPV; uniform vec3 uEye; uniform float uPx; out float vA;
void main(){ vec3 p=aG.xyz+normalize(uEye-aG.xyz)*min(aG.w*.35,.9); gl_Position=uPV*vec4(p,1.); gl_PointSize=clamp(uPx*aG.w/gl_Position.w,1.,256.); vA=aG.w; }`;
const GL_FS = `#version 300 es
precision mediump float; in float vA; uniform float uFade; out vec4 o;
void main(){ vec2 c=gl_PointCoord-.5; float r=length(c)*2.; float g=exp(-r*r*5.)*.9+exp(-r*r*40.)*.8; o=vec4(vec3(.5,.8,1.)*g*uFade*.7,0.); }`;
function sceneNeurons(gl, mob) {
  const PR = compile(gl, NE_VS, NE_FS), GP = compile(gl, GL_VS, GL_FS), R = rng(77);
  const P = [], Nn = [], A = [], I = [], axons = [];
  const SIDES = 7;
  function tube(pts, rad, arc0, id, kind) {  /* parallel-transport tube */
    const base = P.length / 3; let arc = arc0, prevN = null;
    for (let i = 0; i < pts.length; i++) {
      const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)]; let T = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]; const tl = Math.hypot(...T) || 1; T = T.map(v => v / tl);
      let Nv = prevN || (Math.abs(T[1]) < .9 ? [0, 1, 0] : [1, 0, 0]); const dt = Nv[0] * T[0] + Nv[1] * T[1] + Nv[2] * T[2]; Nv = [Nv[0] - T[0] * dt, Nv[1] - T[1] * dt, Nv[2] - T[2] * dt]; const nl = Math.hypot(...Nv) || 1; Nv = Nv.map(v => v / nl); prevN = Nv;
      const Bv = [T[1] * Nv[2] - T[2] * Nv[1], T[2] * Nv[0] - T[0] * Nv[2], T[0] * Nv[1] - T[1] * Nv[0]];
      if (i) arc += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]);
      const r = rad(i / (pts.length - 1));
      for (let s = 0; s < SIDES; s++) { const an = s / SIDES * TAU, c = Math.cos(an), sn = Math.sin(an), nx = Nv[0] * c + Bv[0] * sn, ny = Nv[1] * c + Bv[1] * sn, nz = Nv[2] * c + Bv[2] * sn;
        P.push(pts[i][0] + nx * r, pts[i][1] + ny * r, pts[i][2] + nz * r); Nn.push(nx, ny, nz); A.push(arc, id, kind) } }
    for (let i = 0; i < pts.length - 1; i++)for (let s = 0; s < SIDES; s++) { const a = base + i * SIDES + s, b = base + i * SIDES + (s + 1) % SIDES; I.push(a, a + SIDES, b, b, a + SIDES, b + SIDES) }
    return arc;
  }
  function walk(p0, dir, len, step, curl) { const pts = [p0.slice()]; let p = p0.slice(), d = dir.slice();
    for (let s = 0; s < len; s += step) { d = [d[0] + (R() - .5) * curl, d[1] + (R() - .5) * curl, d[2] + (R() - .5) * curl]; const l = Math.hypot(...d); d = d.map(v => v / l); p = [p[0] + d[0] * step, p[1] + d[1] * step, p[2] + d[2] * step]; pts.push(p) } return pts }
  function sphere(c, r, id, stretch) {  /* lumpy icosphere-ish soma via lat/long */
    const base = P.length / 3, NA = 16, NB = 12;
    for (let j = 0; j <= NB; j++)for (let i = 0; i <= NA; i++) { const th = j / NB * Math.PI, ph = i / NA * TAU, x = Math.sin(th) * Math.cos(ph), z = Math.sin(th) * Math.sin(ph), y = Math.cos(th);
      const k = 1 + .12 * Math.sin(ph * 3 + id * 9) * Math.sin(th * 2), yy = y > 0 ? y * (1 + stretch * y * y) : y;      /* pyramidal apex */
      P.push(c[0] + x * r * k, c[1] + yy * r * k, c[2] + z * r * k); Nn.push(x, y, z); A.push(0, id, 2) }
    for (let j = 0; j < NB; j++)for (let i = 0; i < NA; i++) { const a = base + j * (NA + 1) + i, b = a + NA + 1; I.push(a, b, a + 1, a + 1, b, b + 1) }
  }
  const NN = mob ? 16 : 26, somas = [];
  for (let k = 0, g = 0; k < NN && g < 5000; g++) { const r = 1.5 + Math.cbrt(R()) * 12, a = R() * TAU, b = Math.acos(2 * R() - 1), p = [r * Math.sin(b) * Math.cos(a), r * Math.cos(b) * .6, r * Math.sin(b) * Math.sin(a)];
    if (Math.abs(Math.hypot(p[0], p[2]) - 7) < 2.6 && Math.abs(p[1] - 1.5) < 3.5) continue;                 /* keep the camera's orbit clear */
    somas.push(p); k++ }
  somas.forEach((c, k) => {
    const id = (k + .5) / NN, sr = .42 + R() * .16;
    sphere(c, sr, id, .9);
    /* apical dendrite + basal dendrites, each branching */
    const grow = (p, d, len, rad, depth, arc0) => { const pts = walk(p, d, len, .35, .5); const end = tube(pts, u => rad * (1 - .75 * u) + .012, arc0, id, 0);
      if (depth < 3) for (let b = 0; b < 2; b++) { const q = pts[Math.floor(pts.length * (.5 + R() * .45))] || pts[pts.length - 1]; grow(q, [d[0] + (R() - .5) * 1.4, d[1] + (R() - .5) * 1.4, d[2] + (R() - .5) * 1.4], len * .6, rad * .55, depth + 1, end * .6) } };
    grow(c, [0, 1, 0], 4 + R() * 2.5, .11, 1, 0);
    const nb = 4 + (R() * 3 | 0); for (let i = 0; i < nb; i++) { const a = i / nb * TAU + R(); grow(c, [Math.cos(a), -.35 - R() * .4, Math.sin(a)], 2.2 + R() * 1.6, .08, 1, 0) }
    /* axon toward another neuron, with terminal boutons */
    const tgt = somas[(k * 7 + 3) % NN], dir = [tgt[0] - c[0], tgt[1] - c[1] - 1, tgt[2] - c[2]], dl = Math.hypot(...dir) || 1;
    const ap = walk(c, [dir[0] / dl, -.6, dir[2] / dl], Math.min(26, dl * 1.3 + 4), .45, .28); tube(ap, u => .055 * (1 - .4 * u) + .02, 0, id, 1); axons.push({ pts: ap, id });
  });
  const mesh = vao(gl, [[0, new Float32Array(P), 3], [1, new Float32Array(Nn), 3], [2, new Float32Array(A), 3]], new Uint32Array(I)), NI = I.length;
  /* arc length table for pulse sprites */
  axons.forEach(ax => { let s = 0; ax.arc = [0]; for (let i = 1; i < ax.pts.length; i++) { s += Math.hypot(ax.pts[i][0] - ax.pts[i - 1][0], ax.pts[i][1] - ax.pts[i - 1][1], ax.pts[i][2] - ax.pts[i - 1][2]); ax.arc.push(s) } });
  const gbuf = gl.createBuffer(), gv = gl.createVertexArray(), garr = new Float32Array((axons.length + somas.length) * 4 + 4 * 400);
  gl.bindVertexArray(gv); gl.bindBuffer(gl.ARRAY_BUFFER, gbuf); gl.bufferData(gl.ARRAY_BUFFER, garr.byteLength, gl.DYNAMIC_DRAW); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0); gl.bindVertexArray(null);
  /* floating dust motes */
  const dust = []; for (let i = 0; i < 400; i++)dust.push([(R() - .5) * 34, (R() - .5) * 18, (R() - .5) * 34, .04 + R() * .08]);
  return {
    bg: [.03, .05, .13],
    draw(t, beat, look, aspect, fade, kick, px) {
      const a = t * .045, e = [Math.sin(a) * 7 + look[0] * 1.5, 1.5 + Math.sin(t * .07) * 2 + look[1] * 1.2, Math.cos(a) * 7], tg = [Math.sin(a + 1.1) * 3, Math.sin(t * .05) * 1.5, Math.cos(a + 1.1) * 3];
      const PV = M4.mul(M4.persp(1.0, aspect, .05, 80), M4.look(e, tg, [0, 1, 0]));
      gl.useProgram(PR.p); gl.uniformMatrix4fv(PR.u.uPV, false, PV); gl.uniform3fv(PR.u.uEye, e); gl.uniform1f(PR.u.uT, t); gl.uniform1f(PR.u.uFade, fade); gl.uniform1f(PR.u.uKick, kick);
      gl.bindVertexArray(mesh.v); gl.drawElements(gl.TRIANGLES, NI, gl.UNSIGNED_INT, 0);
      /* glow sprites at every travelling spike + dust, additive, depth-tested */
      let n = 0; axons.forEach(ax => { const rate = .16 + ax.id * .22, ph = (t * rate + ax.id * 7.3) % 1, head = ph * 34; if (ph > .95) return;
        const L = ax.arc, last = L[L.length - 1]; if (head > last) return; let i = 1; while (i < L.length - 1 && L[i] < head) i++; const k = (head - L[i - 1]) / ((L[i] - L[i - 1]) || 1), p0 = ax.pts[i - 1], p1 = ax.pts[i];
        garr.set([p0[0] + (p1[0] - p0[0]) * k, p0[1] + (p1[1] - p0[1]) * k, p0[2] + (p1[2] - p0[2]) * k, 2.2 * (1 + kick)], n * 4); n++ });
      somas.forEach((c, k) => { const id = (k + .5) / NN, ph = (t * (.16 + id * .22) + id * 7.3) % 1; garr.set([c[0], c[1], c[2], 1.6 + 5 * Math.exp(-ph * 9) + kick * 2], n * 4); n++ });
      for (const d of dust) { garr.set([d[0] + Math.sin(t * .2 + d[3] * 50) * .3, d[1] + Math.sin(t * .13 + d[0]) * .3, d[2], d[3]], n * 4); n++ }
      gl.useProgram(GP.p); gl.uniformMatrix4fv(GP.u.uPV, false, PV); gl.uniform3fv(GP.u.uEye, e); gl.uniform1f(GP.u.uPx, px); gl.uniform1f(GP.u.uFade, fade);
      gl.bindBuffer(gl.ARRAY_BUFFER, gbuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, garr, 0, n * 4);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); gl.depthMask(false); gl.bindVertexArray(gv); gl.drawArrays(gl.POINTS, 0, n); gl.depthMask(true); gl.disable(gl.BLEND);
    }
  };
}

/* ======================= mount ======================= */
function supported() { try { const c = document.createElement('canvas'); return !!c.getContext('webgl2') } catch (e) { return false } }
function mount(cv, opt) {
  opt = Object.assign({ scene: 'blood', maxDpr: 1.5 }, opt || {});
  const gl = cv.getContext('webgl2', { antialias: true, alpha: false, powerPreference: 'high-performance' }); if (!gl) return null;
  const mob = innerWidth < 760; let scenes = {}, cur = null, name = null, fade = 0, kick = 0, scale = 1, ft = [], visible = true, raf = 0, alive = true, look = [0, 0], lookG = [0, 0], t = Math.random() * 30, last = performance.now();
  function use(n) { if (n === name) return; name = n; fade = 0; try { cur = scenes[n] = scenes[n] || (n === 'neurons' ? sceneNeurons(gl, mob) : sceneBlood(gl, mob)) } catch (e) { console.warn('bodygl', e); cur = null } }
  use(opt.scene);
  const io = 'IntersectionObserver' in root ? new IntersectionObserver(es => { visible = es[0].isIntersecting }, { rootMargin: '80px' }) : null; io && io.observe(cv);
  const onMove = e => { const r = cv.getBoundingClientRect(); lookG = [clamp((e.clientX - r.left) / r.width - .5, -.6, .6) * 2, -clamp((e.clientY - r.top) / r.height - .5, -.6, .6) * 2] };
  addEventListener('pointermove', onMove, { passive: true }); cv.addEventListener('pointerdown', () => kick = 1);
  function frame(now) {
    raf = requestAnimationFrame(frame); if (!alive || !visible || opt.paused || document.hidden || !cur) { last = now; return }
    const dt = Math.min(.05, (now - last) / 1000); last = now; if (!RM) t += dt;
    ft.push(dt); if (ft.length >= 30) { const a = ft.reduce((s, v) => s + v) / ft.length; ft = []; if (a > 1 / 45 && scale > .5) scale = Math.max(.5, scale - .1); else if (a < 1 / 58 && scale < 1) scale = Math.min(1, scale + .05) }
    const dpr = Math.min(root.devicePixelRatio || 1, opt.maxDpr) * scale, W = Math.max(2, Math.round(cv.clientWidth * dpr)), H = Math.max(2, Math.round(cv.clientHeight * dpr));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H }
    fade = Math.min(1, fade + dt * 1.2); kick = Math.max(0, kick - dt * 1.6);
    for (let i = 0; i < 2; i++)look[i] += (lookG[i] - look[i]) * Math.min(1, dt * 2);
    const hp = (t * 1.2) % 1, beat = RM ? 0 : Math.exp(-Math.pow(hp / .05, 2)) + .55 * Math.exp(-Math.pow((hp - .27) / .06, 2)) + Math.exp(-Math.pow((hp - 1) / .05, 2));
    gl.viewport(0, 0, W, H); const bg = cur.bg; gl.clearColor(bg[0] * fade, bg[1] * fade, bg[2] * fade, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK); if (name === 'blood') gl.disable(gl.CULL_FACE); else gl.disable(gl.CULL_FACE);
    cur.draw(t, beat + kick * .6, look, W / H, fade * (opt.fade == null ? 1 : opt.fade), kick, H * .5 / Math.tan(.5) * .5);
  }
  raf = requestAnimationFrame(frame);
  return { scene: use, get name() { return name }, set fade(v) { opt.fade = v }, set paused(v) { opt.paused = !!v }, kick() { kick = 1 },
    destroy() { alive = false; cancelAnimationFrame(raf); io && io.disconnect(); removeEventListener('pointermove', onMove); const e = gl.getExtension('WEBGL_lose_context'); e && e.loseContext() } };
}
root.BODY = { mount, supported };
})(window);
