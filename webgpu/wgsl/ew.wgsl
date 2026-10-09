// Поэлементные операции (точки входа):
//   concat_in : out[r] = [x[seq(r)] (mel) | cond[r] (mel) | temb[r] (td)]       X0=x(N*mel), X1=cond(R*mel), X2=temb(R*td), Y=out
//   euler     : y += dt * (pc + (pc - pu) * cfg)                                  X0=pred(2*N*mel), Y=y
//   euler1    : y += dt * pc  (без CFG)                                        X0=pred(N*mel)
//   zero_rows : Y[from*C .. to*C) = 0
//   silu_vec  : Y[i] = silu(X0[i])
//   copy      : Y[y_off + i] = X0[x_off + i]
//   gate      : Y[i] = X0[i] * X1[i]
//   swiglu    : Y[i] = silu(X0[i]) * X1[i]
//   add       : Y[y_off + i] += X0[x_off + i]
//   snake     : Y[y_off + i] = x + X2[c] * sin(x * X1[c])^2, x = X0[x_off + i], c = i % a   (X1 = exp(alpha), X2 = 1/(exp(beta)+1e-9))
//   clampcopy : Y[y_off + i] = clamp(X0[x_off + i], -1, 1)
struct P { n: u32, a: u32, b: u32, c: u32, d: u32, x_off: u32, y_off: u32, pad1: u32, f0: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X0: array<f32>;
@group(0) @binding(2) var<storage, read> X1: array<f32>;
@group(0) @binding(3) var<storage, read> X2: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y2: array<f32>;

fn gid1(g: vec3<u32>) -> u32 { return g.x + g.y * 65535u * 64u; }

// sin с настоящей редукцией аргумента и полиномом (cephes sinf): встроенный sin() в WGSL обязан быть точным лишь до 2^-11,
// на практике (SwiftShader, Metal) ошибка доходит до 1e-3 — для Snake-активаций декодера этого слишком много.
fn sin_acc(x: f32) -> f32 {
  let k = round(x * 0.6366197723675814);
  let r = ((x - k * 1.5703125) - k * 4.837512969970703e-4) - k * 7.549789948768648e-8;
  let r2 = r * r;
  let sn = r + r * r2 * (-1.6666654611e-1 + r2 * (8.3321608736e-3 + r2 * -1.9515295891e-4));
  let cs = 1.0 - 0.5 * r2 + r2 * r2 * (4.166664568298827e-2 + r2 * (-1.388731625493765e-3 + r2 * 2.443315711809948e-5));
  let q = u32(i32(k)) & 3u;
  if (q == 0u) { return sn; }
  if (q == 1u) { return cs; }
  if (q == 2u) { return -sn; }
  return -cs;
}

// a = rows (R), b = N, c = mel, d = td; n = R * (2*mel + td)
@compute @workgroup_size(64)
fn concat_in(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  let mel = p.c;
  let td = p.d;
  let wdt = 2u * mel + td;
  let r = i / wdt;
  let c = i % wdt;
  let seq = r / p.b;
  let pos = r % p.b;
  var v = 0.0;
  if (c < mel) { v = X0[pos * mel + c]; }
  else if (c < 2u * mel) { v = X1[r * mel + (c - mel)]; }
  else { v = X2[r * td + (c - 2u * mel)]; }
  Y[i] = v;
}

// n = N*mel, f0 = dt, f1 = cfg
@compute @workgroup_size(64)
fn euler(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  let pc = X0[i];
  let pu = X0[p.n + i];
  let yn = Y[i] + p.f0 * (pc + (pc - pu) * p.f1);
  Y[i] = yn;
}

@compute @workgroup_size(64)
fn euler1(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  let yn = Y[i] + p.f0 * X0[i];
  Y[i] = yn;
}

// n = (to - from) * C элементов, смещение from * C в y_off
@compute @workgroup_size(64)
fn zero_rows(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  Y[p.y_off + i] = 0.0;
}

@compute @workgroup_size(64)
fn silu_vec(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  let v = X0[p.x_off + i];
  Y[p.y_off + i] = v / (1.0 + exp(-v));
}

@compute @workgroup_size(64)
fn copy(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  Y[p.y_off + i] = X0[p.x_off + i];
}

@compute @workgroup_size(64)
fn swiglu(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  let v = X0[p.x_off + i];
  Y[p.y_off + i] = (v / (1.0 + exp(-v))) * X1[i];
}

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  Y[p.y_off + i] = Y[p.y_off + i] + X0[p.x_off + i];
}

@compute @workgroup_size(64)
fn snake(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  let c = i % p.a;
  let v = X0[p.x_off + i];
  let s = sin_acc(v * X1[c]);
  Y[p.y_off + i] = v + X2[c] * s * s;
}

@compute @workgroup_size(64)
fn clampcopy(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  Y[p.y_off + i] = clamp(X0[p.x_off + i], -1.0, 1.0);
}
