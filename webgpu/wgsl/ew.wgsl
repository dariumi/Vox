// Поэлементные операции (точки входа):
//   concat_in : out[r] = [x[seq(r)] (mel) | cond[r] (mel) | temb[r] (td)]       X0=x(N*mel), X1=cond(R*mel), X2=temb(R*td), Y=out
//   euler     : y += dt * (pc + (pc - pu) * cfg)                                  X0=pred(2*N*mel), Y=y
//   euler1    : y += dt * pc  (без CFG)                                        X0=pred(N*mel)
//   zero_rows : Y[from*C .. to*C) = 0
//   silu_vec  : Y[i] = silu(X0[i])
//   copy      : Y[y_off + i] = X0[x_off + i]
//   gate      : Y[i] = X0[i] * X1[i]
struct P { n: u32, a: u32, b: u32, c: u32, d: u32, x_off: u32, y_off: u32, pad1: u32, f0: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X0: array<f32>;
@group(0) @binding(2) var<storage, read> X1: array<f32>;
@group(0) @binding(3) var<storage, read> X2: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y2: array<f32>;

fn gid1(g: vec3<u32>) -> u32 { return g.x + g.y * 65535u * 64u; }

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
