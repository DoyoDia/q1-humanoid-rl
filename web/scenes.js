// Scenes for the interactive browser sim2sim (sim2sim_view.html).
//
// Each scene adds bodies to the robot's own physics model (q1_web.xml): static geoms in the
// world body, free bodies for things that can be knocked over, three balls for throwing. They
// are appended AFTER the robot, so the robot keeps qpos 0..28, qvel 0..27 and ctrl 0..21 in
// every scene. The floor plane stays; only its friction changes.
//
// The physics only knows boxes, cylinders and spheres. How each one is drawn (colour, texture,
// roughness) is kept beside it in `look`, keyed by geom name, for the renderer.
//
// Policies are blind: none of this is in the observation. The scenes are there to push the
// robot into furniture, dance it near a stage edge, stand it on a rug, gravel or ice.

const f = (x) => +x.toFixed(4);
const v3 = (a) => a.map(f).join(' ');

class Builder {
  constructor() { this.xml = []; this.look = {}; this.n = 0; this.decor = []; }
  _name(kind) { return `${kind}_${this.n++}`; }
  // A static geom in the world body. size: half extents (box), [r, half height] (cylinder), [r] (sphere).
  geom(type, pos, size, look, opt = {}) {
    const name = this._name(opt.tag || type);
    const extra = [
      opt.euler ? `euler="${v3(opt.euler)}"` : '',
      opt.friction ? `friction="${opt.friction} 0.005 0.0001"` : '',
      opt.ghost ? 'contype="0" conaffinity="0"' : '',
    ].join(' ');
    this.xml.push(`<geom name="${name}" type="${type}" pos="${v3(pos)}" size="${v3(size)}" ${extra} rgba="${look.rgba || '0.7 0.7 0.7 1'}"/>`);
    this.look[name] = look;
    return name;
  }
  box(pos, half, look, opt) { return this.geom('box', pos, half, look, opt); }
  cyl(pos, r, hh, look, opt) { return this.geom('cylinder', pos, [r, hh], look, opt); }
  // A free body made of boxes/cylinders/spheres given relative to pos. mass: total, kg.
  body(label, pos, parts, mass, opt = {}) {
    const name = this._name('obj');
    const per = mass / parts.length;
    const geoms = parts.map(([type, p, size, look], i) => {
      const gname = `${name}_g${i}`;
      this.look[gname] = look;
      return `<geom name="${gname}" type="${type}" pos="${v3(p)}" size="${v3(size)}" mass="${f(per)}"` +
             ` friction="${opt.friction || 0.7} 0.005 0.0001" rgba="${look.rgba || '0.7 0.7 0.7 1'}"/>`;
    });
    this.xml.push(`<body name="${name}" pos="${v3(pos)}"${opt.yaw ? ` euler="0 0 ${f(opt.yaw)}"` : ''}>` +
                  `<freejoint/>${geoms.join('')}</body>`);
    this.look[name] = { label };
    return name;
  }
  balls() {                                      // the throwing pool, parked far away
    for (let i = 0; i < 3; i++) {
      this.xml.push(`<body name="ball_${i}" pos="${60 + 2 * i} 60 0.11"><freejoint/>` +
                    `<geom name="ball_${i}_g0" type="sphere" size="0.11" mass="0.43" friction="0.8 0.005 0.0001" ` +
                    `solref="0.01 0.6" rgba="1 0.55 0.1 1"/></body>`);
      this.look[`ball_${i}_g0`] = { rgba: '1 0.55 0.1 1', tex: 'ball' };
    }
  }
}

// colours as rgba strings; tex names are drawn by textures() below
const WOOD = { rgba: '0.62 0.45 0.30 1', tex: 'planks', rough: 0.7 };
const DARKWOOD = { rgba: '0.33 0.22 0.15 1', rough: 0.6 };
const WHITE = { rgba: '0.93 0.93 0.91 1', rough: 0.8 };
const WALL = { rgba: '0.90 0.88 0.84 1', rough: 0.95, wall: true };
const FABRIC = { rgba: '0.36 0.45 0.55 1', tex: 'fabric', rough: 1.0 };
const CUSHION = { rgba: '0.85 0.62 0.35 1', tex: 'fabric', rough: 1.0 };
const BLACK = { rgba: '0.07 0.07 0.08 1', rough: 0.3, metal: 0.2 };
const METAL = { rgba: '0.55 0.57 0.6 1', rough: 0.35, metal: 0.8 };
const CARDBOARD = { rgba: '0.72 0.55 0.36 1', tex: 'cardboard', rough: 0.95 };
const PLANT = { rgba: '0.25 0.5 0.28 1', rough: 0.9 };
const POT = { rgba: '0.75 0.42 0.3 1', rough: 0.9 };
const RED_SEAT = { rgba: '0.55 0.1 0.12 1', tex: 'fabric', rough: 1.0 };

function room() {
  const b = new Builder();
  // walls: back (y = +2.6) and left (x = -3.1); the renderer fades one when the camera is behind it
  b.box([0, 2.65, 1.2], [3.2, 0.05, 1.2], { ...WALL, normal: [0, -1, 0] }, { tag: 'wall' });
  b.box([-3.15, 0, 1.2], [0.05, 2.7, 1.2], { ...WALL, normal: [1, 0, 0] }, { tag: 'wall' });
  b.box([0, 2.59, 0.05], [3.2, 0.01, 0.05], WHITE);                     // skirting
  b.box([-3.09, 0, 0.05], [0.01, 2.7, 0.05], WHITE);
  // rug: 8 mm thick, grippier than the floor -- its edge is a small step
  b.box([0.2, 0.1, 0.004], [1.3, 0.9, 0.004], { rgba: '0.62 0.25 0.22 1', tex: 'rug', rough: 1 }, { friction: 0.95, tag: 'rug' });
  // TV wall
  b.box([0, 2.35, 0.25], [1.1, 0.24, 0.25], DARKWOOD);
  b.box([0, 2.5, 1.05], [0.75, 0.03, 0.43], BLACK);
  b.box([0, 2.468, 1.05], [0.72, 0.002, 0.40], { rgba: '0.1 0.16 0.24 1', tex: 'tv', rough: 0.2 }, { ghost: true });
  // sofa facing the TV, behind the robot as seen from the default camera
  b.box([-0.4, -2.05, 0.21], [1.05, 0.45, 0.21], FABRIC);
  b.box([-0.4, -2.42, 0.5], [1.05, 0.1, 0.4], FABRIC);
  b.box([-1.55, -2.05, 0.31], [0.11, 0.47, 0.31], FABRIC);
  b.box([0.75, -2.05, 0.31], [0.11, 0.47, 0.31], FABRIC);
  // coffee table between the rug and the sofa
  b.box([-0.4, -1.05, 0.40], [0.55, 0.3, 0.02], WOOD);
  for (const [x, y] of [[-0.9, -1.3], [0.1, -1.3], [-0.9, -0.8], [0.1, -0.8]]) b.cyl([x, y, 0.19], 0.025, 0.19, DARKWOOD);
  // bookshelf on the left wall
  b.box([-2.95, 1.0, 0.9], [0.15, 0.55, 0.9], DARKWOOD);
  for (let k = 0; k < 4; k++) {
    const z = 0.32 + 0.42 * k;
    for (let j = 0; j < 5; j++) {
      const hgt = 0.12 + 0.05 * ((j * 7 + k * 3) % 4);
      b.box([-2.86, 0.62 + 0.18 * j, z + hgt / 2], [0.05, 0.06, hgt / 2],
            { rgba: ['0.6 0.2 0.2 1', '0.2 0.35 0.6 1', '0.85 0.75 0.4 1', '0.3 0.5 0.35 1'][(j + k) % 4], rough: 0.8 }, { ghost: true });
    }
  }
  // floor lamp and a plant
  b.cyl([-2.6, -2.1, 0.015], 0.15, 0.015, METAL);
  b.cyl([-2.6, -2.1, 0.75], 0.012, 0.75, METAL);
  b.cyl([-2.6, -2.1, 1.55], 0.2, 0.13, { rgba: '0.98 0.94 0.85 1', rough: 1, glow: true });
  b.cyl([2.6, 2.2, 0.18], 0.18, 0.18, POT);
  b.geom('sphere', [2.6, 2.2, 0.6], [0.32], PLANT);
  // things that move: a dining chair, two cardboard boxes, a ball, a cushion on the sofa
  b.body('椅子', [1.7, 0.9, 0], [
    ['box', [0, 0, 0.45], [0.21, 0.21, 0.02], WOOD],
    ['box', [-0.19, 0, 0.75], [0.02, 0.2, 0.28], WOOD],
    ...[[0.18, 0.18], [0.18, -0.18], [-0.18, 0.18], [-0.18, -0.18]].map(([x, y]) => ['box', [x, y, 0.215], [0.018, 0.018, 0.215], DARKWOOD]),
  ], 4.0, { yaw: 2.6 });
  b.body('纸箱', [1.2, -0.7, 0.18], [['box', [0, 0, 0], [0.2, 0.16, 0.18], CARDBOARD]], 1.2, { yaw: 0.3 });
  b.body('小纸箱', [1.25, -0.72, 0.47], [['box', [0, 0, 0], [0.13, 0.11, 0.11], CARDBOARD]], 0.5, { yaw: 0.5 });
  b.body('足球', [0.9, 1.3, 0.11], [['sphere', [0, 0, 0], [0.11], { rgba: '0.95 0.95 0.95 1', tex: 'football' }]], 0.43);
  b.body('抱枕', [-1.0, -2.0, 0.5], [['box', [0, 0, 0], [0.2, 0.2, 0.07], CUSHION]], 0.4, { yaw: 0.2 });
  b.balls();
  return {
    id: 'room', label: '家居客厅', builder: b,
    floor: { tex: 'wood_floor', friction: 0.5 },
    sky: 0xe9e4dc,
    spawns: [
      { label: '地毯中央', x: 0.2, y: 0.1, z: 0.008, yaw: -Math.PI / 2 },
      { label: '木地板（滑一点）', x: 1.6, y: -0.6, z: 0, yaw: Math.PI },
      { label: '茶几旁', x: -0.4, y: -0.45, z: 0.008, yaw: -Math.PI / 2 },
    ],
    camera: [2.0, -2.9, 1.35],
  };
}

function stage() {
  const b = new Builder();
  const H = 0.45;
  // the stage: 6 m wide, 3 m deep, 45 cm up, its front edge at y = 0
  b.box([0, 1.5, H / 2], [3.2, 1.5, H / 2], { rgba: '0.55 0.38 0.24 1', tex: 'stage', rough: 0.55 }, { friction: 0.7, tag: 'stage' });
  b.box([0, -0.004, H / 2], [3.2, 0.004, H / 2 - 0.01], { rgba: '0.15 0.12 0.12 1', rough: 0.9 }, { ghost: true });
  b.box([0, 0.06, H + 0.001], [3.2, 0.03, 0.001], { rgba: '0.95 0.8 0.2 1', rough: 0.6 }, { ghost: true });   // edge marking
  // steps up on the right
  b.box([2.7, -0.45, 0.075], [0.45, 0.15, 0.075], { rgba: '0.45 0.32 0.22 1', rough: 0.6 });
  b.box([2.7, -0.15, 0.15], [0.45, 0.15, 0.15], { rgba: '0.48 0.34 0.23 1', rough: 0.6 });
  b.box([2.7, -0.15, 0.30], [0.45, 0.15, 0.001], { rgba: '0.48 0.34 0.23 1' }, { ghost: true });
  // backdrop with a screen, lectern, a mic stand that can be knocked over
  b.box([0, 3.05, 1.9], [3.3, 0.05, 1.9], { rgba: '0.10 0.12 0.2 1', rough: 0.95 });
  b.box([0, 2.995, 2.05], [1.9, 0.004, 0.95], { rgba: '0.92 0.94 0.98 1', tex: 'screen', rough: 0.4, glow: true }, { ghost: true });
  b.box([-2.1, 1.3, H + 0.5], [0.32, 0.26, 0.5], { rgba: '0.42 0.26 0.16 1', tex: 'planks', rough: 0.6 });
  b.box([-2.1, 1.22, H + 1.02], [0.36, 0.32, 0.02], DARKWOOD, { euler: [0.25, 0, 0] });
  b.box([-2.1, 1.0, H + 0.62], [0.2, 0.004, 0.12], { rgba: '0.8 0.15 0.15 1', tex: 'badge', rough: 0.5 }, { ghost: true });
  b.body('话筒架', [1.0, 0.7, H], [
    ['cylinder', [0, 0, 0.012], [0.13, 0.012], BLACK],
    ['cylinder', [0, 0, 0.55], [0.012, 0.55], METAL],
    ['cylinder', [0, 0, 1.12], [0.02, 0.05], BLACK],
  ], 2.5);
  // audience: three rows of seats on the floor in front
  for (let r = 0; r < 3; r++) {
    for (let k = 0; k < 8; k++) {
      const x = -2.45 + 0.7 * k, y = -1.9 - 0.95 * r;
      b.box([x, y, 0.22], [0.26, 0.24, 0.05], RED_SEAT);
      b.box([x, y - 0.25, 0.5], [0.26, 0.04, 0.3], RED_SEAT);
      b.box([x - 0.29, y - 0.05, 0.28], [0.03, 0.27, 0.28], BLACK);
    }
  }
  b.balls();
  return {
    id: 'stage', label: '礼堂讲台', builder: b,
    floor: { tex: 'carpet', friction: 0.8 },
    sky: 0x1b1e26, dark: true,
    spawns: [
      { label: '舞台中央', x: 0, y: 1.3, z: H, yaw: -Math.PI / 2 },
      { label: '舞台前沿（离边 40 cm）', x: 0, y: 0.45, z: H, yaw: -Math.PI / 2 },
      { label: '台下过道', x: 0, y: -1.0, z: 0, yaw: Math.PI / 2 },
    ],
    camera: [1.2, -4.4, 1.7],
  };
}

function terrain() {
  const b = new Builder();
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  // gravel patch: 10 x 10 tiles, 0.5 to 3 cm proud
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 10; j++) {
      const h = 0.005 + 0.025 * rnd();
      b.box([1.6 + 0.2 * i, -1.0 + 0.2 * j, h / 2], [0.099, 0.099, h / 2],
            { rgba: `${f(0.45 + 0.2 * rnd())} ${f(0.43 + 0.15 * rnd())} 0.38 1`, rough: 0.95 }, { friction: 0.9 });
    }
  }
  // an 8-degree ramp up to a 25 cm plateau: a thick slab, its underside sunk below the floor
  const a = 8 * Math.PI / 180, L = 0.25 / Math.sin(a);
  b.box([-1.5, 2.4 - (L * Math.cos(a)) / 2 - 1.0, 0.125 - 0.125 / Math.cos(a)], [1.0, L / 2, 0.125], { rgba: '0.5 0.52 0.55 1', tex: 'concrete', rough: 0.9 }, { euler: [a, 0, 0] });
  b.box([-1.5, 2.4, 0.115], [1.0, 1.0, 0.135], { rgba: '0.55 0.57 0.6 1', tex: 'concrete', rough: 0.9 });
  // a 5 cm kerb
  b.box([1.5, 2.3, 0.025], [0.9, 0.9, 0.025], { rgba: '0.62 0.62 0.6 1', tex: 'concrete', rough: 0.9 });
  // ice: friction 0.15 (training randomised 0.4 to 1.2 -- this is outside it on purpose)
  b.box([-1.6, -1.6, 0.001], [0.9, 0.9, 0.001], { rgba: '0.72 0.86 0.95 1', rough: 0.1, metal: 0.1 }, { friction: 0.15 });
  // traffic cones to knock over
  for (const [x, y] of [[0.6, 1.2], [0.9, 1.5], [-0.3, -0.9]]) {
    b.body('雪糕筒', [x, y, 0], [
      ['box', [0, 0, 0.01], [0.13, 0.13, 0.01], BLACK],
      ['cylinder', [0, 0, 0.12], [0.08, 0.11], { rgba: '1 0.42 0.05 1', rough: 0.6 }],
      ['cylinder', [0, 0, 0.29], [0.045, 0.07], { rgba: '0.95 0.95 0.95 1', rough: 0.6 }],
    ], 0.8);
  }
  b.balls();
  return {
    id: 'terrain', label: '测试场地', builder: b,
    floor: { tex: 'asphalt', friction: 0.8 },
    sky: 0xcfd8e3,
    spawns: [
      { label: '平地', x: 0, y: 0, z: 0, yaw: 0 },
      { label: '碎石（0.5～3 cm 不平）', x: 2.5, y: -0.1, z: 0.03, yaw: 0 },
      { label: '冰面（摩擦 0.15）', x: -1.6, y: -1.6, z: 0.002, yaw: 0 },
      { label: '斜坡顶（25 cm 高台）', x: -1.5, y: 2.7, z: 0.25, yaw: -Math.PI / 2 },
      { label: '路沿上（5 cm）', x: 1.5, y: 2.0, z: 0.05, yaw: -Math.PI / 2 },
    ],
    camera: [2.8, -1.6, 1.3],
  };
}

function open() {
  const b = new Builder();
  b.balls();
  return {
    id: 'open', label: '空地', builder: b,
    floor: { tex: 'tiles', friction: 0.6 },
    sky: 0xdfe3e8,
    spawns: [{ label: '原点', x: 0, y: 0, z: 0, yaw: 0 }],
    camera: [3.2, -0.9, 1.3],
  };
}

export const SCENES = [open(), room(), stage(), terrain()];

/** The robot's model with the scene's bodies added and the floor's friction set. */
export function sceneXml(baseXml, scene) {
  let xml = baseXml.replace(/(<geom name="floor"[^>]*?)friction="[^"]*"/, `$1friction="${scene.floor.friction}"`);
  const i = xml.lastIndexOf('</worldbody>');
  return xml.slice(0, i) + scene.builder.xml.join('\n') + '\n' + xml.slice(i);
}

// -- textures, drawn on canvases -------------------------------------------------------------

function canvas(w, h, draw) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  return c;
}
function noise(g, w, h, n, alpha, light = 255) {
  for (let i = 0; i < n; i++) {
    const v = Math.random() * light | 0;
    g.fillStyle = `rgba(${v},${v},${v},${alpha * Math.random()})`;
    g.fillRect(Math.random() * w, Math.random() * h, 1 + Math.random() * 2, 1 + Math.random() * 2);
  }
}

/** name -> {canvas, metres, tint?, stretch?}: metres is how much surface one copy covers; tint textures are
 *  greyscale and take the object's colour, the others carry their own; stretch ones cover a face once. */
export function textures() {
  const T = {};
  T.tiles = { metres: 1, canvas: canvas(256, 256, (g, w, h) => {
    g.fillStyle = '#16191e'; g.fillRect(0, 0, w, h);
    noise(g, w, h, 1500, 0.08);
    g.strokeStyle = '#b8bec8'; g.lineWidth = 3; g.strokeRect(0, 0, w, h);
  }) };
  T.wood_floor = { metres: 1.6, canvas: canvas(512, 512, (g, w, h) => {
    const rows = 8;
    for (let r = 0; r < rows; r++) {
      let x = -(r * 97 % 300);
      while (x < w) {
        const len = 220 + (r * 53 + x) % 140;
        const t = 0.85 + 0.25 * Math.random();
        g.fillStyle = `rgb(${158 * t | 0},${112 * t | 0},${72 * t | 0})`;
        g.fillRect(x, r * h / rows, len, h / rows);
        for (let k = 0; k < 6; k++) {
          g.strokeStyle = `rgba(80,45,20,${0.08 + 0.1 * Math.random()})`;
          g.beginPath(); const yy = r * h / rows + Math.random() * h / rows;
          g.moveTo(x, yy); g.bezierCurveTo(x + len / 3, yy + 3, x + 2 * len / 3, yy - 3, x + len, yy); g.stroke();
        }
        g.fillStyle = 'rgba(40,25,10,0.6)'; g.fillRect(x, r * h / rows, 2, h / rows);
        x += len;
      }
      g.fillStyle = 'rgba(40,25,10,0.55)'; g.fillRect(0, r * h / rows, w, 2);
    }
  }) };
  T.planks = { metres: 0.8, canvas: T.wood_floor.canvas };
  T.stage = { metres: 2.0, canvas: canvas(512, 512, (g, w, h) => {
    for (let r = 0; r < 16; r++) {
      const t = 0.8 + 0.3 * Math.random();
      g.fillStyle = `rgb(${120 * t | 0},${80 * t | 0},${50 * t | 0})`;
      g.fillRect(0, r * h / 16, w, h / 16);
      g.fillStyle = 'rgba(20,10,5,0.5)'; g.fillRect(0, r * h / 16, w, 1.5);
    }
    noise(g, w, h, 3000, 0.06, 80);
  }) };
  T.carpet = { metres: 1.0, canvas: canvas(256, 256, (g, w, h) => {
    g.fillStyle = '#4a1d22'; g.fillRect(0, 0, w, h);
    noise(g, w, h, 6000, 0.15, 120);
    g.strokeStyle = 'rgba(200,160,80,0.25)'; g.lineWidth = 2;
    for (let k = 0; k < 4; k++) { g.beginPath(); g.arc(w / 2, h / 2, 30 + 25 * k, 0, 7); g.stroke(); }
  }) };
  T.rug = { metres: 2.6, canvas: canvas(512, 512, (g, w, h) => {
    g.fillStyle = '#9b3b33'; g.fillRect(0, 0, w, h);
    g.strokeStyle = '#e8d3a8'; g.lineWidth = 14; g.strokeRect(24, 24, w - 48, h - 48);
    g.strokeStyle = '#2f3d5c'; g.lineWidth = 8; g.strokeRect(52, 52, w - 104, h - 104);
    g.fillStyle = '#e8d3a8';
    for (let k = 0; k < 5; k++) {
      g.save(); g.translate(w / 2, h / 2); g.rotate(Math.PI / 4);
      const s = 40 + 30 * k; g.strokeStyle = k % 2 ? '#2f3d5c' : '#e8d3a8'; g.lineWidth = 6; g.strokeRect(-s / 2, -s / 2, s, s); g.restore();
    }
    noise(g, w, h, 9000, 0.12, 60);
  }) };
  T.fabric = { tint: true, metres: 0.5, canvas: canvas(128, 128, (g, w, h) => {
    g.fillStyle = '#bbb'; g.fillRect(0, 0, w, h);
    for (let y = 0; y < h; y += 2) { g.fillStyle = `rgba(0,0,0,${0.05 + 0.05 * (y % 4 === 0)})`; g.fillRect(0, y, w, 1); }
    noise(g, w, h, 1500, 0.15);
  }) };
  T.cardboard = { tint: true, metres: 0.4, canvas: canvas(128, 128, (g, w, h) => {
    g.fillStyle = '#ddd'; g.fillRect(0, 0, w, h);
    noise(g, w, h, 800, 0.2, 150);
    g.fillStyle = 'rgba(160,130,90,0.5)'; g.fillRect(0, h / 2 - 8, w, 16);        // tape
  }) };
  T.concrete = { tint: true, metres: 1.0, canvas: canvas(256, 256, (g, w, h) => {
    g.fillStyle = '#ccc'; g.fillRect(0, 0, w, h); noise(g, w, h, 8000, 0.15);
  }) };
  T.asphalt = { metres: 1.5, canvas: canvas(256, 256, (g, w, h) => {
    g.fillStyle = '#5b5f63'; g.fillRect(0, 0, w, h); noise(g, w, h, 9000, 0.25);
    g.fillStyle = 'rgba(240,240,240,0.7)'; g.fillRect(0, h / 2 - 2, w, 4);
  }) };
  T.tv = { metres: 1, stretch: true, canvas: canvas(512, 288, (g, w, h) => {
    const gr = g.createLinearGradient(0, 0, w, h); gr.addColorStop(0, '#16324f'); gr.addColorStop(1, '#0b1522');
    g.fillStyle = gr; g.fillRect(0, 0, w, h);
    g.fillStyle = '#9fc6ff'; g.font = 'bold 44px sans-serif'; g.textAlign = 'center'; g.fillText('Q1 sim2sim', w / 2, h / 2 + 15);
  }) };
  T.screen = { metres: 1, stretch: true, canvas: canvas(1024, 512, (g, w, h) => {
    g.fillStyle = '#f2f5fb'; g.fillRect(0, 0, w, h);
    g.fillStyle = '#1d3b72'; g.fillRect(0, 0, w, 90);
    g.fillStyle = '#fff'; g.font = 'bold 56px "PingFang SC","Noto Sans CJK SC",sans-serif'; g.textAlign = 'center';
    g.fillText('启元 Q1 人形机器人', w / 2, 66);
    g.fillStyle = '#1d3b72'; g.font = 'bold 120px "PingFang SC","Noto Sans CJK SC",sans-serif';
    g.fillText('第九套广播体操', w / 2, 300);
    g.font = '44px sans-serif'; g.fillStyle = '#556'; g.fillText('sim2sim · MuJoCo WASM', w / 2, 420);
  }) };
  T.badge = { metres: 1, stretch: true, canvas: canvas(256, 128, (g, w, h) => {
    g.fillStyle = '#b3201f'; g.fillRect(0, 0, w, h);
    g.fillStyle = '#ffd86b'; g.font = 'bold 72px sans-serif'; g.textAlign = 'center'; g.fillText('Q1', w / 2, 92);
  }) };
  T.ball = { metres: 0.35, stretch: true, canvas: canvas(128, 64, (g, w, h) => {
    g.fillStyle = '#ff8c1a'; g.fillRect(0, 0, w, h);
    g.strokeStyle = '#3a1d00'; g.lineWidth = 3;
    for (const x of [0, w / 4, w / 2, 3 * w / 4]) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); }
    g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
  }) };
  T.football = { metres: 0.35, stretch: true, canvas: canvas(256, 128, (g, w, h) => {
    g.fillStyle = '#f4f4f4'; g.fillRect(0, 0, w, h);
    g.fillStyle = '#111';
    for (let i = 0; i < 12; i++) {
      const x = (i % 6) * w / 6 + (i < 6 ? 0 : w / 12), y = i < 6 ? h * 0.3 : h * 0.72;
      g.beginPath();
      for (let k = 0; k < 5; k++) { const t = k / 5 * 2 * Math.PI - Math.PI / 2; g.lineTo(x + 14 * Math.cos(t), y + 14 * Math.sin(t)); }
      g.fill();
    }
  }) };
  return T;
}
