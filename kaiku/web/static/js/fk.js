// Forward kinematics over the body tree of /api/scene. Same algorithm as kaiku/web/kinematics.py
// (which the tests hold against MuJoCo), written on typed arrays so a frame allocates nothing.
// Quaternions are (w, x, y, z).

export const JNT_FREE = 0;
export const JNT_BALL = 1;
export const JNT_SLIDE = 2;
export const JNT_HINGE = 3;

export class Kinematics {
  constructor(tree) {
    this.nbody = tree.nbody;
    this.nq = tree.nq;
    this.parent = Int32Array.from(tree.parent);
    this.bodyPos = Float64Array.from(tree.pos);
    this.bodyQuat = Float64Array.from(tree.quat);
    this.jntAdr = Int32Array.from(tree.jntadr);
    this.jntNum = Int32Array.from(tree.jntnum);
    this.jntType = Int32Array.from(tree.jtype);
    this.jntQadr = Int32Array.from(tree.jqadr);
    this.jntPos = Float64Array.from(tree.jpos);
    this.jntAxis = Float64Array.from(tree.jaxis);
    this.qpos0 = Float64Array.from(tree.qpos0);
    this.names = tree.names;

    this.xpos = new Float64Array(this.nbody * 3);
    this.xquat = new Float64Array(this.nbody * 4);
    this.xmat = new Float64Array(this.nbody * 9); // row major
    this.xquat[0] = 1;
    this.xmat[0] = this.xmat[4] = this.xmat[8] = 1;
  }

  /** True for a body that moves on its own free joint (a part on the table, a floating base). */
  isFree(body) {
    return this.jntNum[body] === 1 && this.jntType[this.jntAdr[body]] === JNT_FREE;
  }

  bodyId(name) {
    return this.names.indexOf(name);
  }

  /** Fill xpos, xquat and xmat of every body for the joint positions `q`. */
  solve(q) {
    const { xpos, xquat, parent, bodyPos, bodyQuat, jntAdr, jntNum, jntType, jntQadr, jntPos, jntAxis, qpos0 } = this;

    for (let b = 1; b < this.nbody; b++) {
      const j0 = jntAdr[b];
      const nj = jntNum[b];
      let px, py, pz, qw, qx, qy, qz;

      if (nj === 1 && jntType[j0] === JNT_FREE) {
        const a = jntQadr[j0];
        px = q[a]; py = q[a + 1]; pz = q[a + 2];
        qw = q[a + 3]; qx = q[a + 4]; qy = q[a + 5]; qz = q[a + 6];
      } else {
        const p = parent[b];
        const pw = xquat[p * 4], pxq = xquat[p * 4 + 1], pyq = xquat[p * 4 + 2], pzq = xquat[p * 4 + 3];

        // position: parent position + parent rotation * body offset
        rotate(pw, pxq, pyq, pzq, bodyPos[b * 3], bodyPos[b * 3 + 1], bodyPos[b * 3 + 2]);
        px = xpos[p * 3] + rx; py = xpos[p * 3 + 1] + ry; pz = xpos[p * 3 + 2] + rz;

        // orientation: parent orientation * body orientation
        multiply(pw, pxq, pyq, pzq, bodyQuat[b * 4], bodyQuat[b * 4 + 1], bodyQuat[b * 4 + 2], bodyQuat[b * 4 + 3]);
        qw = mw; qx = mx; qy = my; qz = mz;

        for (let j = j0; j < j0 + nj; j++) {
          const a = jntQadr[j];
          const ax = jntAxis[j * 3], ay = jntAxis[j * 3 + 1], az = jntAxis[j * 3 + 2];
          if (jntType[j] === JNT_SLIDE) {
            const d = q[a] - qpos0[a];
            rotate(qw, qx, qy, qz, ax, ay, az);
            px += rx * d; py += ry * d; pz += rz * d;
            continue;
          }
          let lw, lx, ly, lz;
          if (jntType[j] === JNT_HINGE) {
            const half = 0.5 * (q[a] - qpos0[a]);
            const s = Math.sin(half);
            lw = Math.cos(half); lx = ax * s; ly = ay * s; lz = az * s;
          } else {
            lw = q[a]; lx = q[a + 1]; ly = q[a + 2]; lz = q[a + 3];
          }
          // rotate about the joint anchor, which stays where it is
          const jx = jntPos[j * 3], jy = jntPos[j * 3 + 1], jz = jntPos[j * 3 + 2];
          rotate(qw, qx, qy, qz, jx, jy, jz);
          const anchorX = px + rx, anchorY = py + ry, anchorZ = pz + rz;
          multiply(qw, qx, qy, qz, lw, lx, ly, lz);
          qw = mw; qx = mx; qy = my; qz = mz;
          rotate(qw, qx, qy, qz, jx, jy, jz);
          px = anchorX - rx; py = anchorY - ry; pz = anchorZ - rz;
        }
      }

      const norm = Math.hypot(qw, qx, qy, qz);
      if (norm > 1e-12) {
        qw /= norm; qx /= norm; qy /= norm; qz /= norm;
      } else {
        qw = 1; qx = qy = qz = 0;
      }
      xpos[b * 3] = px; xpos[b * 3 + 1] = py; xpos[b * 3 + 2] = pz;
      xquat[b * 4] = qw; xquat[b * 4 + 1] = qx; xquat[b * 4 + 2] = qy; xquat[b * 4 + 3] = qz;
      quatToMat(qw, qx, qy, qz, this.xmat, b * 9);
    }
  }
}

/** Rotation matrix of a unit quaternion, written row major into `out` at `o`. */
export function quatToMat(w, x, y, z, out, o) {
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  out[o] = 1 - 2 * (yy + zz); out[o + 1] = 2 * (xy - wz); out[o + 2] = 2 * (xz + wy);
  out[o + 3] = 2 * (xy + wz); out[o + 4] = 1 - 2 * (xx + zz); out[o + 5] = 2 * (yz - wx);
  out[o + 6] = 2 * (xz - wy); out[o + 7] = 2 * (yz + wx); out[o + 8] = 1 - 2 * (xx + yy);
}

// Results of the two helpers below. Module-level scalars keep the solver free of allocation.
let rx = 0, ry = 0, rz = 0;
let mw = 1, mx = 0, my = 0, mz = 0;

function rotate(w, x, y, z, vx, vy, vz) {
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  rx = vx + w * tx + (y * tz - z * ty);
  ry = vy + w * ty + (z * tx - x * tz);
  rz = vz + w * tz + (x * ty - y * tx);
}

function multiply(aw, ax, ay, az, bw, bx, by, bz) {
  mw = aw * bw - ax * bx - ay * by - az * bz;
  mx = aw * bx + ax * bw + ay * bz - az * by;
  my = aw * by - ax * bz + ay * bw + az * bx;
  mz = aw * bz + ax * by - ay * bx + az * bw;
}
