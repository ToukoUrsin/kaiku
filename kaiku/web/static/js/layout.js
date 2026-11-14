// Where each robot stands. Every environment lives in the same model frame (z up, the pelvis at
// the origin, +x forward, +y left); the stage moves environment k to its own station.
//
// The operator's robot is at the origin. The echoes stand behind it in ranks, like a class
// behind its teacher: all facing the same way, each rank a step higher than the one in front
// and half a station across, so every robot's hands can be seen from the front.

export const RANK_PITCH = 2.0;      // metres between ranks (along x)
export const FILE_PITCH = 1.55;     // metres between neighbours in a rank (along y)
export const FIRST_RANK = -2.7;     // x of the first rank
export const TIER_RISE = 0.34;      // each rank stands this much higher than the one before
export const TIER_LIP = 1.05;       // a tier's front edge, ahead of the robots standing on it

/**
 * @param {number} nEnvs robots, the operator's included
 * @returns {{ offsets: Float32Array, rank: Int16Array, file: Float32Array, ranks: number[], depth: number, width: number }}
 *   offsets: x, y, z of each station; rank: -1 for the operator's robot, else 0.. from the front;
 *   file: position across the rank in stations, 0 at the centre; ranks: robots per rank.
 */
export function formation(nEnvs) {
  const echoes = Math.max(0, nEnvs - 1);
  const ranks = rankSizes(echoes);
  const offsets = new Float32Array(nEnvs * 3);
  const rank = new Int16Array(nEnvs).fill(-1);
  const file = new Float32Array(nEnvs);

  let env = 1;
  ranks.forEach((size, r) => {
    for (let i = 0; i < size; i++, env++) {
      const across = i - (size - 1) / 2;
      offsets[env * 3] = FIRST_RANK - r * RANK_PITCH;
      offsets[env * 3 + 1] = across * FILE_PITCH;
      offsets[env * 3 + 2] = (r + 1) * TIER_RISE;
      rank[env] = r;
      file[env] = across;
    }
  });

  const widest = ranks.length ? Math.max(...ranks) : 1;
  return {
    offsets, rank, file, ranks,
    depth: ranks.length ? -FIRST_RANK + (ranks.length - 1) * RANK_PITCH : 0,
    width: widest * FILE_PITCH,
  };
}

// Ranks grow by one robot towards the back, so the formation fills a camera's widening view
// and neighbouring ranks fall half a station apart.
function rankSizes(echoes) {
  if (echoes === 0) return [];
  const count = Math.max(1, Math.round(Math.sqrt(echoes / 2.2)));
  const first = Math.max(1, Math.floor((echoes - (count * (count - 1)) / 2) / count));
  const sizes = Array.from({ length: count }, (_, r) => first + r);
  let left = echoes - sizes.reduce((a, b) => a + b, 0);
  for (let r = count - 1; left > 0; r = r > 0 ? r - 1 : count - 1, left--) sizes[r] += 1;
  return sizes;
}
