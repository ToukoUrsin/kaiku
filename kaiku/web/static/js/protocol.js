// Decoder of the live state stream. The format is defined in kaiku/web/protocol.py;
// tests/test_web_protocol.py holds the two against each other.

export const MAGIC = 0x4b;
export const KIND_KEY = 1;
export const KIND_DELTA = 2;
const FLAG_STATE = 1;
const HEADER_SIZE = 20;

function readHeader(bytes) {
  if (bytes.length < HEADER_SIZE || bytes[0] !== MAGIC) throw new Error('not a kaiku frame');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return {
    kind: bytes[1],
    flags: bytes[2],
    seq: view.getUint32(4, true),
    baseSeq: view.getUint32(8, true),
    tMs: view.getUint32(12, true),
    nEnvs: view.getUint16(16, true),
    nq: view.getUint16(18, true),
  };
}

/**
 * Apply one frame to the tables `q16` (Int16Array, nEnvs * nq) and `state` (Uint8Array, nEnvs)
 * in place. A key frame overwrites `q16`; a delta frame changes the coordinates it names, so
 * `q16` must hold the frame the delta was made against. Returns the frame header.
 */
export function applyFrame(buffer, q16, state) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const head = readHeader(bytes);
  const { nEnvs, nq } = head;
  if (q16.length !== nEnvs * nq) throw new Error('frame size does not match the scene');
  let o = HEADER_SIZE;

  if (head.flags & FLAG_STATE) {
    state.set(bytes.subarray(o, o + nEnvs));
    o += nEnvs;
  }

  if (head.kind === KIND_KEY) {
    o += o & 1;
    const view = new DataView(bytes.buffer, bytes.byteOffset + o, 2 * nEnvs * nq);
    for (let i = 0; i < q16.length; i++) q16[i] = view.getInt16(2 * i, true);
    return head;
  }
  if (head.kind !== KIND_DELTA) throw new Error(`unknown frame kind ${head.kind}`);

  // Row mask (which environments changed), then one column mask per changed environment,
  // then one zigzag varint per set bit, row by row.
  const rowMask = o;
  const colBytes = (nq + 7) >> 3;
  let colMask = rowMask + ((nEnvs + 7) >> 3);
  let changed = 0;
  for (let e = 0; e < nEnvs; e++) changed += (bytes[rowMask + (e >> 3)] >> (e & 7)) & 1;
  o = colMask + changed * colBytes;

  for (let e = 0; e < nEnvs; e++) {
    if (!((bytes[rowMask + (e >> 3)] >> (e & 7)) & 1)) continue;
    const row = e * nq;
    for (let j = 0; j < nq; j++) {
      if (!((bytes[colMask + (j >> 3)] >> (j & 7)) & 1)) continue;
      let value = 0;
      let shift = 0;
      let byte;
      do {
        byte = bytes[o++];
        value |= (byte & 0x7f) << shift;
        shift += 7;
      } while (byte & 0x80);
      q16[row + j] += (value >>> 1) ^ -(value & 1);
    }
    colMask += colBytes;
  }
  return head;
}

/** Decode one frame into new arrays (tests and tools). `previous` is the frame a delta builds on. */
export function decodeFrame(buffer, previous = null) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const head = readHeader(bytes);
  if (head.kind === KIND_DELTA && !previous?.q16) throw new Error('delta frame without a base');
  const q16 = previous?.q16 ? previous.q16.slice() : new Int16Array(head.nEnvs * head.nq);
  const state = previous?.state ? previous.state.slice() : new Uint8Array(head.nEnvs);
  applyFrame(bytes, q16, state);
  return { ...head, q16, state };
}

/** The acknowledgement the server paces each client by. */
export function ackMessage(seq) {
  const message = new DataView(new ArrayBuffer(5));
  message.setUint8(0, 1);
  message.setUint32(1, seq, true);
  return message.buffer;
}
