/* Kaiku binary stream decoder. Kept byte-for-byte compatible with kaiku.web.protocol. */
export const MAGIC=0x4b, KIND_KEY=1, KIND_DELTA=2, FLAG_STATE=1;
const HEADER=20;
function zigzag(v){return (v>>>1)^-(v&1)}
export function decodeFrame(buffer, previous=null){
  const b=buffer instanceof Uint8Array?buffer:new Uint8Array(buffer), dv=new DataView(b.buffer,b.byteOffset,b.byteLength);
  if(b.length<HEADER||dv.getUint8(0)!==MAGIC) throw new Error('not a kaiku frame');
  const kind=dv.getUint8(1), flags=dv.getUint8(2), seq=dv.getUint32(4,true), baseSeq=dv.getUint32(8,true), tMs=dv.getUint32(12,true), ne=dv.getUint16(16,true), nq=dv.getUint16(18,true);
  let o=HEADER, state=previous?.state||null;
  if(flags&FLAG_STATE){state=b.slice(o,o+ne);o+=ne;}
  let q16;
  if(kind===KIND_KEY){if(ne&1)o++;q16=new Int16Array(b.buffer,b.byteOffset+o,ne*nq).slice();o+=2*ne*nq;}
  else if(kind===KIND_DELTA){
    if(!previous?.q16) throw new Error('delta without base');
    q16=previous.q16.slice(); const rb=Math.ceil(ne/8), rows=[];
    for(let i=0;i<ne;i++) if((b[o+(i>>3)]&(1<<(i&7)))!==0) rows.push(i); o+=rb;
    const cb=Math.ceil(nq/8), masks=[];
    for(let ri=0;ri<rows.length;ri++){const m=[];for(let j=0;j<nq;j++)if(b[o+(ri*cb)+(j>>3)]&(1<<(j&7)))m.push(j);masks.push(m)}o+=rows.length*cb;
    for(let ri=0;ri<rows.length;ri++)for(const j of masks[ri]){let v=0,shift=0,x;do{x=b[o++];v|=(x&127)<<shift;shift+=7}while(x&128);q16[rows[ri]*nq+j]=q16[rows[ri]*nq+j]+zigzag(v)}
  } else throw new Error(`unknown frame kind ${kind}`);
  return {kind,seq,baseSeq,tMs,nEnvs:ne,nq,q16,state};
}
export class StreamDecoder{
  constructor(){this.frame=null}
  decode(buffer){const f=decodeFrame(buffer,this.frame);if(f.kind===KIND_DELTA&&this.frame&&f.baseSeq!==this.frame.seq){this.frame=null;throw new Error('delta base mismatch')}this.frame=f;return f}
  reset(){this.frame=null}
}
export function ack(seq){const a=new ArrayBuffer(5),d=new DataView(a);d.setUint8(0,1);d.setUint32(1,seq,true);return a}
export const decode=decodeFrame;
