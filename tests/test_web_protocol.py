import json, subprocess
from pathlib import Path
import numpy as np
from kaiku.web.protocol import decode, encode_delta, encode_key

def test_key_delta_roundtrip():
    q=np.array([[1,-2,300],[4,5,-6]],dtype=np.int16); st=np.array([4,1],dtype=np.uint8)
    k=encode_key(9,123,q,st); d=decode(k); assert np.array_equal(d['q16'],q); assert np.array_equal(d['state'],st)
    q2=q.copy();q2[0,1]+=13;q2[1,2]-=17
    out=decode(encode_delta(10,9,124,q2,q,st),q.copy(),st.copy()); assert np.array_equal(out['q16'],q2)

def test_js_decoder_matches_python(tmp_path):
    root=Path(__file__).parents[1]; q=np.array([[12,-3,5],[0,7,-9]],dtype=np.int16)
    frame=encode_key(4,88,q,np.array([4,0],dtype=np.uint8)); path=tmp_path/'frame.bin';path.write_bytes(frame)
    js="""import fs from 'fs'; import {decodeFrame} from './kaiku/web/static/js/protocol.js'; const f=decodeFrame(fs.readFileSync(process.argv[1])); console.log(JSON.stringify({seq:f.seq,q:[...f.q16],s:[...f.state]}));"""
    r=subprocess.run(['node','--input-type=module','-e',js,str(path)],cwd=root,text=True,capture_output=True,check=True)
    got=json.loads(r.stdout); assert got=={'seq':4,'q':[12,-3,5,0,7,-9],'s':[4,0]}
