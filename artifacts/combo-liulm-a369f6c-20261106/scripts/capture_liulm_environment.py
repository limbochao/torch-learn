import hashlib,json,os,sys,importlib.util
from pathlib import Path
import torch,torch_npu,triton
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence'
expected=json.loads((ROOT/'expected_source_sha256.json').read_text())
actual={}
for relative,digest in expected.items():
    pkg,rest=relative.split('/',1)
    path=Path(importlib.util.find_spec(pkg).origin).parent/rest
    data=path.read_bytes()
    actual[relative]=hashlib.sha256(data).hexdigest()
    target=OUT/'installed_source'/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(data)
info={'validation_date':'2026-11-06','branch':'liulm/v2.10.0-combokernel-dev','expected_commit':'a369f6c8da77e78a18ccab65e566d0fd6bb0b005','python':sys.version,'torch':torch.__version__,'torch_git':torch.version.git_version,'torch_npu':torch_npu.__version__,'torch_npu_git':torch_npu.version.git_version,'triton':triton.__version__,'device':'NPU_A5 Ascend950PR','cann':'9.1.0','physical_devices':[1,3,4,5],'shared_installation_modified':False,'production_overlay':False,'source_sha256':actual,'matches_expected':actual==expected,'case_source_commit':'be6049a0059677ca408da1360653b694b6d1afb5','selected_methods':62,'expected_instances':409}
(OUT/'environment.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2))
assert info['torch_npu_git']==info['expected_commit']
assert actual==expected, {p:(expected[p],actual[p]) for p in expected if expected[p]!=actual[p]}
