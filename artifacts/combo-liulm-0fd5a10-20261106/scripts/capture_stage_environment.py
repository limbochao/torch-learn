import hashlib,json,os,sys,sysconfig
from pathlib import Path
import torch,torch_npu,triton
root=Path(__file__).resolve().parent;expected=json.loads((root/'expected_source_sha256.json').read_text());site=Path(sysconfig.get_paths()['purelib'])
actual={p:hashlib.sha256((site/p).read_bytes()).hexdigest() for p in expected}
info={'date':'2026-11-06','branch':'liulm/v2.10.0-combokernel-dev','commit':'0fd5a10da2a3c33cab8c200f5778d711e490ff6b','torch':torch.__version__,'torch_git':torch.version.git_version,'torch_npu':torch_npu.__version__,'torch_npu_git':torch_npu.version.git_version,'triton':triton.__version__,'physical_devices':os.environ['COMBO_PHYSICAL_DEVICES'],'prerequisite_fix':'scheduling.py remove extra indentation space at original line 718','existing_f3_overlay':os.environ.get('COMBO_WITH_F3')=='1','source_sha256':actual,'matches_expected':actual==expected}
for p in expected:
 dest=root/'evidence/installed_source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((site/p).read_bytes())
(root/'evidence/environment.json').write_text(json.dumps(info,indent=2)+'\n')
assert actual==expected and info['torch_npu_git']==info['commit']
print(json.dumps({k:v for k,v in info.items() if k not in ['source_sha256']},indent=2))
