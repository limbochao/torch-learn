import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the original NPU FX graph repro")
    parser.add_argument(
        "--execution",
        choices=("static", "dynamic", "group"),
        default="dynamic",
        help="static uses dynamic=False; dynamic/group use mark_dynamic + dynamic=None",
    )
    parser.add_argument("--bs", type=int, default=200, help="Batch size used to construct and compile inputs")
    return parser.parse_args()


SCRIPT_ARGS = parse_args()
if SCRIPT_ARGS.bs <= 0:
    raise ValueError("--bs must be positive")

GROUP_AUTOTUNE_ENV = "INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE"
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

RUN_ID = os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
PROFILE_ROOT = Path(os.environ.get("PROFILE_ROOT", SCRIPT_DIR / "prof_log" / "fx_graph_runable_npu"))
RUN_ROOT = PROFILE_ROOT / RUN_ID / f"bs_{SCRIPT_ARGS.bs}" / SCRIPT_ARGS.execution
RUN_ROOT.mkdir(parents=True, exist_ok=False)

os.environ[GROUP_AUTOTUNE_ENV] = "1" if SCRIPT_ARGS.execution == "group" else "0"
os.environ.setdefault("NPU_INDUCTOR_FALLBACK_LIST", "aten.cat")
os.environ.setdefault("TORCHNPU_PRECOMPILE_THREADS", "32")
os.environ.setdefault("INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE", "1")
os.environ.setdefault("INDUCTOR_INDIRECT_MEMORY_MODE", "fallback")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
os.environ.setdefault("TRITON_ASCEND_ALLOW_IPC_POINTER", "1")
os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "default")
os.environ.setdefault("INDUCTOR_ASCEND_LOG_LEVEL", "INFO")
os.environ.setdefault("TORCHINDUCTOR_ENABLE_FAST_GELU", "1")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")
os.environ.setdefault("TORCH_COMPILE_DEBUG_DIR", str(RUN_ROOT / "torch_compile_debug"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(RUN_ROOT / "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(RUN_ROOT / "triton"))

import torch
import torch_npu
from torch_npu.utils._dynamo import register_inductor_npu

register_inductor_npu()

from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from torch._prims_common import make_contiguous_strides_for
from math import inf, nan
import torch._inductor.inductor_prims

from tools.npu_profiler import ProfileResultParser, TorchNpuProfiler



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.dynamic_shapes = True
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = True
torch._dynamo.config.allow_ignore_mark_dynamic = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.fake_tensor_cache_enabled = False
torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.post_grad_fusion_options = {'fav3_partition': {}}
torch._inductor.config.reorder_for_peak_memory = False
torch._inductor.config.deterministic = False
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.triton.coalesce_tiling_analysis = False
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.triton.mix_order_reduction = False
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.output_code = True
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False


_QIANCHUAN_DEF = torch.library.Library("qianchuan_triton", "DEF")
_QIANCHUAN_NPU = torch.library.Library("qianchuan_triton", "IMPL", "PrivateUse1")
_QIANCHUAN_META = torch.library.Library("qianchuan_triton", "IMPL", "Meta")

_QIANCHUAN_DEF.define("softcap(Tensor x, float cap) -> Tensor")
_QIANCHUAN_DEF.define(
    "fused_swiglu(Tensor x, Tensor w_gate, Tensor w_fc, Tensor b_gate, Tensor b_fc, "
    "bool flag0, bool flag1) -> Tensor"
)
_QIANCHUAN_DEF.define(
    "rope(Tensor x, Tensor positions, Tensor cu_seqlens, SymInt max_seq_len, float base, "
    "bool interleaved) -> Tensor"
)


def _softcap_placeholder(x, cap):
    return x.clone(memory_format=torch.contiguous_format)


def _fused_swiglu_placeholder(x, w_gate, w_fc, b_gate, b_fc, flag0, flag1):
    return x.new_zeros((*x.shape[:-1], w_gate.shape[-1]))


def _rope_placeholder(x, positions, cu_seqlens, max_seq_len, base, interleaved):
    return x.clone(memory_format=torch.contiguous_format)


def _softcap_meta(x, cap):
    return x.new_empty(x.shape)


def _fused_swiglu_meta(x, w_gate, w_fc, b_gate, b_fc, flag0, flag1):
    return x.new_empty((*x.shape[:-1], w_gate.shape[-1]))


def _rope_meta(x, positions, cu_seqlens, max_seq_len, base, interleaved):
    return x.new_empty(x.shape)


_QIANCHUAN_NPU.impl("softcap", _softcap_placeholder)
_QIANCHUAN_NPU.impl("fused_swiglu", _fused_swiglu_placeholder)
_QIANCHUAN_NPU.impl("rope", _rope_placeholder)
_QIANCHUAN_META.impl("softcap", _softcap_meta)
_QIANCHUAN_META.impl("fused_swiglu", _fused_swiglu_meta)
_QIANCHUAN_META.impl("rope", _rope_meta)


class ShapeHint(int):
    def __new__(cls, value, symbol):
        hint = super().__new__(cls, value)
        hint.symbol = symbol
        return hint


batch_size_hint = ShapeHint(SCRIPT_ARGS.bs, "bs")



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.10.0+cpu
# torch cuda version: None
# torch git version: 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('_tensor_constant0', tensor(0.1000))
        self.register_buffer('_tensor_constant1', tensor(0.1000))
        self.register_buffer('_tensor_constant2', tensor(0.1000))
        self.register_buffer('_tensor_constant3', tensor(0.1000))
        self.register_buffer('_tensor_constant4', tensor(0.1000))
        self.register_buffer('_tensor_constant5', tensor(0.1000))
        self.register_buffer('_tensor_constant6', tensor(0.1000))
        self.register_buffer('_tensor_constant7', tensor(0.1000))
        self.register_buffer('_tensor_constant8', tensor(0.1000))
        self.register_buffer('_tensor_constant9', tensor(0.1000))
        self.register_buffer('_tensor_constant10', tensor(0.))
        self.register_buffer('_tensor_constant11', tensor(1000))
        self.register_buffer('_tensor_constant12', tensor(1024., dtype=torch.float32))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1, arg1180_1, arg1181_1, arg1182_1, arg1183_1, arg1184_1, arg1185_1, arg1186_1, arg1187_1, arg1188_1, arg1189_1, arg1190_1, arg1191_1, arg1192_1, arg1193_1, arg1194_1, arg1195_1, arg1196_1, arg1197_1, arg1198_1, arg1199_1, arg1200_1, arg1201_1, arg1202_1, arg1203_1, arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1234_1, arg1235_1, arg1236_1, arg1237_1, arg1238_1, arg1239_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1276_1, arg1277_1, arg1278_1, arg1279_1, arg1280_1, arg1281_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1318_1, arg1319_1, arg1320_1, arg1321_1, arg1322_1, arg1323_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1360_1, arg1361_1, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1, arg1380_1, arg1381_1, arg1382_1, arg1383_1, arg1384_1, arg1385_1, arg1386_1, arg1387_1, arg1388_1, arg1389_1, arg1390_1, arg1391_1, arg1392_1, arg1393_1, arg1394_1, arg1395_1, arg1396_1, arg1397_1, arg1398_1, arg1399_1, arg1400_1, arg1401_1, arg1402_1, arg1403_1, arg1404_1, arg1405_1, arg1406_1, arg1407_1, arg1408_1, arg1409_1, arg1410_1, arg1411_1, arg1412_1, arg1413_1, arg1414_1, arg1415_1, arg1416_1, arg1417_1, arg1418_1, arg1419_1, arg1420_1, arg1421_1, arg1422_1, arg1423_1, arg1424_1, arg1425_1, arg1426_1, arg1427_1, arg1428_1, arg1429_1, arg1430_1, arg1431_1, arg1432_1, arg1433_1, arg1434_1, arg1435_1, arg1436_1, arg1437_1, arg1438_1, arg1439_1, arg1440_1, arg1441_1, arg1442_1, arg1443_1, arg1444_1, arg1445_1, arg1446_1, arg1447_1, arg1448_1, arg1449_1, arg1450_1, arg1451_1, arg1452_1, arg1453_1, arg1454_1, arg1455_1, arg1456_1, arg1457_1, arg1458_1, arg1459_1, arg1460_1, arg1461_1, arg1462_1, arg1463_1, arg1464_1, arg1465_1, arg1466_1, arg1467_1, arg1468_1, arg1469_1, arg1470_1, arg1471_1, arg1472_1, arg1473_1, arg1474_1, arg1475_1, arg1476_1, arg1477_1, arg1478_1, arg1479_1, arg1480_1, arg1481_1, arg1482_1, arg1483_1, arg1484_1, arg1485_1, arg1486_1, arg1487_1, arg1488_1, arg1489_1, arg1490_1, arg1491_1, arg1492_1, arg1493_1, arg1494_1, arg1495_1, arg1496_1, arg1497_1, arg1498_1, arg1499_1, arg1500_1, arg1501_1, arg1502_1, arg1503_1, arg1504_1, arg1505_1, arg1506_1, arg1507_1, arg1508_1, arg1509_1, arg1510_1, arg1511_1, arg1512_1, arg1513_1, arg1514_1, arg1515_1, arg1516_1, arg1517_1, arg1518_1, arg1519_1, arg1520_1, arg1521_1, arg1522_1, arg1523_1, arg1524_1, arg1525_1, arg1526_1, arg1527_1, arg1528_1, arg1529_1, arg1530_1, arg1531_1, arg1532_1, arg1533_1, arg1534_1, arg1535_1, arg1536_1, arg1537_1, arg1538_1, arg1539_1, arg1540_1, arg1541_1, arg1542_1, arg1543_1, arg1544_1, arg1545_1, arg1546_1, arg1547_1, arg1548_1, arg1549_1, arg1550_1, arg1551_1, arg1552_1, arg1553_1, arg1554_1, arg1555_1, arg1556_1, arg1557_1, arg1558_1, arg1559_1, arg1560_1, arg1561_1, arg1562_1, arg1563_1, arg1564_1, arg1565_1, arg1566_1, arg1567_1, arg1568_1, arg1569_1, arg1570_1, arg1571_1, arg1572_1, arg1573_1, arg1574_1, arg1575_1, arg1576_1, arg1577_1, arg1578_1, arg1579_1, arg1580_1, arg1581_1, arg1582_1, arg1583_1, arg1584_1, arg1585_1, arg1586_1, arg1587_1, arg1588_1, arg1589_1, arg1590_1, arg1591_1, arg1592_1, arg1593_1, arg1594_1, arg1595_1, arg1596_1, arg1597_1, arg1598_1, arg1599_1, arg1600_1, arg1601_1, arg1602_1, arg1603_1, arg1604_1, arg1605_1, arg1606_1, arg1607_1, arg1608_1, arg1609_1, arg1610_1, arg1611_1, arg1612_1, arg1613_1, arg1614_1, arg1615_1, arg1616_1, arg1617_1, arg1618_1, arg1619_1, arg1620_1, arg1621_1, arg1622_1, arg1623_1, arg1624_1, arg1625_1, arg1626_1, arg1627_1, arg1628_1, arg1629_1, arg1630_1, arg1631_1, arg1632_1, arg1633_1, arg1634_1, arg1635_1, arg1636_1, arg1637_1, arg1638_1, arg1639_1, arg1640_1, arg1641_1, arg1642_1, arg1643_1, arg1644_1, arg1645_1, arg1646_1, arg1647_1, arg1648_1, arg1649_1, arg1650_1, arg1651_1, arg1652_1, arg1653_1, arg1654_1, arg1655_1, arg1656_1, arg1657_1, arg1658_1, arg1659_1, arg1660_1, arg1661_1, arg1662_1, arg1663_1, arg1664_1, arg1665_1, arg1666_1, arg1667_1, arg1668_1, arg1669_1, arg1670_1, arg1671_1, arg1672_1, arg1673_1, arg1674_1, arg1675_1, arg1676_1, arg1677_1, arg1678_1, arg1679_1, arg1680_1, arg1681_1, arg1682_1, arg1683_1, arg1684_1, arg1685_1, arg1686_1, arg1687_1, arg1688_1, arg1689_1, arg1690_1, arg1691_1, arg1692_1, arg1693_1, arg1694_1, arg1695_1, arg1696_1, arg1697_1, arg1698_1, arg1699_1, arg1700_1, arg1701_1, arg1702_1, arg1703_1, arg1704_1, arg1705_1, arg1706_1, arg1707_1):
        batch_size = arg0_1.shape[0]
        convert_element_type = torch.ops.prims.convert_element_type.default(arg0_1, torch.bool);  arg0_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(arg1_1, torch.bool);  arg1_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(arg2_1, torch.bool);  arg2_1 = None
        logical_or = torch.ops.aten.logical_or.default(convert_element_type, convert_element_type_1);  convert_element_type = None
        logical_or_1 = torch.ops.aten.logical_or.default(convert_element_type_2, logical_or);  logical_or = None
        full_default_6 = torch.ops.aten.full.default([batch_size, 1], 9999, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        eq = torch.ops.aten.eq.Tensor(arg3_1, full_default_6);  full_default_6 = None
        eq_1 = torch.ops.aten.eq.Scalar(arg4_1, 412)
        eq_2 = torch.ops.aten.eq.Scalar(arg5_1, 102)
        logical_and = torch.ops.aten.logical_and.default(eq_1, eq_2);  eq_1 = eq_2 = None
        eq_3 = torch.ops.aten.eq.Scalar(arg4_1, 96)
        logical_or_2 = torch.ops.aten.logical_or.default(eq_3, logical_and);  eq_3 = logical_and = None
        full_default_7 = torch.ops.aten.full.default([batch_size, 1], 9998, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(logical_or_2, full_default_7, arg3_1);  logical_or_2 = full_default_7 = arg3_1 = None
        full_default_8 = torch.ops.aten.full.default([batch_size, 1], 9998, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        eq_4 = torch.ops.aten.eq.Tensor(where, full_default_8);  full_default_8 = None
        logical_not = torch.ops.aten.logical_not.default(eq_4)
        logical_not_1 = torch.ops.aten.logical_not.default(logical_not)
        full_default_9 = torch.ops.aten.full.default([batch_size, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(logical_or_1, arg6_1, full_default_9);  full_default_9 = None
        view = torch.ops.aten.view.default(where_1, [-1]);  where_1 = None
        full_default_10 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(logical_or_1, arg6_1, full_default_10);  arg6_1 = full_default_10 = None
        view_1 = torch.ops.aten.view.default(where_2, [-1]);  where_2 = None
        full_default_11 = torch.ops.aten.full.default([batch_size, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(logical_or_1, arg7_1, full_default_11);  logical_or_1 = arg7_1 = full_default_11 = None
        view_2 = torch.ops.aten.view.default(where_3, [-1]);  where_3 = None
        full_default_12 = torch.ops.aten.full.default([batch_size, 1], 1, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_4 = torch.ops.aten.where.self(convert_element_type_2, arg8_1, full_default_12);  arg8_1 = full_default_12 = None
        view_3 = torch.ops.aten.view.default(where_4, [-1]);  where_4 = None
        full_default_13 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        eq_5 = torch.ops.aten.eq.Tensor(where, full_default_13);  full_default_13 = None
        eq_6 = torch.ops.aten.eq.Scalar(arg9_1, 1128)
        eq_7 = torch.ops.aten.eq.Scalar(arg9_1, 2329)
        eq_8 = torch.ops.aten.eq.Scalar(arg9_1, 6383)
        eq_9 = torch.ops.aten.eq.Scalar(arg9_1, 8663)
        eq_10 = torch.ops.aten.eq.Scalar(arg9_1, 1112)
        eq_11 = torch.ops.aten.eq.Scalar(arg9_1, 1350)
        eq_12 = torch.ops.aten.eq.Scalar(arg9_1, 561124)
        eq_13 = torch.ops.aten.eq.Scalar(arg9_1, 581610);  arg9_1 = None
        logical_or_3 = torch.ops.aten.logical_or.default(eq_13, eq_12);  eq_13 = eq_12 = None
        logical_or_4 = torch.ops.aten.logical_or.default(logical_or_3, eq_9);  logical_or_3 = eq_9 = None
        logical_or_5 = torch.ops.aten.logical_or.default(logical_or_4, eq_10);  logical_or_4 = eq_10 = None
        logical_or_6 = torch.ops.aten.logical_or.default(logical_or_5, eq_11);  logical_or_5 = eq_11 = None
        logical_or_7 = torch.ops.aten.logical_or.default(eq_8, logical_or_6);  eq_8 = logical_or_6 = None
        logical_or_8 = torch.ops.aten.logical_or.default(eq_7, logical_or_7);  eq_7 = logical_or_7 = None
        logical_or_9 = torch.ops.aten.logical_or.default(eq_6, logical_or_8);  eq_6 = logical_or_8 = None
        logical_not_2 = torch.ops.aten.logical_not.default(logical_or_9);  logical_or_9 = None
        logical_and_1 = torch.ops.aten.logical_and.default(eq_5, logical_not_2)
        full_default_14 = torch.ops.aten.full.default([batch_size, 1], 1, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_5 = torch.ops.aten.where.self(logical_and_1, full_default_14, where);  logical_and_1 = full_default_14 = where = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(arg10_1, torch.int32);  arg10_1 = None
        full_default_15 = torch.ops.aten.full.default([batch_size, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        eq_14 = torch.ops.aten.eq.Scalar(convert_element_type_3, 1)
        logical_and_2 = torch.ops.aten.logical_and.default(eq_14, logical_not);  eq_14 = None
        full_default_16 = torch.ops.aten.full.default([batch_size, 1], 2, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_6 = torch.ops.aten.where.self(logical_and_2, full_default_16, where_5);  logical_and_2 = full_default_16 = where_5 = None
        eq_15 = torch.ops.aten.eq.Scalar(convert_element_type_3, 1);  convert_element_type_3 = None
        logical_and_3 = torch.ops.aten.logical_and.default(eq_15, eq_4);  eq_15 = None
        full_default_17 = torch.ops.aten.full.default([batch_size, 1], 3, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_7 = torch.ops.aten.where.self(logical_and_3, full_default_17, where_6);  logical_and_3 = full_default_17 = where_6 = None
        le = torch.ops.aten.le.Scalar(arg11_1, 0);  arg11_1 = None
        full_default_18 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        full_default_19 = torch.ops.aten.full.default([batch_size, 1], 1, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_8 = torch.ops.aten.where.self(le, full_default_18, full_default_19);  le = full_default_18 = full_default_19 = None
        eq_16 = torch.ops.aten.eq.Scalar(arg14_1, 103);  arg14_1 = None
        logical_and_4 = torch.ops.aten.logical_and.default(eq_4, eq_16);  eq_16 = None
        logical_or_10 = torch.ops.aten.logical_or.default(eq, logical_and_4);  eq = logical_and_4 = None
        logical_or_11 = torch.ops.aten.logical_or.default(logical_or_10, full_default_15);  logical_or_10 = full_default_15 = None
        logical_or_12 = torch.ops.aten.logical_or.default(logical_or_11, where_8);  logical_or_11 = None
        slice_3 = torch.ops.aten.slice.Tensor(arg15_1, 1, 55516, 55580)
        ne_1 = torch.ops.aten.ne.Tensor(arg16_1, arg16_1)
        abs_1 = torch.ops.aten.abs.default(arg16_1)
        eq_22 = torch.ops.aten.eq.Scalar(abs_1, inf);  abs_1 = None
        bitwise_or = torch.ops.aten.bitwise_or.Tensor(ne_1, eq_22);  ne_1 = eq_22 = None
        full_default_20 = torch.ops.aten.full.default([batch_size, 512], 1.0013580322265625e-05, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_12 = torch.ops.aten.where.self(bitwise_or, full_default_20, arg16_1);  bitwise_or = full_default_20 = arg16_1 = None
        ne_2 = torch.ops.aten.ne.Tensor(arg17_1, arg17_1)
        abs_2 = torch.ops.aten.abs.default(arg17_1)
        eq_23 = torch.ops.aten.eq.Scalar(abs_2, inf);  abs_2 = None
        bitwise_or_1 = torch.ops.aten.bitwise_or.Tensor(ne_2, eq_23);  ne_2 = eq_23 = None
        full_default_21 = torch.ops.aten.full.default([batch_size, 32], 1.0013580322265625e-05, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_13 = torch.ops.aten.where.self(bitwise_or_1, full_default_21, arg17_1);  bitwise_or_1 = full_default_21 = arg17_1 = None
        ne_3 = torch.ops.aten.ne.Tensor(arg18_1, arg18_1)
        abs_3 = torch.ops.aten.abs.default(arg18_1)
        eq_24 = torch.ops.aten.eq.Scalar(abs_3, inf);  abs_3 = None
        bitwise_or_2 = torch.ops.aten.bitwise_or.Tensor(ne_3, eq_24);  ne_3 = eq_24 = None
        full_default_22 = torch.ops.aten.full.default([batch_size, 32], 1.0013580322265625e-05, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_14 = torch.ops.aten.where.self(bitwise_or_2, full_default_22, arg18_1);  bitwise_or_2 = full_default_22 = arg18_1 = None
        ne_4 = torch.ops.aten.ne.Tensor(arg19_1, arg19_1)
        abs_4 = torch.ops.aten.abs.default(arg19_1)
        eq_25 = torch.ops.aten.eq.Scalar(abs_4, inf);  abs_4 = None
        bitwise_or_3 = torch.ops.aten.bitwise_or.Tensor(ne_4, eq_25);  ne_4 = eq_25 = None
        full_default_23 = torch.ops.aten.full.default([batch_size, 32], 1.0013580322265625e-05, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_15 = torch.ops.aten.where.self(bitwise_or_3, full_default_23, arg19_1);  bitwise_or_3 = full_default_23 = arg19_1 = None
        ne_5 = torch.ops.aten.ne.Tensor(arg20_1, arg20_1)
        abs_5 = torch.ops.aten.abs.default(arg20_1)
        eq_26 = torch.ops.aten.eq.Scalar(abs_5, inf);  abs_5 = None
        bitwise_or_4 = torch.ops.aten.bitwise_or.Tensor(ne_5, eq_26);  ne_5 = eq_26 = None
        full_default_24 = torch.ops.aten.full.default([batch_size, 32], 1.0013580322265625e-05, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_16 = torch.ops.aten.where.self(bitwise_or_4, full_default_24, arg20_1);  bitwise_or_4 = full_default_24 = arg20_1 = None
        cat = torch.ops.aten.cat.default([where_12, where_13, where_14, where_15, where_16], 1)
        addmm = torch.ops.aten.addmm.default(arg22_1, cat, arg21_1);  arg22_1 = cat = arg21_1 = None
        relu = torch.ops.aten.relu.default(addmm);  addmm = None
        addmm_1 = torch.ops.aten.addmm.default(arg24_1, relu, arg23_1);  arg24_1 = relu = arg23_1 = None
        ne_6 = torch.ops.aten.ne.Tensor(arg25_1, arg25_1)
        abs_6 = torch.ops.aten.abs.default(arg25_1)
        eq_27 = torch.ops.aten.eq.Scalar(abs_6, inf);  abs_6 = None
        bitwise_or_5 = torch.ops.aten.bitwise_or.Tensor(ne_6, eq_27);  ne_6 = eq_27 = None
        full_default_25 = torch.ops.aten.full.default([batch_size, 328], 1.0013580322265625e-05, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_17 = torch.ops.aten.where.self(bitwise_or_5, full_default_25, arg25_1);  bitwise_or_5 = full_default_25 = arg25_1 = None
        slice_4 = torch.ops.aten.slice.Tensor(where_17, 1, 0, 128);  where_17 = None
        addmm_2 = torch.ops.aten.addmm.default(arg27_1, slice_4, arg26_1);  arg27_1 = slice_4 = arg26_1 = None
        relu_1 = torch.ops.aten.relu.default(addmm_2);  addmm_2 = None
        addmm_3 = torch.ops.aten.addmm.default(arg29_1, relu_1, arg28_1);  arg29_1 = relu_1 = arg28_1 = None
        ne_7 = torch.ops.aten.ne.Tensor(arg30_1, arg30_1)
        abs_7 = torch.ops.aten.abs.default(arg30_1)
        eq_28 = torch.ops.aten.eq.Scalar(abs_7, inf);  abs_7 = None
        bitwise_or_6 = torch.ops.aten.bitwise_or.Tensor(ne_7, eq_28);  ne_7 = eq_28 = None
        full_default_26 = torch.ops.aten.full.default([batch_size, 256], 1.0013580322265625e-05, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_18 = torch.ops.aten.where.self(bitwise_or_6, full_default_26, arg30_1);  bitwise_or_6 = full_default_26 = arg30_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg32_1, where_18, arg31_1);  arg32_1 = where_18 = arg31_1 = None
        relu_2 = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None
        addmm_5 = torch.ops.aten.addmm.default(arg34_1, relu_2, arg33_1);  arg34_1 = relu_2 = arg33_1 = None
        cat_1 = torch.ops.aten.cat.default([addmm_3, addmm_5, addmm_1], 1);  addmm_5 = addmm_1 = None
        view_7 = torch.ops.aten.view.default(arg35_1, [1, batch_size, 256]);  arg35_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(view_7, [0]);  view_7 = None
        mul_13 = torch.ops.aten.mul.Tensor(cat_1, sum_1)
        mul_14 = torch.ops.aten.mul.Tensor(arg36_1, mul_13);  arg36_1 = mul_13 = None
        ne_8 = torch.ops.aten.ne.Tensor(arg37_1, arg37_1)
        abs_8 = torch.ops.aten.abs.default(arg37_1)
        eq_29 = torch.ops.aten.eq.Scalar(abs_8, inf);  abs_8 = None
        bitwise_or_7 = torch.ops.aten.bitwise_or.Tensor(ne_8, eq_29);  ne_8 = eq_29 = None
        full_default_27 = torch.ops.aten.full.default([batch_size, 128], 1.0013580322265625e-05, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_19 = torch.ops.aten.where.self(bitwise_or_7, full_default_27, arg37_1);  bitwise_or_7 = full_default_27 = arg37_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg39_1, where_19, arg38_1);  arg39_1 = where_19 = arg38_1 = None
        relu_3 = torch.ops.aten.relu.default(addmm_6);  addmm_6 = None
        addmm_7 = torch.ops.aten.addmm.default(arg41_1, relu_3, arg40_1);  arg41_1 = relu_3 = arg40_1 = None
        mul_16 = torch.ops.aten.mul.Tensor(arg42_1, addmm_7);  arg42_1 = None
        slice_5 = torch.ops.aten.slice.Tensor(arg15_1, 1, 55818, 55834)
        slice_6 = torch.ops.aten.slice.Tensor(arg15_1, 1, 56436, 56452)
        slice_7 = torch.ops.aten.slice.Tensor(arg15_1, 1, 45405, 45421)
        slice_8 = torch.ops.aten.slice.Tensor(arg15_1, 1, 44187, 44203)
        slice_9 = torch.ops.aten.slice.Tensor(arg15_1, 1, 44728, 44744)
        slice_10 = torch.ops.aten.slice.Tensor(arg15_1, 1, 45946, 45962)
        slice_11 = torch.ops.aten.slice.Tensor(arg15_1, 1, 61312, 61328)
        slice_12 = torch.ops.aten.slice.Tensor(arg15_1, 1, 62251, 62267)
        slice_13 = torch.ops.aten.slice.Tensor(arg15_1, 1, 69044, 69060)
        slice_14 = torch.ops.aten.slice.Tensor(arg15_1, 1, 55834, 55962)
        slice_15 = torch.ops.aten.slice.Tensor(arg15_1, 1, 56452, 56580)
        slice_16 = torch.ops.aten.slice.Tensor(arg15_1, 1, 45421, 45549)
        slice_17 = torch.ops.aten.slice.Tensor(arg15_1, 1, 44203, 44331)
        slice_18 = torch.ops.aten.slice.Tensor(arg15_1, 1, 44744, 44872)
        slice_19 = torch.ops.aten.slice.Tensor(arg15_1, 1, 45962, 46090)
        slice_20 = torch.ops.aten.slice.Tensor(arg15_1, 1, 60741, 60869)
        slice_21 = torch.ops.aten.slice.Tensor(arg15_1, 1, 61328, 61456)
        slice_22 = torch.ops.aten.slice.Tensor(arg15_1, 1, 62267, 62395)
        slice_23 = torch.ops.aten.slice.Tensor(arg15_1, 1, 65602, 65730)
        slice_24 = torch.ops.aten.slice.Tensor(arg15_1, 1, 66495, 66623)
        slice_25 = torch.ops.aten.slice.Tensor(arg15_1, 1, 66956, 67084)
        slice_26 = torch.ops.aten.slice.Tensor(arg15_1, 1, 67445, 67573)
        slice_27 = torch.ops.aten.slice.Tensor(arg15_1, 1, 68021, 68149)
        slice_28 = torch.ops.aten.slice.Tensor(arg15_1, 1, 69060, 69188)
        slice_29 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32419, 32427)
        squeeze_8 = torch.ops.aten.squeeze.default(logical_not)
        view_15 = torch.ops.aten.view.default(squeeze_8, [-1, 1]);  squeeze_8 = None
        slice_30 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27573, 27581)
        full_default_28 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_20 = torch.ops.aten.where.self(view_15, full_default_28, slice_30);  full_default_28 = slice_30 = None
        slice_31 = torch.ops.aten.slice.Tensor(arg15_1, 1, 26772, 26780)
        full_default_29 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_21 = torch.ops.aten.where.self(view_15, full_default_29, slice_31);  full_default_29 = slice_31 = None
        slice_32 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23504, 23512)
        full_default_30 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_22 = torch.ops.aten.where.self(view_15, full_default_30, slice_32);  full_default_30 = slice_32 = None
        slice_33 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23313, 23321)
        full_default_31 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_23 = torch.ops.aten.where.self(view_15, full_default_31, slice_33);  full_default_31 = slice_33 = None
        slice_34 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23369, 23377)
        full_default_32 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_24 = torch.ops.aten.where.self(view_15, full_default_32, slice_34);  full_default_32 = slice_34 = None
        slice_35 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23425, 23433)
        full_default_33 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_25 = torch.ops.aten.where.self(view_15, full_default_33, slice_35);  full_default_33 = slice_35 = None
        slice_36 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23465, 23473)
        full_default_34 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_26 = torch.ops.aten.where.self(view_15, full_default_34, slice_36);  full_default_34 = slice_36 = None
        slice_37 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23567, 23575)
        full_default_35 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_27 = torch.ops.aten.where.self(view_15, full_default_35, slice_37);  full_default_35 = slice_37 = None
        slice_38 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23596, 23604)
        full_default_36 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_28 = torch.ops.aten.where.self(view_15, full_default_36, slice_38);  full_default_36 = slice_38 = None
        slice_39 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24442, 24450)
        full_default_37 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_29 = torch.ops.aten.where.self(view_15, full_default_37, slice_39);  full_default_37 = slice_39 = None
        slice_40 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25272, 25280)
        full_default_38 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_30 = torch.ops.aten.where.self(view_15, full_default_38, slice_40);  view_15 = full_default_38 = slice_40 = None
        slice_41 = torch.ops.aten.slice.Tensor(arg15_1, 1, 55962, 55978)
        slice_42 = torch.ops.aten.slice.Tensor(arg15_1, 1, 56215, 56231)
        slice_43 = torch.ops.aten.slice.Tensor(arg15_1, 1, 56580, 56596)
        slice_44 = torch.ops.aten.slice.Tensor(arg15_1, 1, 56833, 56849)
        slice_45 = torch.ops.aten.slice.Tensor(arg15_1, 1, 57054, 57070)
        slice_46 = torch.ops.aten.slice.Tensor(arg15_1, 1, 57275, 57291)
        slice_47 = torch.ops.aten.slice.Tensor(arg15_1, 1, 57496, 57512)
        slice_48 = torch.ops.aten.slice.Tensor(arg15_1, 1, 57982, 57998)
        squeeze_11 = torch.ops.aten.squeeze.default(logical_not)
        view_18 = torch.ops.aten.view.default(squeeze_11, [-1, 1]);  squeeze_11 = None
        slice_49 = torch.ops.aten.slice.Tensor(arg15_1, 1, 17868, 17884)
        full_default_39 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_31 = torch.ops.aten.where.self(view_18, full_default_39, slice_49);  full_default_39 = slice_49 = None
        slice_50 = torch.ops.aten.slice.Tensor(arg15_1, 1, 31506, 31522)
        full_default_40 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_32 = torch.ops.aten.where.self(view_18, full_default_40, slice_50);  full_default_40 = slice_50 = None
        slice_51 = torch.ops.aten.slice.Tensor(arg15_1, 1, 28338, 28354)
        full_default_41 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_33 = torch.ops.aten.where.self(view_18, full_default_41, slice_51);  full_default_41 = slice_51 = None
        slice_52 = torch.ops.aten.slice.Tensor(arg15_1, 1, 29184, 29200)
        full_default_42 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_34 = torch.ops.aten.where.self(view_18, full_default_42, slice_52);  full_default_42 = slice_52 = None
        slice_53 = torch.ops.aten.slice.Tensor(arg15_1, 1, 30014, 30030)
        full_default_43 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_35 = torch.ops.aten.where.self(view_18, full_default_43, slice_53);  full_default_43 = slice_53 = None
        slice_54 = torch.ops.aten.slice.Tensor(arg15_1, 1, 30772, 30788)
        full_default_44 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_36 = torch.ops.aten.where.self(view_18, full_default_44, slice_54);  view_18 = full_default_44 = slice_54 = None
        slice_55 = torch.ops.aten.slice.Tensor(arg15_1, 1, 55978, 56010)
        slice_56 = torch.ops.aten.slice.Tensor(arg15_1, 1, 56596, 56628)
        slice_57 = torch.ops.aten.slice.Tensor(arg15_1, 1, 45549, 45581)
        slice_58 = torch.ops.aten.slice.Tensor(arg15_1, 1, 44331, 44363)
        slice_59 = torch.ops.aten.slice.Tensor(arg15_1, 1, 44872, 44904)
        slice_60 = torch.ops.aten.slice.Tensor(arg15_1, 1, 46090, 46122)
        slice_61 = torch.ops.aten.slice.Tensor(arg15_1, 1, 61456, 61488)
        slice_62 = torch.ops.aten.slice.Tensor(arg15_1, 1, 62395, 62427)
        slice_63 = torch.ops.aten.slice.Tensor(arg15_1, 1, 69188, 69220)
        slice_64 = torch.ops.aten.slice.Tensor(arg15_1, 1, 68199, 68215)
        slice_65 = torch.ops.aten.slice.Tensor(arg15_1, 1, 68221, 68237)
        slice_66 = torch.ops.aten.slice.Tensor(arg15_1, 1, 69822, 69838)
        slice_67 = torch.ops.aten.slice.Tensor(arg15_1, 1, 55750, 55766)
        squeeze_20 = torch.ops.aten.squeeze.default(logical_not)
        view_27 = torch.ops.aten.view.default(squeeze_20, [-1, 1]);  squeeze_20 = None
        slice_68 = torch.ops.aten.slice.Tensor(arg15_1, 1, 55483, 55499)
        full_default_45 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_37 = torch.ops.aten.where.self(view_27, full_default_45, slice_68);  view_27 = full_default_45 = slice_68 = None
        slice_69 = torch.ops.aten.slice.Tensor(arg15_1, 1, 4528, 4544)
        slice_70 = torch.ops.aten.slice.Tensor(arg15_1, 1, 4511, 4527)
        slice_71 = torch.ops.aten.slice.Tensor(arg15_1, 1, 4616, 4632)
        slice_72 = torch.ops.aten.slice.Tensor(arg15_1, 1, 4599, 4615)
        slice_73 = torch.ops.aten.slice.Tensor(arg15_1, 1, 4633, 4649)
        slice_74 = torch.ops.aten.slice.Tensor(arg15_1, 1, 142, 158)
        slice_75 = torch.ops.aten.slice.Tensor(arg15_1, 1, 4569, 4585)
        slice_76 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32480, 32496)
        slice_77 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32430, 32446)
        slice_78 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32463, 32479)
        slice_79 = torch.ops.aten.slice.Tensor(arg15_1, 1, 55580, 55708)
        slice_80 = torch.ops.aten.slice.Tensor(arg15_1, 1, 59302, 59430)
        slice_81 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48025, 48033)
        slice_82 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47595, 47603)
        slice_83 = torch.ops.aten.slice.Tensor(arg15_1, 1, 46868, 46876)
        slice_84 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47890, 47898)
        slice_85 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48779, 48787)
        slice_86 = torch.ops.aten.slice.Tensor(arg15_1, 1, 49346, 49354)
        slice_87 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47769, 47777)
        slice_88 = torch.ops.aten.slice.Tensor(arg15_1, 1, 49082, 49090)
        slice_89 = torch.ops.aten.slice.Tensor(arg15_1, 1, 49180, 49188)
        slice_90 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48337, 48345)
        slice_91 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48467, 48475)
        slice_92 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48597, 48605)
        slice_93 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48402, 48410)
        slice_94 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48532, 48540)
        slice_95 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48714, 48722)
        slice_96 = torch.ops.aten.slice.Tensor(arg15_1, 1, 49240, 49248)
        slice_97 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48070, 48078)
        slice_98 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48204, 48212)
        slice_99 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48119, 48127)
        slice_100 = torch.ops.aten.slice.Tensor(arg15_1, 1, 11076, 11084)
        slice_101 = torch.ops.aten.slice.Tensor(arg15_1, 1, 14257, 14265)
        slice_102 = torch.ops.aten.slice.Tensor(arg15_1, 1, 18643, 18651)
        slice_103 = torch.ops.aten.slice.Tensor(arg15_1, 1, 11922, 11930)
        slice_104 = torch.ops.aten.slice.Tensor(arg15_1, 1, 15103, 15111)
        slice_105 = torch.ops.aten.slice.Tensor(arg15_1, 1, 19489, 19497)
        slice_106 = torch.ops.aten.slice.Tensor(arg15_1, 1, 12752, 12760)
        slice_107 = torch.ops.aten.slice.Tensor(arg15_1, 1, 15933, 15941)
        slice_108 = torch.ops.aten.slice.Tensor(arg15_1, 1, 20319, 20327)
        squeeze_33 = torch.ops.aten.squeeze.default(logical_not)
        view_40 = torch.ops.aten.view.default(squeeze_33, [-1, 1]);  squeeze_33 = None
        squeeze_34 = torch.ops.aten.squeeze.default(logical_not_2)
        view_41 = torch.ops.aten.view.default(squeeze_34, [-1, 1]);  squeeze_34 = None
        slice_109 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47098, 47114)
        full_default_46 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_38 = torch.ops.aten.where.self(view_40, full_default_46, slice_109);  full_default_46 = slice_109 = None
        slice_110 = torch.ops.aten.slice.Tensor(arg15_1, 1, 49061, 49077)
        full_default_47 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_39 = torch.ops.aten.where.self(view_40, full_default_47, slice_110);  full_default_47 = slice_110 = None
        slice_111 = torch.ops.aten.slice.Tensor(arg15_1, 1, 46847, 46863)
        full_default_48 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_40 = torch.ops.aten.where.self(view_40, full_default_48, slice_111);  full_default_48 = slice_111 = None
        slice_112 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47119, 47135)
        full_default_49 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_41 = torch.ops.aten.where.self(view_40, full_default_49, slice_112);  full_default_49 = slice_112 = None
        slice_113 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47574, 47590)
        full_default_50 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_42 = torch.ops.aten.where.self(view_40, full_default_50, slice_113);  full_default_50 = slice_113 = None
        slice_114 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47553, 47569)
        full_default_51 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_43 = torch.ops.aten.where.self(view_40, full_default_51, slice_114);  full_default_51 = slice_114 = None
        slice_115 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47511, 47527)
        full_default_52 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_44 = torch.ops.aten.where.self(view_40, full_default_52, slice_115);  full_default_52 = slice_115 = None
        slice_116 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47532, 47548)
        full_default_53 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_45 = torch.ops.aten.where.self(view_40, full_default_53, slice_116);  full_default_53 = slice_116 = None
        slice_117 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42865, 42881)
        full_default_54 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_46 = torch.ops.aten.where.self(view_41, full_default_54, slice_117);  full_default_54 = slice_117 = None
        slice_118 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42886, 42902)
        full_default_55 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_47 = torch.ops.aten.where.self(view_41, full_default_55, slice_118);  full_default_55 = slice_118 = None
        slice_119 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42908, 42924)
        full_default_56 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_48 = torch.ops.aten.where.self(view_41, full_default_56, slice_119);  full_default_56 = slice_119 = None
        slice_120 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42574, 42590)
        full_default_57 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_49 = torch.ops.aten.where.self(view_41, full_default_57, slice_120);  full_default_57 = slice_120 = None
        slice_121 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42531, 42547)
        full_default_58 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_50 = torch.ops.aten.where.self(view_41, full_default_58, slice_121);  full_default_58 = slice_121 = None
        slice_122 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42552, 42568)
        full_default_59 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_51 = torch.ops.aten.where.self(view_41, full_default_59, slice_122);  full_default_59 = slice_122 = None
        slice_123 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42595, 42611)
        full_default_60 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_52 = torch.ops.aten.where.self(view_41, full_default_60, slice_123);  full_default_60 = slice_123 = None
        slice_124 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42700, 42716)
        full_default_61 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_53 = torch.ops.aten.where.self(view_41, full_default_61, slice_124);  full_default_61 = slice_124 = None
        slice_125 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42679, 42695)
        full_default_62 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_54 = torch.ops.aten.where.self(view_41, full_default_62, slice_125);  full_default_62 = slice_125 = None
        slice_126 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42637, 42653)
        full_default_63 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_55 = torch.ops.aten.where.self(view_41, full_default_63, slice_126);  full_default_63 = slice_126 = None
        slice_127 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42827, 42843)
        full_default_64 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_56 = torch.ops.aten.where.self(view_41, full_default_64, slice_127);  full_default_64 = slice_127 = None
        slice_128 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42742, 42758)
        full_default_65 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_57 = torch.ops.aten.where.self(view_41, full_default_65, slice_128);  full_default_65 = slice_128 = None
        slice_129 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42763, 42779)
        full_default_66 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_58 = torch.ops.aten.where.self(view_41, full_default_66, slice_129);  full_default_66 = slice_129 = None
        slice_130 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42785, 42801)
        full_default_67 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_59 = torch.ops.aten.where.self(view_41, full_default_67, slice_130);  full_default_67 = slice_130 = None
        slice_131 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42489, 42505)
        full_default_68 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_60 = torch.ops.aten.where.self(view_41, full_default_68, slice_131);  full_default_68 = slice_131 = None
        slice_132 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42721, 42737)
        full_default_69 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_61 = torch.ops.aten.where.self(view_41, full_default_69, slice_132);  full_default_69 = slice_132 = None
        slice_133 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42447, 42463)
        full_default_70 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_62 = torch.ops.aten.where.self(view_41, full_default_70, slice_133);  full_default_70 = slice_133 = None
        slice_134 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42468, 42484)
        full_default_71 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_63 = torch.ops.aten.where.self(view_41, full_default_71, slice_134);  full_default_71 = slice_134 = None
        slice_135 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42510, 42526)
        full_default_72 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_64 = torch.ops.aten.where.self(view_41, full_default_72, slice_135);  full_default_72 = slice_135 = None
        slice_136 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2265, 2281)
        slice_137 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3583, 3599)
        slice_138 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3069, 3085)
        slice_139 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2207, 2223)
        slice_140 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3549, 3565)
        slice_141 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2971, 2987)
        slice_142 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2228, 2244)
        slice_143 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3566, 3582)
        slice_144 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3052, 3068)
        slice_145 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2286, 2302)
        slice_146 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3600, 3616)
        slice_147 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3150, 3166)
        slice_148 = torch.ops.aten.slice.Tensor(arg15_1, 1, 1981, 1997)
        slice_149 = torch.ops.aten.slice.Tensor(arg15_1, 1, 1849, 1865)
        slice_150 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2073, 2089)
        slice_151 = torch.ops.aten.slice.Tensor(arg15_1, 1, 1677, 1693)
        slice_152 = torch.ops.aten.slice.Tensor(arg15_1, 1, 1733, 1749)
        slice_153 = torch.ops.aten.slice.Tensor(arg15_1, 1, 1791, 1807)
        slice_154 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2143, 2159)
        slice_155 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2017, 2033)
        slice_156 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3458, 3474)
        slice_157 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3389, 3405)
        slice_158 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3506, 3522)
        slice_159 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3260, 3276)
        slice_160 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3316, 3332)
        slice_161 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3334, 3350)
        slice_162 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3527, 3543)
        slice_163 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3489, 3505)
        slice_164 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2639, 2655)
        slice_165 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2519, 2535)
        slice_166 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2806, 2822)
        slice_167 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2430, 2446)
        slice_168 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2447, 2463)
        slice_169 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2502, 2518)
        slice_170 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2861, 2877)
        slice_171 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2725, 2741)
        slice_172 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42658, 42674)
        full_default_73 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_65 = torch.ops.aten.where.self(view_41, full_default_73, slice_172);  full_default_73 = slice_172 = None
        slice_173 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42616, 42632)
        full_default_74 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_66 = torch.ops.aten.where.self(view_41, full_default_74, slice_173);  full_default_74 = slice_173 = None
        slice_174 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42806, 42822)
        full_default_75 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_67 = torch.ops.aten.where.self(view_41, full_default_75, slice_174);  view_41 = full_default_75 = slice_174 = None
        slice_175 = torch.ops.aten.slice.Tensor(arg15_1, 1, 1960, 1976)
        slice_176 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3437, 3453)
        slice_177 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2558, 2574)
        slice_178 = torch.ops.aten.slice.Tensor(arg15_1, 1, 1907, 1923)
        slice_179 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3420, 3436)
        slice_180 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2541, 2557)
        slice_181 = torch.ops.aten.slice.Tensor(arg15_1, 1, 6165, 6181)
        slice_182 = torch.ops.aten.slice.Tensor(arg15_1, 1, 6228, 6244)
        slice_183 = torch.ops.aten.slice.Tensor(arg15_1, 1, 6186, 6202)
        slice_184 = torch.ops.aten.slice.Tensor(arg15_1, 1, 6207, 6223)
        slice_185 = torch.ops.aten.slice.Tensor(arg15_1, 1, 46397, 46413)
        full_default_76 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_68 = torch.ops.aten.where.self(view_40, full_default_76, slice_185);  full_default_76 = slice_185 = None
        slice_186 = torch.ops.aten.slice.Tensor(arg15_1, 1, 46418, 46434)
        full_default_77 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_69 = torch.ops.aten.where.self(view_40, full_default_77, slice_186);  full_default_77 = slice_186 = None
        slice_187 = torch.ops.aten.slice.Tensor(arg15_1, 1, 46439, 46455)
        full_default_78 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_70 = torch.ops.aten.where.self(view_40, full_default_78, slice_187);  full_default_78 = slice_187 = None
        slice_188 = torch.ops.aten.slice.Tensor(arg15_1, 1, 46376, 46392)
        full_default_79 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_71 = torch.ops.aten.where.self(view_40, full_default_79, slice_188);  full_default_79 = slice_188 = None
        slice_189 = torch.ops.aten.slice.Tensor(arg15_1, 1, 46481, 46497)
        full_default_80 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_72 = torch.ops.aten.where.self(view_40, full_default_80, slice_189);  full_default_80 = slice_189 = None
        slice_190 = torch.ops.aten.slice.Tensor(arg15_1, 1, 46502, 46518)
        full_default_81 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_73 = torch.ops.aten.where.self(view_40, full_default_81, slice_190);  full_default_81 = slice_190 = None
        slice_191 = torch.ops.aten.slice.Tensor(arg15_1, 1, 46523, 46539)
        full_default_82 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_74 = torch.ops.aten.where.self(view_40, full_default_82, slice_191);  full_default_82 = slice_191 = None
        slice_192 = torch.ops.aten.slice.Tensor(arg15_1, 1, 46460, 46476)
        full_default_83 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_75 = torch.ops.aten.where.self(view_40, full_default_83, slice_192);  full_default_83 = slice_192 = None
        slice_193 = torch.ops.aten.slice.Tensor(arg15_1, 1, 59848, 59864)
        slice_194 = torch.ops.aten.slice.Tensor(arg15_1, 1, 59873, 59889)
        slice_195 = torch.ops.aten.slice.Tensor(arg15_1, 1, 62834, 62850)
        slice_196 = torch.ops.aten.slice.Tensor(arg15_1, 1, 46826, 46842)
        full_default_84 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_76 = torch.ops.aten.where.self(view_40, full_default_84, slice_196);  full_default_84 = slice_196 = None
        slice_197 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47035, 47051)
        full_default_85 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_77 = torch.ops.aten.where.self(view_40, full_default_85, slice_197);  full_default_85 = slice_197 = None
        slice_198 = torch.ops.aten.slice.Tensor(arg15_1, 1, 63083, 63099)
        slice_199 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47469, 47485)
        full_default_86 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_78 = torch.ops.aten.where.self(view_40, full_default_86, slice_199);  full_default_86 = slice_199 = None
        slice_200 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47490, 47506)
        full_default_87 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_79 = torch.ops.aten.where.self(view_40, full_default_87, slice_200);  full_default_87 = slice_200 = None
        slice_201 = torch.ops.aten.slice.Tensor(arg15_1, 1, 63685, 63701)
        slice_202 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47056, 47072)
        full_default_88 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_80 = torch.ops.aten.where.self(view_40, full_default_88, slice_202);  full_default_88 = slice_202 = None
        slice_203 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47014, 47030)
        full_default_89 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_81 = torch.ops.aten.where.self(view_40, full_default_89, slice_203);  full_default_89 = slice_203 = None
        slice_204 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47077, 47093)
        full_default_90 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_82 = torch.ops.aten.where.self(view_40, full_default_90, slice_204);  view_40 = full_default_90 = slice_204 = None
        squeeze_35 = torch.ops.aten.squeeze.default(logical_not)
        view_42 = torch.ops.aten.view.default(squeeze_35, [-1, 1]);  squeeze_35 = None
        slice_205 = torch.ops.aten.slice.Tensor(arg15_1, 1, 8732, 8748)
        full_default_91 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_83 = torch.ops.aten.where.self(view_42, full_default_91, slice_205);  view_42 = full_default_91 = slice_205 = None
        slice_206 = torch.ops.aten.slice.Tensor(arg15_1, 1, 9494, 9510)
        slice_207 = torch.ops.aten.slice.Tensor(arg15_1, 1, 7152, 7168)
        slice_208 = torch.ops.aten.slice.Tensor(arg15_1, 1, 7982, 7998)
        slice_209 = torch.ops.aten.slice.Tensor(arg15_1, 1, 10324, 10340)
        slice_210 = torch.ops.aten.slice.Tensor(arg15_1, 1, 1693, 1725)
        slice_211 = torch.ops.aten.slice.Tensor(arg15_1, 1, 1749, 1781)
        slice_212 = torch.ops.aten.slice.Tensor(arg15_1, 1, 1807, 1839)
        slice_213 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2033, 2065)
        slice_214 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3276, 3308)
        slice_215 = torch.ops.aten.slice.Tensor(arg15_1, 1, 3350, 3382)
        slice_216 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2463, 2495)
        slice_217 = torch.ops.aten.slice.Tensor(arg15_1, 1, 62858, 62890)
        slice_218 = torch.ops.aten.slice.Tensor(arg15_1, 1, 64850, 64882)
        slice_219 = torch.ops.aten.slice.Tensor(arg15_1, 1, 64901, 64933)
        slice_220 = torch.ops.aten.slice.Tensor(arg15_1, 1, 64940, 64972)
        slice_221 = torch.ops.aten.slice.Tensor(arg15_1, 1, 58205, 58221)
        slice_222 = torch.ops.aten.slice.Tensor(arg15_1, 1, 58980, 58996)
        slice_223 = torch.ops.aten.slice.Tensor(arg15_1, 1, 65045, 65061)
        slice_224 = torch.ops.aten.slice.Tensor(arg15_1, 1, 62964, 62980)
        slice_225 = torch.ops.aten.slice.Tensor(arg15_1, 1, 67651, 67667)
        slice_226 = torch.ops.aten.slice.Tensor(arg15_1, 1, 2159, 2175)
        slice_227 = torch.ops.aten.slice.Tensor(arg15_1, 1, 58744, 58760)
        slice_228 = torch.ops.aten.slice.Tensor(arg15_1, 1, 62801, 62817)
        slice_229 = torch.ops.aten.slice.Tensor(arg15_1, 1, 63099, 63115)
        slice_230 = torch.ops.aten.slice.Tensor(arg15_1, 1, 63353, 63369)
        slice_231 = torch.ops.aten.slice.Tensor(arg15_1, 1, 64802, 64818)
        slice_232 = torch.ops.aten.slice.Tensor(arg15_1, 1, 64977, 64993)
        slice_233 = torch.ops.aten.slice.Tensor(arg15_1, 1, 65816, 65832)
        slice_234 = torch.ops.aten.slice.Tensor(arg15_1, 1, 49430, 49446)
        slice_235 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47253, 47269)
        slice_236 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47213, 47229)
        slice_237 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47322, 47338)
        slice_238 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47143, 47159)
        slice_239 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47175, 47191)
        slice_240 = torch.ops.aten.slice.Tensor(arg15_1, 1, 49319, 49335)
        slice_241 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48984, 49000)
        slice_242 = torch.ops.aten.slice.Tensor(arg15_1, 1, 49024, 49040)
        slice_243 = torch.ops.aten.slice.Tensor(arg15_1, 1, 48871, 48887)
        slice_244 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43789, 43805)
        slice_245 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43373, 43389)
        slice_246 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43339, 43355)
        slice_247 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43356, 43372)
        slice_248 = torch.ops.aten.slice.Tensor(arg15_1, 1, 49616, 49632)
        slice_249 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42998, 43014)
        slice_250 = torch.ops.aten.slice.Tensor(arg15_1, 1, 49511, 49527)
        slice_251 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42943, 42959)
        slice_252 = torch.ops.aten.slice.Tensor(arg15_1, 1, 47603, 47619)
        cat_2 = torch.ops.aten.cat.default([slice_79, slice_80], -1)
        cat_3 = torch.ops.aten.cat.default([slice_14, slice_15, slice_16, slice_17, slice_18, slice_19, slice_20, slice_21, slice_22, slice_23, slice_24, slice_25, slice_26, slice_27, slice_28], -1)
        cat_4 = torch.ops.aten.cat.default([slice_79, slice_80])
        view_59 = torch.ops.aten.view.default(cat_4, [2, batch_size, 128]);  cat_4 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(view_59, [0]);  view_59 = None
        cat_5 = torch.ops.aten.cat.default([slice_14, slice_15, slice_16, slice_17, slice_18, slice_19, slice_20, slice_21, slice_22, slice_23, slice_24, slice_25, slice_26, slice_27, slice_28])
        view_60 = torch.ops.aten.view.default(cat_5, [15, batch_size, 128]);  cat_5 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(view_60, [0]);  view_60 = None
        mul_17 = torch.ops.aten.mul.Tensor(sum_2, sum_3);  sum_3 = None
        addmm_8 = torch.ops.aten.addmm.default(arg44_1, cat_2, arg43_1);  arg44_1 = cat_2 = arg43_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg46_1, cat_3, arg45_1);  arg46_1 = cat_3 = arg45_1 = None
        mul_18 = torch.ops.aten.mul.Tensor(addmm_8, addmm_9);  addmm_8 = addmm_9 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_17, [1])
        cat_6 = torch.ops.aten.cat.default([slice_210, slice_211, slice_212, slice_213, slice_214, slice_215, slice_216, slice_217, slice_218, slice_219, slice_220], -1)
        cat_7 = torch.ops.aten.cat.default([slice_55, slice_56, slice_57, slice_58, slice_59, slice_60, slice_61, slice_62, slice_63], -1)
        cat_8 = torch.ops.aten.cat.default([slice_210, slice_211, slice_212, slice_213, slice_214, slice_215, slice_216, slice_217, slice_218, slice_219, slice_220])
        view_61 = torch.ops.aten.view.default(cat_8, [11, batch_size, 32]);  cat_8 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(view_61, [0]);  view_61 = None
        cat_9 = torch.ops.aten.cat.default([slice_55, slice_56, slice_57, slice_58, slice_59, slice_60, slice_61, slice_62, slice_63])
        view_62 = torch.ops.aten.view.default(cat_9, [9, batch_size, 32]);  cat_9 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(view_62, [0]);  view_62 = None
        mul_19 = torch.ops.aten.mul.Tensor(sum_5, sum_6);  sum_6 = None
        addmm_10 = torch.ops.aten.addmm.default(arg48_1, cat_6, arg47_1);  arg48_1 = cat_6 = arg47_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg50_1, cat_7, arg49_1);  arg50_1 = cat_7 = arg49_1 = None
        mul_20 = torch.ops.aten.mul.Tensor(addmm_10, addmm_11);  addmm_10 = addmm_11 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_19, [1])
        cat_10 = torch.ops.aten.cat.default([slice_221, slice_222, slice_223], -1)
        cat_11 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13, slice_64, slice_65, slice_66], -1)
        cat_12 = torch.ops.aten.cat.default([slice_221, slice_222, slice_223])
        view_63 = torch.ops.aten.view.default(cat_12, [3, batch_size, 16]);  cat_12 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(view_63, [0]);  view_63 = None
        cat_13 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13, slice_64, slice_65, slice_66])
        view_64 = torch.ops.aten.view.default(cat_13, [12, batch_size, 16]);  cat_13 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(view_64, [0]);  view_64 = None
        mul_21 = torch.ops.aten.mul.Tensor(sum_8, sum_9);  sum_9 = None
        addmm_12 = torch.ops.aten.addmm.default(arg52_1, cat_10, arg51_1);  arg52_1 = cat_10 = arg51_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg54_1, cat_11, arg53_1);  arg54_1 = cat_11 = arg53_1 = None
        mul_22 = torch.ops.aten.mul.Tensor(addmm_12, addmm_13);  addmm_12 = addmm_13 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_21, [1])
        cat_14 = torch.ops.aten.cat.default([slice_224, slice_225], -1)
        cat_15 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13], -1)
        cat_16 = torch.ops.aten.cat.default([slice_224, slice_225])
        view_65 = torch.ops.aten.view.default(cat_16, [2, batch_size, 16]);  cat_16 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(view_65, [0]);  view_65 = None
        cat_17 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13])
        view_66 = torch.ops.aten.view.default(cat_17, [9, batch_size, 16]);  cat_17 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(view_66, [0]);  view_66 = None
        mul_23 = torch.ops.aten.mul.Tensor(sum_11, sum_12);  sum_12 = None
        addmm_14 = torch.ops.aten.addmm.default(arg56_1, cat_14, arg55_1);  arg56_1 = cat_14 = arg55_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg58_1, cat_15, arg57_1);  arg58_1 = cat_15 = arg57_1 = None
        mul_24 = torch.ops.aten.mul.Tensor(addmm_14, addmm_15);  addmm_14 = addmm_15 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_23, [1])
        cat_18 = torch.ops.aten.cat.default([slice_226, slice_227, slice_228, slice_229, slice_230, slice_231, slice_232, slice_233], -1)
        cat_19 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13, slice_67], -1)
        cat_20 = torch.ops.aten.cat.default([slice_226, slice_227, slice_228, slice_229, slice_230, slice_231, slice_232, slice_233])
        view_67 = torch.ops.aten.view.default(cat_20, [8, batch_size, 16]);  cat_20 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(view_67, [0]);  view_67 = None
        cat_21 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13, slice_67])
        view_68 = torch.ops.aten.view.default(cat_21, [10, batch_size, 16]);  cat_21 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(view_68, [0]);  view_68 = None
        mul_25 = torch.ops.aten.mul.Tensor(sum_14, sum_15);  sum_15 = None
        addmm_16 = torch.ops.aten.addmm.default(arg60_1, cat_18, arg59_1);  arg60_1 = cat_18 = arg59_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg62_1, cat_19, arg61_1);  arg62_1 = cat_19 = arg61_1 = None
        mul_26 = torch.ops.aten.mul.Tensor(addmm_16, addmm_17);  addmm_16 = addmm_17 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(mul_25, [1])
        cat_22 = torch.ops.aten.cat.default([slice_234, slice_235, slice_236, slice_237, slice_238, slice_239, slice_240, slice_241, slice_242, slice_243], -1)
        cat_23 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13, where_37], -1)
        cat_24 = torch.ops.aten.cat.default([slice_234, slice_235, slice_236, slice_237, slice_238, slice_239, slice_240, slice_241, slice_242, slice_243])
        view_69 = torch.ops.aten.view.default(cat_24, [10, batch_size, 16]);  cat_24 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(view_69, [0]);  view_69 = None
        cat_25 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13, where_37])
        view_70 = torch.ops.aten.view.default(cat_25, [10, batch_size, 16]);  cat_25 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(view_70, [0]);  view_70 = None
        mul_27 = torch.ops.aten.mul.Tensor(sum_17, sum_18);  sum_18 = None
        addmm_18 = torch.ops.aten.addmm.default(arg64_1, cat_22, arg63_1);  arg64_1 = cat_22 = arg63_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg66_1, cat_23, arg65_1);  arg66_1 = cat_23 = arg65_1 = None
        mul_28 = torch.ops.aten.mul.Tensor(addmm_18, addmm_19);  addmm_18 = addmm_19 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_27, [1])
        cat_26 = torch.ops.aten.cat.default([slice_244, slice_245, slice_246, slice_247], -1)
        cat_27 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13], -1)
        cat_28 = torch.ops.aten.cat.default([slice_244, slice_245, slice_246, slice_247])
        view_71 = torch.ops.aten.view.default(cat_28, [4, batch_size, 16]);  cat_28 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(view_71, [0]);  view_71 = None
        cat_29 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13])
        view_72 = torch.ops.aten.view.default(cat_29, [9, batch_size, 16]);  cat_29 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(view_72, [0]);  view_72 = None
        mul_29 = torch.ops.aten.mul.Tensor(sum_20, sum_21);  sum_21 = None
        addmm_20 = torch.ops.aten.addmm.default(arg68_1, cat_26, arg67_1);  arg68_1 = cat_26 = arg67_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg70_1, cat_27, arg69_1);  arg70_1 = cat_27 = arg69_1 = None
        mul_30 = torch.ops.aten.mul.Tensor(addmm_20, addmm_21);  addmm_20 = addmm_21 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(mul_29, [1])
        cat_30 = torch.ops.aten.cat.default([slice_248, slice_249, slice_250, slice_251], -1)
        cat_31 = torch.ops.aten.cat.default([slice_69, slice_70, slice_71, slice_72, slice_73, slice_74], -1)
        cat_32 = torch.ops.aten.cat.default([slice_248, slice_249, slice_250, slice_251])
        view_73 = torch.ops.aten.view.default(cat_32, [4, batch_size, 16]);  cat_32 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(view_73, [0]);  view_73 = None
        cat_33 = torch.ops.aten.cat.default([slice_69, slice_70, slice_71, slice_72, slice_73, slice_74])
        view_74 = torch.ops.aten.view.default(cat_33, [6, batch_size, 16]);  cat_33 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(view_74, [0]);  view_74 = None
        mul_31 = torch.ops.aten.mul.Tensor(sum_23, sum_24);  sum_24 = None
        addmm_22 = torch.ops.aten.addmm.default(arg72_1, cat_30, arg71_1);  arg72_1 = cat_30 = arg71_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg74_1, cat_31, arg73_1);  arg74_1 = cat_31 = arg73_1 = None
        mul_32 = torch.ops.aten.mul.Tensor(addmm_22, addmm_23);  addmm_22 = addmm_23 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_31, [1])
        clone_2 = torch.ops.aten.clone.default(slice_252)
        cat_34 = torch.ops.aten.cat.default([slice_75, slice_76, slice_77, slice_78], -1)
        clone_3 = torch.ops.aten.clone.default(slice_252)
        view_75 = torch.ops.aten.view.default(clone_3, [1, batch_size, 16]);  clone_3 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(view_75, [0]);  view_75 = None
        cat_35 = torch.ops.aten.cat.default([slice_75, slice_76, slice_77, slice_78])
        view_76 = torch.ops.aten.view.default(cat_35, [4, batch_size, 16]);  cat_35 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(view_76, [0]);  view_76 = None
        mul_33 = torch.ops.aten.mul.Tensor(sum_26, sum_27);  sum_27 = None
        addmm_24 = torch.ops.aten.addmm.default(arg76_1, clone_2, arg75_1);  arg76_1 = clone_2 = arg75_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg78_1, cat_34, arg77_1);  arg78_1 = cat_34 = arg77_1 = None
        mul_34 = torch.ops.aten.mul.Tensor(addmm_24, addmm_25);  addmm_24 = addmm_25 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(mul_33, [1])
        cat_36 = torch.ops.aten.cat.default([slice_81, slice_82, slice_83, slice_84, slice_85, slice_86, slice_87, slice_88, slice_89, slice_90, slice_91, slice_92, slice_93, slice_94, slice_95, slice_96, slice_97, slice_98, slice_99, slice_100, slice_101, slice_102, slice_103, slice_104, slice_105, slice_106, slice_107, slice_108], -1)
        cat_37 = torch.ops.aten.cat.default([slice_29, where_20, where_21, where_22, where_23, where_24, where_25, where_26, where_27, where_28, where_29, where_30], -1)
        cat_38 = torch.ops.aten.cat.default([slice_81, slice_82, slice_83, slice_84, slice_85, slice_86, slice_87, slice_88, slice_89, slice_90, slice_91, slice_92, slice_93, slice_94, slice_95, slice_96, slice_97, slice_98, slice_99, slice_100, slice_101, slice_102, slice_103, slice_104, slice_105, slice_106, slice_107, slice_108]);  slice_100 = slice_101 = slice_102 = slice_103 = slice_104 = slice_105 = slice_106 = slice_107 = slice_108 = None
        view_77 = torch.ops.aten.view.default(cat_38, [28, batch_size, 8]);  cat_38 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(view_77, [0]);  view_77 = None
        cat_39 = torch.ops.aten.cat.default([slice_29, where_20, where_21, where_22, where_23, where_24, where_25, where_26, where_27, where_28, where_29, where_30]);  where_20 = where_21 = where_22 = where_23 = where_24 = where_25 = where_26 = where_27 = where_28 = where_29 = where_30 = None
        view_78 = torch.ops.aten.view.default(cat_39, [12, batch_size, 8]);  cat_39 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(view_78, [0]);  view_78 = None
        mul_35 = torch.ops.aten.mul.Tensor(sum_29, sum_30);  sum_30 = None
        addmm_26 = torch.ops.aten.addmm.default(arg80_1, cat_36, arg79_1);  arg80_1 = cat_36 = arg79_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg82_1, cat_37, arg81_1);  arg82_1 = cat_37 = arg81_1 = None
        mul_36 = torch.ops.aten.mul.Tensor(addmm_26, addmm_27);  addmm_26 = addmm_27 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_35, [1])
        cat_40 = torch.ops.aten.cat.default([where_38, where_39, where_40, where_41, where_42, where_43, where_44, where_45, where_46, where_47, where_48, where_49, where_50, where_51, where_52, where_53, where_54, where_55, where_56, where_57, where_58, where_59, where_60, where_61, where_62, where_63, where_64, slice_136, slice_137, slice_138, slice_139, slice_140, slice_141, slice_142, slice_143, slice_144, slice_145, slice_146, slice_147, slice_148, slice_149, slice_150, slice_151, slice_152, slice_153, slice_154, slice_155, slice_156, slice_157, slice_158, slice_159, slice_160, slice_161, slice_162, slice_163, slice_164, slice_165, slice_166, slice_167, slice_168, slice_169, slice_170, slice_171, where_65, where_66, where_67, slice_175, slice_176, slice_177, slice_178, slice_179, slice_180, slice_181, slice_182, slice_183, slice_184, where_68, where_69, where_70, where_71, where_72, where_73, where_74, where_75, slice_193, slice_194, slice_195, where_76, where_77, slice_198, where_78, where_79, slice_201, where_80, where_81, where_82, where_83, slice_206, slice_207, slice_208, slice_209], -1)
        cat_41 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13, slice_41, slice_42, slice_43, slice_44, slice_45, slice_46, slice_47, slice_48, where_31, where_32, where_33, where_34, where_35, where_36], -1)
        cat_42 = torch.ops.aten.cat.default([where_38, where_39, where_40, where_41, where_42, where_43, where_44, where_45, where_46, where_47, where_48, where_49, where_50, where_51, where_52, where_53, where_54, where_55, where_56, where_57, where_58, where_59, where_60, where_61, where_62, where_63, where_64, slice_136, slice_137, slice_138, slice_139, slice_140, slice_141, slice_142, slice_143, slice_144, slice_145, slice_146, slice_147, slice_148, slice_149, slice_150, slice_151, slice_152, slice_153, slice_154, slice_155, slice_156, slice_157, slice_158, slice_159, slice_160, slice_161, slice_162, slice_163, slice_164, slice_165, slice_166, slice_167, slice_168, slice_169, slice_170, slice_171, where_65, where_66, where_67, slice_175, slice_176, slice_177, slice_178, slice_179, slice_180, slice_181, slice_182, slice_183, slice_184, where_68, where_69, where_70, where_71, where_72, where_73, where_74, where_75, slice_193, slice_194, slice_195, where_76, where_77, slice_198, where_78, where_79, slice_201, where_80, where_81, where_82, where_83, slice_206, slice_207, slice_208, slice_209]);  where_83 = slice_206 = slice_207 = slice_208 = slice_209 = None
        view_79 = torch.ops.aten.view.default(cat_42, [101, batch_size, 16]);  cat_42 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(view_79, [0]);  view_79 = None
        cat_43 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13, slice_41, slice_42, slice_43, slice_44, slice_45, slice_46, slice_47, slice_48, where_31, where_32, where_33, where_34, where_35, where_36]);  where_31 = where_32 = where_33 = where_34 = where_35 = where_36 = None
        view_80 = torch.ops.aten.view.default(cat_43, [23, batch_size, 16]);  cat_43 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(view_80, [0]);  view_80 = None
        mul_37 = torch.ops.aten.mul.Tensor(sum_32, sum_33);  sum_33 = None
        addmm_28 = torch.ops.aten.addmm.default(arg84_1, cat_40, arg83_1);  arg84_1 = cat_40 = arg83_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg86_1, cat_41, arg85_1);  arg86_1 = cat_41 = arg85_1 = None
        mul_38 = torch.ops.aten.mul.Tensor(addmm_28, addmm_29);  addmm_28 = addmm_29 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(mul_37, [1])
        cat_44 = torch.ops.aten.cat.default([mul_17, mul_19, mul_21, mul_23, mul_25, mul_37], 1)
        cat_45 = torch.ops.aten.cat.default([mul_18, mul_20, mul_22, mul_24, mul_26, mul_38], 1)
        cat_46 = torch.ops.aten.cat.default([mul_27, mul_29, mul_31, mul_33, mul_35], 1)
        cat_47 = torch.ops.aten.cat.default([mul_28, mul_30, mul_32, mul_34, mul_36], 1)
        cat_48 = torch.ops.aten.cat.default([slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13], 1)
        cat_49 = torch.ops.aten.cat.default([cat_48, slice_16, slice_17, slice_18, slice_19, slice_20, slice_21, slice_22, slice_23, slice_24, slice_25, slice_26, slice_28, slice_57, slice_58, slice_59, slice_60, slice_61, slice_62, slice_63, slice_64, slice_65, slice_66, slice_79, slice_80, slice_210, slice_211, slice_212, slice_213, slice_214, slice_215, slice_216, slice_221, slice_222, slice_224, slice_226, slice_231], 1);  cat_48 = None
        addmm_30 = torch.ops.aten.addmm.default(arg88_1, cat_49, arg87_1);  arg88_1 = arg87_1 = None
        slice_253 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43405, 43469)
        slice_254 = torch.ops.aten.slice.Tensor(arg15_1, 1, 158, 222)
        clone_4 = torch.ops.aten.clone.default(arg89_1);  arg89_1 = None
        squeeze_54 = torch.ops.aten.squeeze.default(logical_not)
        view_83 = torch.ops.aten.view.default(squeeze_54, [-1, 1]);  squeeze_54 = None
        slice_255 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43469, 43485)
        slice_256 = torch.ops.aten.slice.Tensor(arg15_1, 1, 222, 238)
        full_default_92 = torch.ops.aten.full.default([batch_size, 12], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_84 = torch.ops.aten.where.self(view_83, full_default_92, arg90_1);  full_default_92 = arg90_1 = None
        full_default_93 = torch.ops.aten.full.default([batch_size, 2], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_85 = torch.ops.aten.where.self(view_83, full_default_93, arg91_1);  full_default_93 = arg91_1 = None
        full_default_94 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_86 = torch.ops.aten.where.self(view_83, full_default_94, arg92_1);  view_83 = full_default_94 = arg92_1 = None
        cat_50 = torch.ops.aten.cat.default([arg93_1, where_84, arg94_1, where_85, arg95_1, where_86, arg96_1], -1);  arg93_1 = where_84 = arg94_1 = where_85 = arg95_1 = where_86 = arg96_1 = None
        clone_5 = torch.ops.aten.clone.default(arg97_1);  arg97_1 = None
        cat_51 = torch.ops.aten.cat.default([cat_50, clone_5], 1);  cat_50 = None
        cat_52 = torch.ops.aten.cat.default([slice_255, slice_256], 1)
        addmm_31 = torch.ops.aten.addmm.default(arg99_1, clone_4, arg98_1);  arg99_1 = arg98_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg101_1, cat_51, arg100_1);  arg101_1 = arg100_1 = None
        squeeze_57 = torch.ops.aten.squeeze.default(logical_not)
        view_86 = torch.ops.aten.view.default(squeeze_57, [-1, 1]);  squeeze_57 = None
        squeeze_58 = torch.ops.aten.squeeze.default(logical_not_2)
        view_87 = torch.ops.aten.view.default(squeeze_58, [-1, 1]);  squeeze_58 = None
        full_default_95 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_87 = torch.ops.aten.where.self(view_86, full_default_95, arg102_1);  full_default_95 = arg102_1 = None
        full_default_96 = torch.ops.aten.full.default([batch_size, 12], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_88 = torch.ops.aten.where.self(view_86, full_default_96, arg103_1);  full_default_96 = arg103_1 = None
        full_default_97 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_89 = torch.ops.aten.where.self(view_86, full_default_97, arg104_1);  full_default_97 = arg104_1 = None
        full_default_98 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_90 = torch.ops.aten.where.self(view_86, full_default_98, arg105_1);  full_default_98 = arg105_1 = None
        full_default_99 = torch.ops.aten.full.default([batch_size, 44], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_91 = torch.ops.aten.where.self(view_87, full_default_99, arg106_1);  full_default_99 = arg106_1 = None
        full_default_100 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_92 = torch.ops.aten.where.self(view_87, full_default_100, arg107_1);  full_default_100 = arg107_1 = None
        full_default_101 = torch.ops.aten.full.default([batch_size, 12], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_93 = torch.ops.aten.where.self(view_87, full_default_101, arg108_1);  view_87 = full_default_101 = arg108_1 = None
        full_default_102 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_94 = torch.ops.aten.where.self(view_86, full_default_102, arg109_1);  full_default_102 = arg109_1 = None
        full_default_103 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_95 = torch.ops.aten.where.self(view_86, full_default_103, arg110_1);  full_default_103 = arg110_1 = None
        full_default_104 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_96 = torch.ops.aten.where.self(view_86, full_default_104, arg111_1);  full_default_104 = arg111_1 = None
        full_default_105 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_97 = torch.ops.aten.where.self(view_86, full_default_105, arg112_1);  full_default_105 = arg112_1 = None
        full_default_106 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_98 = torch.ops.aten.where.self(view_86, full_default_106, arg113_1);  full_default_106 = arg113_1 = None
        full_default_107 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_99 = torch.ops.aten.where.self(view_86, full_default_107, arg114_1);  full_default_107 = arg114_1 = None
        full_default_108 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_100 = torch.ops.aten.where.self(view_86, full_default_108, arg115_1);  full_default_108 = arg115_1 = None
        full_default_109 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_101 = torch.ops.aten.where.self(view_86, full_default_109, arg116_1);  view_86 = full_default_109 = arg116_1 = None
        cat_53 = torch.ops.aten.cat.default([where_87, arg117_1, where_88, arg118_1, where_89, arg119_1, where_90, arg120_1, where_91, arg121_1, where_92, arg122_1, where_93, arg123_1, where_94, arg124_1, where_95, arg125_1, where_96, arg126_1, where_97, arg127_1, where_98, arg128_1, where_99, arg129_1, where_100, arg130_1, where_101], -1);  where_87 = arg117_1 = where_88 = arg118_1 = where_89 = arg119_1 = where_90 = arg120_1 = where_91 = arg121_1 = where_92 = arg122_1 = where_93 = arg123_1 = where_94 = arg124_1 = where_95 = arg125_1 = where_96 = arg126_1 = where_97 = arg127_1 = where_98 = arg128_1 = where_99 = arg129_1 = where_100 = arg130_1 = where_101 = None
        clone_6 = torch.ops.aten.clone.default(arg131_1);  arg131_1 = None
        cat_54 = torch.ops.aten.cat.default([clone_6, cat_46], 1)
        full_default_110 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        eq_30 = torch.ops.aten.eq.Scalar(where_7, 1)
        full_default_111 = torch.ops.aten.full.default([batch_size, 1], 1, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_102 = torch.ops.aten.where.self(eq_30, full_default_111, full_default_110);  eq_30 = full_default_111 = full_default_110 = None
        eq_31 = torch.ops.aten.eq.Scalar(where_7, 2)
        full_default_112 = torch.ops.aten.full.default([batch_size, 1], 2, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_103 = torch.ops.aten.where.self(eq_31, full_default_112, where_102);  eq_31 = full_default_112 = where_102 = None
        eq_32 = torch.ops.aten.eq.Scalar(where_7, 3)
        full_default_113 = torch.ops.aten.full.default([batch_size, 1], 3, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_104 = torch.ops.aten.where.self(eq_32, full_default_113, where_103);  eq_32 = full_default_113 = where_103 = None
        eq_33 = torch.ops.aten.eq.Scalar(where_7, 9998)
        full_default_114 = torch.ops.aten.full.default([batch_size, 1], 4, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_105 = torch.ops.aten.where.self(eq_33, full_default_114, where_104);  eq_33 = full_default_114 = where_104 = None
        eq_34 = torch.ops.aten.eq.Scalar(where_7, 9999)
        full_default_115 = torch.ops.aten.full.default([batch_size, 1], 5, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_106 = torch.ops.aten.where.self(eq_34, full_default_115, where_105);  eq_34 = full_default_115 = where_105 = None
        embedding = torch.ops.aten.embedding.default(arg132_1, where_106);  arg132_1 = where_106 = None
        squeeze_61 = torch.ops.aten.squeeze.dim(embedding, 1);  embedding = None
        slice_257 = torch.ops.aten.slice.Tensor(arg15_1, 1, 238, 366)
        squeeze_64 = torch.ops.aten.squeeze.default(logical_not)
        view_92 = torch.ops.aten.view.default(squeeze_64, [-1, 1]);  squeeze_64 = None
        full_default_116 = torch.ops.aten.full.default([batch_size, 640], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_107 = torch.ops.aten.where.self(view_92, full_default_116, arg134_1);  view_92 = full_default_116 = arg134_1 = None
        cat_55 = torch.ops.aten.cat.default([squeeze_61, arg133_1, where_107], 1);  squeeze_61 = arg133_1 = where_107 = None
        addmm_33 = torch.ops.aten.addmm.default(arg136_1, cat_55, arg135_1);  arg136_1 = arg135_1 = None
        full_default_117 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        eq_35 = torch.ops.aten.eq.Scalar(where_7, 1)
        full_default_118 = torch.ops.aten.full.default([batch_size, 1], 1, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_108 = torch.ops.aten.where.self(eq_35, full_default_118, full_default_117);  eq_35 = full_default_118 = full_default_117 = None
        eq_36 = torch.ops.aten.eq.Scalar(where_7, 2)
        full_default_119 = torch.ops.aten.full.default([batch_size, 1], 2, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_109 = torch.ops.aten.where.self(eq_36, full_default_119, where_108);  eq_36 = full_default_119 = where_108 = None
        eq_37 = torch.ops.aten.eq.Scalar(where_7, 3)
        full_default_120 = torch.ops.aten.full.default([batch_size, 1], 3, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_110 = torch.ops.aten.where.self(eq_37, full_default_120, where_109);  eq_37 = full_default_120 = where_109 = None
        eq_38 = torch.ops.aten.eq.Scalar(where_7, 9998)
        full_default_121 = torch.ops.aten.full.default([batch_size, 1], 4, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_111 = torch.ops.aten.where.self(eq_38, full_default_121, where_110);  eq_38 = full_default_121 = where_110 = None
        eq_39 = torch.ops.aten.eq.Scalar(where_7, 9999)
        full_default_122 = torch.ops.aten.full.default([batch_size, 1], 5, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_112 = torch.ops.aten.where.self(eq_39, full_default_122, where_111);  eq_39 = full_default_122 = where_111 = None
        embedding_1 = torch.ops.aten.embedding.default(arg137_1, where_112);  arg137_1 = where_112 = None
        squeeze_65 = torch.ops.aten.squeeze.dim(embedding_1, 1);  embedding_1 = None
        squeeze_66 = torch.ops.aten.squeeze.default(logical_not)
        view_93 = torch.ops.aten.view.default(squeeze_66, [-1, 1]);  squeeze_66 = None
        slice_258 = torch.ops.aten.slice.Tensor(arg15_1, 1, 366, 430)
        full_default_123 = torch.ops.aten.full.default([batch_size, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_113 = torch.ops.aten.where.self(view_93, full_default_123, arg138_1);  view_93 = full_default_123 = arg138_1 = None
        cat_56 = torch.ops.aten.cat.default([arg139_1, where_113, arg140_1], -1);  arg139_1 = where_113 = arg140_1 = None
        squeeze_68 = torch.ops.aten.squeeze.default(logical_not)
        view_95 = torch.ops.aten.view.default(squeeze_68, [-1, 1]);  squeeze_68 = None
        full_default_124 = torch.ops.aten.full.default([batch_size, 320], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_114 = torch.ops.aten.where.self(view_95, full_default_124, arg141_1);  view_95 = full_default_124 = arg141_1 = None
        cat_57 = torch.ops.aten.cat.default([squeeze_65, cat_56, where_114], 1);  squeeze_65 = cat_56 = where_114 = None
        cat_58 = torch.ops.aten.cat.default([addmm_3, arg142_1, sum_2, sum_5, sum_8, sum_11, sum_14, sum_17, sum_20, sum_23, sum_26, sum_29, sum_32], 1);  addmm_3 = arg142_1 = sum_2 = sum_5 = sum_8 = sum_11 = sum_14 = sum_17 = sum_20 = sum_23 = sum_26 = sum_29 = sum_32 = None
        addmm_34 = torch.ops.aten.addmm.default(arg144_1, cat_58, arg143_1);  arg144_1 = arg143_1 = None
        relu_4 = torch.ops.aten.relu.default(addmm_34);  addmm_34 = None
        addmm_35 = torch.ops.aten.addmm.default(arg146_1, relu_4, arg145_1);  arg146_1 = relu_4 = arg145_1 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(arg147_1, torch.int32)
        slice_259 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32621, 32653)
        slice_260 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34906, 34922)
        slice_261 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35278, 35294)
        slice_262 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34264, 34296)
        slice_263 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35650, 35666)
        slice_264 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33188, 33204)
        slice_265 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33674, 33706)
        slice_266 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36680, 36696)
        slice_267 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38912, 38944)
        slice_268 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39529, 39561)
        slice_269 = torch.ops.aten.slice.Tensor(arg15_1, 1, 40146, 40178)
        slice_270 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36981, 37013)
        slice_271 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37662, 37694)
        slice_272 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36108, 36140)
        slice_273 = torch.ops.aten.slice.Tensor(arg15_1, 1, 430, 462)
        slice_274 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38151, 38183)
        squeeze_74 = torch.ops.aten.squeeze.default(logical_not)
        view_101 = torch.ops.aten.view.default(squeeze_74, [-1, 1]);  squeeze_74 = None
        slice_275 = torch.ops.aten.slice.Tensor(arg15_1, 1, 22814, 22846)
        full_default_125 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_115 = torch.ops.aten.where.self(view_101, full_default_125, slice_275);  view_101 = full_default_125 = slice_275 = None
        slice_276 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27692, 27724)
        squeeze_76 = torch.ops.aten.squeeze.default(logical_not)
        view_103 = torch.ops.aten.view.default(squeeze_76, [-1, 1]);  squeeze_76 = None
        slice_277 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27051, 27083)
        full_default_126 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_116 = torch.ops.aten.where.self(view_103, full_default_126, slice_277);  full_default_126 = slice_277 = None
        slice_278 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23836, 23868)
        full_default_127 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_117 = torch.ops.aten.where.self(view_103, full_default_127, slice_278);  full_default_127 = slice_278 = None
        slice_279 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24682, 24714)
        full_default_128 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_118 = torch.ops.aten.where.self(view_103, full_default_128, slice_279);  full_default_128 = slice_279 = None
        slice_280 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25512, 25544)
        full_default_129 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_119 = torch.ops.aten.where.self(view_103, full_default_129, slice_280);  full_default_129 = slice_280 = None
        slice_281 = torch.ops.aten.slice.Tensor(arg15_1, 1, 26262, 26294)
        full_default_130 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_120 = torch.ops.aten.where.self(view_103, full_default_130, slice_281);  view_103 = full_default_130 = slice_281 = None
        cat_59 = torch.ops.aten.cat.default([addmm_35, slice_259, slice_260, slice_261, slice_262, slice_263, slice_264, slice_265, slice_266, slice_267, slice_268, slice_269, slice_270, slice_271, slice_272, slice_273, slice_274], 1);  slice_274 = None
        cat_60 = torch.ops.aten.cat.default([addmm_35, where_115, slice_276, where_116, where_117, where_118, where_119, where_120], 1);  where_116 = where_117 = where_118 = where_119 = where_120 = None
        addmm_36 = torch.ops.aten.addmm.default(arg149_1, cat_59, arg148_1);  arg149_1 = arg148_1 = None
        relu_5 = torch.ops.aten.relu.default(addmm_36);  addmm_36 = None
        addmm_37 = torch.ops.aten.addmm.default(arg151_1, relu_5, arg150_1);  arg151_1 = relu_5 = arg150_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg153_1, cat_60, arg152_1);  arg153_1 = arg152_1 = None
        relu_6 = torch.ops.aten.relu.default(addmm_38);  addmm_38 = None
        addmm_39 = torch.ops.aten.addmm.default(arg155_1, relu_6, arg154_1);  arg155_1 = relu_6 = arg154_1 = None
        eq_40 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_41 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_18 = torch.ops.aten.logical_or.default(eq_40, eq_41);  eq_40 = eq_41 = None
        repeat = torch.ops.aten.repeat.default(logical_or_18, [1, 32]);  logical_or_18 = None
        where_121 = torch.ops.aten.where.self(repeat, addmm_39, addmm_37);  repeat = addmm_39 = addmm_37 = None
        slice_282 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32653, 32685)
        slice_283 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34922, 34938)
        slice_284 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35294, 35310)
        slice_285 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34296, 34328)
        slice_286 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35666, 35682)
        slice_287 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33204, 33220)
        slice_288 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33706, 33738)
        slice_289 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36696, 36712)
        slice_290 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38944, 38976)
        slice_291 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39561, 39593)
        slice_292 = torch.ops.aten.slice.Tensor(arg15_1, 1, 40178, 40210)
        slice_293 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37013, 37045)
        slice_294 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37694, 37726)
        slice_295 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36140, 36172)
        slice_296 = torch.ops.aten.slice.Tensor(arg15_1, 1, 462, 494)
        slice_297 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38183, 38215)
        squeeze_80 = torch.ops.aten.squeeze.default(logical_not)
        view_107 = torch.ops.aten.view.default(squeeze_80, [-1, 1]);  squeeze_80 = None
        slice_298 = torch.ops.aten.slice.Tensor(arg15_1, 1, 22846, 22878)
        full_default_131 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_122 = torch.ops.aten.where.self(view_107, full_default_131, slice_298);  view_107 = full_default_131 = slice_298 = None
        slice_299 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27724, 27756)
        squeeze_82 = torch.ops.aten.squeeze.default(logical_not)
        view_109 = torch.ops.aten.view.default(squeeze_82, [-1, 1]);  squeeze_82 = None
        slice_300 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27083, 27115)
        full_default_132 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_123 = torch.ops.aten.where.self(view_109, full_default_132, slice_300);  full_default_132 = slice_300 = None
        slice_301 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23868, 23900)
        full_default_133 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_124 = torch.ops.aten.where.self(view_109, full_default_133, slice_301);  full_default_133 = slice_301 = None
        slice_302 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24714, 24746)
        full_default_134 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_125 = torch.ops.aten.where.self(view_109, full_default_134, slice_302);  full_default_134 = slice_302 = None
        slice_303 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25544, 25576)
        full_default_135 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_126 = torch.ops.aten.where.self(view_109, full_default_135, slice_303);  full_default_135 = slice_303 = None
        slice_304 = torch.ops.aten.slice.Tensor(arg15_1, 1, 26294, 26326)
        full_default_136 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_127 = torch.ops.aten.where.self(view_109, full_default_136, slice_304);  view_109 = full_default_136 = slice_304 = None
        cat_61 = torch.ops.aten.cat.default([addmm_35, slice_282, slice_283, slice_284, slice_285, slice_286, slice_287, slice_288, slice_289, slice_290, slice_291, slice_292, slice_293, slice_294, slice_295, slice_296, slice_297], 1);  slice_297 = None
        cat_62 = torch.ops.aten.cat.default([addmm_35, where_122, slice_299, where_123, where_124, where_125, where_126, where_127], 1);  where_123 = where_124 = where_125 = where_126 = where_127 = None
        addmm_40 = torch.ops.aten.addmm.default(arg157_1, cat_61, arg156_1);  arg157_1 = arg156_1 = None
        relu_7 = torch.ops.aten.relu.default(addmm_40);  addmm_40 = None
        addmm_41 = torch.ops.aten.addmm.default(arg159_1, relu_7, arg158_1);  arg159_1 = relu_7 = arg158_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg161_1, cat_62, arg160_1);  arg161_1 = arg160_1 = None
        relu_8 = torch.ops.aten.relu.default(addmm_42);  addmm_42 = None
        addmm_43 = torch.ops.aten.addmm.default(arg163_1, relu_8, arg162_1);  arg163_1 = relu_8 = arg162_1 = None
        eq_42 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_43 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_19 = torch.ops.aten.logical_or.default(eq_42, eq_43);  eq_42 = eq_43 = None
        repeat_1 = torch.ops.aten.repeat.default(logical_or_19, [1, 32]);  logical_or_19 = None
        where_128 = torch.ops.aten.where.self(repeat_1, addmm_43, addmm_41);  repeat_1 = addmm_43 = addmm_41 = None
        slice_305 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32685, 32717)
        slice_306 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34938, 34954)
        slice_307 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35310, 35326)
        slice_308 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34328, 34360)
        slice_309 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35682, 35698)
        slice_310 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33220, 33236)
        slice_311 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33738, 33770)
        slice_312 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36712, 36728)
        slice_313 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38976, 39008)
        slice_314 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39593, 39625)
        slice_315 = torch.ops.aten.slice.Tensor(arg15_1, 1, 40210, 40242)
        slice_316 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37045, 37077)
        slice_317 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37726, 37758)
        slice_318 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36172, 36204)
        slice_319 = torch.ops.aten.slice.Tensor(arg15_1, 1, 494, 526)
        slice_320 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38215, 38247)
        squeeze_86 = torch.ops.aten.squeeze.default(logical_not)
        view_113 = torch.ops.aten.view.default(squeeze_86, [-1, 1]);  squeeze_86 = None
        slice_321 = torch.ops.aten.slice.Tensor(arg15_1, 1, 22878, 22910)
        full_default_137 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_129 = torch.ops.aten.where.self(view_113, full_default_137, slice_321);  view_113 = full_default_137 = slice_321 = None
        slice_322 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27756, 27788)
        squeeze_88 = torch.ops.aten.squeeze.default(logical_not)
        view_115 = torch.ops.aten.view.default(squeeze_88, [-1, 1]);  squeeze_88 = None
        slice_323 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27115, 27147)
        full_default_138 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_130 = torch.ops.aten.where.self(view_115, full_default_138, slice_323);  full_default_138 = slice_323 = None
        slice_324 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23900, 23932)
        full_default_139 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_131 = torch.ops.aten.where.self(view_115, full_default_139, slice_324);  full_default_139 = slice_324 = None
        slice_325 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24746, 24778)
        full_default_140 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_132 = torch.ops.aten.where.self(view_115, full_default_140, slice_325);  full_default_140 = slice_325 = None
        slice_326 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25576, 25608)
        full_default_141 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_133 = torch.ops.aten.where.self(view_115, full_default_141, slice_326);  full_default_141 = slice_326 = None
        slice_327 = torch.ops.aten.slice.Tensor(arg15_1, 1, 26326, 26358)
        full_default_142 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_134 = torch.ops.aten.where.self(view_115, full_default_142, slice_327);  view_115 = full_default_142 = slice_327 = None
        cat_63 = torch.ops.aten.cat.default([addmm_35, slice_305, slice_306, slice_307, slice_308, slice_309, slice_310, slice_311, slice_312, slice_313, slice_314, slice_315, slice_316, slice_317, slice_318, slice_319, slice_320], 1);  slice_320 = None
        cat_64 = torch.ops.aten.cat.default([addmm_35, where_129, slice_322, where_130, where_131, where_132, where_133, where_134], 1);  where_130 = where_131 = where_132 = where_133 = where_134 = None
        addmm_44 = torch.ops.aten.addmm.default(arg165_1, cat_63, arg164_1);  arg165_1 = arg164_1 = None
        relu_9 = torch.ops.aten.relu.default(addmm_44);  addmm_44 = None
        addmm_45 = torch.ops.aten.addmm.default(arg167_1, relu_9, arg166_1);  arg167_1 = relu_9 = arg166_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg169_1, cat_64, arg168_1);  arg169_1 = arg168_1 = None
        relu_10 = torch.ops.aten.relu.default(addmm_46);  addmm_46 = None
        addmm_47 = torch.ops.aten.addmm.default(arg171_1, relu_10, arg170_1);  arg171_1 = relu_10 = arg170_1 = None
        eq_44 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_45 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_20 = torch.ops.aten.logical_or.default(eq_44, eq_45);  eq_44 = eq_45 = None
        repeat_2 = torch.ops.aten.repeat.default(logical_or_20, [1, 32]);  logical_or_20 = None
        where_135 = torch.ops.aten.where.self(repeat_2, addmm_47, addmm_45);  repeat_2 = addmm_47 = addmm_45 = None
        slice_328 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32717, 32749)
        slice_329 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34954, 34970)
        slice_330 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35326, 35342)
        slice_331 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34360, 34392)
        slice_332 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35698, 35714)
        slice_333 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33236, 33252)
        slice_334 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33770, 33802)
        slice_335 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36728, 36744)
        slice_336 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39008, 39040)
        slice_337 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39625, 39657)
        slice_338 = torch.ops.aten.slice.Tensor(arg15_1, 1, 40242, 40274)
        slice_339 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37077, 37109)
        slice_340 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37758, 37790)
        slice_341 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36204, 36236)
        slice_342 = torch.ops.aten.slice.Tensor(arg15_1, 1, 526, 558)
        slice_343 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38247, 38279)
        squeeze_92 = torch.ops.aten.squeeze.default(logical_not)
        view_119 = torch.ops.aten.view.default(squeeze_92, [-1, 1]);  squeeze_92 = None
        slice_344 = torch.ops.aten.slice.Tensor(arg15_1, 1, 22910, 22942)
        full_default_143 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_136 = torch.ops.aten.where.self(view_119, full_default_143, slice_344);  view_119 = full_default_143 = slice_344 = None
        slice_345 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27788, 27820)
        squeeze_94 = torch.ops.aten.squeeze.default(logical_not)
        view_121 = torch.ops.aten.view.default(squeeze_94, [-1, 1]);  squeeze_94 = None
        slice_346 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27147, 27179)
        full_default_144 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_137 = torch.ops.aten.where.self(view_121, full_default_144, slice_346);  full_default_144 = slice_346 = None
        slice_347 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23932, 23964)
        full_default_145 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_138 = torch.ops.aten.where.self(view_121, full_default_145, slice_347);  full_default_145 = slice_347 = None
        slice_348 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24778, 24810)
        full_default_146 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_139 = torch.ops.aten.where.self(view_121, full_default_146, slice_348);  full_default_146 = slice_348 = None
        slice_349 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25608, 25640)
        full_default_147 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_140 = torch.ops.aten.where.self(view_121, full_default_147, slice_349);  full_default_147 = slice_349 = None
        slice_350 = torch.ops.aten.slice.Tensor(arg15_1, 1, 26358, 26390)
        full_default_148 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_141 = torch.ops.aten.where.self(view_121, full_default_148, slice_350);  view_121 = full_default_148 = slice_350 = None
        cat_65 = torch.ops.aten.cat.default([addmm_35, slice_328, slice_329, slice_330, slice_331, slice_332, slice_333, slice_334, slice_335, slice_336, slice_337, slice_338, slice_339, slice_340, slice_341, slice_342, slice_343], 1);  slice_343 = None
        cat_66 = torch.ops.aten.cat.default([addmm_35, where_136, slice_345, where_137, where_138, where_139, where_140, where_141], 1);  where_137 = where_138 = where_139 = where_140 = where_141 = None
        addmm_48 = torch.ops.aten.addmm.default(arg173_1, cat_65, arg172_1);  arg173_1 = arg172_1 = None
        relu_11 = torch.ops.aten.relu.default(addmm_48);  addmm_48 = None
        addmm_49 = torch.ops.aten.addmm.default(arg175_1, relu_11, arg174_1);  arg175_1 = relu_11 = arg174_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg177_1, cat_66, arg176_1);  arg177_1 = arg176_1 = None
        relu_12 = torch.ops.aten.relu.default(addmm_50);  addmm_50 = None
        addmm_51 = torch.ops.aten.addmm.default(arg179_1, relu_12, arg178_1);  arg179_1 = relu_12 = arg178_1 = None
        eq_46 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_47 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_21 = torch.ops.aten.logical_or.default(eq_46, eq_47);  eq_46 = eq_47 = None
        repeat_3 = torch.ops.aten.repeat.default(logical_or_21, [1, 32]);  logical_or_21 = None
        where_142 = torch.ops.aten.where.self(repeat_3, addmm_51, addmm_49);  repeat_3 = addmm_51 = addmm_49 = None
        slice_351 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32749, 32781)
        slice_352 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34970, 34986)
        slice_353 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35342, 35358)
        slice_354 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34392, 34424)
        slice_355 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35714, 35730)
        slice_356 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33252, 33268)
        slice_357 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33802, 33834)
        slice_358 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36744, 36760)
        slice_359 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39040, 39072)
        slice_360 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39657, 39689)
        slice_361 = torch.ops.aten.slice.Tensor(arg15_1, 1, 40274, 40306)
        slice_362 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37109, 37141)
        slice_363 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37790, 37822)
        slice_364 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36236, 36268)
        slice_365 = torch.ops.aten.slice.Tensor(arg15_1, 1, 558, 590)
        slice_366 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38279, 38311)
        squeeze_98 = torch.ops.aten.squeeze.default(logical_not)
        view_125 = torch.ops.aten.view.default(squeeze_98, [-1, 1]);  squeeze_98 = None
        slice_367 = torch.ops.aten.slice.Tensor(arg15_1, 1, 22942, 22974)
        full_default_149 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_143 = torch.ops.aten.where.self(view_125, full_default_149, slice_367);  view_125 = full_default_149 = slice_367 = None
        slice_368 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27820, 27852)
        squeeze_100 = torch.ops.aten.squeeze.default(logical_not)
        view_127 = torch.ops.aten.view.default(squeeze_100, [-1, 1]);  squeeze_100 = None
        slice_369 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27179, 27211)
        full_default_150 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_144 = torch.ops.aten.where.self(view_127, full_default_150, slice_369);  full_default_150 = slice_369 = None
        slice_370 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23964, 23996)
        full_default_151 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_145 = torch.ops.aten.where.self(view_127, full_default_151, slice_370);  full_default_151 = slice_370 = None
        slice_371 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24810, 24842)
        full_default_152 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_146 = torch.ops.aten.where.self(view_127, full_default_152, slice_371);  full_default_152 = slice_371 = None
        slice_372 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25640, 25672)
        full_default_153 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_147 = torch.ops.aten.where.self(view_127, full_default_153, slice_372);  full_default_153 = slice_372 = None
        slice_373 = torch.ops.aten.slice.Tensor(arg15_1, 1, 26390, 26422)
        full_default_154 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_148 = torch.ops.aten.where.self(view_127, full_default_154, slice_373);  view_127 = full_default_154 = slice_373 = None
        cat_67 = torch.ops.aten.cat.default([addmm_35, slice_351, slice_352, slice_353, slice_354, slice_355, slice_356, slice_357, slice_358, slice_359, slice_360, slice_361, slice_362, slice_363, slice_364, slice_365, slice_366], 1);  slice_366 = None
        cat_68 = torch.ops.aten.cat.default([addmm_35, where_143, slice_368, where_144, where_145, where_146, where_147, where_148], 1);  where_144 = where_145 = where_146 = where_147 = where_148 = None
        addmm_52 = torch.ops.aten.addmm.default(arg181_1, cat_67, arg180_1);  arg181_1 = cat_67 = arg180_1 = None
        relu_13 = torch.ops.aten.relu.default(addmm_52);  addmm_52 = None
        addmm_53 = torch.ops.aten.addmm.default(arg183_1, relu_13, arg182_1);  arg183_1 = relu_13 = arg182_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg185_1, cat_68, arg184_1);  arg185_1 = cat_68 = arg184_1 = None
        relu_14 = torch.ops.aten.relu.default(addmm_54);  addmm_54 = None
        addmm_55 = torch.ops.aten.addmm.default(arg187_1, relu_14, arg186_1);  arg187_1 = relu_14 = arg186_1 = None
        eq_48 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_49 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_22 = torch.ops.aten.logical_or.default(eq_48, eq_49);  eq_48 = eq_49 = None
        repeat_4 = torch.ops.aten.repeat.default(logical_or_22, [1, 32]);  logical_or_22 = None
        where_149 = torch.ops.aten.where.self(repeat_4, addmm_55, addmm_53);  repeat_4 = addmm_55 = addmm_53 = None
        slice_374 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32781, 32813)
        slice_375 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34986, 35002)
        slice_376 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35358, 35374)
        slice_377 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34424, 34456)
        slice_378 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35730, 35746)
        slice_379 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33268, 33284)
        slice_380 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33834, 33866)
        slice_381 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36760, 36776)
        slice_382 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39072, 39104)
        slice_383 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39689, 39721)
        slice_384 = torch.ops.aten.slice.Tensor(arg15_1, 1, 40306, 40338)
        slice_385 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37141, 37173)
        slice_386 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37822, 37854)
        slice_387 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36268, 36300)
        slice_388 = torch.ops.aten.slice.Tensor(arg15_1, 1, 590, 622)
        slice_389 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38311, 38343)
        squeeze_104 = torch.ops.aten.squeeze.default(logical_not)
        view_131 = torch.ops.aten.view.default(squeeze_104, [-1, 1]);  squeeze_104 = None
        slice_390 = torch.ops.aten.slice.Tensor(arg15_1, 1, 22974, 23006)
        full_default_155 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_150 = torch.ops.aten.where.self(view_131, full_default_155, slice_390);  view_131 = full_default_155 = slice_390 = None
        slice_391 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27852, 27884)
        squeeze_106 = torch.ops.aten.squeeze.default(logical_not)
        view_133 = torch.ops.aten.view.default(squeeze_106, [-1, 1]);  squeeze_106 = None
        slice_392 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27211, 27243)
        full_default_156 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_151 = torch.ops.aten.where.self(view_133, full_default_156, slice_392);  full_default_156 = slice_392 = None
        slice_393 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23996, 24028)
        full_default_157 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_152 = torch.ops.aten.where.self(view_133, full_default_157, slice_393);  full_default_157 = slice_393 = None
        slice_394 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24842, 24874)
        full_default_158 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_153 = torch.ops.aten.where.self(view_133, full_default_158, slice_394);  full_default_158 = slice_394 = None
        slice_395 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25672, 25704)
        full_default_159 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_154 = torch.ops.aten.where.self(view_133, full_default_159, slice_395);  full_default_159 = slice_395 = None
        slice_396 = torch.ops.aten.slice.Tensor(arg15_1, 1, 26422, 26454)
        full_default_160 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_155 = torch.ops.aten.where.self(view_133, full_default_160, slice_396);  view_133 = full_default_160 = slice_396 = None
        cat_69 = torch.ops.aten.cat.default([addmm_35, slice_374, slice_375, slice_376, slice_377, slice_378, slice_379, slice_380, slice_381, slice_382, slice_383, slice_384, slice_385, slice_386, slice_387, slice_388, slice_389], 1);  slice_389 = None
        cat_70 = torch.ops.aten.cat.default([addmm_35, where_150, slice_391, where_151, where_152, where_153, where_154, where_155], 1);  where_151 = where_152 = where_153 = where_154 = where_155 = None
        addmm_56 = torch.ops.aten.addmm.default(arg189_1, cat_69, arg188_1);  arg189_1 = arg188_1 = None
        relu_15 = torch.ops.aten.relu.default(addmm_56);  addmm_56 = None
        addmm_57 = torch.ops.aten.addmm.default(arg191_1, relu_15, arg190_1);  arg191_1 = relu_15 = arg190_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg193_1, cat_70, arg192_1);  arg193_1 = arg192_1 = None
        relu_16 = torch.ops.aten.relu.default(addmm_58);  addmm_58 = None
        addmm_59 = torch.ops.aten.addmm.default(arg195_1, relu_16, arg194_1);  arg195_1 = relu_16 = arg194_1 = None
        eq_50 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_51 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_23 = torch.ops.aten.logical_or.default(eq_50, eq_51);  eq_50 = eq_51 = None
        repeat_5 = torch.ops.aten.repeat.default(logical_or_23, [1, 32]);  logical_or_23 = None
        where_156 = torch.ops.aten.where.self(repeat_5, addmm_59, addmm_57);  repeat_5 = addmm_59 = addmm_57 = None
        slice_397 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32813, 32877)
        slice_398 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35002, 35034)
        slice_399 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35374, 35406)
        slice_400 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34456, 34520)
        slice_401 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35746, 35778)
        slice_402 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33284, 33316)
        slice_403 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33866, 33930)
        slice_404 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36776, 36808)
        slice_405 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39104, 39168)
        slice_406 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39721, 39785)
        slice_407 = torch.ops.aten.slice.Tensor(arg15_1, 1, 40338, 40402)
        slice_408 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37173, 37237)
        slice_409 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37854, 37918)
        slice_410 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36300, 36364)
        slice_411 = torch.ops.aten.slice.Tensor(arg15_1, 1, 622, 686)
        slice_412 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38343, 38407)
        squeeze_110 = torch.ops.aten.squeeze.default(logical_not)
        view_137 = torch.ops.aten.view.default(squeeze_110, [-1, 1]);  squeeze_110 = None
        slice_413 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23006, 23070)
        full_default_161 = torch.ops.aten.full.default([batch_size, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_157 = torch.ops.aten.where.self(view_137, full_default_161, slice_413);  view_137 = full_default_161 = slice_413 = None
        slice_414 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27884, 27948)
        squeeze_112 = torch.ops.aten.squeeze.default(logical_not)
        view_139 = torch.ops.aten.view.default(squeeze_112, [-1, 1]);  squeeze_112 = None
        slice_415 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27243, 27307)
        full_default_162 = torch.ops.aten.full.default([batch_size, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_158 = torch.ops.aten.where.self(view_139, full_default_162, slice_415);  full_default_162 = slice_415 = None
        slice_416 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24028, 24092)
        full_default_163 = torch.ops.aten.full.default([batch_size, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_159 = torch.ops.aten.where.self(view_139, full_default_163, slice_416);  full_default_163 = slice_416 = None
        slice_417 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24874, 24938)
        full_default_164 = torch.ops.aten.full.default([batch_size, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_160 = torch.ops.aten.where.self(view_139, full_default_164, slice_417);  full_default_164 = slice_417 = None
        slice_418 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25704, 25768)
        full_default_165 = torch.ops.aten.full.default([batch_size, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_161 = torch.ops.aten.where.self(view_139, full_default_165, slice_418);  full_default_165 = slice_418 = None
        slice_419 = torch.ops.aten.slice.Tensor(arg15_1, 1, 26454, 26518)
        full_default_166 = torch.ops.aten.full.default([batch_size, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_162 = torch.ops.aten.where.self(view_139, full_default_166, slice_419);  view_139 = full_default_166 = slice_419 = None
        cat_71 = torch.ops.aten.cat.default([addmm_35, slice_397, slice_398, slice_399, slice_400, slice_401, slice_402, slice_403, slice_404, slice_405, slice_406, slice_407, slice_408, slice_409, slice_410, slice_411, slice_412], 1);  slice_412 = None
        cat_72 = torch.ops.aten.cat.default([addmm_35, where_157, slice_414, where_158, where_159, where_160, where_161, where_162], 1);  where_158 = where_159 = where_160 = where_161 = where_162 = None
        addmm_60 = torch.ops.aten.addmm.default(arg197_1, cat_71, arg196_1);  arg197_1 = arg196_1 = None
        relu_17 = torch.ops.aten.relu.default(addmm_60);  addmm_60 = None
        addmm_61 = torch.ops.aten.addmm.default(arg199_1, relu_17, arg198_1);  arg199_1 = relu_17 = arg198_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg201_1, cat_72, arg200_1);  arg201_1 = arg200_1 = None
        relu_18 = torch.ops.aten.relu.default(addmm_62);  addmm_62 = None
        addmm_63 = torch.ops.aten.addmm.default(arg203_1, relu_18, arg202_1);  arg203_1 = relu_18 = arg202_1 = None
        eq_52 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_53 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_24 = torch.ops.aten.logical_or.default(eq_52, eq_53);  eq_52 = eq_53 = None
        repeat_6 = torch.ops.aten.repeat.default(logical_or_24, [1, 64]);  logical_or_24 = None
        where_163 = torch.ops.aten.where.self(repeat_6, addmm_63, addmm_61);  repeat_6 = addmm_63 = addmm_61 = None
        slice_420 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32877, 32909)
        slice_421 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35034, 35050)
        slice_422 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35406, 35422)
        slice_423 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34520, 34552)
        slice_424 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35778, 35794)
        slice_425 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33316, 33332)
        slice_426 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33930, 33962)
        slice_427 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36808, 36824)
        slice_428 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39168, 39200)
        slice_429 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39785, 39817)
        slice_430 = torch.ops.aten.slice.Tensor(arg15_1, 1, 40402, 40434)
        slice_431 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37237, 37269)
        slice_432 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37918, 37950)
        slice_433 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36364, 36396)
        slice_434 = torch.ops.aten.slice.Tensor(arg15_1, 1, 686, 718)
        slice_435 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38407, 38439)
        squeeze_116 = torch.ops.aten.squeeze.default(logical_not)
        view_143 = torch.ops.aten.view.default(squeeze_116, [-1, 1]);  squeeze_116 = None
        slice_436 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23070, 23102)
        full_default_167 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_164 = torch.ops.aten.where.self(view_143, full_default_167, slice_436);  view_143 = full_default_167 = slice_436 = None
        slice_437 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27948, 27980)
        squeeze_118 = torch.ops.aten.squeeze.default(logical_not)
        view_145 = torch.ops.aten.view.default(squeeze_118, [-1, 1]);  squeeze_118 = None
        slice_438 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27307, 27339)
        full_default_168 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_165 = torch.ops.aten.where.self(view_145, full_default_168, slice_438);  full_default_168 = slice_438 = None
        slice_439 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24092, 24124)
        full_default_169 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_166 = torch.ops.aten.where.self(view_145, full_default_169, slice_439);  full_default_169 = slice_439 = None
        slice_440 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24938, 24970)
        full_default_170 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_167 = torch.ops.aten.where.self(view_145, full_default_170, slice_440);  full_default_170 = slice_440 = None
        slice_441 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25768, 25800)
        full_default_171 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_168 = torch.ops.aten.where.self(view_145, full_default_171, slice_441);  full_default_171 = slice_441 = None
        slice_442 = torch.ops.aten.slice.Tensor(arg15_1, 1, 26518, 26550)
        full_default_172 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_169 = torch.ops.aten.where.self(view_145, full_default_172, slice_442);  view_145 = full_default_172 = slice_442 = None
        cat_73 = torch.ops.aten.cat.default([addmm_35, slice_420, slice_421, slice_422, slice_423, slice_424, slice_425, slice_426, slice_427, slice_428, slice_429, slice_430, slice_431, slice_432, slice_433, slice_434, slice_435], 1);  slice_435 = None
        cat_74 = torch.ops.aten.cat.default([addmm_35, where_164, slice_437, where_165, where_166, where_167, where_168, where_169], 1);  where_165 = where_166 = where_167 = where_168 = where_169 = None
        addmm_64 = torch.ops.aten.addmm.default(arg205_1, cat_73, arg204_1);  arg205_1 = cat_73 = arg204_1 = None
        relu_19 = torch.ops.aten.relu.default(addmm_64);  addmm_64 = None
        addmm_65 = torch.ops.aten.addmm.default(arg207_1, relu_19, arg206_1);  arg207_1 = relu_19 = arg206_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg209_1, cat_74, arg208_1);  arg209_1 = cat_74 = arg208_1 = None
        relu_20 = torch.ops.aten.relu.default(addmm_66);  addmm_66 = None
        addmm_67 = torch.ops.aten.addmm.default(arg211_1, relu_20, arg210_1);  arg211_1 = relu_20 = arg210_1 = None
        eq_54 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_55 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_25 = torch.ops.aten.logical_or.default(eq_54, eq_55);  eq_54 = eq_55 = None
        repeat_7 = torch.ops.aten.repeat.default(logical_or_25, [1, 32]);  logical_or_25 = None
        where_170 = torch.ops.aten.where.self(repeat_7, addmm_67, addmm_65);  repeat_7 = addmm_67 = addmm_65 = None
        slice_443 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32909, 32941)
        slice_444 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35050, 35066)
        slice_445 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35422, 35438)
        slice_446 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34552, 34584)
        slice_447 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35794, 35810)
        slice_448 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33332, 33348)
        slice_449 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33962, 33994)
        slice_450 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36824, 36840)
        slice_451 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39200, 39232)
        slice_452 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39817, 39849)
        slice_453 = torch.ops.aten.slice.Tensor(arg15_1, 1, 40434, 40466)
        slice_454 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37269, 37301)
        slice_455 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37950, 37982)
        slice_456 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36396, 36428)
        slice_457 = torch.ops.aten.slice.Tensor(arg15_1, 1, 718, 750)
        slice_458 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38439, 38471)
        squeeze_122 = torch.ops.aten.squeeze.default(logical_not)
        view_149 = torch.ops.aten.view.default(squeeze_122, [-1, 1]);  squeeze_122 = None
        slice_459 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23102, 23134)
        full_default_173 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_171 = torch.ops.aten.where.self(view_149, full_default_173, slice_459);  view_149 = full_default_173 = slice_459 = None
        slice_460 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27980, 28012)
        squeeze_124 = torch.ops.aten.squeeze.default(logical_not)
        view_151 = torch.ops.aten.view.default(squeeze_124, [-1, 1]);  squeeze_124 = None
        slice_461 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27339, 27371)
        full_default_174 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_172 = torch.ops.aten.where.self(view_151, full_default_174, slice_461);  full_default_174 = slice_461 = None
        slice_462 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24124, 24156)
        full_default_175 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_173 = torch.ops.aten.where.self(view_151, full_default_175, slice_462);  full_default_175 = slice_462 = None
        slice_463 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24970, 25002)
        full_default_176 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_174 = torch.ops.aten.where.self(view_151, full_default_176, slice_463);  full_default_176 = slice_463 = None
        slice_464 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25800, 25832)
        full_default_177 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_175 = torch.ops.aten.where.self(view_151, full_default_177, slice_464);  full_default_177 = slice_464 = None
        slice_465 = torch.ops.aten.slice.Tensor(arg15_1, 1, 26550, 26582)
        full_default_178 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_176 = torch.ops.aten.where.self(view_151, full_default_178, slice_465);  view_151 = full_default_178 = slice_465 = None
        cat_75 = torch.ops.aten.cat.default([addmm_35, slice_443, slice_444, slice_445, slice_446, slice_447, slice_448, slice_449, slice_450, slice_451, slice_452, slice_453, slice_454, slice_455, slice_456, slice_457, slice_458], 1);  slice_458 = None
        cat_76 = torch.ops.aten.cat.default([addmm_35, where_171, slice_460, where_172, where_173, where_174, where_175, where_176], 1);  where_172 = where_173 = where_174 = where_175 = where_176 = None
        addmm_68 = torch.ops.aten.addmm.default(arg213_1, cat_75, arg212_1);  arg213_1 = cat_75 = arg212_1 = None
        relu_21 = torch.ops.aten.relu.default(addmm_68);  addmm_68 = None
        addmm_69 = torch.ops.aten.addmm.default(arg215_1, relu_21, arg214_1);  arg215_1 = relu_21 = arg214_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg217_1, cat_76, arg216_1);  arg217_1 = cat_76 = arg216_1 = None
        relu_22 = torch.ops.aten.relu.default(addmm_70);  addmm_70 = None
        addmm_71 = torch.ops.aten.addmm.default(arg219_1, relu_22, arg218_1);  arg219_1 = relu_22 = arg218_1 = None
        eq_56 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_57 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_26 = torch.ops.aten.logical_or.default(eq_56, eq_57);  eq_56 = eq_57 = None
        repeat_8 = torch.ops.aten.repeat.default(logical_or_26, [1, 32]);  logical_or_26 = None
        where_177 = torch.ops.aten.where.self(repeat_8, addmm_71, addmm_69);  repeat_8 = addmm_71 = addmm_69 = None
        slice_466 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32941, 32973)
        slice_467 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35066, 35082)
        slice_468 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35438, 35454)
        slice_469 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34584, 34616)
        slice_470 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35810, 35826)
        slice_471 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33348, 33364)
        slice_472 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33994, 34026)
        slice_473 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36840, 36856)
        slice_474 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39232, 39264)
        slice_475 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39849, 39881)
        slice_476 = torch.ops.aten.slice.Tensor(arg15_1, 1, 40466, 40498)
        slice_477 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37301, 37333)
        slice_478 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37982, 38014)
        slice_479 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36428, 36460)
        slice_480 = torch.ops.aten.slice.Tensor(arg15_1, 1, 750, 782)
        slice_481 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38471, 38503)
        squeeze_128 = torch.ops.aten.squeeze.default(logical_not)
        view_155 = torch.ops.aten.view.default(squeeze_128, [-1, 1]);  squeeze_128 = None
        slice_482 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23134, 23166)
        full_default_179 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_178 = torch.ops.aten.where.self(view_155, full_default_179, slice_482);  view_155 = full_default_179 = slice_482 = None
        slice_483 = torch.ops.aten.slice.Tensor(arg15_1, 1, 28012, 28044)
        squeeze_130 = torch.ops.aten.squeeze.default(logical_not)
        view_157 = torch.ops.aten.view.default(squeeze_130, [-1, 1]);  squeeze_130 = None
        slice_484 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27371, 27403)
        full_default_180 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_179 = torch.ops.aten.where.self(view_157, full_default_180, slice_484);  full_default_180 = slice_484 = None
        slice_485 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24156, 24188)
        full_default_181 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_180 = torch.ops.aten.where.self(view_157, full_default_181, slice_485);  full_default_181 = slice_485 = None
        slice_486 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25002, 25034)
        full_default_182 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_181 = torch.ops.aten.where.self(view_157, full_default_182, slice_486);  full_default_182 = slice_486 = None
        slice_487 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25832, 25864)
        full_default_183 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_182 = torch.ops.aten.where.self(view_157, full_default_183, slice_487);  full_default_183 = slice_487 = None
        slice_488 = torch.ops.aten.slice.Tensor(arg15_1, 1, 26582, 26614)
        full_default_184 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_183 = torch.ops.aten.where.self(view_157, full_default_184, slice_488);  view_157 = full_default_184 = slice_488 = None
        cat_77 = torch.ops.aten.cat.default([addmm_35, slice_466, slice_467, slice_468, slice_469, slice_470, slice_471, slice_472, slice_473, slice_474, slice_475, slice_476, slice_477, slice_478, slice_479, slice_480, slice_481], 1);  slice_481 = None
        cat_78 = torch.ops.aten.cat.default([addmm_35, where_178, slice_483, where_179, where_180, where_181, where_182, where_183], 1);  where_179 = where_180 = where_181 = where_182 = where_183 = None
        addmm_72 = torch.ops.aten.addmm.default(arg221_1, cat_77, arg220_1);  arg221_1 = cat_77 = arg220_1 = None
        relu_23 = torch.ops.aten.relu.default(addmm_72);  addmm_72 = None
        addmm_73 = torch.ops.aten.addmm.default(arg223_1, relu_23, arg222_1);  arg223_1 = relu_23 = arg222_1 = None
        addmm_74 = torch.ops.aten.addmm.default(arg225_1, cat_78, arg224_1);  arg225_1 = cat_78 = arg224_1 = None
        relu_24 = torch.ops.aten.relu.default(addmm_74);  addmm_74 = None
        addmm_75 = torch.ops.aten.addmm.default(arg227_1, relu_24, arg226_1);  arg227_1 = relu_24 = arg226_1 = None
        eq_58 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_59 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_27 = torch.ops.aten.logical_or.default(eq_58, eq_59);  eq_58 = eq_59 = None
        repeat_9 = torch.ops.aten.repeat.default(logical_or_27, [1, 32]);  logical_or_27 = None
        where_184 = torch.ops.aten.where.self(repeat_9, addmm_75, addmm_73);  repeat_9 = addmm_75 = addmm_73 = None
        slice_489 = torch.ops.aten.slice.Tensor(arg15_1, 1, 32973, 33005)
        slice_490 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35082, 35098)
        slice_491 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35454, 35470)
        slice_492 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34616, 34648)
        slice_493 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35826, 35842)
        slice_494 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33364, 33380)
        slice_495 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34026, 34058)
        slice_496 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36856, 36872)
        slice_497 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39264, 39296)
        slice_498 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39881, 39913)
        slice_499 = torch.ops.aten.slice.Tensor(arg15_1, 1, 40498, 40530)
        slice_500 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37333, 37365)
        slice_501 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38014, 38046)
        slice_502 = torch.ops.aten.slice.Tensor(arg15_1, 1, 36460, 36492)
        slice_503 = torch.ops.aten.slice.Tensor(arg15_1, 1, 782, 814)
        slice_504 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38503, 38535)
        squeeze_134 = torch.ops.aten.squeeze.default(logical_not)
        view_161 = torch.ops.aten.view.default(squeeze_134, [-1, 1]);  squeeze_134 = None
        slice_505 = torch.ops.aten.slice.Tensor(arg15_1, 1, 23166, 23198)
        full_default_185 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_185 = torch.ops.aten.where.self(view_161, full_default_185, slice_505);  view_161 = full_default_185 = slice_505 = None
        slice_506 = torch.ops.aten.slice.Tensor(arg15_1, 1, 28044, 28076)
        squeeze_136 = torch.ops.aten.squeeze.default(logical_not)
        view_163 = torch.ops.aten.view.default(squeeze_136, [-1, 1]);  squeeze_136 = None
        slice_507 = torch.ops.aten.slice.Tensor(arg15_1, 1, 27403, 27435)
        full_default_186 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_186 = torch.ops.aten.where.self(view_163, full_default_186, slice_507);  full_default_186 = slice_507 = None
        slice_508 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24188, 24220)
        full_default_187 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_187 = torch.ops.aten.where.self(view_163, full_default_187, slice_508);  full_default_187 = slice_508 = None
        slice_509 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25034, 25066)
        full_default_188 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_188 = torch.ops.aten.where.self(view_163, full_default_188, slice_509);  full_default_188 = slice_509 = None
        slice_510 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25864, 25896)
        full_default_189 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_189 = torch.ops.aten.where.self(view_163, full_default_189, slice_510);  full_default_189 = slice_510 = None
        slice_511 = torch.ops.aten.slice.Tensor(arg15_1, 1, 26614, 26646)
        full_default_190 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_190 = torch.ops.aten.where.self(view_163, full_default_190, slice_511);  view_163 = full_default_190 = slice_511 = None
        cat_79 = torch.ops.aten.cat.default([addmm_35, slice_489, slice_490, slice_491, slice_492, slice_493, slice_494, slice_495, slice_496, slice_497, slice_498, slice_499, slice_500, slice_501, slice_502, slice_503, slice_504], 1);  slice_504 = None
        cat_80 = torch.ops.aten.cat.default([addmm_35, where_185, slice_506, where_186, where_187, where_188, where_189, where_190], 1);  addmm_35 = where_186 = where_187 = where_188 = where_189 = where_190 = None
        addmm_76 = torch.ops.aten.addmm.default(arg229_1, cat_79, arg228_1);  arg229_1 = cat_79 = arg228_1 = None
        relu_25 = torch.ops.aten.relu.default(addmm_76);  addmm_76 = None
        addmm_77 = torch.ops.aten.addmm.default(arg231_1, relu_25, arg230_1);  arg231_1 = relu_25 = arg230_1 = None
        addmm_78 = torch.ops.aten.addmm.default(arg233_1, cat_80, arg232_1);  arg233_1 = cat_80 = arg232_1 = None
        relu_26 = torch.ops.aten.relu.default(addmm_78);  addmm_78 = None
        addmm_79 = torch.ops.aten.addmm.default(arg235_1, relu_26, arg234_1);  arg235_1 = relu_26 = arg234_1 = None
        eq_60 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_61 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_28 = torch.ops.aten.logical_or.default(eq_60, eq_61);  eq_60 = eq_61 = None
        repeat_10 = torch.ops.aten.repeat.default(logical_or_28, [1, 32]);  logical_or_28 = None
        where_191 = torch.ops.aten.where.self(repeat_10, addmm_79, addmm_77);  repeat_10 = addmm_79 = addmm_77 = None
        addmm_80 = torch.ops.aten.addmm.default(arg237_1, where_156, arg236_1);  arg237_1 = arg236_1 = None
        slice_512 = torch.ops.aten.slice.Tensor(arg238_1, 2, 1, 34)
        sign = torch.ops.aten.sign.default(arg239_1)
        slice_513 = torch.ops.aten.slice.Tensor(slice_512, 2, 0, 32)
        slice_514 = torch.ops.aten.slice.Tensor(slice_512, 2, 32, 9223372036854775807);  slice_512 = None
        cumsum = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted = torch.ops.aten.searchsorted.Tensor(cumsum, iota, out_int32 = True, right = True);  cumsum = iota = None
        clamp_max = torch.ops.aten.clamp_max.default(searchsorted, 6);  searchsorted = None
        index = torch.ops.aten.index.Tensor(slice_514, [clamp_max]);  slice_514 = clamp_max = None
        slice_515 = torch.ops.aten.slice.Tensor(arg240_1, 2, 0, 32)
        slice_516 = torch.ops.aten.slice.Tensor(arg240_1, 2, 32, 9223372036854775807);  arg240_1 = None
        slice_517 = torch.ops.aten.slice.Tensor(slice_513, 1, 0, 64);  slice_513 = None
        slice_518 = torch.ops.aten.slice.Tensor(sign, 1, 0, 64);  sign = None
        slice_519 = torch.ops.aten.slice.Tensor(slice_515, 1, 0, 64);  slice_515 = None
        cumsum_1 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_1 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_1 = torch.ops.aten.searchsorted.Tensor(cumsum_1, iota_1, out_int32 = True, right = True);  cumsum_1 = iota_1 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(searchsorted_1, 6);  searchsorted_1 = None
        index_1 = torch.ops.aten.index.Tensor(slice_517, [clamp_max_1]);  slice_517 = clamp_max_1 = None
        cumsum_2 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_2 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_2 = torch.ops.aten.searchsorted.Tensor(cumsum_2, iota_2, out_int32 = True, right = True);  cumsum_2 = iota_2 = None
        clamp_max_2 = torch.ops.aten.clamp_max.default(searchsorted_2, 6);  searchsorted_2 = None
        index_2 = torch.ops.aten.index.Tensor(slice_518, [clamp_max_2]);  slice_518 = clamp_max_2 = None
        slice_520 = torch.ops.aten.slice.Tensor(index_1, 1, 0, 2)
        sum_35 = torch.ops.aten.sum.dim_IntList(slice_520, [1]);  slice_520 = None
        slice_521 = torch.ops.aten.slice.Tensor(index_1, 1, 0, 4)
        sum_36 = torch.ops.aten.sum.dim_IntList(slice_521, [1]);  slice_521 = None
        slice_522 = torch.ops.aten.slice.Tensor(index_1, 1, 0, 8)
        sum_37 = torch.ops.aten.sum.dim_IntList(slice_522, [1]);  slice_522 = None
        slice_523 = torch.ops.aten.slice.Tensor(index_1, 1, 0, 16)
        sum_38 = torch.ops.aten.sum.dim_IntList(slice_523, [1]);  slice_523 = None
        slice_524 = torch.ops.aten.slice.Tensor(index_1, 1, 0, 32)
        sum_39 = torch.ops.aten.sum.dim_IntList(slice_524, [1]);  slice_524 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(index_1, [1])
        unsqueeze = torch.ops.aten.unsqueeze.default(index_2, 2)
        mul_47 = torch.ops.aten.mul.Tensor(index_1, unsqueeze);  unsqueeze = None
        sum_41 = torch.ops.aten.sum.dim_IntList(mul_47, [-2]);  mul_47 = None
        view_164 = torch.ops.aten.view.default(index_1, [1, batch_size, 64, 32]);  index_1 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(view_164, [0]);  view_164 = None
        addmm_81 = torch.ops.aten.addmm.default(arg242_1, cat_57, arg241_1);  arg242_1 = arg241_1 = None
        relu_27 = torch.ops.aten.relu.default(addmm_81);  addmm_81 = None
        addmm_82 = torch.ops.aten.addmm.default(arg244_1, relu_27, arg243_1);  arg244_1 = relu_27 = arg243_1 = None
        view_165 = torch.ops.aten.view.default(addmm_82, [-1, 32, 32]);  addmm_82 = None
        addmm_83 = torch.ops.aten.addmm.default(arg246_1, cat_57, arg245_1);  arg246_1 = arg245_1 = None
        relu_28 = torch.ops.aten.relu.default(addmm_83);  addmm_83 = None
        addmm_84 = torch.ops.aten.addmm.default(arg248_1, relu_28, arg247_1);  arg248_1 = relu_28 = arg247_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(addmm_84, 1);  addmm_84 = None
        expand = torch.ops.aten.expand.default(sum_42, [batch_size, 64, 32]);  sum_42 = None
        expand_1 = torch.ops.aten.expand.default(view_165, [batch_size, 32, 32]);  view_165 = None
        bmm = torch.ops.aten.bmm.default(expand, expand_1);  expand = expand_1 = None
        add = torch.ops.aten.add.Tensor(bmm, unsqueeze_1);  bmm = unsqueeze_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where_156, 1)
        expand_2 = torch.ops.aten.expand.default(unsqueeze_2, [-1, 64, -1]);  unsqueeze_2 = None
        sub = torch.ops.aten.sub.Tensor(expand_2, add)
        mul_48 = torch.ops.aten.mul.Tensor(expand_2, add)
        cat_81 = torch.ops.aten.cat.default([expand_2, add, sub, mul_48], -1);  expand_2 = sub = mul_48 = None
        view_169 = torch.ops.aten.view.default(cat_81, [12800, 128]);  cat_81 = None
        addmm_85 = torch.ops.aten.addmm.default(arg250_1, view_169, arg249_1);  arg250_1 = view_169 = arg249_1 = None
        view_170 = torch.ops.aten.view.default(addmm_85, [batch_size, 64, 256]);  addmm_85 = None
        relu_29 = torch.ops.aten.relu.default(view_170);  view_170 = None
        view_171 = torch.ops.aten.view.default(relu_29, [12800, 256]);  relu_29 = None
        addmm_86 = torch.ops.aten.addmm.default(arg252_1, view_171, arg251_1);  arg252_1 = view_171 = arg251_1 = None
        view_172 = torch.ops.aten.view.default(addmm_86, [batch_size, 64, 128]);  addmm_86 = None
        relu_30 = torch.ops.aten.relu.default(view_172);  view_172 = None
        view_173 = torch.ops.aten.view.default(relu_30, [12800, 128]);  relu_30 = None
        addmm_87 = torch.ops.aten.addmm.default(arg254_1, view_173, arg253_1);  arg254_1 = view_173 = arg253_1 = None
        view_174 = torch.ops.aten.view.default(addmm_87, [batch_size, 64, 32]);  addmm_87 = None
        mul_49 = torch.ops.aten.mul.Tensor(view_174, arg255_1);  view_174 = arg255_1 = None
        sigmoid = torch.ops.aten.sigmoid.default(mul_49);  mul_49 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(index_2, 2);  index_2 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(unsqueeze_3, torch.float16);  unsqueeze_3 = None
        mul_50 = torch.ops.aten.mul.Tensor(sigmoid, convert_element_type_8);  sigmoid = convert_element_type_8 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, add);  mul_50 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_51, [1]);  mul_51 = None
        mul_52 = torch.ops.aten.mul.Tensor(slice_519, add);  slice_519 = add = None
        sum_44 = torch.ops.aten.sum.dim_IntList(mul_52, [1]);  mul_52 = None
        cat_82 = torch.ops.aten.cat.default([sum_43, sum_44], 1);  sum_43 = sum_44 = None
        cat_83 = torch.ops.aten.cat.default([cat_82, addmm_80, sum_35, sum_36, sum_37, sum_38, sum_39, sum_40], -1);  addmm_80 = sum_35 = sum_36 = sum_37 = sum_38 = sum_39 = sum_40 = None
        full_default_191 = torch.ops.aten.full.default([batch_size, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_192 = torch.ops.aten.where.self(logical_or_12, full_default_191, cat_82);  full_default_191 = cat_82 = None
        full_default_192 = torch.ops.aten.full.default([batch_size, 320], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_193 = torch.ops.aten.where.self(logical_or_12, full_default_192, cat_83);  full_default_192 = cat_83 = None
        addmm_88 = torch.ops.aten.addmm.default(arg257_1, where_170, arg256_1);  arg257_1 = arg256_1 = None
        slice_525 = torch.ops.aten.slice.Tensor(arg258_1, 2, 1, 130);  arg258_1 = None
        sign_1 = torch.ops.aten.sign.default(arg259_1);  arg259_1 = None
        slice_526 = torch.ops.aten.slice.Tensor(slice_525, 2, 0, 128)
        slice_527 = torch.ops.aten.slice.Tensor(slice_525, 2, 128, 9223372036854775807);  slice_525 = None
        slice_528 = torch.ops.aten.slice.Tensor(arg260_1, 2, 0, 32)
        slice_529 = torch.ops.aten.slice.Tensor(arg260_1, 2, 32, 9223372036854775807);  arg260_1 = None
        slice_530 = torch.ops.aten.slice.Tensor(slice_526, 1, 0, 2)
        sum_45 = torch.ops.aten.sum.dim_IntList(slice_530, [1]);  slice_530 = None
        slice_531 = torch.ops.aten.slice.Tensor(slice_526, 1, 0, 4)
        sum_46 = torch.ops.aten.sum.dim_IntList(slice_531, [1]);  slice_531 = None
        slice_532 = torch.ops.aten.slice.Tensor(slice_526, 1, 0, 8)
        sum_47 = torch.ops.aten.sum.dim_IntList(slice_532, [1]);  slice_532 = None
        slice_533 = torch.ops.aten.slice.Tensor(slice_526, 1, 0, 16)
        sum_48 = torch.ops.aten.sum.dim_IntList(slice_533, [1]);  slice_533 = None
        sum_49 = torch.ops.aten.sum.dim_IntList(slice_526, [1])
        sum_50 = torch.ops.aten.sum.dim_IntList(slice_526, [1])
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(sign_1, 2)
        mul_53 = torch.ops.aten.mul.Tensor(slice_526, unsqueeze_4);  unsqueeze_4 = None
        sum_51 = torch.ops.aten.sum.dim_IntList(mul_53, [-2]);  mul_53 = None
        clone_12 = torch.ops.aten.clone.default(slice_526);  slice_526 = None
        view_175 = torch.ops.aten.view.default(clone_12, [1, batch_size, 32, 128]);  clone_12 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(view_175, [0]);  view_175 = None
        addmm_89 = torch.ops.aten.addmm.default(arg262_1, cat_57, arg261_1);  arg262_1 = arg261_1 = None
        relu_31 = torch.ops.aten.relu.default(addmm_89);  addmm_89 = None
        addmm_90 = torch.ops.aten.addmm.default(arg264_1, relu_31, arg263_1);  arg264_1 = relu_31 = arg263_1 = None
        view_176 = torch.ops.aten.view.default(addmm_90, [-1, 128, 32]);  addmm_90 = None
        addmm_91 = torch.ops.aten.addmm.default(arg266_1, cat_57, arg265_1);  arg266_1 = arg265_1 = None
        relu_32 = torch.ops.aten.relu.default(addmm_91);  addmm_91 = None
        addmm_92 = torch.ops.aten.addmm.default(arg268_1, relu_32, arg267_1);  arg268_1 = relu_32 = arg267_1 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(addmm_92, 1);  addmm_92 = None
        expand_3 = torch.ops.aten.expand.default(sum_52, [batch_size, 32, 128]);  sum_52 = None
        expand_4 = torch.ops.aten.expand.default(view_176, [batch_size, 128, 32]);  view_176 = None
        bmm_1 = torch.ops.aten.bmm.default(expand_3, expand_4);  expand_3 = expand_4 = None
        add_1 = torch.ops.aten.add.Tensor(bmm_1, unsqueeze_5);  bmm_1 = unsqueeze_5 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(where_170, 1)
        expand_5 = torch.ops.aten.expand.default(unsqueeze_6, [-1, 32, -1]);  unsqueeze_6 = None
        sub_1 = torch.ops.aten.sub.Tensor(expand_5, add_1)
        mul_54 = torch.ops.aten.mul.Tensor(expand_5, add_1)
        cat_84 = torch.ops.aten.cat.default([expand_5, add_1, sub_1, mul_54], -1);  expand_5 = sub_1 = mul_54 = None
        view_180 = torch.ops.aten.view.default(cat_84, [6400, 128]);  cat_84 = None
        addmm_93 = torch.ops.aten.addmm.default(arg270_1, view_180, arg269_1);  arg270_1 = view_180 = arg269_1 = None
        view_181 = torch.ops.aten.view.default(addmm_93, [batch_size, 32, 256]);  addmm_93 = None
        relu_33 = torch.ops.aten.relu.default(view_181);  view_181 = None
        view_182 = torch.ops.aten.view.default(relu_33, [6400, 256]);  relu_33 = None
        addmm_94 = torch.ops.aten.addmm.default(arg272_1, view_182, arg271_1);  arg272_1 = view_182 = arg271_1 = None
        view_183 = torch.ops.aten.view.default(addmm_94, [batch_size, 32, 128]);  addmm_94 = None
        relu_34 = torch.ops.aten.relu.default(view_183);  view_183 = None
        view_184 = torch.ops.aten.view.default(relu_34, [6400, 128]);  relu_34 = None
        addmm_95 = torch.ops.aten.addmm.default(arg274_1, view_184, arg273_1);  arg274_1 = view_184 = arg273_1 = None
        view_185 = torch.ops.aten.view.default(addmm_95, [batch_size, 32, 32]);  addmm_95 = None
        mul_55 = torch.ops.aten.mul.Tensor(view_185, arg275_1);  view_185 = arg275_1 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(mul_55);  mul_55 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(sign_1, 2);  sign_1 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(unsqueeze_7, torch.float16);  unsqueeze_7 = None
        mul_56 = torch.ops.aten.mul.Tensor(sigmoid_1, convert_element_type_11);  sigmoid_1 = convert_element_type_11 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, add_1);  mul_56 = None
        sum_53 = torch.ops.aten.sum.dim_IntList(mul_57, [1]);  mul_57 = None
        mul_58 = torch.ops.aten.mul.Tensor(slice_528, add_1);  slice_528 = add_1 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(mul_58, [1]);  mul_58 = None
        cat_85 = torch.ops.aten.cat.default([sum_53, sum_54], 1);  sum_53 = sum_54 = None
        cat_86 = torch.ops.aten.cat.default([cat_85, addmm_88, sum_45, sum_46, sum_47, sum_48, sum_49, sum_50], -1);  addmm_88 = sum_45 = sum_46 = sum_47 = sum_48 = sum_49 = sum_50 = None
        addmm_96 = torch.ops.aten.addmm.default(arg277_1, where_177, arg276_1);  arg277_1 = arg276_1 = None
        slice_534 = torch.ops.aten.slice.Tensor(arg278_1, 2, 1, 130);  arg278_1 = None
        sign_2 = torch.ops.aten.sign.default(arg279_1);  arg279_1 = None
        slice_535 = torch.ops.aten.slice.Tensor(slice_534, 2, 0, 128)
        slice_536 = torch.ops.aten.slice.Tensor(slice_534, 2, 128, 9223372036854775807);  slice_534 = None
        slice_537 = torch.ops.aten.slice.Tensor(arg280_1, 2, 0, 32)
        slice_538 = torch.ops.aten.slice.Tensor(arg280_1, 2, 32, 9223372036854775807);  arg280_1 = None
        slice_539 = torch.ops.aten.slice.Tensor(slice_535, 1, 0, 2)
        sum_55 = torch.ops.aten.sum.dim_IntList(slice_539, [1]);  slice_539 = None
        slice_540 = torch.ops.aten.slice.Tensor(slice_535, 1, 0, 4)
        sum_56 = torch.ops.aten.sum.dim_IntList(slice_540, [1]);  slice_540 = None
        slice_541 = torch.ops.aten.slice.Tensor(slice_535, 1, 0, 8)
        sum_57 = torch.ops.aten.sum.dim_IntList(slice_541, [1]);  slice_541 = None
        slice_542 = torch.ops.aten.slice.Tensor(slice_535, 1, 0, 16)
        sum_58 = torch.ops.aten.sum.dim_IntList(slice_542, [1]);  slice_542 = None
        slice_543 = torch.ops.aten.slice.Tensor(slice_535, 1, 0, 32)
        sum_59 = torch.ops.aten.sum.dim_IntList(slice_543, [1]);  slice_543 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(slice_535, [1])
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(sign_2, 2)
        mul_59 = torch.ops.aten.mul.Tensor(slice_535, unsqueeze_8);  unsqueeze_8 = None
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_59, [-2]);  mul_59 = None
        clone_13 = torch.ops.aten.clone.default(slice_535);  slice_535 = None
        view_186 = torch.ops.aten.view.default(clone_13, [1, batch_size, 64, 128]);  clone_13 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(view_186, [0]);  view_186 = None
        addmm_97 = torch.ops.aten.addmm.default(arg282_1, cat_57, arg281_1);  arg282_1 = arg281_1 = None
        relu_35 = torch.ops.aten.relu.default(addmm_97);  addmm_97 = None
        addmm_98 = torch.ops.aten.addmm.default(arg284_1, relu_35, arg283_1);  arg284_1 = relu_35 = arg283_1 = None
        view_187 = torch.ops.aten.view.default(addmm_98, [-1, 128, 32]);  addmm_98 = None
        addmm_99 = torch.ops.aten.addmm.default(arg286_1, cat_57, arg285_1);  arg286_1 = arg285_1 = None
        relu_36 = torch.ops.aten.relu.default(addmm_99);  addmm_99 = None
        addmm_100 = torch.ops.aten.addmm.default(arg288_1, relu_36, arg287_1);  arg288_1 = relu_36 = arg287_1 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(addmm_100, 1);  addmm_100 = None
        expand_6 = torch.ops.aten.expand.default(sum_62, [batch_size, 64, 128]);  sum_62 = None
        expand_7 = torch.ops.aten.expand.default(view_187, [batch_size, 128, 32]);  view_187 = None
        bmm_2 = torch.ops.aten.bmm.default(expand_6, expand_7);  expand_6 = expand_7 = None
        add_2 = torch.ops.aten.add.Tensor(bmm_2, unsqueeze_9);  bmm_2 = unsqueeze_9 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(where_177, 1)
        expand_8 = torch.ops.aten.expand.default(unsqueeze_10, [-1, 64, -1]);  unsqueeze_10 = None
        sub_2 = torch.ops.aten.sub.Tensor(expand_8, add_2)
        mul_60 = torch.ops.aten.mul.Tensor(expand_8, add_2)
        cat_87 = torch.ops.aten.cat.default([expand_8, add_2, sub_2, mul_60], -1);  expand_8 = sub_2 = mul_60 = None
        view_191 = torch.ops.aten.view.default(cat_87, [12800, 128]);  cat_87 = None
        addmm_101 = torch.ops.aten.addmm.default(arg290_1, view_191, arg289_1);  arg290_1 = view_191 = arg289_1 = None
        view_192 = torch.ops.aten.view.default(addmm_101, [batch_size, 64, 256]);  addmm_101 = None
        relu_37 = torch.ops.aten.relu.default(view_192);  view_192 = None
        view_193 = torch.ops.aten.view.default(relu_37, [12800, 256]);  relu_37 = None
        addmm_102 = torch.ops.aten.addmm.default(arg292_1, view_193, arg291_1);  arg292_1 = view_193 = arg291_1 = None
        view_194 = torch.ops.aten.view.default(addmm_102, [batch_size, 64, 128]);  addmm_102 = None
        relu_38 = torch.ops.aten.relu.default(view_194);  view_194 = None
        view_195 = torch.ops.aten.view.default(relu_38, [12800, 128]);  relu_38 = None
        addmm_103 = torch.ops.aten.addmm.default(arg294_1, view_195, arg293_1);  arg294_1 = view_195 = arg293_1 = None
        view_196 = torch.ops.aten.view.default(addmm_103, [batch_size, 64, 32]);  addmm_103 = None
        mul_61 = torch.ops.aten.mul.Tensor(view_196, arg295_1);  view_196 = arg295_1 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(mul_61);  mul_61 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(sign_2, 2);  sign_2 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(unsqueeze_11, torch.float16);  unsqueeze_11 = None
        mul_62 = torch.ops.aten.mul.Tensor(sigmoid_2, convert_element_type_14);  sigmoid_2 = convert_element_type_14 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_62, add_2);  mul_62 = None
        sum_63 = torch.ops.aten.sum.dim_IntList(mul_63, [1]);  mul_63 = None
        mul_64 = torch.ops.aten.mul.Tensor(slice_537, add_2);  slice_537 = add_2 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(mul_64, [1]);  mul_64 = None
        cat_88 = torch.ops.aten.cat.default([sum_63, sum_64], 1);  sum_63 = sum_64 = None
        cat_89 = torch.ops.aten.cat.default([cat_88, addmm_96, sum_55, sum_56, sum_57, sum_58, sum_59, sum_60], -1);  addmm_96 = sum_55 = sum_56 = sum_57 = sum_58 = sum_59 = sum_60 = None
        addmm_104 = torch.ops.aten.addmm.default(arg297_1, where_135, arg296_1);  arg297_1 = arg296_1 = None
        slice_544 = torch.ops.aten.slice.Tensor(arg298_1, 2, 1, 34)
        sign_3 = torch.ops.aten.sign.default(arg299_1)
        slice_545 = torch.ops.aten.slice.Tensor(slice_544, 2, 0, 32)
        slice_546 = torch.ops.aten.slice.Tensor(slice_544, 2, 32, 9223372036854775807);  slice_544 = None
        cumsum_3 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_3 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_3 = torch.ops.aten.searchsorted.Tensor(cumsum_3, iota_3, out_int32 = True, right = True);  cumsum_3 = iota_3 = None
        clamp_max_3 = torch.ops.aten.clamp_max.default(searchsorted_3, 6);  searchsorted_3 = None
        index_3 = torch.ops.aten.index.Tensor(slice_546, [clamp_max_3]);  slice_546 = clamp_max_3 = None
        slice_547 = torch.ops.aten.slice.Tensor(arg300_1, 2, 0, 32)
        slice_548 = torch.ops.aten.slice.Tensor(arg300_1, 2, 32, 9223372036854775807);  arg300_1 = None
        slice_549 = torch.ops.aten.slice.Tensor(slice_545, 1, 0, 64);  slice_545 = None
        slice_550 = torch.ops.aten.slice.Tensor(sign_3, 1, 0, 64);  sign_3 = None
        slice_551 = torch.ops.aten.slice.Tensor(slice_547, 1, 0, 64);  slice_547 = None
        cumsum_4 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_4 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_4 = torch.ops.aten.searchsorted.Tensor(cumsum_4, iota_4, out_int32 = True, right = True);  cumsum_4 = iota_4 = None
        clamp_max_4 = torch.ops.aten.clamp_max.default(searchsorted_4, 6);  searchsorted_4 = None
        index_4 = torch.ops.aten.index.Tensor(slice_549, [clamp_max_4]);  slice_549 = clamp_max_4 = None
        cumsum_5 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_5 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_5 = torch.ops.aten.searchsorted.Tensor(cumsum_5, iota_5, out_int32 = True, right = True);  cumsum_5 = iota_5 = None
        clamp_max_5 = torch.ops.aten.clamp_max.default(searchsorted_5, 6);  searchsorted_5 = None
        index_5 = torch.ops.aten.index.Tensor(slice_550, [clamp_max_5]);  slice_550 = clamp_max_5 = None
        slice_552 = torch.ops.aten.slice.Tensor(index_4, 1, 0, 2)
        sum_65 = torch.ops.aten.sum.dim_IntList(slice_552, [1]);  slice_552 = None
        slice_553 = torch.ops.aten.slice.Tensor(index_4, 1, 0, 4)
        sum_66 = torch.ops.aten.sum.dim_IntList(slice_553, [1]);  slice_553 = None
        slice_554 = torch.ops.aten.slice.Tensor(index_4, 1, 0, 8)
        sum_67 = torch.ops.aten.sum.dim_IntList(slice_554, [1]);  slice_554 = None
        slice_555 = torch.ops.aten.slice.Tensor(index_4, 1, 0, 16)
        sum_68 = torch.ops.aten.sum.dim_IntList(slice_555, [1]);  slice_555 = None
        slice_556 = torch.ops.aten.slice.Tensor(index_4, 1, 0, 32)
        sum_69 = torch.ops.aten.sum.dim_IntList(slice_556, [1]);  slice_556 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(index_4, [1])
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(index_5, 2)
        mul_65 = torch.ops.aten.mul.Tensor(index_4, unsqueeze_12);  unsqueeze_12 = None
        sum_71 = torch.ops.aten.sum.dim_IntList(mul_65, [-2]);  mul_65 = None
        view_197 = torch.ops.aten.view.default(index_4, [1, batch_size, 64, 32]);  index_4 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(view_197, [0]);  view_197 = None
        addmm_105 = torch.ops.aten.addmm.default(arg302_1, cat_57, arg301_1);  arg302_1 = arg301_1 = None
        relu_39 = torch.ops.aten.relu.default(addmm_105);  addmm_105 = None
        addmm_106 = torch.ops.aten.addmm.default(arg304_1, relu_39, arg303_1);  arg304_1 = relu_39 = arg303_1 = None
        view_198 = torch.ops.aten.view.default(addmm_106, [-1, 32, 32]);  addmm_106 = None
        addmm_107 = torch.ops.aten.addmm.default(arg306_1, cat_57, arg305_1);  arg306_1 = arg305_1 = None
        relu_40 = torch.ops.aten.relu.default(addmm_107);  addmm_107 = None
        addmm_108 = torch.ops.aten.addmm.default(arg308_1, relu_40, arg307_1);  arg308_1 = relu_40 = arg307_1 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(addmm_108, 1);  addmm_108 = None
        expand_9 = torch.ops.aten.expand.default(sum_72, [batch_size, 64, 32]);  sum_72 = None
        expand_10 = torch.ops.aten.expand.default(view_198, [batch_size, 32, 32]);  view_198 = None
        bmm_3 = torch.ops.aten.bmm.default(expand_9, expand_10);  expand_9 = expand_10 = None
        add_3 = torch.ops.aten.add.Tensor(bmm_3, unsqueeze_13);  bmm_3 = unsqueeze_13 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(where_135, 1)
        expand_11 = torch.ops.aten.expand.default(unsqueeze_14, [-1, 64, -1]);  unsqueeze_14 = None
        sub_3 = torch.ops.aten.sub.Tensor(expand_11, add_3)
        mul_66 = torch.ops.aten.mul.Tensor(expand_11, add_3)
        cat_90 = torch.ops.aten.cat.default([expand_11, add_3, sub_3, mul_66], -1);  expand_11 = sub_3 = mul_66 = None
        view_202 = torch.ops.aten.view.default(cat_90, [12800, 128]);  cat_90 = None
        addmm_109 = torch.ops.aten.addmm.default(arg310_1, view_202, arg309_1);  arg310_1 = view_202 = arg309_1 = None
        view_203 = torch.ops.aten.view.default(addmm_109, [batch_size, 64, 256]);  addmm_109 = None
        relu_41 = torch.ops.aten.relu.default(view_203);  view_203 = None
        view_204 = torch.ops.aten.view.default(relu_41, [12800, 256]);  relu_41 = None
        addmm_110 = torch.ops.aten.addmm.default(arg312_1, view_204, arg311_1);  arg312_1 = view_204 = arg311_1 = None
        view_205 = torch.ops.aten.view.default(addmm_110, [batch_size, 64, 128]);  addmm_110 = None
        relu_42 = torch.ops.aten.relu.default(view_205);  view_205 = None
        view_206 = torch.ops.aten.view.default(relu_42, [12800, 128]);  relu_42 = None
        addmm_111 = torch.ops.aten.addmm.default(arg314_1, view_206, arg313_1);  arg314_1 = view_206 = arg313_1 = None
        view_207 = torch.ops.aten.view.default(addmm_111, [batch_size, 64, 32]);  addmm_111 = None
        mul_67 = torch.ops.aten.mul.Tensor(view_207, arg315_1);  view_207 = arg315_1 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(mul_67);  mul_67 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(index_5, 2);  index_5 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(unsqueeze_15, torch.float16);  unsqueeze_15 = None
        mul_68 = torch.ops.aten.mul.Tensor(sigmoid_3, convert_element_type_17);  sigmoid_3 = convert_element_type_17 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, add_3);  mul_68 = None
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_69, [1]);  mul_69 = None
        mul_70 = torch.ops.aten.mul.Tensor(slice_551, add_3);  slice_551 = add_3 = None
        sum_74 = torch.ops.aten.sum.dim_IntList(mul_70, [1]);  mul_70 = None
        cat_91 = torch.ops.aten.cat.default([sum_73, sum_74], 1);  sum_73 = sum_74 = None
        cat_92 = torch.ops.aten.cat.default([cat_91, addmm_104, sum_65, sum_66, sum_67, sum_68, sum_69, sum_70], -1);  addmm_104 = sum_65 = sum_66 = sum_67 = sum_68 = sum_69 = sum_70 = None
        full_default_193 = torch.ops.aten.full.default([batch_size, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_194 = torch.ops.aten.where.self(logical_or_12, full_default_193, cat_91);  full_default_193 = cat_91 = None
        full_default_194 = torch.ops.aten.full.default([batch_size, 320], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_195 = torch.ops.aten.where.self(logical_or_12, full_default_194, cat_92);  full_default_194 = cat_92 = None
        addmm_112 = torch.ops.aten.addmm.default(arg317_1, where_184, arg316_1);  arg317_1 = arg316_1 = None
        slice_557 = torch.ops.aten.slice.Tensor(arg318_1, 2, 1, 34);  arg318_1 = None
        sign_4 = torch.ops.aten.sign.default(arg319_1);  arg319_1 = None
        slice_558 = torch.ops.aten.slice.Tensor(slice_557, 2, 0, 32)
        slice_559 = torch.ops.aten.slice.Tensor(slice_557, 2, 32, 9223372036854775807);  slice_557 = None
        slice_560 = torch.ops.aten.slice.Tensor(arg320_1, 2, 0, 32)
        slice_561 = torch.ops.aten.slice.Tensor(arg320_1, 2, 32, 9223372036854775807);  arg320_1 = None
        slice_562 = torch.ops.aten.slice.Tensor(slice_558, 1, 0, 2)
        sum_75 = torch.ops.aten.sum.dim_IntList(slice_562, [1]);  slice_562 = None
        slice_563 = torch.ops.aten.slice.Tensor(slice_558, 1, 0, 4)
        sum_76 = torch.ops.aten.sum.dim_IntList(slice_563, [1]);  slice_563 = None
        slice_564 = torch.ops.aten.slice.Tensor(slice_558, 1, 0, 8)
        sum_77 = torch.ops.aten.sum.dim_IntList(slice_564, [1]);  slice_564 = None
        slice_565 = torch.ops.aten.slice.Tensor(slice_558, 1, 0, 16)
        sum_78 = torch.ops.aten.sum.dim_IntList(slice_565, [1]);  slice_565 = None
        slice_566 = torch.ops.aten.slice.Tensor(slice_558, 1, 0, 32)
        sum_79 = torch.ops.aten.sum.dim_IntList(slice_566, [1]);  slice_566 = None
        sum_80 = torch.ops.aten.sum.dim_IntList(slice_558, [1])
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(sign_4, 2)
        mul_71 = torch.ops.aten.mul.Tensor(slice_558, unsqueeze_16);  unsqueeze_16 = None
        sum_81 = torch.ops.aten.sum.dim_IntList(mul_71, [-2]);  mul_71 = None
        clone_15 = torch.ops.aten.clone.default(slice_558);  slice_558 = None
        view_208 = torch.ops.aten.view.default(clone_15, [1, batch_size, 64, 32]);  clone_15 = None
        sum_82 = torch.ops.aten.sum.dim_IntList(view_208, [0]);  view_208 = None
        addmm_113 = torch.ops.aten.addmm.default(arg322_1, cat_57, arg321_1);  arg322_1 = arg321_1 = None
        relu_43 = torch.ops.aten.relu.default(addmm_113);  addmm_113 = None
        addmm_114 = torch.ops.aten.addmm.default(arg324_1, relu_43, arg323_1);  arg324_1 = relu_43 = arg323_1 = None
        view_209 = torch.ops.aten.view.default(addmm_114, [-1, 32, 32]);  addmm_114 = None
        addmm_115 = torch.ops.aten.addmm.default(arg326_1, cat_57, arg325_1);  arg326_1 = arg325_1 = None
        relu_44 = torch.ops.aten.relu.default(addmm_115);  addmm_115 = None
        addmm_116 = torch.ops.aten.addmm.default(arg328_1, relu_44, arg327_1);  arg328_1 = relu_44 = arg327_1 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(addmm_116, 1);  addmm_116 = None
        expand_12 = torch.ops.aten.expand.default(sum_82, [batch_size, 64, 32]);  sum_82 = None
        expand_13 = torch.ops.aten.expand.default(view_209, [batch_size, 32, 32]);  view_209 = None
        bmm_4 = torch.ops.aten.bmm.default(expand_12, expand_13);  expand_12 = expand_13 = None
        add_4 = torch.ops.aten.add.Tensor(bmm_4, unsqueeze_17);  bmm_4 = unsqueeze_17 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(where_184, 1)
        expand_14 = torch.ops.aten.expand.default(unsqueeze_18, [-1, 64, -1]);  unsqueeze_18 = None
        sub_4 = torch.ops.aten.sub.Tensor(expand_14, add_4)
        mul_72 = torch.ops.aten.mul.Tensor(expand_14, add_4)
        cat_93 = torch.ops.aten.cat.default([expand_14, add_4, sub_4, mul_72], -1);  expand_14 = sub_4 = mul_72 = None
        view_213 = torch.ops.aten.view.default(cat_93, [12800, 128]);  cat_93 = None
        addmm_117 = torch.ops.aten.addmm.default(arg330_1, view_213, arg329_1);  arg330_1 = view_213 = arg329_1 = None
        view_214 = torch.ops.aten.view.default(addmm_117, [batch_size, 64, 256]);  addmm_117 = None
        relu_45 = torch.ops.aten.relu.default(view_214);  view_214 = None
        view_215 = torch.ops.aten.view.default(relu_45, [12800, 256]);  relu_45 = None
        addmm_118 = torch.ops.aten.addmm.default(arg332_1, view_215, arg331_1);  arg332_1 = view_215 = arg331_1 = None
        view_216 = torch.ops.aten.view.default(addmm_118, [batch_size, 64, 128]);  addmm_118 = None
        relu_46 = torch.ops.aten.relu.default(view_216);  view_216 = None
        view_217 = torch.ops.aten.view.default(relu_46, [12800, 128]);  relu_46 = None
        addmm_119 = torch.ops.aten.addmm.default(arg334_1, view_217, arg333_1);  arg334_1 = view_217 = arg333_1 = None
        view_218 = torch.ops.aten.view.default(addmm_119, [batch_size, 64, 32]);  addmm_119 = None
        mul_73 = torch.ops.aten.mul.Tensor(view_218, arg335_1);  view_218 = arg335_1 = None
        sigmoid_4 = torch.ops.aten.sigmoid.default(mul_73);  mul_73 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(sign_4, 2);  sign_4 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(unsqueeze_19, torch.float16);  unsqueeze_19 = None
        mul_74 = torch.ops.aten.mul.Tensor(sigmoid_4, convert_element_type_20);  sigmoid_4 = convert_element_type_20 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, add_4);  mul_74 = None
        sum_83 = torch.ops.aten.sum.dim_IntList(mul_75, [1]);  mul_75 = None
        mul_76 = torch.ops.aten.mul.Tensor(slice_560, add_4);  slice_560 = add_4 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(mul_76, [1]);  mul_76 = None
        cat_94 = torch.ops.aten.cat.default([sum_83, sum_84], 1);  sum_83 = sum_84 = None
        cat_95 = torch.ops.aten.cat.default([cat_94, addmm_112, sum_75, sum_76, sum_77, sum_78, sum_79, sum_80], -1);  addmm_112 = sum_75 = sum_76 = sum_77 = sum_78 = sum_79 = sum_80 = None
        addmm_120 = torch.ops.aten.addmm.default(arg337_1, where_184, arg336_1);  arg337_1 = arg336_1 = None
        slice_567 = torch.ops.aten.slice.Tensor(arg338_1, 2, 1, 34);  arg338_1 = None
        sign_5 = torch.ops.aten.sign.default(arg339_1);  arg339_1 = None
        slice_568 = torch.ops.aten.slice.Tensor(slice_567, 2, 0, 32)
        slice_569 = torch.ops.aten.slice.Tensor(slice_567, 2, 32, 9223372036854775807);  slice_567 = None
        slice_570 = torch.ops.aten.slice.Tensor(arg340_1, 2, 0, 32)
        slice_571 = torch.ops.aten.slice.Tensor(arg340_1, 2, 32, 9223372036854775807);  arg340_1 = None
        slice_572 = torch.ops.aten.slice.Tensor(slice_568, 1, 0, 2)
        sum_85 = torch.ops.aten.sum.dim_IntList(slice_572, [1]);  slice_572 = None
        slice_573 = torch.ops.aten.slice.Tensor(slice_568, 1, 0, 4)
        sum_86 = torch.ops.aten.sum.dim_IntList(slice_573, [1]);  slice_573 = None
        slice_574 = torch.ops.aten.slice.Tensor(slice_568, 1, 0, 8)
        sum_87 = torch.ops.aten.sum.dim_IntList(slice_574, [1]);  slice_574 = None
        slice_575 = torch.ops.aten.slice.Tensor(slice_568, 1, 0, 16)
        sum_88 = torch.ops.aten.sum.dim_IntList(slice_575, [1]);  slice_575 = None
        slice_576 = torch.ops.aten.slice.Tensor(slice_568, 1, 0, 32)
        sum_89 = torch.ops.aten.sum.dim_IntList(slice_576, [1]);  slice_576 = None
        sum_90 = torch.ops.aten.sum.dim_IntList(slice_568, [1])
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(sign_5, 2)
        mul_77 = torch.ops.aten.mul.Tensor(slice_568, unsqueeze_20);  unsqueeze_20 = None
        sum_91 = torch.ops.aten.sum.dim_IntList(mul_77, [-2]);  mul_77 = None
        clone_16 = torch.ops.aten.clone.default(slice_568);  slice_568 = None
        view_219 = torch.ops.aten.view.default(clone_16, [1, batch_size, 64, 32]);  clone_16 = None
        sum_92 = torch.ops.aten.sum.dim_IntList(view_219, [0]);  view_219 = None
        addmm_121 = torch.ops.aten.addmm.default(arg342_1, cat_57, arg341_1);  arg342_1 = arg341_1 = None
        relu_47 = torch.ops.aten.relu.default(addmm_121);  addmm_121 = None
        addmm_122 = torch.ops.aten.addmm.default(arg344_1, relu_47, arg343_1);  arg344_1 = relu_47 = arg343_1 = None
        view_220 = torch.ops.aten.view.default(addmm_122, [-1, 32, 32]);  addmm_122 = None
        addmm_123 = torch.ops.aten.addmm.default(arg346_1, cat_57, arg345_1);  arg346_1 = arg345_1 = None
        relu_48 = torch.ops.aten.relu.default(addmm_123);  addmm_123 = None
        addmm_124 = torch.ops.aten.addmm.default(arg348_1, relu_48, arg347_1);  arg348_1 = relu_48 = arg347_1 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(addmm_124, 1);  addmm_124 = None
        expand_15 = torch.ops.aten.expand.default(sum_92, [batch_size, 64, 32]);  sum_92 = None
        expand_16 = torch.ops.aten.expand.default(view_220, [batch_size, 32, 32]);  view_220 = None
        bmm_5 = torch.ops.aten.bmm.default(expand_15, expand_16);  expand_15 = expand_16 = None
        add_5 = torch.ops.aten.add.Tensor(bmm_5, unsqueeze_21);  bmm_5 = unsqueeze_21 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(where_184, 1)
        expand_17 = torch.ops.aten.expand.default(unsqueeze_22, [-1, 64, -1]);  unsqueeze_22 = None
        sub_5 = torch.ops.aten.sub.Tensor(expand_17, add_5)
        mul_78 = torch.ops.aten.mul.Tensor(expand_17, add_5)
        cat_96 = torch.ops.aten.cat.default([expand_17, add_5, sub_5, mul_78], -1);  expand_17 = sub_5 = mul_78 = None
        view_224 = torch.ops.aten.view.default(cat_96, [12800, 128]);  cat_96 = None
        addmm_125 = torch.ops.aten.addmm.default(arg350_1, view_224, arg349_1);  arg350_1 = view_224 = arg349_1 = None
        view_225 = torch.ops.aten.view.default(addmm_125, [batch_size, 64, 256]);  addmm_125 = None
        relu_49 = torch.ops.aten.relu.default(view_225);  view_225 = None
        view_226 = torch.ops.aten.view.default(relu_49, [12800, 256]);  relu_49 = None
        addmm_126 = torch.ops.aten.addmm.default(arg352_1, view_226, arg351_1);  arg352_1 = view_226 = arg351_1 = None
        view_227 = torch.ops.aten.view.default(addmm_126, [batch_size, 64, 128]);  addmm_126 = None
        relu_50 = torch.ops.aten.relu.default(view_227);  view_227 = None
        view_228 = torch.ops.aten.view.default(relu_50, [12800, 128]);  relu_50 = None
        addmm_127 = torch.ops.aten.addmm.default(arg354_1, view_228, arg353_1);  arg354_1 = view_228 = arg353_1 = None
        view_229 = torch.ops.aten.view.default(addmm_127, [batch_size, 64, 32]);  addmm_127 = None
        mul_79 = torch.ops.aten.mul.Tensor(view_229, arg355_1);  view_229 = arg355_1 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(mul_79);  mul_79 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(sign_5, 2);  sign_5 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(unsqueeze_23, torch.float16);  unsqueeze_23 = None
        mul_80 = torch.ops.aten.mul.Tensor(sigmoid_5, convert_element_type_23);  sigmoid_5 = convert_element_type_23 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, add_5);  mul_80 = None
        sum_93 = torch.ops.aten.sum.dim_IntList(mul_81, [1]);  mul_81 = None
        mul_82 = torch.ops.aten.mul.Tensor(slice_570, add_5);  slice_570 = add_5 = None
        sum_94 = torch.ops.aten.sum.dim_IntList(mul_82, [1]);  mul_82 = None
        cat_97 = torch.ops.aten.cat.default([sum_93, sum_94], 1);  sum_93 = sum_94 = None
        cat_98 = torch.ops.aten.cat.default([cat_97, addmm_120, sum_85, sum_86, sum_87, sum_88, sum_89, sum_90], -1);  addmm_120 = sum_85 = sum_86 = sum_87 = sum_88 = sum_89 = sum_90 = None
        addmm_128 = torch.ops.aten.addmm.default(arg357_1, where_191, arg356_1);  arg357_1 = arg356_1 = None
        slice_577 = torch.ops.aten.slice.Tensor(arg358_1, 2, 1, 130);  arg358_1 = None
        sign_6 = torch.ops.aten.sign.default(arg359_1);  arg359_1 = None
        slice_578 = torch.ops.aten.slice.Tensor(slice_577, 2, 0, 128)
        slice_579 = torch.ops.aten.slice.Tensor(slice_577, 2, 128, 9223372036854775807);  slice_577 = None
        slice_580 = torch.ops.aten.slice.Tensor(arg360_1, 2, 0, 32)
        slice_581 = torch.ops.aten.slice.Tensor(arg360_1, 2, 32, 9223372036854775807);  arg360_1 = None
        slice_582 = torch.ops.aten.slice.Tensor(slice_578, 1, 0, 2)
        sum_95 = torch.ops.aten.sum.dim_IntList(slice_582, [1]);  slice_582 = None
        slice_583 = torch.ops.aten.slice.Tensor(slice_578, 1, 0, 4)
        sum_96 = torch.ops.aten.sum.dim_IntList(slice_583, [1]);  slice_583 = None
        slice_584 = torch.ops.aten.slice.Tensor(slice_578, 1, 0, 8)
        sum_97 = torch.ops.aten.sum.dim_IntList(slice_584, [1]);  slice_584 = None
        slice_585 = torch.ops.aten.slice.Tensor(slice_578, 1, 0, 16)
        sum_98 = torch.ops.aten.sum.dim_IntList(slice_585, [1]);  slice_585 = None
        slice_586 = torch.ops.aten.slice.Tensor(slice_578, 1, 0, 32)
        sum_99 = torch.ops.aten.sum.dim_IntList(slice_586, [1]);  slice_586 = None
        sum_100 = torch.ops.aten.sum.dim_IntList(slice_578, [1])
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(sign_6, 2)
        mul_83 = torch.ops.aten.mul.Tensor(slice_578, unsqueeze_24);  unsqueeze_24 = None
        sum_101 = torch.ops.aten.sum.dim_IntList(mul_83, [-2]);  mul_83 = None
        clone_17 = torch.ops.aten.clone.default(slice_578);  slice_578 = None
        view_230 = torch.ops.aten.view.default(clone_17, [1, batch_size, 64, 128]);  clone_17 = None
        sum_102 = torch.ops.aten.sum.dim_IntList(view_230, [0]);  view_230 = None
        addmm_129 = torch.ops.aten.addmm.default(arg362_1, cat_57, arg361_1);  arg362_1 = arg361_1 = None
        relu_51 = torch.ops.aten.relu.default(addmm_129);  addmm_129 = None
        addmm_130 = torch.ops.aten.addmm.default(arg364_1, relu_51, arg363_1);  arg364_1 = relu_51 = arg363_1 = None
        view_231 = torch.ops.aten.view.default(addmm_130, [-1, 128, 32]);  addmm_130 = None
        addmm_131 = torch.ops.aten.addmm.default(arg366_1, cat_57, arg365_1);  arg366_1 = arg365_1 = None
        relu_52 = torch.ops.aten.relu.default(addmm_131);  addmm_131 = None
        addmm_132 = torch.ops.aten.addmm.default(arg368_1, relu_52, arg367_1);  arg368_1 = relu_52 = arg367_1 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(addmm_132, 1);  addmm_132 = None
        expand_18 = torch.ops.aten.expand.default(sum_102, [batch_size, 64, 128]);  sum_102 = None
        expand_19 = torch.ops.aten.expand.default(view_231, [batch_size, 128, 32]);  view_231 = None
        bmm_6 = torch.ops.aten.bmm.default(expand_18, expand_19);  expand_18 = expand_19 = None
        add_6 = torch.ops.aten.add.Tensor(bmm_6, unsqueeze_25);  bmm_6 = unsqueeze_25 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(where_191, 1)
        expand_20 = torch.ops.aten.expand.default(unsqueeze_26, [-1, 64, -1]);  unsqueeze_26 = None
        sub_6 = torch.ops.aten.sub.Tensor(expand_20, add_6)
        mul_84 = torch.ops.aten.mul.Tensor(expand_20, add_6)
        cat_99 = torch.ops.aten.cat.default([expand_20, add_6, sub_6, mul_84], -1);  expand_20 = sub_6 = mul_84 = None
        view_235 = torch.ops.aten.view.default(cat_99, [12800, 128]);  cat_99 = None
        addmm_133 = torch.ops.aten.addmm.default(arg370_1, view_235, arg369_1);  arg370_1 = view_235 = arg369_1 = None
        view_236 = torch.ops.aten.view.default(addmm_133, [batch_size, 64, 256]);  addmm_133 = None
        relu_53 = torch.ops.aten.relu.default(view_236);  view_236 = None
        view_237 = torch.ops.aten.view.default(relu_53, [12800, 256]);  relu_53 = None
        addmm_134 = torch.ops.aten.addmm.default(arg372_1, view_237, arg371_1);  arg372_1 = view_237 = arg371_1 = None
        view_238 = torch.ops.aten.view.default(addmm_134, [batch_size, 64, 128]);  addmm_134 = None
        relu_54 = torch.ops.aten.relu.default(view_238);  view_238 = None
        view_239 = torch.ops.aten.view.default(relu_54, [12800, 128]);  relu_54 = None
        addmm_135 = torch.ops.aten.addmm.default(arg374_1, view_239, arg373_1);  arg374_1 = view_239 = arg373_1 = None
        view_240 = torch.ops.aten.view.default(addmm_135, [batch_size, 64, 32]);  addmm_135 = None
        mul_85 = torch.ops.aten.mul.Tensor(view_240, arg375_1);  view_240 = arg375_1 = None
        sigmoid_6 = torch.ops.aten.sigmoid.default(mul_85);  mul_85 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(sign_6, 2);  sign_6 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(unsqueeze_27, torch.float16);  unsqueeze_27 = None
        mul_86 = torch.ops.aten.mul.Tensor(sigmoid_6, convert_element_type_26);  sigmoid_6 = convert_element_type_26 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_86, add_6);  mul_86 = None
        sum_103 = torch.ops.aten.sum.dim_IntList(mul_87, [1]);  mul_87 = None
        mul_88 = torch.ops.aten.mul.Tensor(slice_580, add_6);  slice_580 = add_6 = None
        sum_104 = torch.ops.aten.sum.dim_IntList(mul_88, [1]);  mul_88 = None
        cat_100 = torch.ops.aten.cat.default([sum_103, sum_104], 1);  sum_103 = sum_104 = None
        cat_101 = torch.ops.aten.cat.default([cat_100, addmm_128, sum_95, sum_96, sum_97, sum_98, sum_99, sum_100], -1);  addmm_128 = sum_95 = sum_96 = sum_97 = sum_98 = sum_99 = sum_100 = None
        addmm_136 = torch.ops.aten.addmm.default(arg377_1, where_149, arg376_1);  arg377_1 = arg376_1 = None
        slice_587 = torch.ops.aten.slice.Tensor(arg378_1, 2, 1, 66);  arg378_1 = None
        sign_7 = torch.ops.aten.sign.default(arg379_1);  arg379_1 = None
        slice_588 = torch.ops.aten.slice.Tensor(slice_587, 2, 0, 64)
        slice_589 = torch.ops.aten.slice.Tensor(slice_587, 2, 64, 9223372036854775807);  slice_587 = None
        slice_590 = torch.ops.aten.slice.Tensor(arg380_1, 2, 0, 32)
        slice_591 = torch.ops.aten.slice.Tensor(arg380_1, 2, 32, 9223372036854775807);  arg380_1 = None
        slice_592 = torch.ops.aten.slice.Tensor(slice_588, 1, 0, 2)
        sum_105 = torch.ops.aten.sum.dim_IntList(slice_592, [1]);  slice_592 = None
        slice_593 = torch.ops.aten.slice.Tensor(slice_588, 1, 0, 4)
        sum_106 = torch.ops.aten.sum.dim_IntList(slice_593, [1]);  slice_593 = None
        slice_594 = torch.ops.aten.slice.Tensor(slice_588, 1, 0, 8)
        sum_107 = torch.ops.aten.sum.dim_IntList(slice_594, [1]);  slice_594 = None
        slice_595 = torch.ops.aten.slice.Tensor(slice_588, 1, 0, 16)
        sum_108 = torch.ops.aten.sum.dim_IntList(slice_595, [1]);  slice_595 = None
        slice_596 = torch.ops.aten.slice.Tensor(slice_588, 1, 0, 32)
        sum_109 = torch.ops.aten.sum.dim_IntList(slice_596, [1]);  slice_596 = None
        slice_597 = torch.ops.aten.slice.Tensor(slice_588, 1, 0, 64)
        sum_110 = torch.ops.aten.sum.dim_IntList(slice_597, [1]);  slice_597 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(sign_7, 2)
        mul_89 = torch.ops.aten.mul.Tensor(slice_588, unsqueeze_28);  unsqueeze_28 = None
        sum_111 = torch.ops.aten.sum.dim_IntList(mul_89, [-2]);  mul_89 = None
        clone_18 = torch.ops.aten.clone.default(slice_588);  slice_588 = None
        view_241 = torch.ops.aten.view.default(clone_18, [1, batch_size, 80, 64]);  clone_18 = None
        sum_112 = torch.ops.aten.sum.dim_IntList(view_241, [0]);  view_241 = None
        addmm_137 = torch.ops.aten.addmm.default(arg382_1, cat_57, arg381_1);  arg382_1 = arg381_1 = None
        relu_55 = torch.ops.aten.relu.default(addmm_137);  addmm_137 = None
        addmm_138 = torch.ops.aten.addmm.default(arg384_1, relu_55, arg383_1);  arg384_1 = relu_55 = arg383_1 = None
        view_242 = torch.ops.aten.view.default(addmm_138, [-1, 64, 32]);  addmm_138 = None
        addmm_139 = torch.ops.aten.addmm.default(arg386_1, cat_57, arg385_1);  arg386_1 = arg385_1 = None
        relu_56 = torch.ops.aten.relu.default(addmm_139);  addmm_139 = None
        addmm_140 = torch.ops.aten.addmm.default(arg388_1, relu_56, arg387_1);  arg388_1 = relu_56 = arg387_1 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(addmm_140, 1);  addmm_140 = None
        expand_21 = torch.ops.aten.expand.default(sum_112, [batch_size, 80, 64]);  sum_112 = None
        expand_22 = torch.ops.aten.expand.default(view_242, [batch_size, 64, 32]);  view_242 = None
        bmm_7 = torch.ops.aten.bmm.default(expand_21, expand_22);  expand_21 = expand_22 = None
        add_7 = torch.ops.aten.add.Tensor(bmm_7, unsqueeze_29);  bmm_7 = unsqueeze_29 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(where_149, 1)
        expand_23 = torch.ops.aten.expand.default(unsqueeze_30, [-1, 80, -1]);  unsqueeze_30 = None
        sub_7 = torch.ops.aten.sub.Tensor(expand_23, add_7)
        mul_90 = torch.ops.aten.mul.Tensor(expand_23, add_7)
        cat_102 = torch.ops.aten.cat.default([expand_23, add_7, sub_7, mul_90], -1);  expand_23 = sub_7 = mul_90 = None
        view_246 = torch.ops.aten.view.default(cat_102, [16000, 128]);  cat_102 = None
        addmm_141 = torch.ops.aten.addmm.default(arg390_1, view_246, arg389_1);  arg390_1 = view_246 = arg389_1 = None
        view_247 = torch.ops.aten.view.default(addmm_141, [batch_size, 80, 256]);  addmm_141 = None
        relu_57 = torch.ops.aten.relu.default(view_247);  view_247 = None
        view_248 = torch.ops.aten.view.default(relu_57, [16000, 256]);  relu_57 = None
        addmm_142 = torch.ops.aten.addmm.default(arg392_1, view_248, arg391_1);  arg392_1 = view_248 = arg391_1 = None
        view_249 = torch.ops.aten.view.default(addmm_142, [batch_size, 80, 128]);  addmm_142 = None
        relu_58 = torch.ops.aten.relu.default(view_249);  view_249 = None
        view_250 = torch.ops.aten.view.default(relu_58, [16000, 128]);  relu_58 = None
        addmm_143 = torch.ops.aten.addmm.default(arg394_1, view_250, arg393_1);  arg394_1 = view_250 = arg393_1 = None
        view_251 = torch.ops.aten.view.default(addmm_143, [batch_size, 80, 32]);  addmm_143 = None
        mul_91 = torch.ops.aten.mul.Tensor(view_251, arg395_1);  view_251 = arg395_1 = None
        sigmoid_7 = torch.ops.aten.sigmoid.default(mul_91);  mul_91 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(sign_7, 2);  sign_7 = None
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(unsqueeze_31, torch.float16);  unsqueeze_31 = None
        mul_92 = torch.ops.aten.mul.Tensor(sigmoid_7, convert_element_type_29);  sigmoid_7 = convert_element_type_29 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, add_7);  mul_92 = None
        sum_113 = torch.ops.aten.sum.dim_IntList(mul_93, [1]);  mul_93 = None
        mul_94 = torch.ops.aten.mul.Tensor(slice_590, add_7);  slice_590 = add_7 = None
        sum_114 = torch.ops.aten.sum.dim_IntList(mul_94, [1]);  mul_94 = None
        cat_103 = torch.ops.aten.cat.default([sum_113, sum_114], 1);  sum_113 = sum_114 = None
        cat_104 = torch.ops.aten.cat.default([cat_103, addmm_136, sum_105, sum_106, sum_107, sum_108, sum_109, sum_110], -1);  addmm_136 = sum_105 = sum_106 = sum_107 = sum_108 = sum_109 = sum_110 = None
        addmm_144 = torch.ops.aten.addmm.default(arg397_1, where_163, arg396_1);  arg397_1 = arg396_1 = None
        slice_598 = torch.ops.aten.slice.Tensor(arg398_1, 2, 1, 66)
        sign_8 = torch.ops.aten.sign.default(arg399_1)
        slice_599 = torch.ops.aten.slice.Tensor(slice_598, 2, 0, 64)
        slice_600 = torch.ops.aten.slice.Tensor(slice_598, 2, 64, 9223372036854775807);  slice_598 = None
        cumsum_6 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_6 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_6 = torch.ops.aten.searchsorted.Tensor(cumsum_6, iota_6, out_int32 = True, right = True);  cumsum_6 = iota_6 = None
        clamp_max_6 = torch.ops.aten.clamp_max.default(searchsorted_6, 6);  searchsorted_6 = None
        index_6 = torch.ops.aten.index.Tensor(slice_600, [clamp_max_6]);  slice_600 = clamp_max_6 = None
        slice_601 = torch.ops.aten.slice.Tensor(arg400_1, 2, 0, 64)
        slice_602 = torch.ops.aten.slice.Tensor(arg400_1, 2, 64, 9223372036854775807);  arg400_1 = None
        slice_603 = torch.ops.aten.slice.Tensor(slice_599, 1, 0, 64);  slice_599 = None
        slice_604 = torch.ops.aten.slice.Tensor(sign_8, 1, 0, 64);  sign_8 = None
        slice_605 = torch.ops.aten.slice.Tensor(slice_601, 1, 0, 64);  slice_601 = None
        cumsum_7 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_7 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_7 = torch.ops.aten.searchsorted.Tensor(cumsum_7, iota_7, out_int32 = True, right = True);  cumsum_7 = iota_7 = None
        clamp_max_7 = torch.ops.aten.clamp_max.default(searchsorted_7, 6);  searchsorted_7 = None
        index_7 = torch.ops.aten.index.Tensor(slice_603, [clamp_max_7]);  slice_603 = clamp_max_7 = None
        cumsum_8 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_8 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_8 = torch.ops.aten.searchsorted.Tensor(cumsum_8, iota_8, out_int32 = True, right = True);  cumsum_8 = iota_8 = None
        clamp_max_8 = torch.ops.aten.clamp_max.default(searchsorted_8, 6);  searchsorted_8 = None
        index_8 = torch.ops.aten.index.Tensor(slice_604, [clamp_max_8]);  slice_604 = clamp_max_8 = None
        slice_606 = torch.ops.aten.slice.Tensor(index_7, 1, 0, 2)
        sum_115 = torch.ops.aten.sum.dim_IntList(slice_606, [1]);  slice_606 = None
        slice_607 = torch.ops.aten.slice.Tensor(index_7, 1, 0, 4)
        sum_116 = torch.ops.aten.sum.dim_IntList(slice_607, [1]);  slice_607 = None
        slice_608 = torch.ops.aten.slice.Tensor(index_7, 1, 0, 8)
        sum_117 = torch.ops.aten.sum.dim_IntList(slice_608, [1]);  slice_608 = None
        slice_609 = torch.ops.aten.slice.Tensor(index_7, 1, 0, 16)
        sum_118 = torch.ops.aten.sum.dim_IntList(slice_609, [1]);  slice_609 = None
        slice_610 = torch.ops.aten.slice.Tensor(index_7, 1, 0, 32)
        sum_119 = torch.ops.aten.sum.dim_IntList(slice_610, [1]);  slice_610 = None
        sum_120 = torch.ops.aten.sum.dim_IntList(index_7, [1])
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(index_8, 2)
        mul_95 = torch.ops.aten.mul.Tensor(index_7, unsqueeze_32);  unsqueeze_32 = None
        sum_121 = torch.ops.aten.sum.dim_IntList(mul_95, [-2]);  mul_95 = None
        view_252 = torch.ops.aten.view.default(index_7, [1, batch_size, 64, 64]);  index_7 = None
        sum_122 = torch.ops.aten.sum.dim_IntList(view_252, [0]);  view_252 = None
        addmm_145 = torch.ops.aten.addmm.default(arg402_1, cat_57, arg401_1);  arg402_1 = arg401_1 = None
        relu_59 = torch.ops.aten.relu.default(addmm_145);  addmm_145 = None
        addmm_146 = torch.ops.aten.addmm.default(arg404_1, relu_59, arg403_1);  arg404_1 = relu_59 = arg403_1 = None
        view_253 = torch.ops.aten.view.default(addmm_146, [-1, 64, 64]);  addmm_146 = None
        addmm_147 = torch.ops.aten.addmm.default(arg406_1, cat_57, arg405_1);  arg406_1 = arg405_1 = None
        relu_60 = torch.ops.aten.relu.default(addmm_147);  addmm_147 = None
        addmm_148 = torch.ops.aten.addmm.default(arg408_1, relu_60, arg407_1);  arg408_1 = relu_60 = arg407_1 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(addmm_148, 1);  addmm_148 = None
        expand_24 = torch.ops.aten.expand.default(sum_122, [batch_size, 64, 64]);  sum_122 = None
        expand_25 = torch.ops.aten.expand.default(view_253, [batch_size, 64, 64]);  view_253 = None
        bmm_8 = torch.ops.aten.bmm.default(expand_24, expand_25);  expand_24 = expand_25 = None
        add_8 = torch.ops.aten.add.Tensor(bmm_8, unsqueeze_33);  bmm_8 = unsqueeze_33 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(where_163, 1)
        expand_26 = torch.ops.aten.expand.default(unsqueeze_34, [-1, 64, -1]);  unsqueeze_34 = None
        sub_8 = torch.ops.aten.sub.Tensor(expand_26, add_8)
        mul_96 = torch.ops.aten.mul.Tensor(expand_26, add_8)
        cat_105 = torch.ops.aten.cat.default([expand_26, add_8, sub_8, mul_96], -1);  expand_26 = sub_8 = mul_96 = None
        view_257 = torch.ops.aten.view.default(cat_105, [12800, 256]);  cat_105 = None
        addmm_149 = torch.ops.aten.addmm.default(arg410_1, view_257, arg409_1);  arg410_1 = view_257 = arg409_1 = None
        view_258 = torch.ops.aten.view.default(addmm_149, [batch_size, 64, 256]);  addmm_149 = None
        relu_61 = torch.ops.aten.relu.default(view_258);  view_258 = None
        view_259 = torch.ops.aten.view.default(relu_61, [12800, 256]);  relu_61 = None
        addmm_150 = torch.ops.aten.addmm.default(arg412_1, view_259, arg411_1);  arg412_1 = view_259 = arg411_1 = None
        view_260 = torch.ops.aten.view.default(addmm_150, [batch_size, 64, 128]);  addmm_150 = None
        relu_62 = torch.ops.aten.relu.default(view_260);  view_260 = None
        view_261 = torch.ops.aten.view.default(relu_62, [12800, 128]);  relu_62 = None
        addmm_151 = torch.ops.aten.addmm.default(arg414_1, view_261, arg413_1);  arg414_1 = view_261 = arg413_1 = None
        view_262 = torch.ops.aten.view.default(addmm_151, [batch_size, 64, 64]);  addmm_151 = None
        mul_97 = torch.ops.aten.mul.Tensor(view_262, arg415_1);  view_262 = arg415_1 = None
        sigmoid_8 = torch.ops.aten.sigmoid.default(mul_97);  mul_97 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(index_8, 2);  index_8 = None
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(unsqueeze_35, torch.float16);  unsqueeze_35 = None
        mul_98 = torch.ops.aten.mul.Tensor(sigmoid_8, convert_element_type_32);  sigmoid_8 = convert_element_type_32 = None
        mul_99 = torch.ops.aten.mul.Tensor(mul_98, add_8);  mul_98 = None
        sum_123 = torch.ops.aten.sum.dim_IntList(mul_99, [1]);  mul_99 = None
        mul_100 = torch.ops.aten.mul.Tensor(slice_605, add_8);  slice_605 = add_8 = None
        sum_124 = torch.ops.aten.sum.dim_IntList(mul_100, [1]);  mul_100 = None
        cat_106 = torch.ops.aten.cat.default([sum_123, sum_124], 1);  sum_123 = sum_124 = None
        cat_107 = torch.ops.aten.cat.default([cat_106, addmm_144, sum_115, sum_116, sum_117, sum_118, sum_119, sum_120], -1);  addmm_144 = sum_115 = sum_116 = sum_117 = sum_118 = sum_119 = sum_120 = None
        full_default_195 = torch.ops.aten.full.default([batch_size, 128], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_196 = torch.ops.aten.where.self(logical_or_12, full_default_195, cat_106);  full_default_195 = cat_106 = None
        full_default_196 = torch.ops.aten.full.default([batch_size, 640], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_197 = torch.ops.aten.where.self(logical_or_12, full_default_196, cat_107);  full_default_196 = cat_107 = None
        addmm_152 = torch.ops.aten.addmm.default(arg417_1, where_128, arg416_1);  arg417_1 = arg416_1 = None
        full_196 = torch.ops.aten.full.default([7, 696, 33], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        full_197 = torch.ops.aten.full.default([7, 696], 0, dtype = torch.int32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        slice_612 = torch.ops.aten.slice.Tensor(full_196, 2, 0, 32)
        slice_613 = torch.ops.aten.slice.Tensor(full_196, 2, 32, 9223372036854775807);  full_196 = None
        cumsum_9 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_9 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_9 = torch.ops.aten.searchsorted.Tensor(cumsum_9, iota_9, out_int32 = True, right = True);  cumsum_9 = iota_9 = None
        clamp_max_9 = torch.ops.aten.clamp_max.default(searchsorted_9, 6);  searchsorted_9 = None
        index_9 = torch.ops.aten.index.Tensor(slice_613, [clamp_max_9]);  slice_613 = clamp_max_9 = None
        slice_614 = torch.ops.aten.slice.Tensor(arg420_1, 2, 0, 32)
        slice_615 = torch.ops.aten.slice.Tensor(arg420_1, 2, 32, 9223372036854775807);  arg420_1 = None
        slice_616 = torch.ops.aten.slice.Tensor(slice_612, 1, 0, 64);  slice_612 = None
        slice_617 = torch.ops.aten.slice.Tensor(full_197, 1, 0, 64);  full_197 = None
        slice_618 = torch.ops.aten.slice.Tensor(slice_614, 1, 0, 64);  slice_614 = None
        cumsum_10 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_10 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_10 = torch.ops.aten.searchsorted.Tensor(cumsum_10, iota_10, out_int32 = True, right = True);  cumsum_10 = iota_10 = None
        clamp_max_10 = torch.ops.aten.clamp_max.default(searchsorted_10, 6);  searchsorted_10 = None
        index_10 = torch.ops.aten.index.Tensor(slice_616, [clamp_max_10]);  slice_616 = clamp_max_10 = None
        cumsum_11 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_11 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_11 = torch.ops.aten.searchsorted.Tensor(cumsum_11, iota_11, out_int32 = True, right = True);  cumsum_11 = iota_11 = None
        clamp_max_11 = torch.ops.aten.clamp_max.default(searchsorted_11, 6);  searchsorted_11 = None
        index_11 = torch.ops.aten.index.Tensor(slice_617, [clamp_max_11]);  slice_617 = clamp_max_11 = None
        slice_619 = torch.ops.aten.slice.Tensor(index_10, 1, 0, 2)
        sum_125 = torch.ops.aten.sum.dim_IntList(slice_619, [1]);  slice_619 = None
        slice_620 = torch.ops.aten.slice.Tensor(index_10, 1, 0, 4)
        sum_126 = torch.ops.aten.sum.dim_IntList(slice_620, [1]);  slice_620 = None
        slice_621 = torch.ops.aten.slice.Tensor(index_10, 1, 0, 8)
        sum_127 = torch.ops.aten.sum.dim_IntList(slice_621, [1]);  slice_621 = None
        slice_622 = torch.ops.aten.slice.Tensor(index_10, 1, 0, 16)
        sum_128 = torch.ops.aten.sum.dim_IntList(slice_622, [1]);  slice_622 = None
        slice_623 = torch.ops.aten.slice.Tensor(index_10, 1, 0, 32)
        sum_129 = torch.ops.aten.sum.dim_IntList(slice_623, [1]);  slice_623 = None
        sum_130 = torch.ops.aten.sum.dim_IntList(index_10, [1])
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(index_11, 2)
        mul_101 = torch.ops.aten.mul.Tensor(index_10, unsqueeze_36);  unsqueeze_36 = None
        sum_131 = torch.ops.aten.sum.dim_IntList(mul_101, [-2]);  mul_101 = None
        view_263 = torch.ops.aten.view.default(index_10, [1, batch_size, 64, 32]);  index_10 = None
        sum_132 = torch.ops.aten.sum.dim_IntList(view_263, [0]);  view_263 = None
        addmm_153 = torch.ops.aten.addmm.default(arg422_1, cat_57, arg421_1);  arg422_1 = arg421_1 = None
        relu_63 = torch.ops.aten.relu.default(addmm_153);  addmm_153 = None
        addmm_154 = torch.ops.aten.addmm.default(arg424_1, relu_63, arg423_1);  arg424_1 = relu_63 = arg423_1 = None
        view_264 = torch.ops.aten.view.default(addmm_154, [-1, 32, 32]);  addmm_154 = None
        addmm_155 = torch.ops.aten.addmm.default(arg426_1, cat_57, arg425_1);  arg426_1 = arg425_1 = None
        relu_64 = torch.ops.aten.relu.default(addmm_155);  addmm_155 = None
        addmm_156 = torch.ops.aten.addmm.default(arg428_1, relu_64, arg427_1);  arg428_1 = relu_64 = arg427_1 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(addmm_156, 1);  addmm_156 = None
        expand_27 = torch.ops.aten.expand.default(sum_132, [batch_size, 64, 32]);  sum_132 = None
        expand_28 = torch.ops.aten.expand.default(view_264, [batch_size, 32, 32]);  view_264 = None
        bmm_9 = torch.ops.aten.bmm.default(expand_27, expand_28);  expand_27 = expand_28 = None
        add_9 = torch.ops.aten.add.Tensor(bmm_9, unsqueeze_37);  bmm_9 = unsqueeze_37 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(where_128, 1)
        expand_29 = torch.ops.aten.expand.default(unsqueeze_38, [-1, 64, -1]);  unsqueeze_38 = None
        sub_9 = torch.ops.aten.sub.Tensor(expand_29, add_9)
        mul_102 = torch.ops.aten.mul.Tensor(expand_29, add_9)
        cat_108 = torch.ops.aten.cat.default([expand_29, add_9, sub_9, mul_102], -1);  expand_29 = sub_9 = mul_102 = None
        view_268 = torch.ops.aten.view.default(cat_108, [12800, 128]);  cat_108 = None
        addmm_157 = torch.ops.aten.addmm.default(arg430_1, view_268, arg429_1);  arg430_1 = view_268 = arg429_1 = None
        view_269 = torch.ops.aten.view.default(addmm_157, [batch_size, 64, 256]);  addmm_157 = None
        relu_65 = torch.ops.aten.relu.default(view_269);  view_269 = None
        view_270 = torch.ops.aten.view.default(relu_65, [12800, 256]);  relu_65 = None
        addmm_158 = torch.ops.aten.addmm.default(arg432_1, view_270, arg431_1);  arg432_1 = view_270 = arg431_1 = None
        view_271 = torch.ops.aten.view.default(addmm_158, [batch_size, 64, 128]);  addmm_158 = None
        relu_66 = torch.ops.aten.relu.default(view_271);  view_271 = None
        view_272 = torch.ops.aten.view.default(relu_66, [12800, 128]);  relu_66 = None
        addmm_159 = torch.ops.aten.addmm.default(arg434_1, view_272, arg433_1);  arg434_1 = view_272 = arg433_1 = None
        view_273 = torch.ops.aten.view.default(addmm_159, [batch_size, 64, 32]);  addmm_159 = None
        mul_103 = torch.ops.aten.mul.Tensor(view_273, arg435_1);  view_273 = arg435_1 = None
        sigmoid_9 = torch.ops.aten.sigmoid.default(mul_103);  mul_103 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(index_11, 2);  index_11 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(unsqueeze_39, torch.float16);  unsqueeze_39 = None
        mul_104 = torch.ops.aten.mul.Tensor(sigmoid_9, convert_element_type_35);  sigmoid_9 = convert_element_type_35 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, add_9);  mul_104 = None
        sum_133 = torch.ops.aten.sum.dim_IntList(mul_105, [1]);  mul_105 = None
        mul_106 = torch.ops.aten.mul.Tensor(slice_618, add_9);  slice_618 = add_9 = None
        sum_134 = torch.ops.aten.sum.dim_IntList(mul_106, [1]);  mul_106 = None
        cat_109 = torch.ops.aten.cat.default([sum_133, sum_134], 1);  sum_133 = sum_134 = None
        cat_110 = torch.ops.aten.cat.default([cat_109, addmm_152, sum_125, sum_126, sum_127, sum_128, sum_129, sum_130], -1);  addmm_152 = sum_125 = sum_126 = sum_127 = sum_128 = sum_129 = sum_130 = None
        full_default_197 = torch.ops.aten.full.default([batch_size, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_198 = torch.ops.aten.where.self(logical_or_12, full_default_197, cat_109);  full_default_197 = cat_109 = None
        full_default_198 = torch.ops.aten.full.default([batch_size, 320], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_199 = torch.ops.aten.where.self(logical_or_12, full_default_198, cat_110);  full_default_198 = cat_110 = None
        addmm_160 = torch.ops.aten.addmm.default(arg437_1, where_121, arg436_1);  arg437_1 = arg436_1 = None
        slice_624 = torch.ops.aten.slice.Tensor(arg438_1, 2, 1, 34)
        sign_10 = torch.ops.aten.sign.default(arg439_1)
        slice_625 = torch.ops.aten.slice.Tensor(slice_624, 2, 0, 32)
        slice_626 = torch.ops.aten.slice.Tensor(slice_624, 2, 32, 9223372036854775807);  slice_624 = None
        cumsum_12 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_12 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_12 = torch.ops.aten.searchsorted.Tensor(cumsum_12, iota_12, out_int32 = True, right = True);  cumsum_12 = iota_12 = None
        clamp_max_12 = torch.ops.aten.clamp_max.default(searchsorted_12, 6);  searchsorted_12 = None
        index_12 = torch.ops.aten.index.Tensor(slice_626, [clamp_max_12]);  slice_626 = clamp_max_12 = None
        slice_627 = torch.ops.aten.slice.Tensor(arg440_1, 2, 0, 32)
        slice_628 = torch.ops.aten.slice.Tensor(arg440_1, 2, 32, 9223372036854775807);  arg440_1 = None
        slice_629 = torch.ops.aten.slice.Tensor(slice_625, 1, 0, 64);  slice_625 = None
        slice_630 = torch.ops.aten.slice.Tensor(sign_10, 1, 0, 64);  sign_10 = None
        slice_631 = torch.ops.aten.slice.Tensor(slice_627, 1, 0, 64);  slice_627 = None
        cumsum_13 = torch.ops.aten.cumsum.default(convert_element_type_5, 0)
        iota_13 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_13 = torch.ops.aten.searchsorted.Tensor(cumsum_13, iota_13, out_int32 = True, right = True);  cumsum_13 = iota_13 = None
        clamp_max_13 = torch.ops.aten.clamp_max.default(searchsorted_13, 6);  searchsorted_13 = None
        index_13 = torch.ops.aten.index.Tensor(slice_629, [clamp_max_13]);  slice_629 = clamp_max_13 = None
        cumsum_14 = torch.ops.aten.cumsum.default(convert_element_type_5, 0);  convert_element_type_5 = None
        iota_14 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        searchsorted_14 = torch.ops.aten.searchsorted.Tensor(cumsum_14, iota_14, out_int32 = True, right = True);  cumsum_14 = iota_14 = None
        clamp_max_14 = torch.ops.aten.clamp_max.default(searchsorted_14, 6);  searchsorted_14 = None
        index_14 = torch.ops.aten.index.Tensor(slice_630, [clamp_max_14]);  slice_630 = clamp_max_14 = None
        slice_632 = torch.ops.aten.slice.Tensor(index_13, 1, 0, 2)
        sum_135 = torch.ops.aten.sum.dim_IntList(slice_632, [1]);  slice_632 = None
        slice_633 = torch.ops.aten.slice.Tensor(index_13, 1, 0, 4)
        sum_136 = torch.ops.aten.sum.dim_IntList(slice_633, [1]);  slice_633 = None
        slice_634 = torch.ops.aten.slice.Tensor(index_13, 1, 0, 8)
        sum_137 = torch.ops.aten.sum.dim_IntList(slice_634, [1]);  slice_634 = None
        slice_635 = torch.ops.aten.slice.Tensor(index_13, 1, 0, 16)
        sum_138 = torch.ops.aten.sum.dim_IntList(slice_635, [1]);  slice_635 = None
        slice_636 = torch.ops.aten.slice.Tensor(index_13, 1, 0, 32)
        sum_139 = torch.ops.aten.sum.dim_IntList(slice_636, [1]);  slice_636 = None
        sum_140 = torch.ops.aten.sum.dim_IntList(index_13, [1])
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(index_14, 2)
        mul_107 = torch.ops.aten.mul.Tensor(index_13, unsqueeze_40);  unsqueeze_40 = None
        sum_141 = torch.ops.aten.sum.dim_IntList(mul_107, [-2]);  mul_107 = None
        view_274 = torch.ops.aten.view.default(index_13, [1, batch_size, 64, 32]);  index_13 = None
        sum_142 = torch.ops.aten.sum.dim_IntList(view_274, [0]);  view_274 = None
        addmm_161 = torch.ops.aten.addmm.default(arg442_1, cat_57, arg441_1);  arg442_1 = arg441_1 = None
        relu_67 = torch.ops.aten.relu.default(addmm_161);  addmm_161 = None
        addmm_162 = torch.ops.aten.addmm.default(arg444_1, relu_67, arg443_1);  arg444_1 = relu_67 = arg443_1 = None
        view_275 = torch.ops.aten.view.default(addmm_162, [-1, 32, 32]);  addmm_162 = None
        addmm_163 = torch.ops.aten.addmm.default(arg446_1, cat_57, arg445_1);  arg446_1 = arg445_1 = None
        relu_68 = torch.ops.aten.relu.default(addmm_163);  addmm_163 = None
        addmm_164 = torch.ops.aten.addmm.default(arg448_1, relu_68, arg447_1);  arg448_1 = relu_68 = arg447_1 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(addmm_164, 1);  addmm_164 = None
        expand_30 = torch.ops.aten.expand.default(sum_142, [batch_size, 64, 32]);  sum_142 = None
        expand_31 = torch.ops.aten.expand.default(view_275, [batch_size, 32, 32]);  view_275 = None
        bmm_10 = torch.ops.aten.bmm.default(expand_30, expand_31);  expand_30 = expand_31 = None
        add_10 = torch.ops.aten.add.Tensor(bmm_10, unsqueeze_41);  bmm_10 = unsqueeze_41 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(where_121, 1)
        expand_32 = torch.ops.aten.expand.default(unsqueeze_42, [-1, 64, -1]);  unsqueeze_42 = None
        sub_10 = torch.ops.aten.sub.Tensor(expand_32, add_10)
        mul_108 = torch.ops.aten.mul.Tensor(expand_32, add_10)
        cat_111 = torch.ops.aten.cat.default([expand_32, add_10, sub_10, mul_108], -1);  expand_32 = sub_10 = mul_108 = None
        view_279 = torch.ops.aten.view.default(cat_111, [12800, 128]);  cat_111 = None
        addmm_165 = torch.ops.aten.addmm.default(arg450_1, view_279, arg449_1);  arg450_1 = view_279 = arg449_1 = None
        view_280 = torch.ops.aten.view.default(addmm_165, [batch_size, 64, 256]);  addmm_165 = None
        relu_69 = torch.ops.aten.relu.default(view_280);  view_280 = None
        view_281 = torch.ops.aten.view.default(relu_69, [12800, 256]);  relu_69 = None
        addmm_166 = torch.ops.aten.addmm.default(arg452_1, view_281, arg451_1);  arg452_1 = view_281 = arg451_1 = None
        view_282 = torch.ops.aten.view.default(addmm_166, [batch_size, 64, 128]);  addmm_166 = None
        relu_70 = torch.ops.aten.relu.default(view_282);  view_282 = None
        view_283 = torch.ops.aten.view.default(relu_70, [12800, 128]);  relu_70 = None
        addmm_167 = torch.ops.aten.addmm.default(arg454_1, view_283, arg453_1);  arg454_1 = view_283 = arg453_1 = None
        view_284 = torch.ops.aten.view.default(addmm_167, [batch_size, 64, 32]);  addmm_167 = None
        mul_109 = torch.ops.aten.mul.Tensor(view_284, arg455_1);  view_284 = arg455_1 = None
        sigmoid_10 = torch.ops.aten.sigmoid.default(mul_109);  mul_109 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(index_14, 2);  index_14 = None
        convert_element_type_38 = torch.ops.prims.convert_element_type.default(unsqueeze_43, torch.float16);  unsqueeze_43 = None
        mul_110 = torch.ops.aten.mul.Tensor(sigmoid_10, convert_element_type_38);  sigmoid_10 = convert_element_type_38 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_110, add_10);  mul_110 = None
        sum_143 = torch.ops.aten.sum.dim_IntList(mul_111, [1]);  mul_111 = None
        mul_112 = torch.ops.aten.mul.Tensor(slice_631, add_10);  slice_631 = add_10 = None
        sum_144 = torch.ops.aten.sum.dim_IntList(mul_112, [1]);  mul_112 = None
        cat_112 = torch.ops.aten.cat.default([sum_143, sum_144], 1);  sum_143 = sum_144 = None
        cat_113 = torch.ops.aten.cat.default([cat_112, addmm_160, sum_135, sum_136, sum_137, sum_138, sum_139, sum_140], -1);  addmm_160 = sum_135 = sum_136 = sum_137 = sum_138 = sum_139 = sum_140 = None
        full_default_199 = torch.ops.aten.full.default([batch_size, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_200 = torch.ops.aten.where.self(logical_or_12, full_default_199, cat_112);  full_default_199 = cat_112 = None
        full_default_200 = torch.ops.aten.full.default([batch_size, 320], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_201 = torch.ops.aten.where.self(logical_or_12, full_default_200, cat_113);  full_default_200 = cat_113 = None
        addmm_168 = torch.ops.aten.addmm.default(arg457_1, cat_65, arg456_1);  arg457_1 = cat_65 = arg456_1 = None
        addmm_169 = torch.ops.aten.addmm.default(arg459_1, cat_66, arg458_1);  arg459_1 = cat_66 = arg458_1 = None
        eq_62 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_63 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_29 = torch.ops.aten.logical_or.default(eq_62, eq_63);  eq_62 = eq_63 = None
        repeat_11 = torch.ops.aten.repeat.default(logical_or_29, [1, 256]);  logical_or_29 = None
        where_202 = torch.ops.aten.where.self(repeat_11, addmm_169, addmm_168);  repeat_11 = addmm_169 = addmm_168 = None
        slice_637 = torch.ops.aten.slice.Tensor(arg460_1, 2, 1, 130);  arg460_1 = None
        sign_11 = torch.ops.aten.sign.default(arg461_1);  arg461_1 = None
        slice_638 = torch.ops.aten.slice.Tensor(slice_637, 2, 0, 128)
        slice_639 = torch.ops.aten.slice.Tensor(slice_637, 2, 128, 9223372036854775807);  slice_637 = None
        clamp_min = torch.ops.aten.clamp_min.default(sign_11, 0);  sign_11 = None
        clamp_max_15 = torch.ops.aten.clamp_max.default(clamp_min, 1);  clamp_min = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(clamp_max_15, 2)
        mul_113 = torch.ops.aten.mul.Tensor(slice_638, unsqueeze_44);  unsqueeze_44 = None
        sum_145 = torch.ops.aten.sum.dim_IntList(mul_113, [-2]);  mul_113 = None
        slice_640 = torch.ops.aten.slice.Tensor(arg463_1, 2, 1, 130);  arg463_1 = None
        sign_12 = torch.ops.aten.sign.default(arg464_1);  arg464_1 = None
        slice_641 = torch.ops.aten.slice.Tensor(slice_640, 2, 0, 128)
        slice_642 = torch.ops.aten.slice.Tensor(slice_640, 2, 128, 9223372036854775807);  slice_640 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(sign_12, 0);  sign_12 = None
        clamp_max_16 = torch.ops.aten.clamp_max.default(clamp_min_1, 1);  clamp_min_1 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(clamp_max_16, 2);  clamp_max_16 = None
        mul_114 = torch.ops.aten.mul.Tensor(slice_641, unsqueeze_45);  unsqueeze_45 = None
        sum_146 = torch.ops.aten.sum.dim_IntList(mul_114, [-2]);  mul_114 = None
        slice_643 = torch.ops.aten.slice.Tensor(arg466_1, 2, 1, 130);  arg466_1 = None
        sign_13 = torch.ops.aten.sign.default(arg467_1);  arg467_1 = None
        slice_644 = torch.ops.aten.slice.Tensor(slice_643, 2, 0, 128)
        slice_645 = torch.ops.aten.slice.Tensor(slice_643, 2, 128, 9223372036854775807);  slice_643 = None
        clamp_min_2 = torch.ops.aten.clamp_min.default(sign_13, 0);  sign_13 = None
        clamp_max_17 = torch.ops.aten.clamp_max.default(clamp_min_2, 1);  clamp_min_2 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(clamp_max_17, 2);  clamp_max_17 = None
        mul_115 = torch.ops.aten.mul.Tensor(slice_644, unsqueeze_46);  unsqueeze_46 = None
        sum_147 = torch.ops.aten.sum.dim_IntList(mul_115, [-2]);  mul_115 = None
        slice_646 = torch.ops.aten.slice.Tensor(arg469_1, 2, 1, 130);  arg469_1 = None
        sign_14 = torch.ops.aten.sign.default(arg470_1);  arg470_1 = None
        slice_647 = torch.ops.aten.slice.Tensor(slice_646, 2, 0, 128)
        slice_648 = torch.ops.aten.slice.Tensor(slice_646, 2, 128, 9223372036854775807);  slice_646 = None
        clamp_min_3 = torch.ops.aten.clamp_min.default(sign_14, 0);  sign_14 = None
        clamp_max_18 = torch.ops.aten.clamp_max.default(clamp_min_3, 1);  clamp_min_3 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(clamp_max_18, 2);  clamp_max_18 = None
        mul_116 = torch.ops.aten.mul.Tensor(slice_647, unsqueeze_47);  unsqueeze_47 = None
        sum_148 = torch.ops.aten.sum.dim_IntList(mul_116, [-2]);  mul_116 = None
        slice_649 = torch.ops.aten.slice.Tensor(arg472_1, 2, 1, 130);  arg472_1 = None
        sign_15 = torch.ops.aten.sign.default(arg473_1);  arg473_1 = None
        slice_650 = torch.ops.aten.slice.Tensor(slice_649, 2, 0, 128)
        slice_651 = torch.ops.aten.slice.Tensor(slice_649, 2, 128, 9223372036854775807);  slice_649 = None
        clamp_min_4 = torch.ops.aten.clamp_min.default(sign_15, 0);  sign_15 = None
        clamp_max_19 = torch.ops.aten.clamp_max.default(clamp_min_4, 1);  clamp_min_4 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(clamp_max_19, 2);  clamp_max_19 = None
        mul_117 = torch.ops.aten.mul.Tensor(slice_650, unsqueeze_48);  unsqueeze_48 = None
        sum_149 = torch.ops.aten.sum.dim_IntList(mul_117, [-2]);  mul_117 = None
        slice_652 = torch.ops.aten.slice.Tensor(arg475_1, 2, 1, 130);  arg475_1 = None
        sign_16 = torch.ops.aten.sign.default(arg476_1);  arg476_1 = None
        slice_653 = torch.ops.aten.slice.Tensor(slice_652, 2, 0, 128)
        slice_654 = torch.ops.aten.slice.Tensor(slice_652, 2, 128, 9223372036854775807);  slice_652 = None
        clamp_min_5 = torch.ops.aten.clamp_min.default(sign_16, 0);  sign_16 = None
        clamp_max_20 = torch.ops.aten.clamp_max.default(clamp_min_5, 1);  clamp_min_5 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(clamp_max_20, 2);  clamp_max_20 = None
        mul_118 = torch.ops.aten.mul.Tensor(slice_653, unsqueeze_49);  unsqueeze_49 = None
        sum_150 = torch.ops.aten.sum.dim_IntList(mul_118, [-2]);  mul_118 = None
        cat_114 = torch.ops.aten.cat.default([slice_638, slice_641, slice_644, slice_647, slice_650, slice_653]);  slice_638 = slice_641 = slice_644 = slice_647 = slice_650 = slice_653 = None
        view_285 = torch.ops.aten.view.default(cat_114, [6, batch_size, 64, 128]);  cat_114 = None
        sum_151 = torch.ops.aten.sum.dim_IntList(view_285, [0]);  view_285 = None
        slice_655 = torch.ops.aten.slice.Tensor(sum_151, 1, 0, 2)
        sum_152 = torch.ops.aten.sum.dim_IntList(slice_655, [1]);  slice_655 = None
        slice_656 = torch.ops.aten.slice.Tensor(sum_151, 1, 0, 4)
        sum_153 = torch.ops.aten.sum.dim_IntList(slice_656, [1]);  slice_656 = None
        slice_657 = torch.ops.aten.slice.Tensor(sum_151, 1, 0, 8)
        sum_154 = torch.ops.aten.sum.dim_IntList(slice_657, [1]);  slice_657 = None
        slice_658 = torch.ops.aten.slice.Tensor(sum_151, 1, 0, 16)
        sum_155 = torch.ops.aten.sum.dim_IntList(slice_658, [1]);  slice_658 = None
        slice_659 = torch.ops.aten.slice.Tensor(sum_151, 1, 0, 32)
        sum_156 = torch.ops.aten.sum.dim_IntList(slice_659, [1]);  slice_659 = None
        sum_157 = torch.ops.aten.sum.dim_IntList(sum_151, [1])
        view_286 = torch.ops.aten.view.default(sum_151, [1, batch_size, 64, 128]);  sum_151 = None
        sum_158 = torch.ops.aten.sum.dim_IntList(view_286, [0]);  view_286 = None
        addmm_170 = torch.ops.aten.addmm.default(arg480_1, cat_57, arg479_1);  arg480_1 = arg479_1 = None
        relu_71 = torch.ops.aten.relu.default(addmm_170);  addmm_170 = None
        addmm_171 = torch.ops.aten.addmm.default(arg482_1, relu_71, arg481_1);  arg482_1 = relu_71 = arg481_1 = None
        view_287 = torch.ops.aten.view.default(addmm_171, [-1, 128, 32]);  addmm_171 = None
        addmm_172 = torch.ops.aten.addmm.default(arg484_1, cat_57, arg483_1);  arg484_1 = cat_57 = arg483_1 = None
        relu_72 = torch.ops.aten.relu.default(addmm_172);  addmm_172 = None
        addmm_173 = torch.ops.aten.addmm.default(arg486_1, relu_72, arg485_1);  arg486_1 = relu_72 = arg485_1 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(addmm_173, 1);  addmm_173 = None
        expand_33 = torch.ops.aten.expand.default(sum_158, [batch_size, 64, 128]);  sum_158 = None
        expand_34 = torch.ops.aten.expand.default(view_287, [batch_size, 128, 32]);  view_287 = None
        bmm_11 = torch.ops.aten.bmm.default(expand_33, expand_34);  expand_33 = expand_34 = None
        add_11 = torch.ops.aten.add.Tensor(bmm_11, unsqueeze_50);  bmm_11 = unsqueeze_50 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(where_142, 1)
        expand_35 = torch.ops.aten.expand.default(unsqueeze_51, [-1, 64, -1]);  unsqueeze_51 = None
        sub_11 = torch.ops.aten.sub.Tensor(expand_35, add_11)
        mul_119 = torch.ops.aten.mul.Tensor(expand_35, add_11)
        cat_115 = torch.ops.aten.cat.default([expand_35, add_11, sub_11, mul_119], -1);  expand_35 = sub_11 = mul_119 = None
        view_291 = torch.ops.aten.view.default(cat_115, [12800, 128]);  cat_115 = None
        addmm_174 = torch.ops.aten.addmm.default(arg488_1, view_291, arg487_1);  arg488_1 = view_291 = arg487_1 = None
        view_292 = torch.ops.aten.view.default(addmm_174, [batch_size, 64, 256]);  addmm_174 = None
        relu_73 = torch.ops.aten.relu.default(view_292);  view_292 = None
        view_293 = torch.ops.aten.view.default(relu_73, [12800, 256]);  relu_73 = None
        addmm_175 = torch.ops.aten.addmm.default(arg490_1, view_293, arg489_1);  arg490_1 = view_293 = arg489_1 = None
        view_294 = torch.ops.aten.view.default(addmm_175, [batch_size, 64, 128]);  addmm_175 = None
        relu_74 = torch.ops.aten.relu.default(view_294);  view_294 = None
        view_295 = torch.ops.aten.view.default(relu_74, [12800, 128]);  relu_74 = None
        addmm_176 = torch.ops.aten.addmm.default(arg492_1, view_295, arg491_1);  arg492_1 = view_295 = arg491_1 = None
        view_296 = torch.ops.aten.view.default(addmm_176, [batch_size, 64, 32]);  addmm_176 = None
        mul_120 = torch.ops.aten.mul.Tensor(view_296, arg493_1);  view_296 = arg493_1 = None
        sigmoid_11 = torch.ops.aten.sigmoid.default(mul_120);  mul_120 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(clamp_max_15, 2);  clamp_max_15 = None
        convert_element_type_41 = torch.ops.prims.convert_element_type.default(unsqueeze_52, torch.float16);  unsqueeze_52 = None
        mul_121 = torch.ops.aten.mul.Tensor(sigmoid_11, convert_element_type_41);  sigmoid_11 = convert_element_type_41 = None
        mul_122 = torch.ops.aten.mul.Tensor(mul_121, add_11);  mul_121 = None
        sum_159 = torch.ops.aten.sum.dim_IntList(mul_122, [1]);  mul_122 = None
        mul_123 = torch.ops.aten.mul.Tensor(arg478_1, add_11);  arg478_1 = add_11 = None
        sum_160 = torch.ops.aten.sum.dim_IntList(mul_123, [1]);  mul_123 = None
        cat_116 = torch.ops.aten.cat.default([sum_159, sum_160], 1);  sum_159 = sum_160 = None
        cat_117 = torch.ops.aten.cat.default([cat_116, where_202, sum_152, sum_153, sum_154, sum_155, sum_156, sum_157], -1);  where_202 = sum_152 = sum_153 = sum_154 = sum_155 = sum_156 = sum_157 = None
        addmm_177 = torch.ops.aten.addmm.default(arg495_1, slice_3, arg494_1);  arg495_1 = arg494_1 = None
        relu_75 = torch.ops.aten.relu.default(addmm_177);  addmm_177 = None
        addmm_178 = torch.ops.aten.addmm.default(arg497_1, cat_71, arg496_1);  arg497_1 = cat_71 = arg496_1 = None
        relu_76 = torch.ops.aten.relu.default(addmm_178);  addmm_178 = None
        addmm_179 = torch.ops.aten.addmm.default(arg499_1, cat_72, arg498_1);  arg499_1 = cat_72 = arg498_1 = None
        relu_77 = torch.ops.aten.relu.default(addmm_179);  addmm_179 = None
        eq_64 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_65 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_30 = torch.ops.aten.logical_or.default(eq_64, eq_65);  eq_64 = eq_65 = None
        repeat_12 = torch.ops.aten.repeat.default(logical_or_30, [1, 512]);  logical_or_30 = None
        where_203 = torch.ops.aten.where.self(repeat_12, relu_77, relu_76);  repeat_12 = relu_77 = relu_76 = None
        cat_118 = torch.ops.aten.cat.default([where_203, relu_75], 1);  where_203 = relu_75 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(cat_118, 1);  cat_118 = None
        view_297 = torch.ops.aten.view.default(unsqueeze_53, [batch_size, 1024]);  unsqueeze_53 = None
        addmm_180 = torch.ops.aten.addmm.default(arg501_1, view_297, arg500_1);  arg501_1 = view_297 = arg500_1 = None
        slice_660 = torch.ops.aten.slice.Tensor(arg398_1, 2, 1, 66);  arg398_1 = None
        slice_661 = torch.ops.aten.slice.Tensor(slice_660, 2, 0, 64);  slice_660 = None
        sign_17 = torch.ops.aten.sign.default(arg399_1);  arg399_1 = None
        expand_36 = torch.ops.aten.expand.default(slice_661, [7, 4800, 64]);  slice_661 = None
        expand_37 = torch.ops.aten.expand.default(arg502_1, [7, 64, 256]);  arg502_1 = None
        bmm_12 = torch.ops.aten.bmm.default(expand_36, expand_37);  expand_36 = expand_37 = None
        add_12 = torch.ops.aten.add.Tensor(bmm_12, arg503_1);  bmm_12 = arg503_1 = None
        view_302 = torch.ops.aten.view.default(add_12, [-1, 2400, 512]);  add_12 = None
        view_303 = torch.ops.aten.view.default(sign_17, [-1, 2400, 2]);  sign_17 = None
        amax = torch.ops.aten.amax.default(view_303, [2]);  view_303 = None
        view_304 = torch.ops.aten.view.default(amax, [-1])
        gt_2 = torch.ops.aten.gt.Scalar(view_304, 0);  view_304 = None
        nonzero = torch.ops.aten.nonzero.default(gt_2);  gt_2 = None
        sym_size_int = torch.ops.aten.sym_size.int(nonzero, 0)
        ge_1 = sym_size_int >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge_1 = _assert_scalar = None
        le_1 = sym_size_int <= 16800
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(le_1, "Runtime assertion failed for expression u0 <= 16800 on node 'le_5'");  le_1 = _assert_scalar_1 = None
        iota_15 = torch.ops.prims.iota.default(2400, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        repeat_13 = torch.ops.aten.repeat.default(iota_15, [7]);  iota_15 = None
        index_15 = torch.ops.aten.index.Tensor(repeat_13, [nonzero]);  repeat_13 = None
        sum_161 = torch.ops.aten.sum.dim_IntList(amax, [1]);  amax = None
        cumsum_15 = torch.ops.aten.cumsum.default(sum_161, 0);  sum_161 = None
        constant_pad_nd_2 = torch.ops.aten.constant_pad_nd.default(cumsum_15, [1, 0], 0.0);  cumsum_15 = None
        view_305 = torch.ops.aten.view.default(view_302, [-1, 512]);  view_302 = None
        index_16 = torch.ops.aten.index.Tensor(view_305, [nonzero]);  view_305 = nonzero = None
        full_default = torch.ops.aten.full.default([sym_size_int, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(arg147_1, torch.int32)
        iota_16 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        add_29 = torch.ops.aten.add.Tensor(iota_16, 1);  iota_16 = None
        iota_17 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        repeat_15 = torch.ops.aten.repeat.default(iota_17, [batch_size]);  iota_17 = None
        cumsum_16 = torch.ops.aten.cumsum.default(convert_element_type_44, 0);  convert_element_type_44 = None
        constant_pad_nd_3 = torch.ops.aten.constant_pad_nd.default(cumsum_16, [1, 0], 0.0);  cumsum_16 = None
        mul_135 = torch.ops.aten.mul.Tensor(constant_pad_nd_3, 1);  constant_pad_nd_3 = None
        ascend_create_position_offset = torch.ops.ascend_triton.ascend_create_position_offset.default(repeat_15, mul_135)
        ascend_seq_tensor_concat = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(addmm_180, index_16, mul_135, constant_pad_nd_2);  addmm_180 = index_16 = None
        add_33 = torch.ops.aten.add.Tensor(mul_135, constant_pad_nd_2)
        ascend_position_concat = torch.ops.ascend_triton.ascend_position_concat.default(repeat_15, index_15, mul_135, constant_pad_nd_2, ascend_create_position_offset);  repeat_15 = index_15 = ascend_create_position_offset = None
        sym_size_int_2 = torch.ops.aten.sym_size.int(ascend_position_concat, 0);  ascend_position_concat = None
        ascend_seq_tensor_concat_1 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(add_29, full_default, mul_135, constant_pad_nd_2);  add_29 = full_default = None
        full_default_201 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        full_204 = torch.ops.aten.full.default([sym_size_int], 1, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  sym_size_int = None
        ascend_seq_tensor_concat_2 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(full_default_201, full_204, mul_135, constant_pad_nd_2);  full_default_201 = full_204 = mul_135 = constant_pad_nd_2 = None
        eq_90 = torch.ops.aten.eq.Scalar(ascend_seq_tensor_concat_2, 0);  ascend_seq_tensor_concat_2 = None
        nonzero_1 = torch.ops.aten.nonzero.default(eq_90);  eq_90 = None
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression u1 >= batch_size on node 'ge_1'");  _assert_scalar_2 = None
        _assert_scalar_3 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression u1 <= batch_size on node 'le_6'");  _assert_scalar_3 = None
        _assert_scalar_4 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression Eq(u1, batch_size) on node 'eq_128'");  _assert_scalar_4 = None
        squeeze_137 = torch.ops.aten.squeeze.dim(nonzero_1, -1);  nonzero_1 = None
        native_layer_norm = torch.ops.aten.native_layer_norm.default(ascend_seq_tensor_concat, [512], arg504_1, arg505_1, 1e-06);  ascend_seq_tensor_concat = arg504_1 = arg505_1 = None
        getitem = native_layer_norm[0];  native_layer_norm = None
        native_layer_norm_1 = torch.ops.aten.native_layer_norm.default(getitem, [512], arg506_1, arg507_1, 1e-06);  arg506_1 = arg507_1 = None
        getitem_3 = native_layer_norm_1[0];  native_layer_norm_1 = None
        addmm_181 = torch.ops.aten.addmm.default(arg509_1, getitem_3, arg508_1);  arg509_1 = arg508_1 = None
        addmm_182 = torch.ops.aten.addmm.default(arg511_1, getitem_3, arg510_1);  arg511_1 = arg510_1 = None
        addmm_183 = torch.ops.aten.addmm.default(arg513_1, getitem_3, arg512_1);  arg513_1 = getitem_3 = arg512_1 = None
        view_309 = torch.ops.aten.view.default(addmm_181, [-1, 8, 64]);  addmm_181 = None
        view_310 = torch.ops.aten.view.default(addmm_182, [-1, 8, 64]);  addmm_182 = None
        view_311 = torch.ops.aten.view.default(addmm_183, [-1, 8, 64]);  addmm_183 = None
        ascend_flash_attention = torch.ops.ascend_triton.ascend_flash_attention.default(view_309, view_310, view_311, ascend_seq_tensor_concat_1, ascend_seq_tensor_concat_1, add_33, add_33, 2600, 2600, 0.125, 1);  view_309 = view_310 = view_311 = None
        view_312 = torch.ops.aten.view.default(ascend_flash_attention, [-1, 512]);  ascend_flash_attention = None
        addmm_184 = torch.ops.aten.addmm.default(arg515_1, view_312, arg514_1);  arg515_1 = view_312 = arg514_1 = None
        add_96 = torch.ops.aten.add.Tensor(addmm_184, getitem);  addmm_184 = getitem = None
        softcap = torch.ops.qianchuan_triton.softcap.default(add_96, 50.0);  add_96 = None
        native_layer_norm_2 = torch.ops.aten.native_layer_norm.default(softcap, [512], arg516_1, arg517_1, 1e-06);  arg516_1 = arg517_1 = None
        getitem_6 = native_layer_norm_2[0];  native_layer_norm_2 = None
        fused_swiglu = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_6, arg518_1, arg519_1, arg520_1, arg521_1, False, False);  getitem_6 = arg518_1 = arg519_1 = arg520_1 = arg521_1 = None
        addmm_185 = torch.ops.aten.addmm.default(arg523_1, fused_swiglu, arg522_1);  arg523_1 = fused_swiglu = arg522_1 = None
        add_118 = torch.ops.aten.add.Tensor(addmm_185, softcap);  addmm_185 = softcap = None
        softcap_1 = torch.ops.qianchuan_triton.softcap.default(add_118, 50.0);  add_118 = None
        native_layer_norm_3 = torch.ops.aten.native_layer_norm.default(softcap_1, [512], arg524_1, arg525_1, 1e-06);  arg524_1 = arg525_1 = None
        getitem_9 = native_layer_norm_3[0];  native_layer_norm_3 = None
        addmm_186 = torch.ops.aten.addmm.default(arg527_1, getitem_9, arg526_1);  arg527_1 = arg526_1 = None
        addmm_187 = torch.ops.aten.addmm.default(arg529_1, getitem_9, arg528_1);  arg529_1 = arg528_1 = None
        addmm_188 = torch.ops.aten.addmm.default(arg531_1, getitem_9, arg530_1);  arg531_1 = getitem_9 = arg530_1 = None
        view_313 = torch.ops.aten.view.default(addmm_186, [-1, 8, 64]);  addmm_186 = None
        view_314 = torch.ops.aten.view.default(addmm_187, [-1, 8, 64]);  addmm_187 = None
        view_315 = torch.ops.aten.view.default(addmm_188, [-1, 8, 64]);  addmm_188 = None
        ascend_flash_attention_1 = torch.ops.ascend_triton.ascend_flash_attention.default(view_313, view_314, view_315, ascend_seq_tensor_concat_1, ascend_seq_tensor_concat_1, add_33, add_33, 2600, 2600, 0.125, 1);  view_313 = view_314 = view_315 = None
        view_316 = torch.ops.aten.view.default(ascend_flash_attention_1, [-1, 512]);  ascend_flash_attention_1 = None
        addmm_189 = torch.ops.aten.addmm.default(arg533_1, view_316, arg532_1);  arg533_1 = view_316 = arg532_1 = None
        add_165 = torch.ops.aten.add.Tensor(addmm_189, softcap_1);  addmm_189 = softcap_1 = None
        softcap_2 = torch.ops.qianchuan_triton.softcap.default(add_165, 50.0);  add_165 = None
        native_layer_norm_4 = torch.ops.aten.native_layer_norm.default(softcap_2, [512], arg534_1, arg535_1, 1e-06);  arg534_1 = arg535_1 = None
        getitem_12 = native_layer_norm_4[0];  native_layer_norm_4 = None
        fused_swiglu_1 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_12, arg536_1, arg537_1, arg538_1, arg539_1, False, False);  getitem_12 = arg536_1 = arg537_1 = arg538_1 = arg539_1 = None
        addmm_190 = torch.ops.aten.addmm.default(arg541_1, fused_swiglu_1, arg540_1);  arg541_1 = fused_swiglu_1 = arg540_1 = None
        add_187 = torch.ops.aten.add.Tensor(addmm_190, softcap_2);  addmm_190 = softcap_2 = None
        softcap_3 = torch.ops.qianchuan_triton.softcap.default(add_187, 50.0);  add_187 = None
        native_layer_norm_5 = torch.ops.aten.native_layer_norm.default(softcap_3, [512], arg542_1, arg543_1, 1e-06);  arg542_1 = arg543_1 = None
        getitem_15 = native_layer_norm_5[0];  native_layer_norm_5 = None
        addmm_191 = torch.ops.aten.addmm.default(arg545_1, getitem_15, arg544_1);  arg545_1 = arg544_1 = None
        addmm_192 = torch.ops.aten.addmm.default(arg547_1, getitem_15, arg546_1);  arg547_1 = arg546_1 = None
        addmm_193 = torch.ops.aten.addmm.default(arg549_1, getitem_15, arg548_1);  arg549_1 = getitem_15 = arg548_1 = None
        view_317 = torch.ops.aten.view.default(addmm_191, [-1, 8, 64]);  addmm_191 = None
        view_318 = torch.ops.aten.view.default(addmm_192, [-1, 8, 64]);  addmm_192 = None
        view_319 = torch.ops.aten.view.default(addmm_193, [-1, 8, 64]);  addmm_193 = None
        ascend_flash_attention_2 = torch.ops.ascend_triton.ascend_flash_attention.default(view_317, view_318, view_319, ascend_seq_tensor_concat_1, ascend_seq_tensor_concat_1, add_33, add_33, 2600, 2600, 0.125, 1);  view_317 = view_318 = view_319 = None
        view_320 = torch.ops.aten.view.default(ascend_flash_attention_2, [-1, 512]);  ascend_flash_attention_2 = None
        addmm_194 = torch.ops.aten.addmm.default(arg551_1, view_320, arg550_1);  arg551_1 = view_320 = arg550_1 = None
        add_234 = torch.ops.aten.add.Tensor(addmm_194, softcap_3);  addmm_194 = softcap_3 = None
        softcap_4 = torch.ops.qianchuan_triton.softcap.default(add_234, 50.0);  add_234 = None
        native_layer_norm_6 = torch.ops.aten.native_layer_norm.default(softcap_4, [512], arg552_1, arg553_1, 1e-06);  arg552_1 = arg553_1 = None
        getitem_18 = native_layer_norm_6[0];  native_layer_norm_6 = None
        fused_swiglu_2 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_18, arg554_1, arg555_1, arg556_1, arg557_1, False, False);  getitem_18 = arg554_1 = arg555_1 = arg556_1 = arg557_1 = None
        addmm_195 = torch.ops.aten.addmm.default(arg559_1, fused_swiglu_2, arg558_1);  arg559_1 = fused_swiglu_2 = arg558_1 = None
        add_256 = torch.ops.aten.add.Tensor(addmm_195, softcap_4);  addmm_195 = softcap_4 = None
        softcap_5 = torch.ops.qianchuan_triton.softcap.default(add_256, 50.0);  add_256 = None
        native_layer_norm_7 = torch.ops.aten.native_layer_norm.default(softcap_5, [512], arg560_1, arg561_1, 1e-06);  arg560_1 = arg561_1 = None
        getitem_21 = native_layer_norm_7[0];  native_layer_norm_7 = None
        addmm_196 = torch.ops.aten.addmm.default(arg563_1, getitem_21, arg562_1);  arg563_1 = arg562_1 = None
        addmm_197 = torch.ops.aten.addmm.default(arg565_1, getitem_21, arg564_1);  arg565_1 = arg564_1 = None
        addmm_198 = torch.ops.aten.addmm.default(arg567_1, getitem_21, arg566_1);  arg567_1 = getitem_21 = arg566_1 = None
        view_321 = torch.ops.aten.view.default(addmm_196, [-1, 8, 64]);  addmm_196 = None
        view_322 = torch.ops.aten.view.default(addmm_197, [-1, 8, 64]);  addmm_197 = None
        view_323 = torch.ops.aten.view.default(addmm_198, [-1, 8, 64]);  addmm_198 = None
        ascend_flash_attention_3 = torch.ops.ascend_triton.ascend_flash_attention.default(view_321, view_322, view_323, ascend_seq_tensor_concat_1, ascend_seq_tensor_concat_1, add_33, add_33, 2600, 2600, 0.125, 1);  view_321 = view_322 = view_323 = None
        view_324 = torch.ops.aten.view.default(ascend_flash_attention_3, [-1, 512]);  ascend_flash_attention_3 = None
        addmm_199 = torch.ops.aten.addmm.default(arg569_1, view_324, arg568_1);  arg569_1 = view_324 = arg568_1 = None
        add_303 = torch.ops.aten.add.Tensor(addmm_199, softcap_5);  addmm_199 = softcap_5 = None
        softcap_6 = torch.ops.qianchuan_triton.softcap.default(add_303, 50.0);  add_303 = None
        native_layer_norm_8 = torch.ops.aten.native_layer_norm.default(softcap_6, [512], arg570_1, arg571_1, 1e-06);  arg570_1 = arg571_1 = None
        getitem_24 = native_layer_norm_8[0];  native_layer_norm_8 = None
        fused_swiglu_3 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_24, arg572_1, arg573_1, arg574_1, arg575_1, False, False);  getitem_24 = arg572_1 = arg573_1 = arg574_1 = arg575_1 = None
        addmm_200 = torch.ops.aten.addmm.default(arg577_1, fused_swiglu_3, arg576_1);  arg577_1 = fused_swiglu_3 = arg576_1 = None
        add_325 = torch.ops.aten.add.Tensor(addmm_200, softcap_6);  addmm_200 = softcap_6 = None
        softcap_7 = torch.ops.qianchuan_triton.softcap.default(add_325, 50.0);  add_325 = None
        native_layer_norm_9 = torch.ops.aten.native_layer_norm.default(softcap_7, [512], arg578_1, arg579_1, 1e-06);  arg578_1 = arg579_1 = None
        getitem_27 = native_layer_norm_9[0];  native_layer_norm_9 = None
        addmm_201 = torch.ops.aten.addmm.default(arg581_1, getitem_27, arg580_1);  arg581_1 = arg580_1 = None
        addmm_202 = torch.ops.aten.addmm.default(arg583_1, getitem_27, arg582_1);  arg583_1 = arg582_1 = None
        addmm_203 = torch.ops.aten.addmm.default(arg585_1, getitem_27, arg584_1);  arg585_1 = getitem_27 = arg584_1 = None
        view_325 = torch.ops.aten.view.default(addmm_201, [-1, 8, 64]);  addmm_201 = None
        view_326 = torch.ops.aten.view.default(addmm_202, [-1, 8, 64]);  addmm_202 = None
        view_327 = torch.ops.aten.view.default(addmm_203, [-1, 8, 64]);  addmm_203 = None
        ascend_flash_attention_4 = torch.ops.ascend_triton.ascend_flash_attention.default(view_325, view_326, view_327, ascend_seq_tensor_concat_1, ascend_seq_tensor_concat_1, add_33, add_33, 2600, 2600, 0.125, 1);  view_325 = view_326 = view_327 = None
        view_328 = torch.ops.aten.view.default(ascend_flash_attention_4, [-1, 512]);  ascend_flash_attention_4 = None
        addmm_204 = torch.ops.aten.addmm.default(arg587_1, view_328, arg586_1);  arg587_1 = view_328 = arg586_1 = None
        add_372 = torch.ops.aten.add.Tensor(addmm_204, softcap_7);  addmm_204 = softcap_7 = None
        softcap_8 = torch.ops.qianchuan_triton.softcap.default(add_372, 50.0);  add_372 = None
        native_layer_norm_10 = torch.ops.aten.native_layer_norm.default(softcap_8, [512], arg588_1, arg589_1, 1e-06);  arg588_1 = arg589_1 = None
        getitem_30 = native_layer_norm_10[0];  native_layer_norm_10 = None
        fused_swiglu_4 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_30, arg590_1, arg591_1, arg592_1, arg593_1, False, False);  getitem_30 = arg590_1 = arg591_1 = arg592_1 = arg593_1 = None
        addmm_205 = torch.ops.aten.addmm.default(arg595_1, fused_swiglu_4, arg594_1);  arg595_1 = fused_swiglu_4 = arg594_1 = None
        add_394 = torch.ops.aten.add.Tensor(addmm_205, softcap_8);  addmm_205 = softcap_8 = None
        softcap_9 = torch.ops.qianchuan_triton.softcap.default(add_394, 50.0);  add_394 = None
        index_17 = torch.ops.aten.index.Tensor(softcap_9, [squeeze_137])
        native_layer_norm_11 = torch.ops.aten.native_layer_norm.default(softcap_9, [512], arg596_1, arg597_1, 1e-06);  softcap_9 = arg596_1 = arg597_1 = None
        getitem_33 = native_layer_norm_11[0];  native_layer_norm_11 = None
        index_21 = torch.ops.aten.index.Tensor(getitem_33, [squeeze_137])
        full_206 = torch.ops.aten.full.default([sym_size_int_2], False, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  sym_size_int_2 = None
        full_default_202 = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        index_put_1 = torch.ops.aten.index_put.default(full_206, [squeeze_137], full_default_202);  full_206 = full_default_202 = None
        convert_element_type_48 = torch.ops.prims.convert_element_type.default(index_put_1, torch.int64);  index_put_1 = None
        cumsum_18 = torch.ops.aten.cumsum.default(convert_element_type_48, 0);  convert_element_type_48 = None
        constant_pad_nd_5 = torch.ops.aten.constant_pad_nd.default(cumsum_18, [1, 0], 0.0);  cumsum_18 = None
        index_22 = torch.ops.aten.index.Tensor(constant_pad_nd_5, [add_33]);  constant_pad_nd_5 = None
        slice_666 = torch.ops.aten.slice.Tensor(index_22, 0, 1, 9223372036854775807)
        slice_667 = torch.ops.aten.slice.Tensor(index_22, 0, 0, -1)
        sub_149 = torch.ops.aten.sub.Tensor(slice_666, slice_667);  slice_666 = slice_667 = None
        max_2 = torch.ops.aten.max.default(sub_149);  sub_149 = None
        _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(max_2);  max_2 = None
        index_24 = torch.ops.aten.index.Tensor(ascend_seq_tensor_concat_1, [squeeze_137]);  squeeze_137 = None
        addmm_206 = torch.ops.aten.addmm.default(arg599_1, index_21, arg598_1);  arg599_1 = index_21 = arg598_1 = None
        addmm_207 = torch.ops.aten.addmm.default(arg601_1, getitem_33, arg600_1);  arg601_1 = arg600_1 = None
        addmm_208 = torch.ops.aten.addmm.default(arg603_1, getitem_33, arg602_1);  arg603_1 = getitem_33 = arg602_1 = None
        view_329 = torch.ops.aten.view.default(addmm_206, [-1, 16, 32]);  addmm_206 = None
        view_330 = torch.ops.aten.view.default(addmm_207, [-1, 16, 32]);  addmm_207 = None
        view_331 = torch.ops.aten.view.default(addmm_208, [-1, 16, 32]);  addmm_208 = None
        ascend_flash_attention_5 = torch.ops.ascend_triton.ascend_flash_attention.default(view_329, view_330, view_331, index_24, ascend_seq_tensor_concat_1, index_22, add_33, _local_scalar_dense, 2600, 0.17677669529663687, 1);  view_329 = view_330 = view_331 = index_24 = ascend_seq_tensor_concat_1 = index_22 = add_33 = _local_scalar_dense = None
        view_332 = torch.ops.aten.view.default(ascend_flash_attention_5, [-1, 512]);  ascend_flash_attention_5 = None
        addmm_209 = torch.ops.aten.addmm.default(arg605_1, view_332, arg604_1);  arg605_1 = view_332 = arg604_1 = None
        add_448 = torch.ops.aten.add.Tensor(addmm_209, index_17);  addmm_209 = index_17 = None
        softcap_10 = torch.ops.qianchuan_triton.softcap.default(add_448, 50.0);  add_448 = None
        native_layer_norm_12 = torch.ops.aten.native_layer_norm.default(softcap_10, [512], arg606_1, arg607_1, 1e-06);  arg606_1 = arg607_1 = None
        getitem_36 = native_layer_norm_12[0];  native_layer_norm_12 = None
        fused_swiglu_5 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_36, arg608_1, arg609_1, arg610_1, arg611_1, False, False);  getitem_36 = arg608_1 = arg609_1 = arg610_1 = arg611_1 = None
        addmm_210 = torch.ops.aten.addmm.default(arg613_1, fused_swiglu_5, arg612_1);  arg613_1 = fused_swiglu_5 = arg612_1 = None
        add_449 = torch.ops.aten.add.Tensor(addmm_210, softcap_10);  addmm_210 = softcap_10 = None
        softcap_11 = torch.ops.qianchuan_triton.softcap.default(add_449, 50.0);  add_449 = None
        full_default_203 = torch.ops.aten.full.default([batch_size, 512], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_204 = torch.ops.aten.where.self(logical_or_12, full_default_203, softcap_11);  full_default_203 = softcap_11 = None
        addmm_211 = torch.ops.aten.addmm.default(arg615_1, where_204, arg614_1);  arg615_1 = arg614_1 = None
        relu_78 = torch.ops.aten.relu.default(addmm_211);  addmm_211 = None
        addmm_212 = torch.ops.aten.addmm.default(arg617_1, relu_78, arg616_1);  arg617_1 = relu_78 = arg616_1 = None
        squeeze_138 = torch.ops.aten.squeeze.dim(addmm_212, 1);  addmm_212 = None
        addmm_213 = torch.ops.aten.addmm.default(arg619_1, where_204, arg618_1);  arg619_1 = arg618_1 = None
        relu_79 = torch.ops.aten.relu.default(addmm_213);  addmm_213 = None
        addmm_214 = torch.ops.aten.addmm.default(arg621_1, relu_79, arg620_1);  arg621_1 = relu_79 = arg620_1 = None
        squeeze_139 = torch.ops.aten.squeeze.dim(addmm_214, 1);  addmm_214 = None
        addmm_215 = torch.ops.aten.addmm.default(arg623_1, slice_3, arg622_1);  arg623_1 = arg622_1 = None
        relu_80 = torch.ops.aten.relu.default(addmm_215);  addmm_215 = None
        addmm_216 = torch.ops.aten.addmm.default(arg625_1, cat_59, arg624_1);  arg625_1 = cat_59 = arg624_1 = None
        relu_81 = torch.ops.aten.relu.default(addmm_216);  addmm_216 = None
        addmm_217 = torch.ops.aten.addmm.default(arg627_1, cat_60, arg626_1);  arg627_1 = cat_60 = arg626_1 = None
        relu_82 = torch.ops.aten.relu.default(addmm_217);  addmm_217 = None
        eq_404 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_405 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_31 = torch.ops.aten.logical_or.default(eq_404, eq_405);  eq_404 = eq_405 = None
        repeat_16 = torch.ops.aten.repeat.default(logical_or_31, [1, 512]);  logical_or_31 = None
        where_205 = torch.ops.aten.where.self(repeat_16, relu_82, relu_81);  repeat_16 = relu_82 = relu_81 = None
        cat_119 = torch.ops.aten.cat.default([where_205, relu_80], 1);  where_205 = relu_80 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(cat_119, 1);  cat_119 = None
        view_335 = torch.ops.aten.view.default(unsqueeze_54, [batch_size, 1024]);  unsqueeze_54 = None
        addmm_218 = torch.ops.aten.addmm.default(arg629_1, view_335, arg628_1);  arg629_1 = view_335 = arg628_1 = None
        slice_668 = torch.ops.aten.slice.Tensor(arg438_1, 2, 1, 34);  arg438_1 = None
        slice_669 = torch.ops.aten.slice.Tensor(slice_668, 2, 0, 32);  slice_668 = None
        sign_18 = torch.ops.aten.sign.default(arg439_1);  arg439_1 = None
        expand_38 = torch.ops.aten.expand.default(slice_669, [7, 688, 32]);  slice_669 = None
        expand_39 = torch.ops.aten.expand.default(arg630_1, [7, 32, 64]);  arg630_1 = None
        bmm_13 = torch.ops.aten.bmm.default(expand_38, expand_39);  expand_38 = expand_39 = None
        add_450 = torch.ops.aten.add.Tensor(bmm_13, arg631_1);  bmm_13 = arg631_1 = None
        view_340 = torch.ops.aten.view.default(add_450, [-1, 86, 512]);  add_450 = None
        view_341 = torch.ops.aten.view.default(sign_18, [-1, 86, 8]);  sign_18 = None
        amax_1 = torch.ops.aten.amax.default(view_341, [2]);  view_341 = None
        view_342 = torch.ops.aten.view.default(amax_1, [-1])
        gt_3 = torch.ops.aten.gt.Scalar(view_342, 0);  view_342 = None
        nonzero_2 = torch.ops.aten.nonzero.default(gt_3);  gt_3 = None
        sym_size_int_32 = torch.ops.aten.sym_size.int(nonzero_2, 0)
        ge_6 = sym_size_int_32 >= 0
        _assert_scalar_5 = torch.ops.aten._assert_scalar.default(ge_6, "Runtime assertion failed for expression u4 >= 0 on node 'ge_2'");  ge_6 = _assert_scalar_5 = None
        le_2 = sym_size_int_32 <= 602
        _assert_scalar_6 = torch.ops.aten._assert_scalar.default(le_2, "Runtime assertion failed for expression u4 <= 602 on node 'le_7'");  le_2 = _assert_scalar_6 = None
        iota_18 = torch.ops.prims.iota.default(86, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        repeat_17 = torch.ops.aten.repeat.default(iota_18, [7]);  iota_18 = None
        index_25 = torch.ops.aten.index.Tensor(repeat_17, [nonzero_2]);  repeat_17 = None
        sum_162 = torch.ops.aten.sum.dim_IntList(amax_1, [1]);  amax_1 = None
        cumsum_19 = torch.ops.aten.cumsum.default(sum_162, 0);  sum_162 = None
        constant_pad_nd_6 = torch.ops.aten.constant_pad_nd.default(cumsum_19, [1, 0], 0.0);  cumsum_19 = None
        view_343 = torch.ops.aten.view.default(view_340, [-1, 512]);  view_340 = None
        index_26 = torch.ops.aten.index.Tensor(view_343, [nonzero_2]);  view_343 = nonzero_2 = None
        full_default_1 = torch.ops.aten.full.default([sym_size_int_32, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        convert_element_type_51 = torch.ops.prims.convert_element_type.default(arg147_1, torch.int32)
        iota_19 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        add_467 = torch.ops.aten.add.Tensor(iota_19, 1);  iota_19 = None
        iota_20 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        repeat_19 = torch.ops.aten.repeat.default(iota_20, [batch_size]);  iota_20 = None
        cumsum_20 = torch.ops.aten.cumsum.default(convert_element_type_51, 0);  convert_element_type_51 = None
        constant_pad_nd_7 = torch.ops.aten.constant_pad_nd.default(cumsum_20, [1, 0], 0.0);  cumsum_20 = None
        mul_665 = torch.ops.aten.mul.Tensor(constant_pad_nd_7, 1);  constant_pad_nd_7 = None
        ascend_create_position_offset_1 = torch.ops.ascend_triton.ascend_create_position_offset.default(repeat_19, mul_665)
        ascend_seq_tensor_concat_3 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(addmm_218, index_26, mul_665, constant_pad_nd_6);  addmm_218 = index_26 = None
        add_471 = torch.ops.aten.add.Tensor(mul_665, constant_pad_nd_6)
        ascend_position_concat_1 = torch.ops.ascend_triton.ascend_position_concat.default(repeat_19, index_25, mul_665, constant_pad_nd_6, ascend_create_position_offset_1);  repeat_19 = index_25 = ascend_create_position_offset_1 = None
        sym_size_int_34 = torch.ops.aten.sym_size.int(ascend_position_concat_1, 0);  ascend_position_concat_1 = None
        ascend_seq_tensor_concat_4 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(add_467, full_default_1, mul_665, constant_pad_nd_6);  add_467 = full_default_1 = None
        full_default_204 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        full_210 = torch.ops.aten.full.default([sym_size_int_32], 1, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  sym_size_int_32 = None
        ascend_seq_tensor_concat_5 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(full_default_204, full_210, mul_665, constant_pad_nd_6);  full_default_204 = full_210 = mul_665 = constant_pad_nd_6 = None
        eq_430 = torch.ops.aten.eq.Scalar(ascend_seq_tensor_concat_5, 0);  ascend_seq_tensor_concat_5 = None
        nonzero_3 = torch.ops.aten.nonzero.default(eq_430);  eq_430 = None
        _assert_scalar_7 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression u5 >= batch_size on node 'ge_3'");  _assert_scalar_7 = None
        _assert_scalar_8 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression u5 <= batch_size on node 'le_8'");  _assert_scalar_8 = None
        _assert_scalar_9 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression Eq(u5, batch_size) on node 'eq_129'");  _assert_scalar_9 = None
        squeeze_140 = torch.ops.aten.squeeze.dim(nonzero_3, -1);  nonzero_3 = None
        native_layer_norm_13 = torch.ops.aten.native_layer_norm.default(ascend_seq_tensor_concat_3, [512], arg632_1, arg633_1, 1e-06);  ascend_seq_tensor_concat_3 = arg632_1 = arg633_1 = None
        getitem_39 = native_layer_norm_13[0];  native_layer_norm_13 = None
        native_layer_norm_14 = torch.ops.aten.native_layer_norm.default(getitem_39, [512], arg634_1, arg635_1, 1e-06);  arg634_1 = arg635_1 = None
        getitem_42 = native_layer_norm_14[0];  native_layer_norm_14 = None
        addmm_219 = torch.ops.aten.addmm.default(arg637_1, getitem_42, arg636_1);  arg637_1 = arg636_1 = None
        addmm_220 = torch.ops.aten.addmm.default(arg639_1, getitem_42, arg638_1);  arg639_1 = arg638_1 = None
        addmm_221 = torch.ops.aten.addmm.default(arg641_1, getitem_42, arg640_1);  arg641_1 = getitem_42 = arg640_1 = None
        view_347 = torch.ops.aten.view.default(addmm_219, [-1, 4, 128]);  addmm_219 = None
        view_348 = torch.ops.aten.view.default(addmm_220, [-1, 4, 128]);  addmm_220 = None
        view_349 = torch.ops.aten.view.default(addmm_221, [-1, 4, 128]);  addmm_221 = None
        ascend_flash_attention_6 = torch.ops.ascend_triton.ascend_flash_attention.default(view_347, view_348, view_349, ascend_seq_tensor_concat_4, ascend_seq_tensor_concat_4, add_471, add_471, 286, 286, 0.08838834764831843, 1);  view_347 = view_348 = view_349 = None
        view_350 = torch.ops.aten.view.default(ascend_flash_attention_6, [-1, 512]);  ascend_flash_attention_6 = None
        addmm_222 = torch.ops.aten.addmm.default(arg643_1, view_350, arg642_1);  arg643_1 = view_350 = arg642_1 = None
        add_534 = torch.ops.aten.add.Tensor(addmm_222, getitem_39);  addmm_222 = getitem_39 = None
        softcap_12 = torch.ops.qianchuan_triton.softcap.default(add_534, 50.0);  add_534 = None
        native_layer_norm_15 = torch.ops.aten.native_layer_norm.default(softcap_12, [512], arg644_1, arg645_1, 1e-06);  arg644_1 = arg645_1 = None
        getitem_45 = native_layer_norm_15[0];  native_layer_norm_15 = None
        fused_swiglu_6 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_45, arg646_1, arg647_1, arg648_1, arg649_1, False, False);  getitem_45 = arg646_1 = arg647_1 = arg648_1 = arg649_1 = None
        addmm_223 = torch.ops.aten.addmm.default(arg651_1, fused_swiglu_6, arg650_1);  arg651_1 = fused_swiglu_6 = arg650_1 = None
        add_556 = torch.ops.aten.add.Tensor(addmm_223, softcap_12);  addmm_223 = softcap_12 = None
        softcap_13 = torch.ops.qianchuan_triton.softcap.default(add_556, 50.0);  add_556 = None
        index_27 = torch.ops.aten.index.Tensor(softcap_13, [squeeze_140])
        native_layer_norm_16 = torch.ops.aten.native_layer_norm.default(softcap_13, [512], arg652_1, arg653_1, 1e-06);  softcap_13 = arg652_1 = arg653_1 = None
        getitem_48 = native_layer_norm_16[0];  native_layer_norm_16 = None
        index_31 = torch.ops.aten.index.Tensor(getitem_48, [squeeze_140])
        full_212 = torch.ops.aten.full.default([sym_size_int_34], False, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  sym_size_int_34 = None
        full_default_205 = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        index_put_3 = torch.ops.aten.index_put.default(full_212, [squeeze_140], full_default_205);  full_212 = full_default_205 = None
        convert_element_type_55 = torch.ops.prims.convert_element_type.default(index_put_3, torch.int64);  index_put_3 = None
        cumsum_22 = torch.ops.aten.cumsum.default(convert_element_type_55, 0);  convert_element_type_55 = None
        constant_pad_nd_9 = torch.ops.aten.constant_pad_nd.default(cumsum_22, [1, 0], 0.0);  cumsum_22 = None
        index_32 = torch.ops.aten.index.Tensor(constant_pad_nd_9, [add_471]);  constant_pad_nd_9 = None
        slice_674 = torch.ops.aten.slice.Tensor(index_32, 0, 1, 9223372036854775807)
        slice_675 = torch.ops.aten.slice.Tensor(index_32, 0, 0, -1)
        sub_207 = torch.ops.aten.sub.Tensor(slice_674, slice_675);  slice_674 = slice_675 = None
        max_4 = torch.ops.aten.max.default(sub_207);  sub_207 = None
        _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense.default(max_4);  max_4 = None
        index_34 = torch.ops.aten.index.Tensor(ascend_seq_tensor_concat_4, [squeeze_140]);  squeeze_140 = None
        addmm_224 = torch.ops.aten.addmm.default(arg655_1, index_31, arg654_1);  arg655_1 = index_31 = arg654_1 = None
        addmm_225 = torch.ops.aten.addmm.default(arg657_1, getitem_48, arg656_1);  arg657_1 = arg656_1 = None
        addmm_226 = torch.ops.aten.addmm.default(arg659_1, getitem_48, arg658_1);  arg659_1 = getitem_48 = arg658_1 = None
        view_351 = torch.ops.aten.view.default(addmm_224, [-1, 8, 64]);  addmm_224 = None
        view_352 = torch.ops.aten.view.default(addmm_225, [-1, 8, 64]);  addmm_225 = None
        view_353 = torch.ops.aten.view.default(addmm_226, [-1, 8, 64]);  addmm_226 = None
        ascend_flash_attention_7 = torch.ops.ascend_triton.ascend_flash_attention.default(view_351, view_352, view_353, index_34, ascend_seq_tensor_concat_4, index_32, add_471, _local_scalar_dense_1, 286, 0.125, 1);  view_351 = view_352 = view_353 = index_34 = ascend_seq_tensor_concat_4 = index_32 = add_471 = _local_scalar_dense_1 = None
        view_354 = torch.ops.aten.view.default(ascend_flash_attention_7, [-1, 512]);  ascend_flash_attention_7 = None
        addmm_227 = torch.ops.aten.addmm.default(arg661_1, view_354, arg660_1);  arg661_1 = view_354 = arg660_1 = None
        add_610 = torch.ops.aten.add.Tensor(addmm_227, index_27);  addmm_227 = index_27 = None
        softcap_14 = torch.ops.qianchuan_triton.softcap.default(add_610, 50.0);  add_610 = None
        native_layer_norm_17 = torch.ops.aten.native_layer_norm.default(softcap_14, [512], arg662_1, arg663_1, 1e-06);  arg662_1 = arg663_1 = None
        getitem_51 = native_layer_norm_17[0];  native_layer_norm_17 = None
        fused_swiglu_7 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_51, arg664_1, arg665_1, arg666_1, arg667_1, False, False);  getitem_51 = arg664_1 = arg665_1 = arg666_1 = arg667_1 = None
        addmm_228 = torch.ops.aten.addmm.default(arg669_1, fused_swiglu_7, arg668_1);  arg669_1 = fused_swiglu_7 = arg668_1 = None
        add_611 = torch.ops.aten.add.Tensor(addmm_228, softcap_14);  addmm_228 = softcap_14 = None
        softcap_15 = torch.ops.qianchuan_triton.softcap.default(add_611, 50.0);  add_611 = None
        full_default_206 = torch.ops.aten.full.default([batch_size, 512], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_206 = torch.ops.aten.where.self(logical_or_12, full_default_206, softcap_15);  full_default_206 = softcap_15 = None
        addmm_229 = torch.ops.aten.addmm.default(arg671_1, where_206, arg670_1);  arg671_1 = arg670_1 = None
        relu_83 = torch.ops.aten.relu.default(addmm_229);  addmm_229 = None
        addmm_230 = torch.ops.aten.addmm.default(arg673_1, relu_83, arg672_1);  arg673_1 = relu_83 = arg672_1 = None
        squeeze_141 = torch.ops.aten.squeeze.dim(addmm_230, 1);  addmm_230 = None
        addmm_231 = torch.ops.aten.addmm.default(arg675_1, where_206, arg674_1);  arg675_1 = arg674_1 = None
        relu_84 = torch.ops.aten.relu.default(addmm_231);  addmm_231 = None
        addmm_232 = torch.ops.aten.addmm.default(arg677_1, relu_84, arg676_1);  arg677_1 = relu_84 = arg676_1 = None
        squeeze_142 = torch.ops.aten.squeeze.dim(addmm_232, 1);  addmm_232 = None
        addmm_233 = torch.ops.aten.addmm.default(arg679_1, slice_3, arg678_1);  arg679_1 = arg678_1 = None
        relu_85 = torch.ops.aten.relu.default(addmm_233);  addmm_233 = None
        addmm_234 = torch.ops.aten.addmm.default(arg681_1, cat_69, arg680_1);  arg681_1 = cat_69 = arg680_1 = None
        relu_86 = torch.ops.aten.relu.default(addmm_234);  addmm_234 = None
        addmm_235 = torch.ops.aten.addmm.default(arg683_1, cat_70, arg682_1);  arg683_1 = cat_70 = arg682_1 = None
        relu_87 = torch.ops.aten.relu.default(addmm_235);  addmm_235 = None
        eq_524 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_525 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_32 = torch.ops.aten.logical_or.default(eq_524, eq_525);  eq_524 = eq_525 = None
        repeat_20 = torch.ops.aten.repeat.default(logical_or_32, [1, 512]);  logical_or_32 = None
        where_207 = torch.ops.aten.where.self(repeat_20, relu_87, relu_86);  repeat_20 = relu_87 = relu_86 = None
        cat_120 = torch.ops.aten.cat.default([where_207, relu_85], 1);  where_207 = relu_85 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(cat_120, 1);  cat_120 = None
        view_357 = torch.ops.aten.view.default(unsqueeze_55, [batch_size, 1024]);  unsqueeze_55 = None
        addmm_236 = torch.ops.aten.addmm.default(arg685_1, view_357, arg684_1);  arg685_1 = view_357 = arg684_1 = None
        slice_676 = torch.ops.aten.slice.Tensor(arg238_1, 2, 1, 34);  arg238_1 = None
        slice_677 = torch.ops.aten.slice.Tensor(slice_676, 2, 0, 32);  slice_676 = None
        sign_19 = torch.ops.aten.sign.default(arg239_1);  arg239_1 = None
        expand_40 = torch.ops.aten.expand.default(slice_677, [7, 480, 32]);  slice_677 = None
        expand_41 = torch.ops.aten.expand.default(arg686_1, [7, 32, 64]);  arg686_1 = None
        bmm_14 = torch.ops.aten.bmm.default(expand_40, expand_41);  expand_40 = expand_41 = None
        add_612 = torch.ops.aten.add.Tensor(bmm_14, arg687_1);  bmm_14 = arg687_1 = None
        view_362 = torch.ops.aten.view.default(add_612, [-1, 60, 512]);  add_612 = None
        view_363 = torch.ops.aten.view.default(sign_19, [-1, 60, 8]);  sign_19 = None
        amax_2 = torch.ops.aten.amax.default(view_363, [2]);  view_363 = None
        view_364 = torch.ops.aten.view.default(amax_2, [-1])
        gt_4 = torch.ops.aten.gt.Scalar(view_364, 0);  view_364 = None
        nonzero_4 = torch.ops.aten.nonzero.default(gt_4);  gt_4 = None
        sym_size_int_48 = torch.ops.aten.sym_size.int(nonzero_4, 0)
        ge_11 = sym_size_int_48 >= 0
        _assert_scalar_10 = torch.ops.aten._assert_scalar.default(ge_11, "Runtime assertion failed for expression u8 >= 0 on node 'ge_4'");  ge_11 = _assert_scalar_10 = None
        le_3 = sym_size_int_48 <= 420
        _assert_scalar_11 = torch.ops.aten._assert_scalar.default(le_3, "Runtime assertion failed for expression u8 <= 420 on node 'le_9'");  le_3 = _assert_scalar_11 = None
        iota_21 = torch.ops.prims.iota.default(60, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        repeat_21 = torch.ops.aten.repeat.default(iota_21, [7]);  iota_21 = None
        index_35 = torch.ops.aten.index.Tensor(repeat_21, [nonzero_4]);  repeat_21 = None
        sum_163 = torch.ops.aten.sum.dim_IntList(amax_2, [1]);  amax_2 = None
        cumsum_23 = torch.ops.aten.cumsum.default(sum_163, 0);  sum_163 = None
        constant_pad_nd_10 = torch.ops.aten.constant_pad_nd.default(cumsum_23, [1, 0], 0.0);  cumsum_23 = None
        view_365 = torch.ops.aten.view.default(view_362, [-1, 512]);  view_362 = None
        index_36 = torch.ops.aten.index.Tensor(view_365, [nonzero_4]);  view_365 = nonzero_4 = None
        full_default_2 = torch.ops.aten.full.default([sym_size_int_48, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        convert_element_type_58 = torch.ops.prims.convert_element_type.default(arg147_1, torch.int32)
        iota_22 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        add_629 = torch.ops.aten.add.Tensor(iota_22, 1);  iota_22 = None
        iota_23 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        repeat_23 = torch.ops.aten.repeat.default(iota_23, [batch_size]);  iota_23 = None
        cumsum_24 = torch.ops.aten.cumsum.default(convert_element_type_58, 0);  convert_element_type_58 = None
        constant_pad_nd_11 = torch.ops.aten.constant_pad_nd.default(cumsum_24, [1, 0], 0.0);  cumsum_24 = None
        mul_823 = torch.ops.aten.mul.Tensor(constant_pad_nd_11, 1);  constant_pad_nd_11 = None
        ascend_create_position_offset_2 = torch.ops.ascend_triton.ascend_create_position_offset.default(repeat_23, mul_823)
        ascend_seq_tensor_concat_6 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(addmm_236, index_36, mul_823, constant_pad_nd_10);  addmm_236 = index_36 = None
        add_633 = torch.ops.aten.add.Tensor(mul_823, constant_pad_nd_10)
        ascend_position_concat_2 = torch.ops.ascend_triton.ascend_position_concat.default(repeat_23, index_35, mul_823, constant_pad_nd_10, ascend_create_position_offset_2);  repeat_23 = index_35 = ascend_create_position_offset_2 = None
        sym_size_int_50 = torch.ops.aten.sym_size.int(ascend_position_concat_2, 0);  ascend_position_concat_2 = None
        ascend_seq_tensor_concat_7 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(add_629, full_default_2, mul_823, constant_pad_nd_10);  add_629 = full_default_2 = None
        full_default_207 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        full_216 = torch.ops.aten.full.default([sym_size_int_48], 1, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  sym_size_int_48 = None
        ascend_seq_tensor_concat_8 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(full_default_207, full_216, mul_823, constant_pad_nd_10);  full_default_207 = full_216 = mul_823 = constant_pad_nd_10 = None
        eq_550 = torch.ops.aten.eq.Scalar(ascend_seq_tensor_concat_8, 0);  ascend_seq_tensor_concat_8 = None
        nonzero_5 = torch.ops.aten.nonzero.default(eq_550);  eq_550 = None
        _assert_scalar_12 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression u9 >= batch_size on node 'ge_5'");  _assert_scalar_12 = None
        _assert_scalar_13 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression u9 <= batch_size on node 'le_10'");  _assert_scalar_13 = None
        _assert_scalar_14 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression Eq(u9, batch_size) on node 'eq_130'");  _assert_scalar_14 = None
        squeeze_143 = torch.ops.aten.squeeze.dim(nonzero_5, -1);  nonzero_5 = None
        native_layer_norm_18 = torch.ops.aten.native_layer_norm.default(ascend_seq_tensor_concat_6, [512], arg688_1, arg689_1, 1e-06);  ascend_seq_tensor_concat_6 = arg688_1 = arg689_1 = None
        getitem_54 = native_layer_norm_18[0];  native_layer_norm_18 = None
        native_layer_norm_19 = torch.ops.aten.native_layer_norm.default(getitem_54, [512], arg690_1, arg691_1, 1e-06);  arg690_1 = arg691_1 = None
        getitem_57 = native_layer_norm_19[0];  native_layer_norm_19 = None
        addmm_237 = torch.ops.aten.addmm.default(arg693_1, getitem_57, arg692_1);  arg693_1 = arg692_1 = None
        addmm_238 = torch.ops.aten.addmm.default(arg695_1, getitem_57, arg694_1);  arg695_1 = arg694_1 = None
        addmm_239 = torch.ops.aten.addmm.default(arg697_1, getitem_57, arg696_1);  arg697_1 = getitem_57 = arg696_1 = None
        view_369 = torch.ops.aten.view.default(addmm_237, [-1, 4, 128]);  addmm_237 = None
        view_370 = torch.ops.aten.view.default(addmm_238, [-1, 4, 128]);  addmm_238 = None
        view_371 = torch.ops.aten.view.default(addmm_239, [-1, 4, 128]);  addmm_239 = None
        ascend_flash_attention_8 = torch.ops.ascend_triton.ascend_flash_attention.default(view_369, view_370, view_371, ascend_seq_tensor_concat_7, ascend_seq_tensor_concat_7, add_633, add_633, 260, 260, 0.08838834764831843, 1);  view_369 = view_370 = view_371 = None
        view_372 = torch.ops.aten.view.default(ascend_flash_attention_8, [-1, 512]);  ascend_flash_attention_8 = None
        addmm_240 = torch.ops.aten.addmm.default(arg699_1, view_372, arg698_1);  arg699_1 = view_372 = arg698_1 = None
        add_696 = torch.ops.aten.add.Tensor(addmm_240, getitem_54);  addmm_240 = getitem_54 = None
        softcap_16 = torch.ops.qianchuan_triton.softcap.default(add_696, 50.0);  add_696 = None
        native_layer_norm_20 = torch.ops.aten.native_layer_norm.default(softcap_16, [512], arg700_1, arg701_1, 1e-06);  arg700_1 = arg701_1 = None
        getitem_60 = native_layer_norm_20[0];  native_layer_norm_20 = None
        fused_swiglu_8 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_60, arg702_1, arg703_1, arg704_1, arg705_1, False, False);  getitem_60 = arg702_1 = arg703_1 = arg704_1 = arg705_1 = None
        addmm_241 = torch.ops.aten.addmm.default(arg707_1, fused_swiglu_8, arg706_1);  arg707_1 = fused_swiglu_8 = arg706_1 = None
        add_718 = torch.ops.aten.add.Tensor(addmm_241, softcap_16);  addmm_241 = softcap_16 = None
        softcap_17 = torch.ops.qianchuan_triton.softcap.default(add_718, 50.0);  add_718 = None
        index_37 = torch.ops.aten.index.Tensor(softcap_17, [squeeze_143])
        native_layer_norm_21 = torch.ops.aten.native_layer_norm.default(softcap_17, [512], arg708_1, arg709_1, 1e-06);  softcap_17 = arg708_1 = arg709_1 = None
        getitem_63 = native_layer_norm_21[0];  native_layer_norm_21 = None
        index_41 = torch.ops.aten.index.Tensor(getitem_63, [squeeze_143])
        full_218 = torch.ops.aten.full.default([sym_size_int_50], False, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  sym_size_int_50 = None
        full_default_208 = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        index_put_5 = torch.ops.aten.index_put.default(full_218, [squeeze_143], full_default_208);  full_218 = full_default_208 = None
        convert_element_type_62 = torch.ops.prims.convert_element_type.default(index_put_5, torch.int64);  index_put_5 = None
        cumsum_26 = torch.ops.aten.cumsum.default(convert_element_type_62, 0);  convert_element_type_62 = None
        constant_pad_nd_13 = torch.ops.aten.constant_pad_nd.default(cumsum_26, [1, 0], 0.0);  cumsum_26 = None
        index_42 = torch.ops.aten.index.Tensor(constant_pad_nd_13, [add_633]);  constant_pad_nd_13 = None
        slice_682 = torch.ops.aten.slice.Tensor(index_42, 0, 1, 9223372036854775807)
        slice_683 = torch.ops.aten.slice.Tensor(index_42, 0, 0, -1)
        sub_265 = torch.ops.aten.sub.Tensor(slice_682, slice_683);  slice_682 = slice_683 = None
        max_6 = torch.ops.aten.max.default(sub_265);  sub_265 = None
        _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense.default(max_6);  max_6 = None
        index_44 = torch.ops.aten.index.Tensor(ascend_seq_tensor_concat_7, [squeeze_143]);  squeeze_143 = None
        addmm_242 = torch.ops.aten.addmm.default(arg711_1, index_41, arg710_1);  arg711_1 = index_41 = arg710_1 = None
        addmm_243 = torch.ops.aten.addmm.default(arg713_1, getitem_63, arg712_1);  arg713_1 = arg712_1 = None
        addmm_244 = torch.ops.aten.addmm.default(arg715_1, getitem_63, arg714_1);  arg715_1 = getitem_63 = arg714_1 = None
        view_373 = torch.ops.aten.view.default(addmm_242, [-1, 4, 128]);  addmm_242 = None
        view_374 = torch.ops.aten.view.default(addmm_243, [-1, 4, 128]);  addmm_243 = None
        view_375 = torch.ops.aten.view.default(addmm_244, [-1, 4, 128]);  addmm_244 = None
        ascend_flash_attention_9 = torch.ops.ascend_triton.ascend_flash_attention.default(view_373, view_374, view_375, index_44, ascend_seq_tensor_concat_7, index_42, add_633, _local_scalar_dense_2, 260, 0.08838834764831843, 1);  view_373 = view_374 = view_375 = index_44 = ascend_seq_tensor_concat_7 = index_42 = add_633 = _local_scalar_dense_2 = None
        view_376 = torch.ops.aten.view.default(ascend_flash_attention_9, [-1, 512]);  ascend_flash_attention_9 = None
        addmm_245 = torch.ops.aten.addmm.default(arg717_1, view_376, arg716_1);  arg717_1 = view_376 = arg716_1 = None
        add_772 = torch.ops.aten.add.Tensor(addmm_245, index_37);  addmm_245 = index_37 = None
        softcap_18 = torch.ops.qianchuan_triton.softcap.default(add_772, 50.0);  add_772 = None
        native_layer_norm_22 = torch.ops.aten.native_layer_norm.default(softcap_18, [512], arg718_1, arg719_1, 1e-06);  arg718_1 = arg719_1 = None
        getitem_66 = native_layer_norm_22[0];  native_layer_norm_22 = None
        fused_swiglu_9 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_66, arg720_1, arg721_1, arg722_1, arg723_1, False, False);  getitem_66 = arg720_1 = arg721_1 = arg722_1 = arg723_1 = None
        addmm_246 = torch.ops.aten.addmm.default(arg725_1, fused_swiglu_9, arg724_1);  arg725_1 = fused_swiglu_9 = arg724_1 = None
        add_773 = torch.ops.aten.add.Tensor(addmm_246, softcap_18);  addmm_246 = softcap_18 = None
        softcap_19 = torch.ops.qianchuan_triton.softcap.default(add_773, 50.0);  add_773 = None
        full_default_209 = torch.ops.aten.full.default([batch_size, 512], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_208 = torch.ops.aten.where.self(logical_or_12, full_default_209, softcap_19);  full_default_209 = softcap_19 = None
        addmm_247 = torch.ops.aten.addmm.default(arg727_1, where_208, arg726_1);  arg727_1 = arg726_1 = None
        relu_88 = torch.ops.aten.relu.default(addmm_247);  addmm_247 = None
        addmm_248 = torch.ops.aten.addmm.default(arg729_1, relu_88, arg728_1);  arg729_1 = relu_88 = arg728_1 = None
        squeeze_144 = torch.ops.aten.squeeze.dim(addmm_248, 1);  addmm_248 = None
        addmm_249 = torch.ops.aten.addmm.default(arg731_1, where_208, arg730_1);  arg731_1 = arg730_1 = None
        relu_89 = torch.ops.aten.relu.default(addmm_249);  addmm_249 = None
        addmm_250 = torch.ops.aten.addmm.default(arg733_1, relu_89, arg732_1);  arg733_1 = relu_89 = arg732_1 = None
        squeeze_145 = torch.ops.aten.squeeze.dim(addmm_250, 1);  addmm_250 = None
        addmm_251 = torch.ops.aten.addmm.default(arg735_1, slice_3, arg734_1);  arg735_1 = arg734_1 = None
        relu_90 = torch.ops.aten.relu.default(addmm_251);  addmm_251 = None
        addmm_252 = torch.ops.aten.addmm.default(arg737_1, cat_61, arg736_1);  arg737_1 = cat_61 = arg736_1 = None
        relu_91 = torch.ops.aten.relu.default(addmm_252);  addmm_252 = None
        addmm_253 = torch.ops.aten.addmm.default(arg739_1, cat_62, arg738_1);  arg739_1 = cat_62 = arg738_1 = None
        relu_92 = torch.ops.aten.relu.default(addmm_253);  addmm_253 = None
        eq_644 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_645 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_33 = torch.ops.aten.logical_or.default(eq_644, eq_645);  eq_644 = eq_645 = None
        repeat_24 = torch.ops.aten.repeat.default(logical_or_33, [1, 512]);  logical_or_33 = None
        where_209 = torch.ops.aten.where.self(repeat_24, relu_92, relu_91);  repeat_24 = relu_92 = relu_91 = None
        cat_121 = torch.ops.aten.cat.default([where_209, relu_90], 1);  where_209 = relu_90 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(cat_121, 1);  cat_121 = None
        view_379 = torch.ops.aten.view.default(unsqueeze_56, [batch_size, 1024]);  unsqueeze_56 = None
        addmm_254 = torch.ops.aten.addmm.default(arg741_1, view_379, arg740_1);  arg741_1 = view_379 = arg740_1 = None
        full_default_210 = torch.ops.aten.full.default([4872, 32], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        addmm_255 = torch.ops.aten.addmm.default(arg743_1, full_default_210, arg742_1);  arg743_1 = full_default_210 = arg742_1 = None
        view_382 = torch.ops.aten.view.default(addmm_255, [7, 696, 64]);  addmm_255 = None
        view_383 = torch.ops.aten.view.default(view_382, [-1, 87, 512]);  view_382 = None
        full_default_211 = torch.ops.aten.full.default([7, 87, 8], 0, dtype = torch.int32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        amax_3 = torch.ops.aten.amax.default(full_default_211, [2]);  full_default_211 = None
        view_385 = torch.ops.aten.view.default(amax_3, [-1])
        gt_5 = torch.ops.aten.gt.Scalar(view_385, 0);  view_385 = None
        nonzero_6 = torch.ops.aten.nonzero.default(gt_5);  gt_5 = None
        sym_size_int_64 = torch.ops.aten.sym_size.int(nonzero_6, 0)
        ge_16 = sym_size_int_64 >= 0
        _assert_scalar_15 = torch.ops.aten._assert_scalar.default(ge_16, "Runtime assertion failed for expression u12 >= 0 on node 'ge_6'");  ge_16 = _assert_scalar_15 = None
        le_4 = sym_size_int_64 <= 609
        _assert_scalar_16 = torch.ops.aten._assert_scalar.default(le_4, "Runtime assertion failed for expression u12 <= 609 on node 'le_11'");  le_4 = _assert_scalar_16 = None
        iota_24 = torch.ops.prims.iota.default(87, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        repeat_25 = torch.ops.aten.repeat.default(iota_24, [7]);  iota_24 = None
        index_45 = torch.ops.aten.index.Tensor(repeat_25, [nonzero_6]);  repeat_25 = None
        sum_164 = torch.ops.aten.sum.dim_IntList(amax_3, [1]);  amax_3 = None
        cumsum_27 = torch.ops.aten.cumsum.default(sum_164, 0);  sum_164 = None
        constant_pad_nd_14 = torch.ops.aten.constant_pad_nd.default(cumsum_27, [1, 0], 0.0);  cumsum_27 = None
        view_386 = torch.ops.aten.view.default(view_383, [-1, 512]);  view_383 = None
        index_46 = torch.ops.aten.index.Tensor(view_386, [nonzero_6]);  view_386 = nonzero_6 = None
        full_default_3 = torch.ops.aten.full.default([sym_size_int_64, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        convert_element_type_63 = torch.ops.prims.convert_element_type.default(arg147_1, torch.int32)
        iota_25 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        add_790 = torch.ops.aten.add.Tensor(iota_25, 1);  iota_25 = None
        iota_26 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        repeat_27 = torch.ops.aten.repeat.default(iota_26, [batch_size]);  iota_26 = None
        cumsum_28 = torch.ops.aten.cumsum.default(convert_element_type_63, 0);  convert_element_type_63 = None
        constant_pad_nd_15 = torch.ops.aten.constant_pad_nd.default(cumsum_28, [1, 0], 0.0);  cumsum_28 = None
        mul_981 = torch.ops.aten.mul.Tensor(constant_pad_nd_15, 1);  constant_pad_nd_15 = None
        ascend_create_position_offset_3 = torch.ops.ascend_triton.ascend_create_position_offset.default(repeat_27, mul_981)
        ascend_seq_tensor_concat_9 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(addmm_254, index_46, mul_981, constant_pad_nd_14);  addmm_254 = index_46 = None
        add_794 = torch.ops.aten.add.Tensor(mul_981, constant_pad_nd_14)
        ascend_position_concat_3 = torch.ops.ascend_triton.ascend_position_concat.default(repeat_27, index_45, mul_981, constant_pad_nd_14, ascend_create_position_offset_3);  repeat_27 = index_45 = ascend_create_position_offset_3 = None
        sym_size_int_66 = torch.ops.aten.sym_size.int(ascend_position_concat_3, 0);  ascend_position_concat_3 = None
        ascend_seq_tensor_concat_10 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(add_790, full_default_3, mul_981, constant_pad_nd_14);  add_790 = full_default_3 = None
        full_default_212 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        full_224 = torch.ops.aten.full.default([sym_size_int_64], 1, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  sym_size_int_64 = None
        ascend_seq_tensor_concat_11 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(full_default_212, full_224, mul_981, constant_pad_nd_14);  full_default_212 = full_224 = mul_981 = constant_pad_nd_14 = None
        eq_670 = torch.ops.aten.eq.Scalar(ascend_seq_tensor_concat_11, 0);  ascend_seq_tensor_concat_11 = None
        nonzero_7 = torch.ops.aten.nonzero.default(eq_670);  eq_670 = None
        _assert_scalar_17 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression u13 >= batch_size on node 'ge_7'");  _assert_scalar_17 = None
        _assert_scalar_18 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression u13 <= batch_size on node 'le_12'");  _assert_scalar_18 = None
        _assert_scalar_19 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression Eq(u13, batch_size) on node 'eq_131'");  _assert_scalar_19 = None
        squeeze_146 = torch.ops.aten.squeeze.dim(nonzero_7, -1);  nonzero_7 = None
        native_layer_norm_23 = torch.ops.aten.native_layer_norm.default(ascend_seq_tensor_concat_9, [512], arg744_1, arg745_1, 1e-06);  ascend_seq_tensor_concat_9 = arg744_1 = arg745_1 = None
        getitem_69 = native_layer_norm_23[0];  native_layer_norm_23 = None
        native_layer_norm_24 = torch.ops.aten.native_layer_norm.default(getitem_69, [512], arg746_1, arg747_1, 1e-06);  arg746_1 = arg747_1 = None
        getitem_72 = native_layer_norm_24[0];  native_layer_norm_24 = None
        addmm_256 = torch.ops.aten.addmm.default(arg749_1, getitem_72, arg748_1);  arg749_1 = arg748_1 = None
        addmm_257 = torch.ops.aten.addmm.default(arg751_1, getitem_72, arg750_1);  arg751_1 = arg750_1 = None
        addmm_258 = torch.ops.aten.addmm.default(arg753_1, getitem_72, arg752_1);  arg753_1 = getitem_72 = arg752_1 = None
        view_390 = torch.ops.aten.view.default(addmm_256, [-1, 4, 128]);  addmm_256 = None
        view_391 = torch.ops.aten.view.default(addmm_257, [-1, 4, 128]);  addmm_257 = None
        view_392 = torch.ops.aten.view.default(addmm_258, [-1, 4, 128]);  addmm_258 = None
        ascend_flash_attention_10 = torch.ops.ascend_triton.ascend_flash_attention.default(view_390, view_391, view_392, ascend_seq_tensor_concat_10, ascend_seq_tensor_concat_10, add_794, add_794, 287, 287, 0.08838834764831843, 1);  view_390 = view_391 = view_392 = None
        view_393 = torch.ops.aten.view.default(ascend_flash_attention_10, [-1, 512]);  ascend_flash_attention_10 = None
        addmm_259 = torch.ops.aten.addmm.default(arg755_1, view_393, arg754_1);  arg755_1 = view_393 = arg754_1 = None
        add_857 = torch.ops.aten.add.Tensor(addmm_259, getitem_69);  addmm_259 = getitem_69 = None
        softcap_20 = torch.ops.qianchuan_triton.softcap.default(add_857, 50.0);  add_857 = None
        native_layer_norm_25 = torch.ops.aten.native_layer_norm.default(softcap_20, [512], arg756_1, arg757_1, 1e-06);  arg756_1 = arg757_1 = None
        getitem_75 = native_layer_norm_25[0];  native_layer_norm_25 = None
        fused_swiglu_10 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_75, arg758_1, arg759_1, arg760_1, arg761_1, False, False);  getitem_75 = arg758_1 = arg759_1 = arg760_1 = arg761_1 = None
        addmm_260 = torch.ops.aten.addmm.default(arg763_1, fused_swiglu_10, arg762_1);  arg763_1 = fused_swiglu_10 = arg762_1 = None
        add_879 = torch.ops.aten.add.Tensor(addmm_260, softcap_20);  addmm_260 = softcap_20 = None
        softcap_21 = torch.ops.qianchuan_triton.softcap.default(add_879, 50.0);  add_879 = None
        index_47 = torch.ops.aten.index.Tensor(softcap_21, [squeeze_146])
        native_layer_norm_26 = torch.ops.aten.native_layer_norm.default(softcap_21, [512], arg764_1, arg765_1, 1e-06);  softcap_21 = arg764_1 = arg765_1 = None
        getitem_78 = native_layer_norm_26[0];  native_layer_norm_26 = None
        index_51 = torch.ops.aten.index.Tensor(getitem_78, [squeeze_146])
        full_226 = torch.ops.aten.full.default([sym_size_int_66], False, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  sym_size_int_66 = None
        full_default_213 = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        index_put_7 = torch.ops.aten.index_put.default(full_226, [squeeze_146], full_default_213);  full_226 = full_default_213 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(index_put_7, torch.int64);  index_put_7 = None
        cumsum_30 = torch.ops.aten.cumsum.default(convert_element_type_67, 0);  convert_element_type_67 = None
        constant_pad_nd_17 = torch.ops.aten.constant_pad_nd.default(cumsum_30, [1, 0], 0.0);  cumsum_30 = None
        index_52 = torch.ops.aten.index.Tensor(constant_pad_nd_17, [add_794]);  constant_pad_nd_17 = None
        slice_690 = torch.ops.aten.slice.Tensor(index_52, 0, 1, 9223372036854775807)
        slice_691 = torch.ops.aten.slice.Tensor(index_52, 0, 0, -1)
        sub_323 = torch.ops.aten.sub.Tensor(slice_690, slice_691);  slice_690 = slice_691 = None
        max_8 = torch.ops.aten.max.default(sub_323);  sub_323 = None
        _local_scalar_dense_3 = torch.ops.aten._local_scalar_dense.default(max_8);  max_8 = None
        index_54 = torch.ops.aten.index.Tensor(ascend_seq_tensor_concat_10, [squeeze_146]);  squeeze_146 = None
        addmm_261 = torch.ops.aten.addmm.default(arg767_1, index_51, arg766_1);  arg767_1 = index_51 = arg766_1 = None
        addmm_262 = torch.ops.aten.addmm.default(arg769_1, getitem_78, arg768_1);  arg769_1 = arg768_1 = None
        addmm_263 = torch.ops.aten.addmm.default(arg771_1, getitem_78, arg770_1);  arg771_1 = getitem_78 = arg770_1 = None
        view_394 = torch.ops.aten.view.default(addmm_261, [-1, 4, 128]);  addmm_261 = None
        view_395 = torch.ops.aten.view.default(addmm_262, [-1, 4, 128]);  addmm_262 = None
        view_396 = torch.ops.aten.view.default(addmm_263, [-1, 4, 128]);  addmm_263 = None
        ascend_flash_attention_11 = torch.ops.ascend_triton.ascend_flash_attention.default(view_394, view_395, view_396, index_54, ascend_seq_tensor_concat_10, index_52, add_794, _local_scalar_dense_3, 287, 0.08838834764831843, 1);  view_394 = view_395 = view_396 = index_54 = ascend_seq_tensor_concat_10 = index_52 = add_794 = _local_scalar_dense_3 = None
        view_397 = torch.ops.aten.view.default(ascend_flash_attention_11, [-1, 512]);  ascend_flash_attention_11 = None
        addmm_264 = torch.ops.aten.addmm.default(arg773_1, view_397, arg772_1);  arg773_1 = view_397 = arg772_1 = None
        add_933 = torch.ops.aten.add.Tensor(addmm_264, index_47);  addmm_264 = index_47 = None
        softcap_22 = torch.ops.qianchuan_triton.softcap.default(add_933, 50.0);  add_933 = None
        native_layer_norm_27 = torch.ops.aten.native_layer_norm.default(softcap_22, [512], arg774_1, arg775_1, 1e-06);  arg774_1 = arg775_1 = None
        getitem_81 = native_layer_norm_27[0];  native_layer_norm_27 = None
        fused_swiglu_11 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_81, arg776_1, arg777_1, arg778_1, arg779_1, False, False);  getitem_81 = arg776_1 = arg777_1 = arg778_1 = arg779_1 = None
        addmm_265 = torch.ops.aten.addmm.default(arg781_1, fused_swiglu_11, arg780_1);  arg781_1 = fused_swiglu_11 = arg780_1 = None
        add_934 = torch.ops.aten.add.Tensor(addmm_265, softcap_22);  addmm_265 = softcap_22 = None
        softcap_23 = torch.ops.qianchuan_triton.softcap.default(add_934, 50.0);  add_934 = None
        full_default_214 = torch.ops.aten.full.default([batch_size, 512], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_210 = torch.ops.aten.where.self(logical_or_12, full_default_214, softcap_23);  full_default_214 = softcap_23 = None
        addmm_266 = torch.ops.aten.addmm.default(arg783_1, where_210, arg782_1);  arg783_1 = arg782_1 = None
        relu_93 = torch.ops.aten.relu.default(addmm_266);  addmm_266 = None
        addmm_267 = torch.ops.aten.addmm.default(arg785_1, relu_93, arg784_1);  arg785_1 = relu_93 = arg784_1 = None
        squeeze_147 = torch.ops.aten.squeeze.dim(addmm_267, 1);  addmm_267 = None
        addmm_268 = torch.ops.aten.addmm.default(arg787_1, where_210, arg786_1);  arg787_1 = arg786_1 = None
        relu_94 = torch.ops.aten.relu.default(addmm_268);  addmm_268 = None
        addmm_269 = torch.ops.aten.addmm.default(arg789_1, relu_94, arg788_1);  arg789_1 = relu_94 = arg788_1 = None
        squeeze_148 = torch.ops.aten.squeeze.dim(addmm_269, 1);  addmm_269 = None
        addmm_270 = torch.ops.aten.addmm.default(arg791_1, slice_3, arg790_1);  arg791_1 = arg790_1 = None
        relu_95 = torch.ops.aten.relu.default(addmm_270);  addmm_270 = None
        addmm_271 = torch.ops.aten.addmm.default(arg793_1, cat_63, arg792_1);  arg793_1 = cat_63 = arg792_1 = None
        relu_96 = torch.ops.aten.relu.default(addmm_271);  addmm_271 = None
        addmm_272 = torch.ops.aten.addmm.default(arg795_1, cat_64, arg794_1);  arg795_1 = cat_64 = arg794_1 = None
        relu_97 = torch.ops.aten.relu.default(addmm_272);  addmm_272 = None
        eq_764 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_765 = torch.ops.aten.eq.Scalar(where_7, 3)
        logical_or_34 = torch.ops.aten.logical_or.default(eq_764, eq_765);  eq_764 = eq_765 = None
        repeat_28 = torch.ops.aten.repeat.default(logical_or_34, [1, 512]);  logical_or_34 = None
        where_211 = torch.ops.aten.where.self(repeat_28, relu_97, relu_96);  repeat_28 = relu_97 = relu_96 = None
        cat_122 = torch.ops.aten.cat.default([where_211, relu_95], 1);  where_211 = relu_95 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(cat_122, 1);  cat_122 = None
        view_400 = torch.ops.aten.view.default(unsqueeze_57, [batch_size, 1024]);  unsqueeze_57 = None
        addmm_273 = torch.ops.aten.addmm.default(arg797_1, view_400, arg796_1);  arg797_1 = view_400 = arg796_1 = None
        slice_692 = torch.ops.aten.slice.Tensor(arg298_1, 2, 1, 34);  arg298_1 = None
        slice_693 = torch.ops.aten.slice.Tensor(slice_692, 2, 0, 32);  slice_692 = None
        sign_21 = torch.ops.aten.sign.default(arg299_1);  arg299_1 = None
        expand_42 = torch.ops.aten.expand.default(slice_693, [7, 432, 32]);  slice_693 = None
        expand_43 = torch.ops.aten.expand.default(arg798_1, [7, 32, 64]);  arg798_1 = None
        bmm_15 = torch.ops.aten.bmm.default(expand_42, expand_43);  expand_42 = expand_43 = None
        add_935 = torch.ops.aten.add.Tensor(bmm_15, arg799_1);  bmm_15 = arg799_1 = None
        view_405 = torch.ops.aten.view.default(add_935, [-1, 54, 512]);  add_935 = None
        view_406 = torch.ops.aten.view.default(sign_21, [-1, 54, 8]);  sign_21 = None
        amax_4 = torch.ops.aten.amax.default(view_406, [2]);  view_406 = None
        view_407 = torch.ops.aten.view.default(amax_4, [-1])
        gt_6 = torch.ops.aten.gt.Scalar(view_407, 0);  view_407 = None
        nonzero_8 = torch.ops.aten.nonzero.default(gt_6);  gt_6 = None
        sym_size_int_80 = torch.ops.aten.sym_size.int(nonzero_8, 0)
        ge_21 = sym_size_int_80 >= 0
        _assert_scalar_20 = torch.ops.aten._assert_scalar.default(ge_21, "Runtime assertion failed for expression u16 >= 0 on node 'ge_8'");  ge_21 = _assert_scalar_20 = None
        le_5 = sym_size_int_80 <= 378
        _assert_scalar_21 = torch.ops.aten._assert_scalar.default(le_5, "Runtime assertion failed for expression u16 <= 378 on node 'le_13'");  le_5 = _assert_scalar_21 = None
        iota_27 = torch.ops.prims.iota.default(54, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        repeat_29 = torch.ops.aten.repeat.default(iota_27, [7]);  iota_27 = None
        index_55 = torch.ops.aten.index.Tensor(repeat_29, [nonzero_8]);  repeat_29 = None
        sum_165 = torch.ops.aten.sum.dim_IntList(amax_4, [1]);  amax_4 = None
        cumsum_31 = torch.ops.aten.cumsum.default(sum_165, 0);  sum_165 = None
        constant_pad_nd_18 = torch.ops.aten.constant_pad_nd.default(cumsum_31, [1, 0], 0.0);  cumsum_31 = None
        view_408 = torch.ops.aten.view.default(view_405, [-1, 512]);  view_405 = None
        index_56 = torch.ops.aten.index.Tensor(view_408, [nonzero_8]);  view_408 = nonzero_8 = None
        full_default_4 = torch.ops.aten.full.default([sym_size_int_80, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(arg147_1, torch.int32)
        iota_28 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        add_952 = torch.ops.aten.add.Tensor(iota_28, 1);  iota_28 = None
        iota_29 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        repeat_31 = torch.ops.aten.repeat.default(iota_29, [batch_size]);  iota_29 = None
        cumsum_32 = torch.ops.aten.cumsum.default(convert_element_type_70, 0);  convert_element_type_70 = None
        constant_pad_nd_19 = torch.ops.aten.constant_pad_nd.default(cumsum_32, [1, 0], 0.0);  cumsum_32 = None
        mul_1139 = torch.ops.aten.mul.Tensor(constant_pad_nd_19, 1);  constant_pad_nd_19 = None
        ascend_create_position_offset_4 = torch.ops.ascend_triton.ascend_create_position_offset.default(repeat_31, mul_1139)
        ascend_seq_tensor_concat_12 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(addmm_273, index_56, mul_1139, constant_pad_nd_18);  addmm_273 = index_56 = None
        add_956 = torch.ops.aten.add.Tensor(mul_1139, constant_pad_nd_18)
        ascend_position_concat_4 = torch.ops.ascend_triton.ascend_position_concat.default(repeat_31, index_55, mul_1139, constant_pad_nd_18, ascend_create_position_offset_4);  repeat_31 = index_55 = ascend_create_position_offset_4 = None
        sym_size_int_82 = torch.ops.aten.sym_size.int(ascend_position_concat_4, 0);  ascend_position_concat_4 = None
        ascend_seq_tensor_concat_13 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(add_952, full_default_4, mul_1139, constant_pad_nd_18);  add_952 = full_default_4 = None
        full_default_215 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        full_230 = torch.ops.aten.full.default([sym_size_int_80], 1, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  sym_size_int_80 = None
        ascend_seq_tensor_concat_14 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(full_default_215, full_230, mul_1139, constant_pad_nd_18);  full_default_215 = full_230 = mul_1139 = constant_pad_nd_18 = None
        eq_790 = torch.ops.aten.eq.Scalar(ascend_seq_tensor_concat_14, 0);  ascend_seq_tensor_concat_14 = None
        nonzero_9 = torch.ops.aten.nonzero.default(eq_790);  eq_790 = None
        _assert_scalar_22 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression u17 >= batch_size on node 'ge_9'");  _assert_scalar_22 = None
        _assert_scalar_23 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression u17 <= batch_size on node 'le_14'");  _assert_scalar_23 = None
        _assert_scalar_24 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression Eq(u17, batch_size) on node 'eq_132'");  _assert_scalar_24 = None
        squeeze_149 = torch.ops.aten.squeeze.dim(nonzero_9, -1);  nonzero_9 = None
        native_layer_norm_28 = torch.ops.aten.native_layer_norm.default(ascend_seq_tensor_concat_12, [512], arg800_1, arg801_1, 1e-06);  ascend_seq_tensor_concat_12 = arg800_1 = arg801_1 = None
        getitem_84 = native_layer_norm_28[0];  native_layer_norm_28 = None
        native_layer_norm_29 = torch.ops.aten.native_layer_norm.default(getitem_84, [512], arg802_1, arg803_1, 1e-06);  arg802_1 = arg803_1 = None
        getitem_87 = native_layer_norm_29[0];  native_layer_norm_29 = None
        addmm_274 = torch.ops.aten.addmm.default(arg805_1, getitem_87, arg804_1);  arg805_1 = arg804_1 = None
        addmm_275 = torch.ops.aten.addmm.default(arg807_1, getitem_87, arg806_1);  arg807_1 = arg806_1 = None
        addmm_276 = torch.ops.aten.addmm.default(arg809_1, getitem_87, arg808_1);  arg809_1 = getitem_87 = arg808_1 = None
        view_412 = torch.ops.aten.view.default(addmm_274, [-1, 4, 128]);  addmm_274 = None
        view_413 = torch.ops.aten.view.default(addmm_275, [-1, 4, 128]);  addmm_275 = None
        view_414 = torch.ops.aten.view.default(addmm_276, [-1, 4, 128]);  addmm_276 = None
        ascend_flash_attention_12 = torch.ops.ascend_triton.ascend_flash_attention.default(view_412, view_413, view_414, ascend_seq_tensor_concat_13, ascend_seq_tensor_concat_13, add_956, add_956, 254, 254, 0.08838834764831843, 1);  view_412 = view_413 = view_414 = None
        view_415 = torch.ops.aten.view.default(ascend_flash_attention_12, [-1, 512]);  ascend_flash_attention_12 = None
        addmm_277 = torch.ops.aten.addmm.default(arg811_1, view_415, arg810_1);  arg811_1 = view_415 = arg810_1 = None
        add_1019 = torch.ops.aten.add.Tensor(addmm_277, getitem_84);  addmm_277 = getitem_84 = None
        softcap_24 = torch.ops.qianchuan_triton.softcap.default(add_1019, 50.0);  add_1019 = None
        native_layer_norm_30 = torch.ops.aten.native_layer_norm.default(softcap_24, [512], arg812_1, arg813_1, 1e-06);  arg812_1 = arg813_1 = None
        getitem_90 = native_layer_norm_30[0];  native_layer_norm_30 = None
        fused_swiglu_12 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_90, arg814_1, arg815_1, arg816_1, arg817_1, False, False);  getitem_90 = arg814_1 = arg815_1 = arg816_1 = arg817_1 = None
        addmm_278 = torch.ops.aten.addmm.default(arg819_1, fused_swiglu_12, arg818_1);  arg819_1 = fused_swiglu_12 = arg818_1 = None
        add_1041 = torch.ops.aten.add.Tensor(addmm_278, softcap_24);  addmm_278 = softcap_24 = None
        softcap_25 = torch.ops.qianchuan_triton.softcap.default(add_1041, 50.0);  add_1041 = None
        index_57 = torch.ops.aten.index.Tensor(softcap_25, [squeeze_149])
        native_layer_norm_31 = torch.ops.aten.native_layer_norm.default(softcap_25, [512], arg820_1, arg821_1, 1e-06);  softcap_25 = arg820_1 = arg821_1 = None
        getitem_93 = native_layer_norm_31[0];  native_layer_norm_31 = None
        index_61 = torch.ops.aten.index.Tensor(getitem_93, [squeeze_149])
        full_232 = torch.ops.aten.full.default([sym_size_int_82], False, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  sym_size_int_82 = None
        full_default_216 = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        index_put_9 = torch.ops.aten.index_put.default(full_232, [squeeze_149], full_default_216);  full_232 = full_default_216 = None
        convert_element_type_74 = torch.ops.prims.convert_element_type.default(index_put_9, torch.int64);  index_put_9 = None
        cumsum_34 = torch.ops.aten.cumsum.default(convert_element_type_74, 0);  convert_element_type_74 = None
        constant_pad_nd_21 = torch.ops.aten.constant_pad_nd.default(cumsum_34, [1, 0], 0.0);  cumsum_34 = None
        index_62 = torch.ops.aten.index.Tensor(constant_pad_nd_21, [add_956]);  constant_pad_nd_21 = None
        slice_698 = torch.ops.aten.slice.Tensor(index_62, 0, 1, 9223372036854775807)
        slice_699 = torch.ops.aten.slice.Tensor(index_62, 0, 0, -1)
        sub_381 = torch.ops.aten.sub.Tensor(slice_698, slice_699);  slice_698 = slice_699 = None
        max_10 = torch.ops.aten.max.default(sub_381);  sub_381 = None
        _local_scalar_dense_4 = torch.ops.aten._local_scalar_dense.default(max_10);  max_10 = None
        index_64 = torch.ops.aten.index.Tensor(ascend_seq_tensor_concat_13, [squeeze_149]);  squeeze_149 = None
        addmm_279 = torch.ops.aten.addmm.default(arg823_1, index_61, arg822_1);  arg823_1 = index_61 = arg822_1 = None
        addmm_280 = torch.ops.aten.addmm.default(arg825_1, getitem_93, arg824_1);  arg825_1 = arg824_1 = None
        addmm_281 = torch.ops.aten.addmm.default(arg827_1, getitem_93, arg826_1);  arg827_1 = getitem_93 = arg826_1 = None
        view_416 = torch.ops.aten.view.default(addmm_279, [-1, 4, 128]);  addmm_279 = None
        view_417 = torch.ops.aten.view.default(addmm_280, [-1, 4, 128]);  addmm_280 = None
        view_418 = torch.ops.aten.view.default(addmm_281, [-1, 4, 128]);  addmm_281 = None
        ascend_flash_attention_13 = torch.ops.ascend_triton.ascend_flash_attention.default(view_416, view_417, view_418, index_64, ascend_seq_tensor_concat_13, index_62, add_956, _local_scalar_dense_4, 254, 0.08838834764831843, 1);  view_416 = view_417 = view_418 = index_64 = ascend_seq_tensor_concat_13 = index_62 = add_956 = _local_scalar_dense_4 = None
        view_419 = torch.ops.aten.view.default(ascend_flash_attention_13, [-1, 512]);  ascend_flash_attention_13 = None
        addmm_282 = torch.ops.aten.addmm.default(arg829_1, view_419, arg828_1);  arg829_1 = view_419 = arg828_1 = None
        add_1095 = torch.ops.aten.add.Tensor(addmm_282, index_57);  addmm_282 = index_57 = None
        softcap_26 = torch.ops.qianchuan_triton.softcap.default(add_1095, 50.0);  add_1095 = None
        native_layer_norm_32 = torch.ops.aten.native_layer_norm.default(softcap_26, [512], arg830_1, arg831_1, 1e-06);  arg830_1 = arg831_1 = None
        getitem_96 = native_layer_norm_32[0];  native_layer_norm_32 = None
        fused_swiglu_13 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_96, arg832_1, arg833_1, arg834_1, arg835_1, False, False);  getitem_96 = arg832_1 = arg833_1 = arg834_1 = arg835_1 = None
        addmm_283 = torch.ops.aten.addmm.default(arg837_1, fused_swiglu_13, arg836_1);  arg837_1 = fused_swiglu_13 = arg836_1 = None
        add_1096 = torch.ops.aten.add.Tensor(addmm_283, softcap_26);  addmm_283 = softcap_26 = None
        softcap_27 = torch.ops.qianchuan_triton.softcap.default(add_1096, 50.0);  add_1096 = None
        full_default_217 = torch.ops.aten.full.default([batch_size, 512], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_212 = torch.ops.aten.where.self(logical_or_12, full_default_217, softcap_27);  full_default_217 = softcap_27 = None
        addmm_284 = torch.ops.aten.addmm.default(arg839_1, where_212, arg838_1);  arg839_1 = arg838_1 = None
        relu_98 = torch.ops.aten.relu.default(addmm_284);  addmm_284 = None
        addmm_285 = torch.ops.aten.addmm.default(arg841_1, relu_98, arg840_1);  arg841_1 = relu_98 = arg840_1 = None
        squeeze_150 = torch.ops.aten.squeeze.dim(addmm_285, 1);  addmm_285 = None
        addmm_286 = torch.ops.aten.addmm.default(arg843_1, where_212, arg842_1);  arg843_1 = arg842_1 = None
        relu_99 = torch.ops.aten.relu.default(addmm_286);  addmm_286 = None
        addmm_287 = torch.ops.aten.addmm.default(arg845_1, relu_99, arg844_1);  arg845_1 = relu_99 = arg844_1 = None
        squeeze_151 = torch.ops.aten.squeeze.dim(addmm_287, 1);  addmm_287 = None
        slice_700 = torch.ops.aten.slice.Tensor(arg15_1, 1, 24220, 24300)
        full_default_218 = torch.ops.aten.full.default([batch_size, 80], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_213 = torch.ops.aten.where.self(logical_not, full_default_218, slice_700);  full_default_218 = slice_700 = None
        slice_701 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35098, 35178)
        repeat_32 = torch.ops.aten.repeat.default(logical_not_1, [1, 80])
        where_214 = torch.ops.aten.where.self(repeat_32, where_213, slice_701);  repeat_32 = where_213 = slice_701 = None
        slice_702 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39296, 39376)
        slice_703 = torch.ops.aten.slice.Tensor(arg15_1, 1, 25066, 25146)
        full_default_219 = torch.ops.aten.full.default([batch_size, 80], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_215 = torch.ops.aten.where.self(logical_not, full_default_219, slice_703);  full_default_219 = slice_703 = None
        slice_704 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35470, 35550)
        repeat_33 = torch.ops.aten.repeat.default(logical_not_1, [1, 80])
        where_216 = torch.ops.aten.where.self(repeat_33, where_215, slice_704);  repeat_33 = where_215 = slice_704 = None
        slice_705 = torch.ops.aten.slice.Tensor(arg15_1, 1, 39913, 39993)
        slice_706 = torch.ops.aten.slice.Tensor(arg15_1, 1, 22385, 22497)
        slice_707 = torch.ops.aten.slice.Tensor(arg15_1, 1, 34648, 34760)
        repeat_34 = torch.ops.aten.repeat.default(logical_not_1, [1, 112])
        where_217 = torch.ops.aten.where.self(repeat_34, slice_706, slice_707);  repeat_34 = slice_706 = slice_707 = None
        slice_708 = torch.ops.aten.slice.Tensor(arg15_1, 1, 38647, 38759)
        slice_709 = torch.ops.aten.slice.Tensor(arg15_1, 1, 28076, 28204)
        slice_710 = torch.ops.aten.slice.Tensor(arg15_1, 1, 35842, 35970)
        repeat_35 = torch.ops.aten.repeat.default(logical_not_1, [1, 128])
        where_218 = torch.ops.aten.where.self(repeat_35, slice_709, slice_710);  repeat_35 = slice_709 = slice_710 = None
        slice_711 = torch.ops.aten.slice.Tensor(arg15_1, 1, 40530, 40658)
        slice_712 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43554, 43682)
        full_default_220 = torch.ops.aten.full.default([batch_size, 128], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_219 = torch.ops.aten.where.self(logical_not, full_default_220, slice_712);  full_default_220 = slice_712 = None
        slice_713 = torch.ops.aten.slice.Tensor(arg15_1, 1, 33380, 33508)
        repeat_36 = torch.ops.aten.repeat.default(logical_not_1, [1, 128])
        where_220 = torch.ops.aten.where.self(repeat_36, where_219, slice_713);  repeat_36 = where_219 = slice_713 = None
        slice_714 = torch.ops.aten.slice.Tensor(arg15_1, 1, 37365, 37493)
        cat_123 = torch.ops.aten.cat.default([arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1], 1);  arg846_1 = arg847_1 = arg848_1 = arg849_1 = arg850_1 = arg851_1 = arg852_1 = arg853_1 = arg854_1 = arg855_1 = None
        cat_124 = torch.ops.aten.cat.default([where_214, slice_702, where_216, slice_705, where_217, slice_708, where_218, slice_711, where_220, slice_714], 1);  where_214 = slice_702 = where_216 = slice_705 = where_217 = slice_708 = where_218 = slice_711 = where_220 = slice_714 = None
        mul_1286 = torch.ops.aten.mul.Tensor(cat_123, cat_124)
        sum_166 = torch.ops.aten.sum.dim_IntList(mul_1286, [1])
        mul_1287 = torch.ops.aten.mul.Tensor(slice_639, arg462_1);  slice_639 = arg462_1 = None
        squeeze_172 = torch.ops.aten.squeeze.dims(mul_1287, [2]);  mul_1287 = None
        sum_167 = torch.ops.aten.sum.dim_IntList(squeeze_172, [1], True);  squeeze_172 = None
        mul_1288 = torch.ops.aten.mul.Tensor(slice_642, arg465_1);  slice_642 = arg465_1 = None
        squeeze_173 = torch.ops.aten.squeeze.dims(mul_1288, [2]);  mul_1288 = None
        sum_168 = torch.ops.aten.sum.dim_IntList(squeeze_173, [1], True);  squeeze_173 = None
        mul_1289 = torch.ops.aten.mul.Tensor(slice_645, arg468_1);  slice_645 = arg468_1 = None
        squeeze_174 = torch.ops.aten.squeeze.dims(mul_1289, [2]);  mul_1289 = None
        sum_169 = torch.ops.aten.sum.dim_IntList(squeeze_174, [1], True);  squeeze_174 = None
        mul_1290 = torch.ops.aten.mul.Tensor(slice_648, arg471_1);  slice_648 = arg471_1 = None
        squeeze_175 = torch.ops.aten.squeeze.dims(mul_1290, [2]);  mul_1290 = None
        sum_170 = torch.ops.aten.sum.dim_IntList(squeeze_175, [1], True);  squeeze_175 = None
        mul_1291 = torch.ops.aten.mul.Tensor(slice_651, arg474_1);  slice_651 = arg474_1 = None
        squeeze_176 = torch.ops.aten.squeeze.dims(mul_1291, [2]);  mul_1291 = None
        sum_171 = torch.ops.aten.sum.dim_IntList(squeeze_176, [1], True);  squeeze_176 = None
        mul_1292 = torch.ops.aten.mul.Tensor(slice_654, arg477_1);  slice_654 = arg477_1 = None
        squeeze_177 = torch.ops.aten.squeeze.dims(mul_1292, [2]);  mul_1292 = None
        sum_172 = torch.ops.aten.sum.dim_IntList(squeeze_177, [1], True);  squeeze_177 = None
        mul_1293 = torch.ops.aten.mul.Tensor(index, slice_516);  index = slice_516 = None
        squeeze_178 = torch.ops.aten.squeeze.dims(mul_1293, [2]);  mul_1293 = None
        sum_173 = torch.ops.aten.sum.dim_IntList(squeeze_178, [1], True);  squeeze_178 = None
        full_default_221 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_221 = torch.ops.aten.where.self(logical_or_12, full_default_221, sum_173);  full_default_221 = sum_173 = None
        mul_1294 = torch.ops.aten.mul.Tensor(slice_527, slice_529);  slice_527 = slice_529 = None
        squeeze_179 = torch.ops.aten.squeeze.dims(mul_1294, [2]);  mul_1294 = None
        sum_174 = torch.ops.aten.sum.dim_IntList(squeeze_179, [1], True);  squeeze_179 = None
        mul_1295 = torch.ops.aten.mul.Tensor(slice_536, slice_538);  slice_536 = slice_538 = None
        squeeze_180 = torch.ops.aten.squeeze.dims(mul_1295, [2]);  mul_1295 = None
        sum_175 = torch.ops.aten.sum.dim_IntList(squeeze_180, [1], True);  squeeze_180 = None
        mul_1296 = torch.ops.aten.mul.Tensor(index_3, slice_548);  index_3 = slice_548 = None
        squeeze_181 = torch.ops.aten.squeeze.dims(mul_1296, [2]);  mul_1296 = None
        sum_176 = torch.ops.aten.sum.dim_IntList(squeeze_181, [1], True);  squeeze_181 = None
        full_default_222 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_222 = torch.ops.aten.where.self(logical_or_12, full_default_222, sum_176);  full_default_222 = sum_176 = None
        mul_1297 = torch.ops.aten.mul.Tensor(slice_559, slice_561);  slice_559 = slice_561 = None
        squeeze_182 = torch.ops.aten.squeeze.dims(mul_1297, [2]);  mul_1297 = None
        sum_177 = torch.ops.aten.sum.dim_IntList(squeeze_182, [1], True);  squeeze_182 = None
        mul_1298 = torch.ops.aten.mul.Tensor(slice_569, slice_571);  slice_569 = slice_571 = None
        squeeze_183 = torch.ops.aten.squeeze.dims(mul_1298, [2]);  mul_1298 = None
        sum_178 = torch.ops.aten.sum.dim_IntList(squeeze_183, [1], True);  squeeze_183 = None
        mul_1299 = torch.ops.aten.mul.Tensor(slice_579, slice_581);  slice_579 = slice_581 = None
        squeeze_184 = torch.ops.aten.squeeze.dims(mul_1299, [2]);  mul_1299 = None
        sum_179 = torch.ops.aten.sum.dim_IntList(squeeze_184, [1], True);  squeeze_184 = None
        mul_1300 = torch.ops.aten.mul.Tensor(slice_589, slice_591);  slice_589 = slice_591 = None
        squeeze_185 = torch.ops.aten.squeeze.dims(mul_1300, [2]);  mul_1300 = None
        sum_180 = torch.ops.aten.sum.dim_IntList(squeeze_185, [1], True);  squeeze_185 = None
        mul_1301 = torch.ops.aten.mul.Tensor(index_6, slice_602);  index_6 = slice_602 = None
        squeeze_186 = torch.ops.aten.squeeze.dims(mul_1301, [2]);  mul_1301 = None
        sum_181 = torch.ops.aten.sum.dim_IntList(squeeze_186, [1], True);  squeeze_186 = None
        full_default_223 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_223 = torch.ops.aten.where.self(logical_or_12, full_default_223, sum_181);  full_default_223 = sum_181 = None
        mul_1302 = torch.ops.aten.mul.Tensor(index_9, slice_615);  index_9 = slice_615 = None
        squeeze_187 = torch.ops.aten.squeeze.dims(mul_1302, [2]);  mul_1302 = None
        sum_182 = torch.ops.aten.sum.dim_IntList(squeeze_187, [1], True);  squeeze_187 = None
        full_default_224 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_224 = torch.ops.aten.where.self(logical_or_12, full_default_224, sum_182);  full_default_224 = sum_182 = None
        mul_1303 = torch.ops.aten.mul.Tensor(index_12, slice_628);  index_12 = slice_628 = None
        squeeze_188 = torch.ops.aten.squeeze.dims(mul_1303, [2]);  mul_1303 = None
        sum_183 = torch.ops.aten.sum.dim_IntList(squeeze_188, [1], True);  squeeze_188 = None
        full_default_225 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_225 = torch.ops.aten.where.self(logical_or_12, full_default_225, sum_183);  full_default_225 = sum_183 = None
        cat_125 = torch.ops.aten.cat.default([sum_167, sum_168, sum_169, sum_170, sum_171, sum_172, where_221, sum_174, sum_175, where_222, sum_177, sum_178, sum_179, sum_180, where_223, where_224, where_225], 1);  sum_167 = sum_168 = sum_169 = sum_170 = sum_171 = sum_172 = where_221 = sum_174 = sum_175 = where_222 = sum_177 = sum_178 = sum_179 = sum_180 = where_223 = where_224 = where_225 = None
        full_default_226 = torch.ops.aten.full.default([batch_size, 50], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        cat_126 = torch.ops.aten.cat.default([arg856_1, cat_125, full_default_226], 1);  full_default_226 = None
        sum_184 = torch.ops.aten.sum.dim_IntList(cat_126, [1])
        full_default_227 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        eq_884 = torch.ops.aten.eq.Scalar(where_7, 1)
        full_default_228 = torch.ops.aten.full.default([batch_size, 1], 1, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_226 = torch.ops.aten.where.self(eq_884, full_default_228, full_default_227);  eq_884 = full_default_228 = full_default_227 = None
        eq_885 = torch.ops.aten.eq.Scalar(where_7, 2)
        full_default_229 = torch.ops.aten.full.default([batch_size, 1], 2, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_227 = torch.ops.aten.where.self(eq_885, full_default_229, where_226);  eq_885 = full_default_229 = where_226 = None
        eq_886 = torch.ops.aten.eq.Scalar(where_7, 3)
        full_default_230 = torch.ops.aten.full.default([batch_size, 1], 3, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_228 = torch.ops.aten.where.self(eq_886, full_default_230, where_227);  eq_886 = full_default_230 = where_227 = None
        eq_887 = torch.ops.aten.eq.Scalar(where_7, 9998)
        full_default_231 = torch.ops.aten.full.default([batch_size, 1], 4, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_229 = torch.ops.aten.where.self(eq_887, full_default_231, where_228);  eq_887 = full_default_231 = where_228 = None
        eq_888 = torch.ops.aten.eq.Scalar(where_7, 9999)
        full_default_232 = torch.ops.aten.full.default([batch_size, 1], 5, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_230 = torch.ops.aten.where.self(eq_888, full_default_232, where_229);  eq_888 = full_default_232 = where_229 = None
        embedding_2 = torch.ops.aten.embedding.default(arg857_1, where_230);  arg857_1 = where_230 = None
        squeeze_189 = torch.ops.aten.squeeze.dim(embedding_2, 1);  embedding_2 = None
        sum_185 = torch.ops.aten.sum.dim_IntList(squeeze_189, [1]);  squeeze_189 = None
        add_1097 = torch.ops.aten.add.Tensor(sum_184, sum_185);  sum_184 = sum_185 = None
        squeeze_190 = torch.ops.aten.squeeze.default(logical_not)
        view_445 = torch.ops.aten.view.default(squeeze_190, [-1, 1]);  squeeze_190 = None
        full_default_233 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_231 = torch.ops.aten.where.self(view_445, full_default_233, arg858_1);  full_default_233 = arg858_1 = None
        full_default_234 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_232 = torch.ops.aten.where.self(view_445, full_default_234, arg859_1);  full_default_234 = arg859_1 = None
        full_default_235 = torch.ops.aten.full.default([batch_size, 48], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_233 = torch.ops.aten.where.self(view_445, full_default_235, arg860_1);  view_445 = full_default_235 = arg860_1 = None
        cat_127 = torch.ops.aten.cat.default([arg861_1, where_231, arg862_1, where_232, arg863_1, where_233, arg864_1], -1);  arg861_1 = where_231 = arg862_1 = where_232 = arg863_1 = where_233 = arg864_1 = None
        squeeze_192 = torch.ops.aten.squeeze.default(logical_not)
        view_447 = torch.ops.aten.view.default(squeeze_192, [-1, 1]);  squeeze_192 = None
        full_default_236 = torch.ops.aten.full.default([batch_size, 80], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_234 = torch.ops.aten.where.self(view_447, full_default_236, arg865_1);  full_default_236 = arg865_1 = None
        full_default_237 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_235 = torch.ops.aten.where.self(view_447, full_default_237, arg866_1);  full_default_237 = arg866_1 = None
        full_default_238 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_236 = torch.ops.aten.where.self(view_447, full_default_238, arg867_1);  full_default_238 = arg867_1 = None
        full_default_239 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_237 = torch.ops.aten.where.self(view_447, full_default_239, arg868_1);  full_default_239 = arg868_1 = None
        full_default_240 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_238 = torch.ops.aten.where.self(view_447, full_default_240, arg869_1);  view_447 = full_default_240 = arg869_1 = None
        cat_128 = torch.ops.aten.cat.default([where_234, arg870_1, where_235, arg871_1, where_236, arg872_1, where_237, arg873_1, where_238], -1);  where_234 = arg870_1 = where_235 = arg871_1 = where_236 = arg872_1 = where_237 = arg873_1 = where_238 = None
        cat_129 = torch.ops.aten.cat.default([cat_127, cat_128], 1);  cat_127 = cat_128 = None
        convert_element_type_75 = torch.ops.prims.convert_element_type.default(cat_129, torch.float32);  cat_129 = None
        clamp_min_6 = torch.ops.aten.clamp_min.default(convert_element_type_75, -20);  convert_element_type_75 = None
        clamp_max_21 = torch.ops.aten.clamp_max.default(clamp_min_6, 20);  clamp_min_6 = None
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(clamp_max_21, torch.float16);  clamp_max_21 = None
        mm = torch.ops.aten.mm.default(convert_element_type_76, arg874_1);  arg874_1 = None
        add_1098 = torch.ops.aten.add.Tensor(mm, arg875_1);  mm = arg875_1 = None
        mul_1308 = torch.ops.aten.mul.Tensor(convert_element_type_76, add_1098);  add_1098 = None
        add_1099 = torch.ops.aten.add.Tensor(mul_1308, convert_element_type_76);  mul_1308 = None
        mm_1 = torch.ops.aten.mm.default(add_1099, arg876_1);  arg876_1 = None
        add_1100 = torch.ops.aten.add.Tensor(mm_1, arg877_1);  mm_1 = arg877_1 = None
        mul_1309 = torch.ops.aten.mul.Tensor(convert_element_type_76, add_1100);  convert_element_type_76 = add_1100 = None
        add_1101 = torch.ops.aten.add.Tensor(mul_1309, add_1099);  mul_1309 = add_1099 = None
        addmm_288 = torch.ops.aten.addmm.default(arg879_1, add_1101, arg878_1);  arg879_1 = add_1101 = arg878_1 = None
        slice_715 = torch.ops.aten.slice.Tensor(arg15_1, 1, 814, 846)
        cat_130 = torch.ops.aten.cat.default([arg880_1, arg881_1], 1);  arg880_1 = arg881_1 = None
        convert_element_type_81 = torch.ops.prims.convert_element_type.default(cat_130, torch.float32);  cat_130 = None
        clamp_min_7 = torch.ops.aten.clamp_min.default(convert_element_type_81, -20);  convert_element_type_81 = None
        clamp_max_22 = torch.ops.aten.clamp_max.default(clamp_min_7, 20);  clamp_min_7 = None
        convert_element_type_82 = torch.ops.prims.convert_element_type.default(clamp_max_22, torch.float16);  clamp_max_22 = None
        mm_2 = torch.ops.aten.mm.default(convert_element_type_82, arg882_1);  arg882_1 = None
        add_1102 = torch.ops.aten.add.Tensor(mm_2, arg883_1);  mm_2 = arg883_1 = None
        mul_1310 = torch.ops.aten.mul.Tensor(convert_element_type_82, add_1102);  add_1102 = None
        add_1103 = torch.ops.aten.add.Tensor(mul_1310, convert_element_type_82);  mul_1310 = None
        mm_3 = torch.ops.aten.mm.default(add_1103, arg884_1);  arg884_1 = None
        add_1104 = torch.ops.aten.add.Tensor(mm_3, arg885_1);  mm_3 = arg885_1 = None
        mul_1311 = torch.ops.aten.mul.Tensor(convert_element_type_82, add_1104);  convert_element_type_82 = add_1104 = None
        add_1105 = torch.ops.aten.add.Tensor(mul_1311, add_1103);  mul_1311 = add_1103 = None
        addmm_289 = torch.ops.aten.addmm.default(arg887_1, add_1105, arg886_1);  arg887_1 = add_1105 = arg886_1 = None
        repeat_37 = torch.ops.aten.repeat.default(logical_not_1, [1, 128]);  logical_not_1 = None
        where_239 = torch.ops.aten.where.self(repeat_37, addmm_288, addmm_289);  repeat_37 = addmm_288 = addmm_289 = None
        cat_131 = torch.ops.aten.cat.default([arg888_1, arg889_1], 1);  arg888_1 = arg889_1 = None
        addmm_290 = torch.ops.aten.addmm.default(arg891_1, cat_131, arg890_1);  arg891_1 = cat_131 = arg890_1 = None
        relu_100 = torch.ops.aten.relu.default(addmm_290);  addmm_290 = None
        addmm_291 = torch.ops.aten.addmm.default(arg893_1, relu_100, arg892_1);  arg893_1 = relu_100 = arg892_1 = None
        convert_element_type_87 = torch.ops.prims.convert_element_type.default(addmm_291, torch.float32)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_87, 2);  convert_element_type_87 = None
        sum_186 = torch.ops.aten.sum.dim_IntList(pow_6, [1], True);  pow_6 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(sum_186, 0.5);  sum_186 = None
        convert_element_type_88 = torch.ops.prims.convert_element_type.default(pow_7, torch.float16);  pow_7 = None
        clamp_min_8 = torch.ops.aten.clamp_min.default(convert_element_type_88, 1e-12);  convert_element_type_88 = None
        expand_44 = torch.ops.aten.expand.default(clamp_min_8, [batch_size, 64]);  clamp_min_8 = None
        div = torch.ops.aten.div.Tensor(addmm_291, expand_44);  addmm_291 = expand_44 = None
        squeeze_199 = torch.ops.aten.squeeze.default(logical_not)
        view_454 = torch.ops.aten.view.default(squeeze_199, [-1, 1]);  squeeze_199 = None
        full_default_241 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_240 = torch.ops.aten.where.self(view_454, full_default_241, arg894_1);  view_454 = full_default_241 = arg894_1 = None
        cat_132 = torch.ops.aten.cat.default([arg895_1, where_240, arg896_1], -1);  arg895_1 = where_240 = arg896_1 = None
        squeeze_201 = torch.ops.aten.squeeze.default(logical_not)
        view_456 = torch.ops.aten.view.default(squeeze_201, [-1, 1]);  squeeze_201 = None
        full_default_242 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_241 = torch.ops.aten.where.self(view_456, full_default_242, arg897_1);  full_default_242 = arg897_1 = None
        full_default_243 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_242 = torch.ops.aten.where.self(view_456, full_default_243, arg898_1);  full_default_243 = arg898_1 = None
        full_default_244 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_243 = torch.ops.aten.where.self(view_456, full_default_244, arg899_1);  full_default_244 = arg899_1 = None
        full_default_245 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_244 = torch.ops.aten.where.self(view_456, full_default_245, arg900_1);  full_default_245 = arg900_1 = None
        full_default_246 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_245 = torch.ops.aten.where.self(view_456, full_default_246, arg901_1);  full_default_246 = arg901_1 = None
        full_default_247 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_246 = torch.ops.aten.where.self(view_456, full_default_247, arg902_1);  full_default_247 = arg902_1 = None
        full_default_248 = torch.ops.aten.full.default([batch_size, 4], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_247 = torch.ops.aten.where.self(view_456, full_default_248, arg903_1);  full_default_248 = arg903_1 = None
        full_default_249 = torch.ops.aten.full.default([batch_size, 20], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_248 = torch.ops.aten.where.self(view_456, full_default_249, arg904_1);  view_456 = full_default_249 = arg904_1 = None
        cat_133 = torch.ops.aten.cat.default([arg905_1, where_241, arg906_1, where_242, arg907_1, where_243, arg908_1, where_244, arg909_1, where_245, arg910_1, where_246, arg911_1, where_247, arg912_1, where_248], -1);  arg905_1 = where_241 = arg906_1 = where_242 = arg907_1 = where_243 = arg908_1 = where_244 = arg909_1 = where_245 = arg910_1 = where_246 = arg911_1 = where_247 = arg912_1 = where_248 = None
        cat_134 = torch.ops.aten.cat.default([cat_132, cat_133], 1);  cat_132 = cat_133 = None
        addmm_292 = torch.ops.aten.addmm.default(arg914_1, cat_134, arg913_1);  arg914_1 = cat_134 = arg913_1 = None
        relu_101 = torch.ops.aten.relu.default(addmm_292);  addmm_292 = None
        addmm_293 = torch.ops.aten.addmm.default(arg916_1, relu_101, arg915_1);  arg916_1 = relu_101 = arg915_1 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(addmm_293, torch.float32)
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_89, 2);  convert_element_type_89 = None
        sum_187 = torch.ops.aten.sum.dim_IntList(pow_8, [1], True);  pow_8 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(sum_187, 0.5);  sum_187 = None
        convert_element_type_90 = torch.ops.prims.convert_element_type.default(pow_9, torch.float16);  pow_9 = None
        clamp_min_9 = torch.ops.aten.clamp_min.default(convert_element_type_90, 1e-12);  convert_element_type_90 = None
        expand_45 = torch.ops.aten.expand.default(clamp_min_9, [batch_size, 64]);  clamp_min_9 = None
        div_1 = torch.ops.aten.div.Tensor(addmm_293, expand_45);  addmm_293 = expand_45 = None
        add_1107 = torch.ops.aten.add.Tensor(div, div_1)
        mul_1314 = torch.ops.aten.mul.Tensor(div, div_1)
        sub_386 = torch.ops.aten.sub.Tensor(div, div_1)
        cat_135 = torch.ops.aten.cat.default([div, div_1, add_1107, mul_1314, sub_386], 1);  add_1107 = mul_1314 = sub_386 = None
        addmm_294 = torch.ops.aten.addmm.default(arg918_1, cat_135, arg917_1);  arg918_1 = cat_135 = arg917_1 = None
        view_459 = torch.ops.aten.view.default(arg919_1, [1, batch_size, 16]);  arg919_1 = None
        sum_189 = torch.ops.aten.sum.dim_IntList(view_459, [0]);  view_459 = None
        view_461 = torch.ops.aten.view.default(arg920_1, [1, batch_size, 16]);  arg920_1 = None
        sum_190 = torch.ops.aten.sum.dim_IntList(view_461, [0]);  view_461 = None
        mul_1315 = torch.ops.aten.mul.Tensor(sum_189, sum_190)
        add_1108 = torch.ops.aten.add.Tensor(sum_189, sum_190)
        cat_136 = torch.ops.aten.cat.default([sum_189, sum_190, mul_1315, add_1108], -1);  mul_1315 = add_1108 = None
        addmm_295 = torch.ops.aten.addmm.default(arg922_1, cat_136, arg921_1);  arg922_1 = cat_136 = arg921_1 = None
        relu_102 = torch.ops.aten.relu.default(addmm_295);  addmm_295 = None
        addmm_296 = torch.ops.aten.addmm.default(arg924_1, relu_102, arg923_1);  arg924_1 = relu_102 = arg923_1 = None
        relu_103 = torch.ops.aten.relu.default(addmm_296);  addmm_296 = None
        addmm_297 = torch.ops.aten.addmm.default(arg926_1, relu_103, arg925_1);  arg926_1 = relu_103 = arg925_1 = None
        squeeze_205 = torch.ops.aten.squeeze.default(logical_not)
        view_462 = torch.ops.aten.view.default(squeeze_205, [-1, 1]);  squeeze_205 = None
        slice_716 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43485, 43517)
        slice_717 = torch.ops.aten.slice.Tensor(arg15_1, 1, 846, 878)
        full_default_250 = torch.ops.aten.full.default([batch_size, 32], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_249 = torch.ops.aten.where.self(view_462, full_default_250, arg927_1);  view_462 = full_default_250 = arg927_1 = None
        cat_137 = torch.ops.aten.cat.default([where_249, arg928_1], -1);  where_249 = arg928_1 = None
        squeeze_208 = torch.ops.aten.squeeze.default(logical_not)
        view_465 = torch.ops.aten.view.default(squeeze_208, [-1, 1]);  squeeze_208 = None
        full_default_251 = torch.ops.aten.full.default([batch_size, 160], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_250 = torch.ops.aten.where.self(view_465, full_default_251, arg930_1);  view_465 = full_default_251 = arg930_1 = None
        cat_138 = torch.ops.aten.cat.default([cat_137, arg929_1, where_250], 1);  cat_137 = arg929_1 = where_250 = None
        addmm_298 = torch.ops.aten.addmm.default(arg932_1, cat_138, arg931_1);  arg932_1 = arg931_1 = None
        relu_104 = torch.ops.aten.relu.default(addmm_298);  addmm_298 = None
        addmm_299 = torch.ops.aten.addmm.default(arg934_1, relu_104, arg933_1);  arg934_1 = relu_104 = arg933_1 = None
        relu_105 = torch.ops.aten.relu.default(addmm_299);  addmm_299 = None
        ne_9 = torch.ops.aten.ne.Tensor(arg935_1, arg935_1)
        abs_9 = torch.ops.aten.abs.default(arg935_1)
        eq_889 = torch.ops.aten.eq.Scalar(abs_9, inf);  abs_9 = None
        bitwise_or_8 = torch.ops.aten.bitwise_or.Tensor(ne_9, eq_889);  ne_9 = eq_889 = None
        full_default_252 = torch.ops.aten.full.default([batch_size, 256], 1.0013580322265625e-05, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_251 = torch.ops.aten.where.self(bitwise_or_8, full_default_252, arg935_1);  bitwise_or_8 = full_default_252 = arg935_1 = None
        mm_4 = torch.ops.aten.mm.default(where_251, arg936_1);  arg936_1 = None
        mm_5 = torch.ops.aten.mm.default(mm_4, arg937_1);  mm_4 = arg937_1 = None
        add_1109 = torch.ops.aten.add.Tensor(where_251, mm_5);  where_251 = mm_5 = None
        mul_1317 = torch.ops.aten.mul.Tensor(add_1109, relu_105);  add_1109 = relu_105 = None
        view_468 = torch.ops.aten.view.default(arg938_1, [1, batch_size, 16]);  arg938_1 = None
        sum_191 = torch.ops.aten.sum.dim_IntList(view_468, [0]);  view_468 = None
        repeat_38 = torch.ops.aten.repeat.default(logical_not_2, [1, 16])
        full_default_253 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_252 = torch.ops.aten.where.self(repeat_38, full_default_253, sum_191);  repeat_38 = full_default_253 = sum_191 = None
        repeat_39 = torch.ops.aten.repeat.default(logical_not_2, [1, 16])
        full_default_254 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_253 = torch.ops.aten.where.self(repeat_39, arg939_1, full_default_254);  repeat_39 = arg939_1 = full_default_254 = None
        repeat_40 = torch.ops.aten.repeat.default(logical_not_2, [1, 16])
        full_default_255 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_254 = torch.ops.aten.where.self(repeat_40, full_default_255, arg940_1);  repeat_40 = full_default_255 = arg940_1 = None
        cat_139 = torch.ops.aten.cat.default([where_252, where_253, where_254], 1);  where_252 = where_253 = where_254 = None
        cat_140 = torch.ops.aten.cat.default([cat_139, cat_53], 1)
        addmm_300 = torch.ops.aten.addmm.default(arg942_1, cat_140, arg941_1);  arg942_1 = arg941_1 = None
        relu_106 = torch.ops.aten.relu.default(addmm_300);  addmm_300 = None
        addmm_301 = torch.ops.aten.addmm.default(arg944_1, relu_106, arg943_1);  arg944_1 = relu_106 = arg943_1 = None
        clone_44 = torch.ops.aten.clone.default(arg945_1);  arg945_1 = None
        split_with_sizes = torch.ops.aten.split_with_sizes.default(clone_44, [1024, 1024, 1024, 1146], 1)
        getitem_99 = split_with_sizes[0]
        getitem_100 = split_with_sizes[1]
        getitem_101 = split_with_sizes[2]
        getitem_102 = split_with_sizes[3];  split_with_sizes = None
        cat_141 = torch.ops.aten.cat.default([clone_4, cat_52], 1)
        addmm_302 = torch.ops.aten.addmm.default(arg947_1, cat_141, arg946_1);  arg947_1 = cat_141 = arg946_1 = None
        cat_142 = torch.ops.aten.cat.default([addmm_297, mul_1286], 1)
        addmm_303 = torch.ops.aten.addmm.default(arg949_1, cat_142, arg948_1);  arg949_1 = cat_142 = arg948_1 = None
        cat_143 = torch.ops.aten.cat.default([cat_49, clone_4], 1)
        cat_144 = torch.ops.aten.cat.default([cat_51, cat_140], 1)
        cat_145 = torch.ops.aten.cat.default([cat_44, where_239], 1)
        cat_146 = torch.ops.aten.cat.default([cat_85, cat_88], 1)
        cat_147 = torch.ops.aten.cat.default([where_196, cat_97], 1)
        cat_148 = torch.ops.aten.cat.default([cat_116, cat_103], 1)
        clone_45 = torch.ops.aten.clone.default(where_204)
        clone_46 = torch.ops.aten.clone.default(cat_94)
        cat_149 = torch.ops.aten.cat.default([cat_46, cat_47], 1)
        clone_47 = torch.ops.aten.clone.default(where_200)
        clone_48 = torch.ops.aten.clone.default(where_204)
        clone_49 = torch.ops.aten.clone.default(where_196)
        native_layer_norm_33 = torch.ops.aten.native_layer_norm.default(where_201, [320], arg950_1, arg951_1, 1e-06);  arg950_1 = arg951_1 = None
        getitem_103 = native_layer_norm_33[0];  native_layer_norm_33 = None
        addmm_304 = torch.ops.aten.addmm.default(arg953_1, getitem_103, arg952_1);  arg953_1 = getitem_103 = arg952_1 = None
        native_layer_norm_34 = torch.ops.aten.native_layer_norm.default(where_206, [512], arg954_1, arg955_1, 1e-06);  arg954_1 = arg955_1 = None
        getitem_106 = native_layer_norm_34[0];  native_layer_norm_34 = None
        addmm_305 = torch.ops.aten.addmm.default(arg957_1, getitem_106, arg956_1);  arg957_1 = getitem_106 = arg956_1 = None
        native_layer_norm_35 = torch.ops.aten.native_layer_norm.default(mul_14, [256], arg958_1, arg959_1, 1e-06);  arg958_1 = arg959_1 = None
        getitem_109 = native_layer_norm_35[0];  native_layer_norm_35 = None
        addmm_306 = torch.ops.aten.addmm.default(arg961_1, getitem_109, arg960_1);  arg961_1 = getitem_109 = arg960_1 = None
        native_layer_norm_36 = torch.ops.aten.native_layer_norm.default(mul_1317, [256], arg962_1, arg963_1, 1e-06);  arg962_1 = arg963_1 = None
        getitem_112 = native_layer_norm_36[0];  native_layer_norm_36 = None
        mm_6 = torch.ops.aten.mm.default(getitem_112, arg964_1);  getitem_112 = arg964_1 = None
        native_layer_norm_37 = torch.ops.aten.native_layer_norm.default(mul_16, [64], arg965_1, arg966_1, 1e-06);  arg965_1 = arg966_1 = None
        getitem_115 = native_layer_norm_37[0];  native_layer_norm_37 = None
        addmm_307 = torch.ops.aten.addmm.default(arg968_1, getitem_115, arg967_1);  arg968_1 = getitem_115 = arg967_1 = None
        native_layer_norm_38 = torch.ops.aten.native_layer_norm.default(cat_45, [1536], arg969_1, arg970_1, 1e-06);  arg969_1 = arg970_1 = None
        getitem_118 = native_layer_norm_38[0];  native_layer_norm_38 = None
        addmm_308 = torch.ops.aten.addmm.default(arg972_1, getitem_118, arg971_1);  arg972_1 = getitem_118 = arg971_1 = None
        native_layer_norm_39 = torch.ops.aten.native_layer_norm.default(cat_49, [2512], arg973_1, arg974_1, 1e-06);  arg973_1 = arg974_1 = None
        getitem_121 = native_layer_norm_39[0];  native_layer_norm_39 = None
        addmm_309 = torch.ops.aten.addmm.default(arg976_1, getitem_121, arg975_1);  arg976_1 = getitem_121 = arg975_1 = None
        native_layer_norm_40 = torch.ops.aten.native_layer_norm.default(addmm_302, [256], arg977_1, arg978_1, 1e-06);  arg977_1 = arg978_1 = None
        getitem_124 = native_layer_norm_40[0];  native_layer_norm_40 = None
        addmm_310 = torch.ops.aten.addmm.default(arg980_1, getitem_124, arg979_1);  arg980_1 = getitem_124 = arg979_1 = None
        native_layer_norm_41 = torch.ops.aten.native_layer_norm.default(cat_51, [1284], arg981_1, arg982_1, 1e-06);  arg981_1 = arg982_1 = None
        getitem_127 = native_layer_norm_41[0];  native_layer_norm_41 = None
        addmm_311 = torch.ops.aten.addmm.default(arg984_1, getitem_127, arg983_1);  arg984_1 = getitem_127 = arg983_1 = None
        native_layer_norm_42 = torch.ops.aten.native_layer_norm.default(cat_126, [1028], arg985_1, arg986_1, 1e-06);  arg985_1 = arg986_1 = None
        getitem_130 = native_layer_norm_42[0];  native_layer_norm_42 = None
        addmm_312 = torch.ops.aten.addmm.default(arg988_1, getitem_130, arg987_1);  arg988_1 = getitem_130 = arg987_1 = None
        native_layer_norm_43 = torch.ops.aten.native_layer_norm.default(cat_89, [896], arg989_1, arg990_1, 1e-06);  arg989_1 = arg990_1 = None
        getitem_133 = native_layer_norm_43[0];  native_layer_norm_43 = None
        addmm_313 = torch.ops.aten.addmm.default(arg992_1, getitem_133, arg991_1);  arg992_1 = getitem_133 = arg991_1 = None
        native_layer_norm_44 = torch.ops.aten.native_layer_norm.default(cat_95, [320], arg993_1, arg994_1, 1e-06);  arg993_1 = arg994_1 = None
        getitem_136 = native_layer_norm_44[0];  native_layer_norm_44 = None
        addmm_314 = torch.ops.aten.addmm.default(arg996_1, getitem_136, arg995_1);  arg996_1 = getitem_136 = arg995_1 = None
        native_layer_norm_45 = torch.ops.aten.native_layer_norm.default(where_197, [640], arg997_1, arg998_1, 1e-06);  arg997_1 = arg998_1 = None
        getitem_139 = native_layer_norm_45[0];  native_layer_norm_45 = None
        addmm_315 = torch.ops.aten.addmm.default(arg1000_1, getitem_139, arg999_1);  arg1000_1 = getitem_139 = arg999_1 = None
        native_layer_norm_46 = torch.ops.aten.native_layer_norm.default(where_204, [512], arg1001_1, arg1002_1, 1e-06);  arg1001_1 = arg1002_1 = None
        getitem_142 = native_layer_norm_46[0];  native_layer_norm_46 = None
        addmm_316 = torch.ops.aten.addmm.default(arg1004_1, getitem_142, arg1003_1);  arg1004_1 = getitem_142 = arg1003_1 = None
        native_layer_norm_47 = torch.ops.aten.native_layer_norm.default(where_193, [320], arg1005_1, arg1006_1, 1e-06);  arg1005_1 = arg1006_1 = None
        getitem_145 = native_layer_norm_47[0];  native_layer_norm_47 = None
        addmm_317 = torch.ops.aten.addmm.default(arg1008_1, getitem_145, arg1007_1);  arg1008_1 = getitem_145 = arg1007_1 = None
        native_layer_norm_48 = torch.ops.aten.native_layer_norm.default(where_208, [512], arg1009_1, arg1010_1, 1e-06);  arg1009_1 = arg1010_1 = None
        getitem_148 = native_layer_norm_48[0];  native_layer_norm_48 = None
        addmm_318 = torch.ops.aten.addmm.default(arg1012_1, getitem_148, arg1011_1);  arg1012_1 = getitem_148 = arg1011_1 = None
        native_layer_norm_49 = torch.ops.aten.native_layer_norm.default(where_156, [32], arg1013_1, arg1014_1, 1e-06);  arg1013_1 = arg1014_1 = None
        getitem_151 = native_layer_norm_49[0];  native_layer_norm_49 = None
        addmm_319 = torch.ops.aten.addmm.default(arg1016_1, getitem_151, arg1015_1);  arg1016_1 = getitem_151 = arg1015_1 = None
        native_layer_norm_50 = torch.ops.aten.native_layer_norm.default(sum_41, [32], arg1017_1, arg1018_1, 1e-06);  arg1017_1 = arg1018_1 = None
        getitem_154 = native_layer_norm_50[0];  native_layer_norm_50 = None
        addmm_320 = torch.ops.aten.addmm.default(arg1020_1, getitem_154, arg1019_1);  arg1020_1 = getitem_154 = arg1019_1 = None
        native_layer_norm_51 = torch.ops.aten.native_layer_norm.default(where_199, [320], arg1021_1, arg1022_1, 1e-06);  arg1021_1 = arg1022_1 = None
        getitem_157 = native_layer_norm_51[0];  native_layer_norm_51 = None
        addmm_321 = torch.ops.aten.addmm.default(arg1024_1, getitem_157, arg1023_1);  arg1024_1 = getitem_157 = arg1023_1 = None
        native_layer_norm_52 = torch.ops.aten.native_layer_norm.default(where_210, [512], arg1025_1, arg1026_1, 1e-06);  arg1025_1 = arg1026_1 = None
        getitem_160 = native_layer_norm_52[0];  native_layer_norm_52 = None
        addmm_322 = torch.ops.aten.addmm.default(arg1028_1, getitem_160, arg1027_1);  arg1028_1 = getitem_160 = arg1027_1 = None
        native_layer_norm_53 = torch.ops.aten.native_layer_norm.default(where_195, [320], arg1029_1, arg1030_1, 1e-06);  arg1029_1 = arg1030_1 = None
        getitem_163 = native_layer_norm_53[0];  native_layer_norm_53 = None
        addmm_323 = torch.ops.aten.addmm.default(arg1032_1, getitem_163, arg1031_1);  arg1032_1 = getitem_163 = arg1031_1 = None
        native_layer_norm_54 = torch.ops.aten.native_layer_norm.default(where_212, [512], arg1033_1, arg1034_1, 1e-06);  arg1033_1 = arg1034_1 = None
        getitem_166 = native_layer_norm_54[0];  native_layer_norm_54 = None
        addmm_324 = torch.ops.aten.addmm.default(arg1036_1, getitem_166, arg1035_1);  arg1036_1 = getitem_166 = arg1035_1 = None
        native_layer_norm_55 = torch.ops.aten.native_layer_norm.default(cat_86, [896], arg1037_1, arg1038_1, 1e-06);  arg1037_1 = arg1038_1 = None
        getitem_169 = native_layer_norm_55[0];  native_layer_norm_55 = None
        addmm_325 = torch.ops.aten.addmm.default(arg1040_1, getitem_169, arg1039_1);  arg1040_1 = getitem_169 = arg1039_1 = None
        native_layer_norm_56 = torch.ops.aten.native_layer_norm.default(cat_98, [320], arg1041_1, arg1042_1, 1e-06);  arg1041_1 = arg1042_1 = None
        getitem_172 = native_layer_norm_56[0];  native_layer_norm_56 = None
        addmm_326 = torch.ops.aten.addmm.default(arg1044_1, getitem_172, arg1043_1);  arg1044_1 = getitem_172 = arg1043_1 = None
        native_layer_norm_57 = torch.ops.aten.native_layer_norm.default(cat_104, [512], arg1045_1, arg1046_1, 1e-06);  arg1045_1 = arg1046_1 = None
        getitem_175 = native_layer_norm_57[0];  native_layer_norm_57 = None
        addmm_327 = torch.ops.aten.addmm.default(arg1048_1, getitem_175, arg1047_1);  arg1048_1 = getitem_175 = arg1047_1 = None
        native_layer_norm_58 = torch.ops.aten.native_layer_norm.default(cat_101, [896], arg1049_1, arg1050_1, 1e-06);  arg1049_1 = arg1050_1 = None
        getitem_178 = native_layer_norm_58[0];  native_layer_norm_58 = None
        addmm_328 = torch.ops.aten.addmm.default(arg1052_1, getitem_178, arg1051_1);  arg1052_1 = getitem_178 = arg1051_1 = None
        native_layer_norm_59 = torch.ops.aten.native_layer_norm.default(cat_140, [490], arg1053_1, arg1054_1, 1e-06);  arg1053_1 = arg1054_1 = None
        getitem_181 = native_layer_norm_59[0];  native_layer_norm_59 = None
        addmm_329 = torch.ops.aten.addmm.default(arg1056_1, getitem_181, arg1055_1);  arg1056_1 = getitem_181 = arg1055_1 = None
        native_layer_norm_60 = torch.ops.aten.native_layer_norm.default(addmm_303, [256], arg1057_1, arg1058_1, 1e-06);  arg1057_1 = arg1058_1 = None
        getitem_184 = native_layer_norm_60[0];  native_layer_norm_60 = None
        addmm_330 = torch.ops.aten.addmm.default(arg1060_1, getitem_184, arg1059_1);  arg1060_1 = getitem_184 = arg1059_1 = None
        native_layer_norm_61 = torch.ops.aten.native_layer_norm.default(where_239, [128], arg1061_1, arg1062_1, 1e-06);  arg1061_1 = arg1062_1 = None
        getitem_187 = native_layer_norm_61[0];  native_layer_norm_61 = None
        addmm_331 = torch.ops.aten.addmm.default(arg1064_1, getitem_187, arg1063_1);  arg1064_1 = getitem_187 = arg1063_1 = None
        native_layer_norm_62 = torch.ops.aten.native_layer_norm.default(cat_143, [5036], arg1065_1, arg1066_1, 1e-06);  arg1065_1 = arg1066_1 = None
        getitem_190 = native_layer_norm_62[0];  native_layer_norm_62 = None
        addmm_332 = torch.ops.aten.addmm.default(arg1068_1, getitem_190, arg1067_1);  arg1068_1 = getitem_190 = arg1067_1 = None
        native_layer_norm_63 = torch.ops.aten.native_layer_norm.default(cat_144, [1774], arg1069_1, arg1070_1, 1e-06);  arg1069_1 = arg1070_1 = None
        getitem_193 = native_layer_norm_63[0];  native_layer_norm_63 = None
        addmm_333 = torch.ops.aten.addmm.default(arg1072_1, getitem_193, arg1071_1);  arg1072_1 = getitem_193 = arg1071_1 = None
        native_layer_norm_64 = torch.ops.aten.native_layer_norm.default(cat_145, [352], arg1073_1, arg1074_1, 1e-06);  arg1073_1 = arg1074_1 = None
        getitem_196 = native_layer_norm_64[0];  native_layer_norm_64 = None
        addmm_334 = torch.ops.aten.addmm.default(arg1076_1, getitem_196, arg1075_1);  arg1076_1 = getitem_196 = arg1075_1 = None
        native_layer_norm_65 = torch.ops.aten.native_layer_norm.default(cat_146, [128], arg1077_1, arg1078_1, 1e-06);  arg1077_1 = arg1078_1 = None
        getitem_199 = native_layer_norm_65[0];  native_layer_norm_65 = None
        addmm_335 = torch.ops.aten.addmm.default(arg1080_1, getitem_199, arg1079_1);  arg1080_1 = getitem_199 = arg1079_1 = None
        native_layer_norm_66 = torch.ops.aten.native_layer_norm.default(cat_147, [192], arg1081_1, arg1082_1, 1e-06);  arg1081_1 = arg1082_1 = None
        getitem_202 = native_layer_norm_66[0];  native_layer_norm_66 = None
        addmm_336 = torch.ops.aten.addmm.default(arg1084_1, getitem_202, arg1083_1);  arg1084_1 = getitem_202 = arg1083_1 = None
        native_layer_norm_67 = torch.ops.aten.native_layer_norm.default(cat_148, [128], arg1085_1, arg1086_1, 1e-06);  arg1085_1 = arg1086_1 = None
        getitem_205 = native_layer_norm_67[0];  native_layer_norm_67 = None
        addmm_337 = torch.ops.aten.addmm.default(arg1088_1, getitem_205, arg1087_1);  arg1088_1 = getitem_205 = arg1087_1 = None
        native_layer_norm_68 = torch.ops.aten.native_layer_norm.default(clone_45, [512], arg1089_1, arg1090_1, 1e-06);  arg1089_1 = arg1090_1 = None
        getitem_208 = native_layer_norm_68[0];  native_layer_norm_68 = None
        addmm_338 = torch.ops.aten.addmm.default(arg1092_1, getitem_208, arg1091_1);  arg1092_1 = getitem_208 = arg1091_1 = None
        native_layer_norm_69 = torch.ops.aten.native_layer_norm.default(clone_46, [64], arg1093_1, arg1094_1, 1e-06);  arg1093_1 = arg1094_1 = None
        getitem_211 = native_layer_norm_69[0];  native_layer_norm_69 = None
        addmm_339 = torch.ops.aten.addmm.default(arg1096_1, getitem_211, arg1095_1);  arg1096_1 = getitem_211 = arg1095_1 = None
        native_layer_norm_70 = torch.ops.aten.native_layer_norm.default(cat_149, [1352], arg1097_1, arg1098_1, 1e-06);  arg1097_1 = arg1098_1 = None
        getitem_214 = native_layer_norm_70[0];  native_layer_norm_70 = None
        addmm_340 = torch.ops.aten.addmm.default(arg1100_1, getitem_214, arg1099_1);  arg1100_1 = getitem_214 = arg1099_1 = None
        native_layer_norm_71 = torch.ops.aten.native_layer_norm.default(clone_47, [64], arg1101_1, arg1102_1, 1e-06);  arg1101_1 = arg1102_1 = None
        getitem_217 = native_layer_norm_71[0];  native_layer_norm_71 = None
        addmm_341 = torch.ops.aten.addmm.default(arg1104_1, getitem_217, arg1103_1);  arg1104_1 = getitem_217 = arg1103_1 = None
        native_layer_norm_72 = torch.ops.aten.native_layer_norm.default(clone_48, [512], arg1105_1, arg1106_1, 1e-06);  arg1105_1 = arg1106_1 = None
        getitem_220 = native_layer_norm_72[0];  native_layer_norm_72 = None
        addmm_342 = torch.ops.aten.addmm.default(arg1108_1, getitem_220, arg1107_1);  arg1108_1 = getitem_220 = arg1107_1 = None
        native_layer_norm_73 = torch.ops.aten.native_layer_norm.default(clone_49, [128], arg1109_1, arg1110_1, 1e-06);  arg1109_1 = arg1110_1 = None
        getitem_223 = native_layer_norm_73[0];  native_layer_norm_73 = None
        addmm_343 = torch.ops.aten.addmm.default(arg1112_1, getitem_223, arg1111_1);  arg1112_1 = getitem_223 = arg1111_1 = None
        native_layer_norm_74 = torch.ops.aten.native_layer_norm.default(where_12, [512], arg1113_1, arg1114_1, 1e-06);  arg1113_1 = arg1114_1 = None
        getitem_226 = native_layer_norm_74[0];  native_layer_norm_74 = None
        addmm_344 = torch.ops.aten.addmm.default(arg1116_1, getitem_226, arg1115_1);  arg1116_1 = getitem_226 = arg1115_1 = None
        native_layer_norm_75 = torch.ops.aten.native_layer_norm.default(where_13, [32], arg1117_1, arg1118_1, 1e-06);  arg1117_1 = arg1118_1 = None
        getitem_229 = native_layer_norm_75[0];  native_layer_norm_75 = None
        addmm_345 = torch.ops.aten.addmm.default(arg1120_1, getitem_229, arg1119_1);  arg1120_1 = getitem_229 = arg1119_1 = None
        native_layer_norm_76 = torch.ops.aten.native_layer_norm.default(where_14, [32], arg1121_1, arg1122_1, 1e-06);  arg1121_1 = arg1122_1 = None
        getitem_232 = native_layer_norm_76[0];  native_layer_norm_76 = None
        addmm_346 = torch.ops.aten.addmm.default(arg1124_1, getitem_232, arg1123_1);  arg1124_1 = getitem_232 = arg1123_1 = None
        native_layer_norm_77 = torch.ops.aten.native_layer_norm.default(where_15, [32], arg1125_1, arg1126_1, 1e-06);  arg1125_1 = arg1126_1 = None
        getitem_235 = native_layer_norm_77[0];  native_layer_norm_77 = None
        addmm_347 = torch.ops.aten.addmm.default(arg1128_1, getitem_235, arg1127_1);  arg1128_1 = getitem_235 = arg1127_1 = None
        native_layer_norm_78 = torch.ops.aten.native_layer_norm.default(cat_1, [256], arg1129_1, arg1130_1, 1e-06);  arg1129_1 = arg1130_1 = None
        getitem_238 = native_layer_norm_78[0];  native_layer_norm_78 = None
        addmm_348 = torch.ops.aten.addmm.default(arg1132_1, getitem_238, arg1131_1);  arg1132_1 = getitem_238 = arg1131_1 = None
        native_layer_norm_79 = torch.ops.aten.native_layer_norm.default(sum_1, [256], arg1133_1, arg1134_1, 1e-06);  arg1133_1 = arg1134_1 = None
        getitem_241 = native_layer_norm_79[0];  native_layer_norm_79 = None
        addmm_349 = torch.ops.aten.addmm.default(arg1136_1, getitem_241, arg1135_1);  arg1136_1 = getitem_241 = arg1135_1 = None
        native_layer_norm_80 = torch.ops.aten.native_layer_norm.default(addmm_7, [64], arg1137_1, arg1138_1, 1e-06);  arg1137_1 = arg1138_1 = None
        getitem_244 = native_layer_norm_80[0];  native_layer_norm_80 = None
        addmm_350 = torch.ops.aten.addmm.default(arg1140_1, getitem_244, arg1139_1);  arg1140_1 = getitem_244 = arg1139_1 = None
        native_layer_norm_81 = torch.ops.aten.native_layer_norm.default(cat_45, [1536], arg1141_1, arg1142_1, 1e-06);  arg1141_1 = arg1142_1 = None
        getitem_247 = native_layer_norm_81[0];  native_layer_norm_81 = None
        addmm_351 = torch.ops.aten.addmm.default(arg1144_1, getitem_247, arg1143_1);  arg1144_1 = getitem_247 = arg1143_1 = None
        native_layer_norm_82 = torch.ops.aten.native_layer_norm.default(cat_47, [1280], arg1145_1, arg1146_1, 1e-06);  arg1145_1 = arg1146_1 = None
        getitem_250 = native_layer_norm_82[0];  native_layer_norm_82 = None
        addmm_352 = torch.ops.aten.addmm.default(arg1148_1, getitem_250, arg1147_1);  arg1148_1 = getitem_250 = arg1147_1 = None
        native_layer_norm_83 = torch.ops.aten.native_layer_norm.default(clone_6, [576], arg1149_1, arg1150_1, 1e-06);  arg1149_1 = arg1150_1 = None
        getitem_253 = native_layer_norm_83[0];  native_layer_norm_83 = None
        addmm_353 = torch.ops.aten.addmm.default(arg1152_1, getitem_253, arg1151_1);  arg1152_1 = getitem_253 = arg1151_1 = None
        native_layer_norm_84 = torch.ops.aten.native_layer_norm.default(cat_55, [4544], arg1153_1, arg1154_1, 1e-06);  arg1153_1 = arg1154_1 = None
        getitem_256 = native_layer_norm_84[0];  native_layer_norm_84 = None
        addmm_354 = torch.ops.aten.addmm.default(arg1156_1, getitem_256, arg1155_1);  arg1156_1 = getitem_256 = arg1155_1 = None
        native_layer_norm_85 = torch.ops.aten.native_layer_norm.default(cat_58, [976], arg1157_1, arg1158_1, 1e-06);  arg1157_1 = arg1158_1 = None
        getitem_259 = native_layer_norm_85[0];  native_layer_norm_85 = None
        addmm_355 = torch.ops.aten.addmm.default(arg1160_1, getitem_259, arg1159_1);  arg1160_1 = getitem_259 = arg1159_1 = None
        native_layer_norm_86 = torch.ops.aten.native_layer_norm.default(where_184, [32], arg1161_1, arg1162_1, 1e-06);  arg1161_1 = arg1162_1 = None
        getitem_262 = native_layer_norm_86[0];  native_layer_norm_86 = None
        addmm_356 = torch.ops.aten.addmm.default(arg1164_1, getitem_262, arg1163_1);  arg1164_1 = getitem_262 = arg1163_1 = None
        native_layer_norm_87 = torch.ops.aten.native_layer_norm.default(sum_81, [32], arg1165_1, arg1166_1, 1e-06);  arg1165_1 = arg1166_1 = None
        getitem_265 = native_layer_norm_87[0];  native_layer_norm_87 = None
        addmm_357 = torch.ops.aten.addmm.default(arg1168_1, getitem_265, arg1167_1);  arg1168_1 = getitem_265 = arg1167_1 = None
        native_layer_norm_88 = torch.ops.aten.native_layer_norm.default(sum_91, [32], arg1169_1, arg1170_1, 1e-06);  arg1169_1 = arg1170_1 = None
        getitem_268 = native_layer_norm_88[0];  native_layer_norm_88 = None
        addmm_358 = torch.ops.aten.addmm.default(arg1172_1, getitem_268, arg1171_1);  arg1172_1 = getitem_268 = arg1171_1 = None
        native_layer_norm_89 = torch.ops.aten.native_layer_norm.default(where_191, [32], arg1173_1, arg1174_1, 1e-06);  arg1173_1 = arg1174_1 = None
        getitem_271 = native_layer_norm_89[0];  native_layer_norm_89 = None
        addmm_359 = torch.ops.aten.addmm.default(arg1176_1, getitem_271, arg1175_1);  arg1176_1 = getitem_271 = arg1175_1 = None
        native_layer_norm_90 = torch.ops.aten.native_layer_norm.default(sum_101, [128], arg1177_1, arg1178_1, 1e-06);  arg1177_1 = arg1178_1 = None
        getitem_274 = native_layer_norm_90[0];  native_layer_norm_90 = None
        addmm_360 = torch.ops.aten.addmm.default(arg1180_1, getitem_274, arg1179_1);  arg1180_1 = getitem_274 = arg1179_1 = None
        native_layer_norm_91 = torch.ops.aten.native_layer_norm.default(where_149, [32], arg1181_1, arg1182_1, 1e-06);  arg1181_1 = arg1182_1 = None
        getitem_277 = native_layer_norm_91[0];  native_layer_norm_91 = None
        addmm_361 = torch.ops.aten.addmm.default(arg1184_1, getitem_277, arg1183_1);  arg1184_1 = getitem_277 = arg1183_1 = None
        native_layer_norm_92 = torch.ops.aten.native_layer_norm.default(sum_111, [64], arg1185_1, arg1186_1, 1e-06);  arg1185_1 = arg1186_1 = None
        getitem_280 = native_layer_norm_92[0];  native_layer_norm_92 = None
        addmm_362 = torch.ops.aten.addmm.default(arg1188_1, getitem_280, arg1187_1);  arg1188_1 = getitem_280 = arg1187_1 = None
        native_layer_norm_93 = torch.ops.aten.native_layer_norm.default(where_163, [64], arg1189_1, arg1190_1, 1e-06);  arg1189_1 = arg1190_1 = None
        getitem_283 = native_layer_norm_93[0];  native_layer_norm_93 = None
        addmm_363 = torch.ops.aten.addmm.default(arg1192_1, getitem_283, arg1191_1);  arg1192_1 = getitem_283 = arg1191_1 = None
        native_layer_norm_94 = torch.ops.aten.native_layer_norm.default(sum_121, [64], arg1193_1, arg1194_1, 1e-06);  arg1193_1 = arg1194_1 = None
        getitem_286 = native_layer_norm_94[0];  native_layer_norm_94 = None
        addmm_364 = torch.ops.aten.addmm.default(arg1196_1, getitem_286, arg1195_1);  arg1196_1 = getitem_286 = arg1195_1 = None
        native_layer_norm_95 = torch.ops.aten.native_layer_norm.default(sum_145, [128], arg1197_1, arg1198_1, 1e-06);  arg1197_1 = arg1198_1 = None
        getitem_289 = native_layer_norm_95[0];  native_layer_norm_95 = None
        addmm_365 = torch.ops.aten.addmm.default(arg1200_1, getitem_289, arg1199_1);  arg1200_1 = getitem_289 = arg1199_1 = None
        native_layer_norm_96 = torch.ops.aten.native_layer_norm.default(sum_150, [128], arg1201_1, arg1202_1, 1e-06);  arg1201_1 = arg1202_1 = None
        getitem_292 = native_layer_norm_96[0];  native_layer_norm_96 = None
        addmm_366 = torch.ops.aten.addmm.default(arg1204_1, getitem_292, arg1203_1);  arg1204_1 = getitem_292 = arg1203_1 = None
        native_layer_norm_97 = torch.ops.aten.native_layer_norm.default(sum_141, [32], arg1205_1, arg1206_1, 1e-06);  arg1205_1 = arg1206_1 = None
        getitem_295 = native_layer_norm_97[0];  native_layer_norm_97 = None
        addmm_367 = torch.ops.aten.addmm.default(arg1208_1, getitem_295, arg1207_1);  arg1208_1 = getitem_295 = arg1207_1 = None
        native_layer_norm_98 = torch.ops.aten.native_layer_norm.default(where_121, [32], arg1209_1, arg1210_1, 1e-06);  arg1209_1 = arg1210_1 = None
        getitem_298 = native_layer_norm_98[0];  native_layer_norm_98 = None
        addmm_368 = torch.ops.aten.addmm.default(arg1212_1, getitem_298, arg1211_1);  arg1212_1 = getitem_298 = arg1211_1 = None
        native_layer_norm_99 = torch.ops.aten.native_layer_norm.default(cat_123, [1056], arg1213_1, arg1214_1, 1e-06);  arg1213_1 = arg1214_1 = None
        getitem_301 = native_layer_norm_99[0];  native_layer_norm_99 = None
        addmm_369 = torch.ops.aten.addmm.default(arg1216_1, getitem_301, arg1215_1);  arg1216_1 = getitem_301 = arg1215_1 = None
        native_layer_norm_100 = torch.ops.aten.native_layer_norm.default(cat_124, [1056], arg1217_1, arg1218_1, 1e-06);  arg1217_1 = arg1218_1 = None
        getitem_304 = native_layer_norm_100[0];  native_layer_norm_100 = None
        addmm_370 = torch.ops.aten.addmm.default(arg1220_1, getitem_304, arg1219_1);  arg1220_1 = getitem_304 = arg1219_1 = None
        native_layer_norm_101 = torch.ops.aten.native_layer_norm.default(arg856_1, [961], arg1221_1, arg1222_1, 1e-06);  arg856_1 = arg1221_1 = arg1222_1 = None
        getitem_307 = native_layer_norm_101[0];  native_layer_norm_101 = None
        addmm_371 = torch.ops.aten.addmm.default(arg1224_1, getitem_307, arg1223_1);  arg1224_1 = getitem_307 = arg1223_1 = None
        native_layer_norm_102 = torch.ops.aten.native_layer_norm.default(cat_125, [17], arg1225_1, arg1226_1, 1e-06);  arg1225_1 = arg1226_1 = None
        getitem_310 = native_layer_norm_102[0];  native_layer_norm_102 = None
        addmm_372 = torch.ops.aten.addmm.default(arg1228_1, getitem_310, arg1227_1);  arg1228_1 = getitem_310 = arg1227_1 = None
        native_layer_norm_103 = torch.ops.aten.native_layer_norm.default(div, [64], arg1229_1, arg1230_1, 1e-06);  arg1229_1 = arg1230_1 = None
        getitem_313 = native_layer_norm_103[0];  native_layer_norm_103 = None
        addmm_373 = torch.ops.aten.addmm.default(arg1232_1, getitem_313, arg1231_1);  arg1232_1 = getitem_313 = arg1231_1 = None
        native_layer_norm_104 = torch.ops.aten.native_layer_norm.default(div_1, [64], arg1233_1, arg1234_1, 1e-06);  arg1233_1 = arg1234_1 = None
        getitem_316 = native_layer_norm_104[0];  native_layer_norm_104 = None
        addmm_374 = torch.ops.aten.addmm.default(arg1236_1, getitem_316, arg1235_1);  arg1236_1 = getitem_316 = arg1235_1 = None
        native_layer_norm_105 = torch.ops.aten.native_layer_norm.default(sum_189, [16], arg1237_1, arg1238_1, 1e-06);  arg1237_1 = arg1238_1 = None
        getitem_319 = native_layer_norm_105[0];  native_layer_norm_105 = None
        addmm_375 = torch.ops.aten.addmm.default(arg1240_1, getitem_319, arg1239_1);  arg1240_1 = getitem_319 = arg1239_1 = None
        native_layer_norm_106 = torch.ops.aten.native_layer_norm.default(sum_190, [16], arg1241_1, arg1242_1, 1e-06);  arg1241_1 = arg1242_1 = None
        getitem_322 = native_layer_norm_106[0];  native_layer_norm_106 = None
        addmm_376 = torch.ops.aten.addmm.default(arg1244_1, getitem_322, arg1243_1);  arg1244_1 = getitem_322 = arg1243_1 = None
        native_layer_norm_107 = torch.ops.aten.native_layer_norm.default(cat_138, [688], arg1245_1, arg1246_1, 1e-06);  arg1245_1 = arg1246_1 = None
        getitem_325 = native_layer_norm_107[0];  native_layer_norm_107 = None
        addmm_377 = torch.ops.aten.addmm.default(arg1248_1, getitem_325, arg1247_1);  arg1248_1 = getitem_325 = arg1247_1 = None
        native_layer_norm_108 = torch.ops.aten.native_layer_norm.default(cat_139, [48], arg1249_1, arg1250_1, 1e-06);  arg1249_1 = arg1250_1 = None
        getitem_328 = native_layer_norm_108[0];  native_layer_norm_108 = None
        addmm_378 = torch.ops.aten.addmm.default(arg1252_1, getitem_328, arg1251_1);  arg1252_1 = getitem_328 = arg1251_1 = None
        native_layer_norm_109 = torch.ops.aten.native_layer_norm.default(getitem_99, [1024], arg1253_1, arg1254_1, 1e-06);  arg1253_1 = arg1254_1 = None
        getitem_331 = native_layer_norm_109[0];  native_layer_norm_109 = None
        addmm_379 = torch.ops.aten.addmm.default(arg1256_1, getitem_331, arg1255_1);  arg1256_1 = getitem_331 = arg1255_1 = None
        native_layer_norm_110 = torch.ops.aten.native_layer_norm.default(getitem_100, [1024], arg1257_1, arg1258_1, 1e-06);  arg1257_1 = arg1258_1 = None
        getitem_334 = native_layer_norm_110[0];  native_layer_norm_110 = None
        addmm_380 = torch.ops.aten.addmm.default(arg1260_1, getitem_334, arg1259_1);  arg1260_1 = getitem_334 = arg1259_1 = None
        native_layer_norm_111 = torch.ops.aten.native_layer_norm.default(getitem_101, [1024], arg1261_1, arg1262_1, 1e-06);  arg1261_1 = arg1262_1 = None
        getitem_337 = native_layer_norm_111[0];  native_layer_norm_111 = None
        addmm_381 = torch.ops.aten.addmm.default(arg1264_1, getitem_337, arg1263_1);  arg1264_1 = getitem_337 = arg1263_1 = None
        native_layer_norm_112 = torch.ops.aten.native_layer_norm.default(getitem_102, [1146], arg1265_1, arg1266_1, 1e-06);  arg1265_1 = arg1266_1 = None
        getitem_340 = native_layer_norm_112[0];  native_layer_norm_112 = None
        addmm_382 = torch.ops.aten.addmm.default(arg1268_1, getitem_340, arg1267_1);  arg1268_1 = getitem_340 = arg1267_1 = None
        cat_150 = torch.ops.aten.cat.default([addmm_304, addmm_305, addmm_306, mm_6, addmm_307, addmm_308, addmm_309, addmm_310, addmm_311, addmm_312, addmm_313, addmm_314, addmm_315, addmm_316, addmm_317, addmm_318, addmm_319, addmm_320, addmm_321, addmm_322, addmm_323, addmm_324, addmm_325, addmm_326, addmm_327, addmm_328, addmm_329, addmm_330, addmm_331, addmm_332, addmm_333, addmm_334, addmm_335, addmm_336, addmm_337, addmm_338, addmm_339, addmm_340, addmm_341, addmm_342, addmm_343, addmm_344, addmm_345, addmm_346, addmm_347, addmm_348, addmm_349, addmm_350, addmm_351, addmm_352, addmm_353, addmm_354, addmm_355, addmm_356, addmm_357, addmm_358, addmm_359, addmm_360, addmm_361, addmm_362, addmm_363, addmm_364, addmm_365, addmm_366, addmm_367, addmm_368, addmm_369, addmm_370, addmm_371, addmm_372, addmm_373, addmm_374, addmm_375, addmm_376, addmm_377, addmm_378, addmm_379, addmm_380, addmm_381, addmm_382], 1);  addmm_304 = addmm_305 = addmm_306 = mm_6 = addmm_307 = addmm_308 = addmm_309 = addmm_310 = addmm_311 = addmm_312 = addmm_313 = addmm_314 = addmm_315 = addmm_316 = addmm_317 = addmm_318 = addmm_319 = addmm_320 = addmm_321 = addmm_322 = addmm_323 = addmm_324 = addmm_325 = addmm_326 = addmm_327 = addmm_328 = addmm_329 = addmm_330 = addmm_331 = addmm_332 = addmm_333 = addmm_334 = addmm_335 = addmm_336 = addmm_337 = addmm_338 = addmm_339 = addmm_340 = addmm_341 = addmm_342 = addmm_343 = addmm_344 = addmm_345 = addmm_346 = addmm_347 = addmm_348 = addmm_349 = addmm_350 = addmm_351 = addmm_352 = addmm_353 = addmm_354 = addmm_355 = addmm_356 = addmm_357 = addmm_358 = addmm_359 = addmm_360 = addmm_361 = addmm_362 = addmm_363 = addmm_364 = addmm_365 = addmm_366 = addmm_367 = addmm_368 = addmm_369 = addmm_370 = addmm_371 = addmm_372 = addmm_373 = addmm_374 = addmm_375 = addmm_376 = addmm_377 = addmm_378 = addmm_379 = addmm_380 = addmm_381 = addmm_382 = None
        view_475 = torch.ops.aten.view.default(cat_150, [batch_size, 80, 640]);  cat_150 = None
        view_476 = torch.ops.aten.view.default(view_475, [-1, 80, 80, 8])
        permute_780 = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
        clone_50 = torch.ops.aten.clone.default(permute_780, memory_format = torch.contiguous_format);  permute_780 = None
        view_477 = torch.ops.aten.view.default(clone_50, [batch_size, 80, 640]);  clone_50 = None
        native_layer_norm_113 = torch.ops.aten.native_layer_norm.default(view_477, [640], arg1269_1, arg1270_1, 1e-06);  view_477 = arg1269_1 = arg1270_1 = None
        getitem_343 = native_layer_norm_113[0];  native_layer_norm_113 = None
        permute_781 = torch.ops.aten.permute.default(getitem_343, [1, 0, 2])
        expand_46 = torch.ops.aten.expand.default(permute_781, [80, batch_size, 640])
        expand_47 = torch.ops.aten.expand.default(arg1271_1, [80, 640, 80]);  arg1271_1 = None
        bmm_16 = torch.ops.aten.bmm.default(expand_46, expand_47);  expand_46 = expand_47 = None
        add_1110 = torch.ops.aten.add.Tensor(bmm_16, arg1272_1);  bmm_16 = arg1272_1 = None
        mul_1318 = torch.ops.aten.mul.Tensor(add_1110, add_1110)
        mul_1319 = torch.ops.aten.mul.Tensor(mul_1318, add_1110);  mul_1318 = None
        mul_1320 = torch.ops.aten.mul.Tensor(mul_1319, 0.044715);  mul_1319 = None
        add_1111 = torch.ops.aten.add.Tensor(add_1110, mul_1320);  mul_1320 = None
        mul_1321 = torch.ops.aten.mul.Tensor(add_1111, 1.5957691216057308);  add_1111 = None
        sigmoid_12 = torch.ops.aten.sigmoid.default(mul_1321);  mul_1321 = None
        mul_1322 = torch.ops.aten.mul.Tensor(add_1110, sigmoid_12);  add_1110 = sigmoid_12 = None
        expand_48 = torch.ops.aten.expand.default(mul_1322, [80, batch_size, 80]);  mul_1322 = None
        expand_49 = torch.ops.aten.expand.default(arg1273_1, [80, 80, 640]);  arg1273_1 = None
        bmm_17 = torch.ops.aten.bmm.default(expand_48, expand_49);  expand_48 = expand_49 = None
        add_1112 = torch.ops.aten.add.Tensor(bmm_17, arg1274_1);  bmm_17 = arg1274_1 = None
        expand_50 = torch.ops.aten.expand.default(permute_781, [80, batch_size, 640])
        expand_51 = torch.ops.aten.expand.default(arg1275_1, [80, 640, 960]);  arg1275_1 = None
        bmm_18 = torch.ops.aten.bmm.default(expand_50, expand_51);  expand_50 = expand_51 = None
        add_1113 = torch.ops.aten.add.Tensor(bmm_18, arg1276_1);  bmm_18 = arg1276_1 = None
        mul_1323 = torch.ops.aten.mul.Tensor(add_1113, add_1113)
        mul_1324 = torch.ops.aten.mul.Tensor(mul_1323, add_1113);  mul_1323 = None
        mul_1325 = torch.ops.aten.mul.Tensor(mul_1324, 0.044715);  mul_1324 = None
        add_1114 = torch.ops.aten.add.Tensor(add_1113, mul_1325);  mul_1325 = None
        mul_1326 = torch.ops.aten.mul.Tensor(add_1114, 1.5957691216057308);  add_1114 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(mul_1326);  mul_1326 = None
        mul_1327 = torch.ops.aten.mul.Tensor(add_1113, sigmoid_13);  add_1113 = sigmoid_13 = None
        expand_52 = torch.ops.aten.expand.default(mul_1327, [80, batch_size, 960]);  mul_1327 = None
        expand_53 = torch.ops.aten.expand.default(arg1277_1, [80, 960, 1280]);  arg1277_1 = None
        bmm_19 = torch.ops.aten.bmm.default(expand_52, expand_53);  expand_52 = expand_53 = None
        add_1115 = torch.ops.aten.add.Tensor(bmm_19, arg1278_1);  bmm_19 = arg1278_1 = None
        tanh = torch.ops.aten.tanh.default(add_1112);  add_1112 = None
        mul_1328 = torch.ops.aten.mul.Tensor(permute_781, tanh);  permute_781 = tanh = None
        expand_54 = torch.ops.aten.expand.default(mul_1328, [80, batch_size, 640]);  mul_1328 = None
        expand_55 = torch.ops.aten.expand.default(arg1279_1, [80, 640, 1280]);  arg1279_1 = None
        bmm_20 = torch.ops.aten.bmm.default(expand_54, expand_55);  expand_54 = expand_55 = None
        add_1116 = torch.ops.aten.add.Tensor(bmm_20, arg1280_1);  bmm_20 = arg1280_1 = None
        mul_1329 = torch.ops.aten.mul.Tensor(add_1116, add_1116)
        mul_1330 = torch.ops.aten.mul.Tensor(mul_1329, add_1116);  mul_1329 = None
        mul_1331 = torch.ops.aten.mul.Tensor(mul_1330, 0.044715);  mul_1330 = None
        add_1117 = torch.ops.aten.add.Tensor(add_1116, mul_1331);  mul_1331 = None
        mul_1332 = torch.ops.aten.mul.Tensor(add_1117, 1.5957691216057308);  add_1117 = None
        sigmoid_14 = torch.ops.aten.sigmoid.default(mul_1332);  mul_1332 = None
        mul_1333 = torch.ops.aten.mul.Tensor(add_1116, sigmoid_14);  add_1116 = sigmoid_14 = None
        tanh_1 = torch.ops.aten.tanh.default(add_1115);  add_1115 = None
        mul_1334 = torch.ops.aten.mul.Tensor(mul_1333, tanh_1);  mul_1333 = tanh_1 = None
        expand_56 = torch.ops.aten.expand.default(mul_1334, [80, batch_size, 1280]);  mul_1334 = None
        expand_57 = torch.ops.aten.expand.default(arg1281_1, [80, 1280, 640]);  arg1281_1 = None
        bmm_21 = torch.ops.aten.bmm.default(expand_56, expand_57);  expand_56 = expand_57 = None
        add_1118 = torch.ops.aten.add.Tensor(bmm_21, arg1282_1);  bmm_21 = arg1282_1 = None
        permute_782 = torch.ops.aten.permute.default(add_1118, [1, 0, 2]);  add_1118 = None
        add_1119 = torch.ops.aten.add.Tensor(getitem_343, permute_782);  getitem_343 = permute_782 = None
        native_layer_norm_114 = torch.ops.aten.native_layer_norm.default(add_1119, [640], arg1283_1, arg1284_1, 1e-06);  add_1119 = arg1283_1 = arg1284_1 = None
        getitem_346 = native_layer_norm_114[0];  native_layer_norm_114 = None
        view_496 = torch.ops.aten.view.default(getitem_346, [-1, 80, 80, 8]);  getitem_346 = None
        permute_783 = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        clone_55 = torch.ops.aten.clone.default(permute_783, memory_format = torch.contiguous_format);  permute_783 = None
        view_497 = torch.ops.aten.view.default(clone_55, [batch_size, 80, 640]);  clone_55 = None
        native_layer_norm_115 = torch.ops.aten.native_layer_norm.default(view_497, [640], arg1285_1, arg1286_1, 1e-06);  view_497 = arg1285_1 = arg1286_1 = None
        getitem_349 = native_layer_norm_115[0];  native_layer_norm_115 = None
        permute_784 = torch.ops.aten.permute.default(getitem_349, [1, 0, 2])
        expand_58 = torch.ops.aten.expand.default(permute_784, [80, batch_size, 640])
        expand_59 = torch.ops.aten.expand.default(arg1287_1, [80, 640, 80]);  arg1287_1 = None
        bmm_22 = torch.ops.aten.bmm.default(expand_58, expand_59);  expand_58 = expand_59 = None
        add_1120 = torch.ops.aten.add.Tensor(bmm_22, arg1288_1);  bmm_22 = arg1288_1 = None
        mul_1335 = torch.ops.aten.mul.Tensor(add_1120, add_1120)
        mul_1336 = torch.ops.aten.mul.Tensor(mul_1335, add_1120);  mul_1335 = None
        mul_1337 = torch.ops.aten.mul.Tensor(mul_1336, 0.044715);  mul_1336 = None
        add_1121 = torch.ops.aten.add.Tensor(add_1120, mul_1337);  mul_1337 = None
        mul_1338 = torch.ops.aten.mul.Tensor(add_1121, 1.5957691216057308);  add_1121 = None
        sigmoid_15 = torch.ops.aten.sigmoid.default(mul_1338);  mul_1338 = None
        mul_1339 = torch.ops.aten.mul.Tensor(add_1120, sigmoid_15);  add_1120 = sigmoid_15 = None
        expand_60 = torch.ops.aten.expand.default(mul_1339, [80, batch_size, 80]);  mul_1339 = None
        expand_61 = torch.ops.aten.expand.default(arg1289_1, [80, 80, 640]);  arg1289_1 = None
        bmm_23 = torch.ops.aten.bmm.default(expand_60, expand_61);  expand_60 = expand_61 = None
        add_1122 = torch.ops.aten.add.Tensor(bmm_23, arg1290_1);  bmm_23 = arg1290_1 = None
        expand_62 = torch.ops.aten.expand.default(permute_784, [80, batch_size, 640])
        expand_63 = torch.ops.aten.expand.default(arg1291_1, [80, 640, 960]);  arg1291_1 = None
        bmm_24 = torch.ops.aten.bmm.default(expand_62, expand_63);  expand_62 = expand_63 = None
        add_1123 = torch.ops.aten.add.Tensor(bmm_24, arg1292_1);  bmm_24 = arg1292_1 = None
        mul_1340 = torch.ops.aten.mul.Tensor(add_1123, add_1123)
        mul_1341 = torch.ops.aten.mul.Tensor(mul_1340, add_1123);  mul_1340 = None
        mul_1342 = torch.ops.aten.mul.Tensor(mul_1341, 0.044715);  mul_1341 = None
        add_1124 = torch.ops.aten.add.Tensor(add_1123, mul_1342);  mul_1342 = None
        mul_1343 = torch.ops.aten.mul.Tensor(add_1124, 1.5957691216057308);  add_1124 = None
        sigmoid_16 = torch.ops.aten.sigmoid.default(mul_1343);  mul_1343 = None
        mul_1344 = torch.ops.aten.mul.Tensor(add_1123, sigmoid_16);  add_1123 = sigmoid_16 = None
        expand_64 = torch.ops.aten.expand.default(mul_1344, [80, batch_size, 960]);  mul_1344 = None
        expand_65 = torch.ops.aten.expand.default(arg1293_1, [80, 960, 1280]);  arg1293_1 = None
        bmm_25 = torch.ops.aten.bmm.default(expand_64, expand_65);  expand_64 = expand_65 = None
        add_1125 = torch.ops.aten.add.Tensor(bmm_25, arg1294_1);  bmm_25 = arg1294_1 = None
        tanh_2 = torch.ops.aten.tanh.default(add_1122);  add_1122 = None
        mul_1345 = torch.ops.aten.mul.Tensor(permute_784, tanh_2);  permute_784 = tanh_2 = None
        expand_66 = torch.ops.aten.expand.default(mul_1345, [80, batch_size, 640]);  mul_1345 = None
        expand_67 = torch.ops.aten.expand.default(arg1295_1, [80, 640, 1280]);  arg1295_1 = None
        bmm_26 = torch.ops.aten.bmm.default(expand_66, expand_67);  expand_66 = expand_67 = None
        add_1126 = torch.ops.aten.add.Tensor(bmm_26, arg1296_1);  bmm_26 = arg1296_1 = None
        mul_1346 = torch.ops.aten.mul.Tensor(add_1126, add_1126)
        mul_1347 = torch.ops.aten.mul.Tensor(mul_1346, add_1126);  mul_1346 = None
        mul_1348 = torch.ops.aten.mul.Tensor(mul_1347, 0.044715);  mul_1347 = None
        add_1127 = torch.ops.aten.add.Tensor(add_1126, mul_1348);  mul_1348 = None
        mul_1349 = torch.ops.aten.mul.Tensor(add_1127, 1.5957691216057308);  add_1127 = None
        sigmoid_17 = torch.ops.aten.sigmoid.default(mul_1349);  mul_1349 = None
        mul_1350 = torch.ops.aten.mul.Tensor(add_1126, sigmoid_17);  add_1126 = sigmoid_17 = None
        tanh_3 = torch.ops.aten.tanh.default(add_1125);  add_1125 = None
        mul_1351 = torch.ops.aten.mul.Tensor(mul_1350, tanh_3);  mul_1350 = tanh_3 = None
        expand_68 = torch.ops.aten.expand.default(mul_1351, [80, batch_size, 1280]);  mul_1351 = None
        expand_69 = torch.ops.aten.expand.default(arg1297_1, [80, 1280, 640]);  arg1297_1 = None
        bmm_27 = torch.ops.aten.bmm.default(expand_68, expand_69);  expand_68 = expand_69 = None
        add_1128 = torch.ops.aten.add.Tensor(bmm_27, arg1298_1);  bmm_27 = arg1298_1 = None
        permute_785 = torch.ops.aten.permute.default(add_1128, [1, 0, 2]);  add_1128 = None
        add_1129 = torch.ops.aten.add.Tensor(getitem_349, permute_785);  getitem_349 = permute_785 = None
        native_layer_norm_116 = torch.ops.aten.native_layer_norm.default(add_1129, [640], arg1299_1, arg1300_1, 1e-06);  add_1129 = arg1299_1 = arg1300_1 = None
        getitem_352 = native_layer_norm_116[0];  native_layer_norm_116 = None
        add_1130 = torch.ops.aten.add.Tensor(view_475, getitem_352);  view_475 = None
        view_516 = torch.ops.aten.view.default(add_1130, [-1, 80, 32, 20]);  add_1130 = None
        permute_786 = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
        clone_60 = torch.ops.aten.clone.default(permute_786, memory_format = torch.contiguous_format);  permute_786 = None
        view_517 = torch.ops.aten.view.default(clone_60, [batch_size, 32, 1600]);  clone_60 = None
        native_layer_norm_117 = torch.ops.aten.native_layer_norm.default(view_517, [1600], arg1301_1, arg1302_1, 1e-06);  view_517 = arg1301_1 = arg1302_1 = None
        getitem_355 = native_layer_norm_117[0];  native_layer_norm_117 = None
        permute_787 = torch.ops.aten.permute.default(getitem_355, [1, 0, 2]);  getitem_355 = None
        expand_70 = torch.ops.aten.expand.default(permute_787, [32, batch_size, 1600])
        expand_71 = torch.ops.aten.expand.default(arg1303_1, [32, 1600, 200]);  arg1303_1 = None
        bmm_28 = torch.ops.aten.bmm.default(expand_70, expand_71);  expand_70 = expand_71 = None
        add_1131 = torch.ops.aten.add.Tensor(bmm_28, arg1304_1);  bmm_28 = arg1304_1 = None
        mul_1352 = torch.ops.aten.mul.Tensor(add_1131, add_1131)
        mul_1353 = torch.ops.aten.mul.Tensor(mul_1352, add_1131);  mul_1352 = None
        mul_1354 = torch.ops.aten.mul.Tensor(mul_1353, 0.044715);  mul_1353 = None
        add_1132 = torch.ops.aten.add.Tensor(add_1131, mul_1354);  mul_1354 = None
        mul_1355 = torch.ops.aten.mul.Tensor(add_1132, 1.5957691216057308);  add_1132 = None
        sigmoid_18 = torch.ops.aten.sigmoid.default(mul_1355);  mul_1355 = None
        mul_1356 = torch.ops.aten.mul.Tensor(add_1131, sigmoid_18);  add_1131 = sigmoid_18 = None
        expand_72 = torch.ops.aten.expand.default(mul_1356, [32, batch_size, 200]);  mul_1356 = None
        expand_73 = torch.ops.aten.expand.default(arg1305_1, [32, 200, 1600]);  arg1305_1 = None
        bmm_29 = torch.ops.aten.bmm.default(expand_72, expand_73);  expand_72 = expand_73 = None
        add_1133 = torch.ops.aten.add.Tensor(bmm_29, arg1306_1);  bmm_29 = arg1306_1 = None
        expand_74 = torch.ops.aten.expand.default(permute_787, [32, batch_size, 1600])
        expand_75 = torch.ops.aten.expand.default(arg1307_1, [32, 1600, 960]);  arg1307_1 = None
        bmm_30 = torch.ops.aten.bmm.default(expand_74, expand_75);  expand_74 = expand_75 = None
        add_1134 = torch.ops.aten.add.Tensor(bmm_30, arg1308_1);  bmm_30 = arg1308_1 = None
        mul_1357 = torch.ops.aten.mul.Tensor(add_1134, add_1134)
        mul_1358 = torch.ops.aten.mul.Tensor(mul_1357, add_1134);  mul_1357 = None
        mul_1359 = torch.ops.aten.mul.Tensor(mul_1358, 0.044715);  mul_1358 = None
        add_1135 = torch.ops.aten.add.Tensor(add_1134, mul_1359);  mul_1359 = None
        mul_1360 = torch.ops.aten.mul.Tensor(add_1135, 1.5957691216057308);  add_1135 = None
        sigmoid_19 = torch.ops.aten.sigmoid.default(mul_1360);  mul_1360 = None
        mul_1361 = torch.ops.aten.mul.Tensor(add_1134, sigmoid_19);  add_1134 = sigmoid_19 = None
        expand_76 = torch.ops.aten.expand.default(mul_1361, [32, batch_size, 960]);  mul_1361 = None
        expand_77 = torch.ops.aten.expand.default(arg1309_1, [32, 960, 1280]);  arg1309_1 = None
        bmm_31 = torch.ops.aten.bmm.default(expand_76, expand_77);  expand_76 = expand_77 = None
        add_1136 = torch.ops.aten.add.Tensor(bmm_31, arg1310_1);  bmm_31 = arg1310_1 = None
        tanh_4 = torch.ops.aten.tanh.default(add_1133);  add_1133 = None
        mul_1362 = torch.ops.aten.mul.Tensor(permute_787, tanh_4);  permute_787 = tanh_4 = None
        expand_78 = torch.ops.aten.expand.default(mul_1362, [32, batch_size, 1600]);  mul_1362 = None
        expand_79 = torch.ops.aten.expand.default(arg1311_1, [32, 1600, 1280]);  arg1311_1 = None
        bmm_32 = torch.ops.aten.bmm.default(expand_78, expand_79);  expand_78 = expand_79 = None
        add_1137 = torch.ops.aten.add.Tensor(bmm_32, arg1312_1);  bmm_32 = arg1312_1 = None
        mul_1363 = torch.ops.aten.mul.Tensor(add_1137, add_1137)
        mul_1364 = torch.ops.aten.mul.Tensor(mul_1363, add_1137);  mul_1363 = None
        mul_1365 = torch.ops.aten.mul.Tensor(mul_1364, 0.044715);  mul_1364 = None
        add_1138 = torch.ops.aten.add.Tensor(add_1137, mul_1365);  mul_1365 = None
        mul_1366 = torch.ops.aten.mul.Tensor(add_1138, 1.5957691216057308);  add_1138 = None
        sigmoid_20 = torch.ops.aten.sigmoid.default(mul_1366);  mul_1366 = None
        mul_1367 = torch.ops.aten.mul.Tensor(add_1137, sigmoid_20);  add_1137 = sigmoid_20 = None
        tanh_5 = torch.ops.aten.tanh.default(add_1136);  add_1136 = None
        mul_1368 = torch.ops.aten.mul.Tensor(mul_1367, tanh_5);  mul_1367 = tanh_5 = None
        expand_80 = torch.ops.aten.expand.default(mul_1368, [32, batch_size, 1280]);  mul_1368 = None
        expand_81 = torch.ops.aten.expand.default(arg1313_1, [32, 1280, 640]);  arg1313_1 = None
        bmm_33 = torch.ops.aten.bmm.default(expand_80, expand_81);  expand_80 = expand_81 = None
        add_1139 = torch.ops.aten.add.Tensor(bmm_33, arg1314_1);  bmm_33 = arg1314_1 = None
        permute_788 = torch.ops.aten.permute.default(add_1139, [1, 0, 2]);  add_1139 = None
        native_layer_norm_118 = torch.ops.aten.native_layer_norm.default(permute_788, [640], arg1315_1, arg1316_1, 1e-06);  permute_788 = arg1315_1 = arg1316_1 = None
        getitem_358 = native_layer_norm_118[0];  native_layer_norm_118 = None
        view_536 = torch.ops.aten.view.default(getitem_358, [-1, 32, 16, 40]);  getitem_358 = None
        permute_789 = torch.ops.aten.permute.default(view_536, [0, 2, 1, 3]);  view_536 = None
        clone_65 = torch.ops.aten.clone.default(permute_789, memory_format = torch.contiguous_format);  permute_789 = None
        view_537 = torch.ops.aten.view.default(clone_65, [batch_size, 16, 1280]);  clone_65 = None
        native_layer_norm_119 = torch.ops.aten.native_layer_norm.default(view_537, [1280], arg1317_1, arg1318_1, 1e-06);  view_537 = arg1317_1 = arg1318_1 = None
        getitem_361 = native_layer_norm_119[0];  native_layer_norm_119 = None
        permute_790 = torch.ops.aten.permute.default(getitem_361, [1, 0, 2]);  getitem_361 = None
        expand_82 = torch.ops.aten.expand.default(permute_790, [16, batch_size, 1280])
        expand_83 = torch.ops.aten.expand.default(arg1319_1, [16, 1280, 160]);  arg1319_1 = None
        bmm_34 = torch.ops.aten.bmm.default(expand_82, expand_83);  expand_82 = expand_83 = None
        add_1140 = torch.ops.aten.add.Tensor(bmm_34, arg1320_1);  bmm_34 = arg1320_1 = None
        mul_1369 = torch.ops.aten.mul.Tensor(add_1140, add_1140)
        mul_1370 = torch.ops.aten.mul.Tensor(mul_1369, add_1140);  mul_1369 = None
        mul_1371 = torch.ops.aten.mul.Tensor(mul_1370, 0.044715);  mul_1370 = None
        add_1141 = torch.ops.aten.add.Tensor(add_1140, mul_1371);  mul_1371 = None
        mul_1372 = torch.ops.aten.mul.Tensor(add_1141, 1.5957691216057308);  add_1141 = None
        sigmoid_21 = torch.ops.aten.sigmoid.default(mul_1372);  mul_1372 = None
        mul_1373 = torch.ops.aten.mul.Tensor(add_1140, sigmoid_21);  add_1140 = sigmoid_21 = None
        expand_84 = torch.ops.aten.expand.default(mul_1373, [16, batch_size, 160]);  mul_1373 = None
        expand_85 = torch.ops.aten.expand.default(arg1321_1, [16, 160, 1280]);  arg1321_1 = None
        bmm_35 = torch.ops.aten.bmm.default(expand_84, expand_85);  expand_84 = expand_85 = None
        add_1142 = torch.ops.aten.add.Tensor(bmm_35, arg1322_1);  bmm_35 = arg1322_1 = None
        expand_86 = torch.ops.aten.expand.default(permute_790, [16, batch_size, 1280])
        expand_87 = torch.ops.aten.expand.default(arg1323_1, [16, 1280, 960]);  arg1323_1 = None
        bmm_36 = torch.ops.aten.bmm.default(expand_86, expand_87);  expand_86 = expand_87 = None
        add_1143 = torch.ops.aten.add.Tensor(bmm_36, arg1324_1);  bmm_36 = arg1324_1 = None
        mul_1374 = torch.ops.aten.mul.Tensor(add_1143, add_1143)
        mul_1375 = torch.ops.aten.mul.Tensor(mul_1374, add_1143);  mul_1374 = None
        mul_1376 = torch.ops.aten.mul.Tensor(mul_1375, 0.044715);  mul_1375 = None
        add_1144 = torch.ops.aten.add.Tensor(add_1143, mul_1376);  mul_1376 = None
        mul_1377 = torch.ops.aten.mul.Tensor(add_1144, 1.5957691216057308);  add_1144 = None
        sigmoid_22 = torch.ops.aten.sigmoid.default(mul_1377);  mul_1377 = None
        mul_1378 = torch.ops.aten.mul.Tensor(add_1143, sigmoid_22);  add_1143 = sigmoid_22 = None
        expand_88 = torch.ops.aten.expand.default(mul_1378, [16, batch_size, 960]);  mul_1378 = None
        expand_89 = torch.ops.aten.expand.default(arg1325_1, [16, 960, 1280]);  arg1325_1 = None
        bmm_37 = torch.ops.aten.bmm.default(expand_88, expand_89);  expand_88 = expand_89 = None
        add_1145 = torch.ops.aten.add.Tensor(bmm_37, arg1326_1);  bmm_37 = arg1326_1 = None
        tanh_6 = torch.ops.aten.tanh.default(add_1142);  add_1142 = None
        mul_1379 = torch.ops.aten.mul.Tensor(permute_790, tanh_6);  permute_790 = tanh_6 = None
        expand_90 = torch.ops.aten.expand.default(mul_1379, [16, batch_size, 1280]);  mul_1379 = None
        expand_91 = torch.ops.aten.expand.default(arg1327_1, [16, 1280, 1280]);  arg1327_1 = None
        bmm_38 = torch.ops.aten.bmm.default(expand_90, expand_91);  expand_90 = expand_91 = None
        add_1146 = torch.ops.aten.add.Tensor(bmm_38, arg1328_1);  bmm_38 = arg1328_1 = None
        mul_1380 = torch.ops.aten.mul.Tensor(add_1146, add_1146)
        mul_1381 = torch.ops.aten.mul.Tensor(mul_1380, add_1146);  mul_1380 = None
        mul_1382 = torch.ops.aten.mul.Tensor(mul_1381, 0.044715);  mul_1381 = None
        add_1147 = torch.ops.aten.add.Tensor(add_1146, mul_1382);  mul_1382 = None
        mul_1383 = torch.ops.aten.mul.Tensor(add_1147, 1.5957691216057308);  add_1147 = None
        sigmoid_23 = torch.ops.aten.sigmoid.default(mul_1383);  mul_1383 = None
        mul_1384 = torch.ops.aten.mul.Tensor(add_1146, sigmoid_23);  add_1146 = sigmoid_23 = None
        tanh_7 = torch.ops.aten.tanh.default(add_1145);  add_1145 = None
        mul_1385 = torch.ops.aten.mul.Tensor(mul_1384, tanh_7);  mul_1384 = tanh_7 = None
        expand_92 = torch.ops.aten.expand.default(mul_1385, [16, batch_size, 1280]);  mul_1385 = None
        expand_93 = torch.ops.aten.expand.default(arg1329_1, [16, 1280, 640]);  arg1329_1 = None
        bmm_39 = torch.ops.aten.bmm.default(expand_92, expand_93);  expand_92 = expand_93 = None
        add_1148 = torch.ops.aten.add.Tensor(bmm_39, arg1330_1);  bmm_39 = arg1330_1 = None
        permute_791 = torch.ops.aten.permute.default(add_1148, [1, 0, 2]);  add_1148 = None
        native_layer_norm_120 = torch.ops.aten.native_layer_norm.default(permute_791, [640], arg1331_1, arg1332_1, 1e-06);  permute_791 = arg1331_1 = arg1332_1 = None
        getitem_364 = native_layer_norm_120[0];  native_layer_norm_120 = None
        div_2 = torch.ops.aten.div.Tensor(getitem_364, 10.0);  getitem_364 = None
        mean = torch.ops.aten.mean.dim(div_2, [1]);  div_2 = None
        tanh_8 = torch.ops.aten.tanh.default(mean);  mean = None
        convert_element_type_145 = torch.ops.prims.convert_element_type.default(getitem_352, torch.float32);  getitem_352 = None
        div_3 = torch.ops.aten.div.Tensor(convert_element_type_145, 10.0);  convert_element_type_145 = None
        mean_1 = torch.ops.aten.mean.dim(div_3, [1]);  div_3 = None
        tanh_9 = torch.ops.aten.tanh.default(mean_1);  mean_1 = None
        convert_element_type_146 = torch.ops.prims.convert_element_type.default(tanh_9, torch.float16);  tanh_9 = None
        cat_151 = torch.ops.aten.cat.default([mul_14, mul_16, cat_44, cat_46, addmm_30, addmm_31, addmm_32, cat_52, cat_126, cat_85, cat_88, where_196, where_192, where_200, where_198, where_194, cat_97, cat_116, cat_103, cat_100, where_204, where_206, where_208, where_210, where_212, cat_94, mul_1317, addmm_297, where_239, addmm_294, mul_1286, addmm_301, tanh_8], 1)
        ne_10 = torch.ops.aten.ne.Tensor(arg1333_1, arg1333_1)
        abs_10 = torch.ops.aten.abs.default(arg1333_1)
        eq_890 = torch.ops.aten.eq.Scalar(abs_10, inf);  abs_10 = None
        bitwise_or_9 = torch.ops.aten.bitwise_or.Tensor(ne_10, eq_890);  ne_10 = eq_890 = None
        full_default_256 = torch.ops.aten.full.default([7, 16384], 1.0013580322265625e-05, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_255 = torch.ops.aten.where.self(bitwise_or_9, full_default_256, arg1333_1);  bitwise_or_9 = full_default_256 = arg1333_1 = None
        view_556 = torch.ops.aten.view.default(where_255, [-1, 256, 64]);  where_255 = None
        slice_718 = torch.ops.aten.slice.Tensor(arg1334_1, 2, 1, 65);  arg1334_1 = None
        clamp_min_10 = torch.ops.aten.clamp_min.default(arg1335_1, 0);  arg1335_1 = None
        clamp_max_23 = torch.ops.aten.clamp_max.default(clamp_min_10, 1);  clamp_min_10 = None
        view_557 = torch.ops.aten.view.default(view_556, [1792, 64]);  view_556 = None
        addmm_383 = torch.ops.aten.addmm.default(arg1337_1, view_557, arg1336_1);  arg1337_1 = view_557 = arg1336_1 = None
        view_558 = torch.ops.aten.view.default(addmm_383, [7, 256, 64]);  addmm_383 = None
        cat_152 = torch.ops.aten.cat.default([slice_718, view_558]);  slice_718 = view_558 = None
        view_559 = torch.ops.aten.view.default(cat_152, [2, 7, 256, 64]);  cat_152 = None
        sum_193 = torch.ops.aten.sum.dim_IntList(view_559, [0]);  view_559 = None
        view_560 = torch.ops.aten.view.default(sum_193, [1792, 64]);  sum_193 = None
        addmm_384 = torch.ops.aten.addmm.default(arg1339_1, view_560, arg1338_1);  arg1339_1 = view_560 = arg1338_1 = None
        addmm_385 = torch.ops.aten.addmm.default(arg1341_1, slice_3, arg1340_1);  arg1341_1 = slice_3 = arg1340_1 = None
        addmm_386 = torch.ops.aten.addmm.default(arg1343_1, cat_151, arg1342_1);  arg1343_1 = cat_151 = arg1342_1 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(addmm_386, 1);  addmm_386 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(addmm_385, 1);  addmm_385 = None
        cat_153 = torch.ops.aten.cat.default([unsqueeze_58, unsqueeze_59], 1);  unsqueeze_58 = unsqueeze_59 = None
        view_562 = torch.ops.aten.view.default(clamp_max_23, [-1])
        gt_7 = torch.ops.aten.gt.Scalar(view_562, 0);  view_562 = None
        nonzero_10 = torch.ops.aten.nonzero.default(gt_7);  gt_7 = None
        sym_size_int_96 = torch.ops.aten.sym_size.int(nonzero_10, 0)
        ge_26 = sym_size_int_96 >= 0
        _assert_scalar_25 = torch.ops.aten._assert_scalar.default(ge_26, "Runtime assertion failed for expression u20 >= 0 on node 'ge_10'");  ge_26 = _assert_scalar_25 = None
        le_6 = sym_size_int_96 <= 1792
        _assert_scalar_26 = torch.ops.aten._assert_scalar.default(le_6, "Runtime assertion failed for expression u20 <= 1792 on node 'le_15'");  le_6 = _assert_scalar_26 = None
        iota_30 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        repeat_41 = torch.ops.aten.repeat.default(iota_30, [7]);  iota_30 = None
        index_65 = torch.ops.aten.index.Tensor(repeat_41, [nonzero_10]);  repeat_41 = None
        sum_194 = torch.ops.aten.sum.dim_IntList(clamp_max_23, [1]);  clamp_max_23 = None
        cumsum_35 = torch.ops.aten.cumsum.default(sum_194, 0);  sum_194 = None
        constant_pad_nd_22 = torch.ops.aten.constant_pad_nd.default(cumsum_35, [1, 0], 0.0);  cumsum_35 = None
        index_66 = torch.ops.aten.index.Tensor(addmm_384, [nonzero_10]);  addmm_384 = nonzero_10 = None
        full_default_5 = torch.ops.aten.full.default([sym_size_int_96, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        convert_element_type_147 = torch.ops.prims.convert_element_type.default(arg147_1, torch.int32);  arg147_1 = None
        iota_31 = torch.ops.prims.iota.default(batch_size, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        add_1165 = torch.ops.aten.add.Tensor(iota_31, 1);  iota_31 = None
        view_564 = torch.ops.aten.view.default(add_1165, [batch_size, 1]);  add_1165 = None
        repeat_42 = torch.ops.aten.repeat.default(view_564, [1, 2]);  view_564 = None
        iota_32 = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        repeat_43 = torch.ops.aten.repeat.default(iota_32, [batch_size]);  iota_32 = None
        cumsum_36 = torch.ops.aten.cumsum.default(convert_element_type_147, 0);  convert_element_type_147 = None
        constant_pad_nd_23 = torch.ops.aten.constant_pad_nd.default(cumsum_36, [1, 0], 0.0);  cumsum_36 = None
        mul_1398 = torch.ops.aten.mul.Tensor(constant_pad_nd_23, 2);  constant_pad_nd_23 = None
        view_565 = torch.ops.aten.view.default(cat_153, [-1, 512]);  cat_153 = None
        view_566 = torch.ops.aten.view.default(repeat_42, [2 * batch_size]);  repeat_42 = None
        ascend_create_position_offset_5 = torch.ops.ascend_triton.ascend_create_position_offset.default(repeat_43, mul_1398)
        ascend_seq_tensor_concat_15 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(view_565, index_66, mul_1398, constant_pad_nd_22);  view_565 = index_66 = None
        add_1169 = torch.ops.aten.add.Tensor(mul_1398, constant_pad_nd_22)
        ascend_position_concat_5 = torch.ops.ascend_triton.ascend_position_concat.default(repeat_43, index_65, mul_1398, constant_pad_nd_22, ascend_create_position_offset_5);  repeat_43 = index_65 = ascend_create_position_offset_5 = None
        sym_size_int_98 = torch.ops.aten.sym_size.int(ascend_position_concat_5, 0)
        ascend_seq_tensor_concat_16 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(view_566, full_default_5, mul_1398, constant_pad_nd_22);  view_566 = full_default_5 = None
        full_default_257 = torch.ops.aten.full.default([2 * batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        full_275 = torch.ops.aten.full.default([sym_size_int_96], 1, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  sym_size_int_96 = None
        ascend_seq_tensor_concat_17 = torch.ops.ascend_triton.ascend_seq_tensor_concat.default(full_default_257, full_275, mul_1398, constant_pad_nd_22);  full_default_257 = full_275 = mul_1398 = constant_pad_nd_22 = None
        eq_915 = torch.ops.aten.eq.Scalar(ascend_seq_tensor_concat_17, 0);  ascend_seq_tensor_concat_17 = None
        nonzero_11 = torch.ops.aten.nonzero.default(eq_915);  eq_915 = None
        _assert_scalar_27 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression u21 >= 2 * batch_size on node 'ge_11'");  _assert_scalar_27 = None
        _assert_scalar_28 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression u21 <= 2 * batch_size on node 'le_16'");  _assert_scalar_28 = None
        _assert_scalar_29 = torch.ops.aten._assert_scalar.default(True, "Runtime assertion failed for expression Eq(u21, 2 * batch_size) on node 'eq_133'");  _assert_scalar_29 = None
        squeeze_217 = torch.ops.aten.squeeze.dim(nonzero_11, -1);  nonzero_11 = None
        native_layer_norm_121 = torch.ops.aten.native_layer_norm.default(ascend_seq_tensor_concat_15, [512], arg1344_1, arg1345_1, 1e-06);  ascend_seq_tensor_concat_15 = arg1344_1 = arg1345_1 = None
        getitem_367 = native_layer_norm_121[0];  native_layer_norm_121 = None
        native_layer_norm_122 = torch.ops.aten.native_layer_norm.default(getitem_367, [512], arg1346_1, arg1347_1, 1e-06);  arg1346_1 = arg1347_1 = None
        getitem_370 = native_layer_norm_122[0];  native_layer_norm_122 = None
        addmm_387 = torch.ops.aten.addmm.default(arg1349_1, getitem_370, arg1348_1);  arg1349_1 = arg1348_1 = None
        addmm_388 = torch.ops.aten.addmm.default(arg1351_1, getitem_370, arg1350_1);  arg1351_1 = arg1350_1 = None
        addmm_389 = torch.ops.aten.addmm.default(arg1353_1, getitem_370, arg1352_1);  arg1353_1 = getitem_370 = arg1352_1 = None
        view_567 = torch.ops.aten.view.default(addmm_387, [-1, 8, 64]);  addmm_387 = None
        view_568 = torch.ops.aten.view.default(addmm_388, [-1, 8, 64]);  addmm_388 = None
        view_569 = torch.ops.aten.view.default(addmm_389, [-1, 8, 64]);  addmm_389 = None
        rope = torch.ops.qianchuan_triton.rope.default(view_567, ascend_position_concat_5, add_1169, 656, 10000.0, False);  view_567 = None
        rope_1 = torch.ops.qianchuan_triton.rope.default(view_568, ascend_position_concat_5, add_1169, 656, 10000.0, False);  view_568 = None
        ascend_flash_attention_14 = torch.ops.ascend_triton.ascend_flash_attention.default(rope, rope_1, view_569, ascend_seq_tensor_concat_16, ascend_seq_tensor_concat_16, add_1169, add_1169, 656, 656, 0.125, 1);  rope = rope_1 = view_569 = None
        view_570 = torch.ops.aten.view.default(ascend_flash_attention_14, [-1, 512]);  ascend_flash_attention_14 = None
        addmm_390 = torch.ops.aten.addmm.default(arg1355_1, view_570, arg1354_1);  arg1355_1 = view_570 = arg1354_1 = None
        add_1240 = torch.ops.aten.add.Tensor(addmm_390, getitem_367);  addmm_390 = getitem_367 = None
        softcap_28 = torch.ops.qianchuan_triton.softcap.default(add_1240, 50.0);  add_1240 = None
        native_layer_norm_123 = torch.ops.aten.native_layer_norm.default(softcap_28, [512], arg1356_1, arg1357_1, 1e-06);  arg1356_1 = arg1357_1 = None
        getitem_373 = native_layer_norm_123[0];  native_layer_norm_123 = None
        fused_swiglu_14 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_373, arg1358_1, arg1359_1, arg1360_1, arg1361_1, False, False);  getitem_373 = arg1358_1 = arg1359_1 = arg1360_1 = arg1361_1 = None
        addmm_391 = torch.ops.aten.addmm.default(arg1363_1, fused_swiglu_14, arg1362_1);  arg1363_1 = fused_swiglu_14 = arg1362_1 = None
        add_1262 = torch.ops.aten.add.Tensor(addmm_391, softcap_28);  addmm_391 = softcap_28 = None
        softcap_29 = torch.ops.qianchuan_triton.softcap.default(add_1262, 50.0);  add_1262 = None
        native_layer_norm_124 = torch.ops.aten.native_layer_norm.default(softcap_29, [512], arg1364_1, arg1365_1, 1e-06);  arg1364_1 = arg1365_1 = None
        getitem_376 = native_layer_norm_124[0];  native_layer_norm_124 = None
        addmm_392 = torch.ops.aten.addmm.default(arg1367_1, getitem_376, arg1366_1);  arg1367_1 = arg1366_1 = None
        addmm_393 = torch.ops.aten.addmm.default(arg1369_1, getitem_376, arg1368_1);  arg1369_1 = arg1368_1 = None
        addmm_394 = torch.ops.aten.addmm.default(arg1371_1, getitem_376, arg1370_1);  arg1371_1 = getitem_376 = arg1370_1 = None
        view_571 = torch.ops.aten.view.default(addmm_392, [-1, 8, 64]);  addmm_392 = None
        view_572 = torch.ops.aten.view.default(addmm_393, [-1, 8, 64]);  addmm_393 = None
        view_573 = torch.ops.aten.view.default(addmm_394, [-1, 8, 64]);  addmm_394 = None
        rope_2 = torch.ops.qianchuan_triton.rope.default(view_571, ascend_position_concat_5, add_1169, 656, 10000.0, False);  view_571 = None
        rope_3 = torch.ops.qianchuan_triton.rope.default(view_572, ascend_position_concat_5, add_1169, 656, 10000.0, False);  view_572 = None
        ascend_flash_attention_15 = torch.ops.ascend_triton.ascend_flash_attention.default(rope_2, rope_3, view_573, ascend_seq_tensor_concat_16, ascend_seq_tensor_concat_16, add_1169, add_1169, 656, 656, 0.125, 1);  rope_2 = rope_3 = view_573 = None
        view_574 = torch.ops.aten.view.default(ascend_flash_attention_15, [-1, 512]);  ascend_flash_attention_15 = None
        addmm_395 = torch.ops.aten.addmm.default(arg1373_1, view_574, arg1372_1);  arg1373_1 = view_574 = arg1372_1 = None
        add_1317 = torch.ops.aten.add.Tensor(addmm_395, softcap_29);  addmm_395 = softcap_29 = None
        softcap_30 = torch.ops.qianchuan_triton.softcap.default(add_1317, 50.0);  add_1317 = None
        native_layer_norm_125 = torch.ops.aten.native_layer_norm.default(softcap_30, [512], arg1374_1, arg1375_1, 1e-06);  arg1374_1 = arg1375_1 = None
        getitem_379 = native_layer_norm_125[0];  native_layer_norm_125 = None
        fused_swiglu_15 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_379, arg1376_1, arg1377_1, arg1378_1, arg1379_1, False, False);  getitem_379 = arg1376_1 = arg1377_1 = arg1378_1 = arg1379_1 = None
        addmm_396 = torch.ops.aten.addmm.default(arg1381_1, fused_swiglu_15, arg1380_1);  arg1381_1 = fused_swiglu_15 = arg1380_1 = None
        add_1339 = torch.ops.aten.add.Tensor(addmm_396, softcap_30);  addmm_396 = softcap_30 = None
        softcap_31 = torch.ops.qianchuan_triton.softcap.default(add_1339, 50.0);  add_1339 = None
        native_layer_norm_126 = torch.ops.aten.native_layer_norm.default(softcap_31, [512], arg1382_1, arg1383_1, 1e-06);  arg1382_1 = arg1383_1 = None
        getitem_382 = native_layer_norm_126[0];  native_layer_norm_126 = None
        addmm_397 = torch.ops.aten.addmm.default(arg1385_1, getitem_382, arg1384_1);  arg1385_1 = arg1384_1 = None
        addmm_398 = torch.ops.aten.addmm.default(arg1387_1, getitem_382, arg1386_1);  arg1387_1 = arg1386_1 = None
        addmm_399 = torch.ops.aten.addmm.default(arg1389_1, getitem_382, arg1388_1);  arg1389_1 = getitem_382 = arg1388_1 = None
        view_575 = torch.ops.aten.view.default(addmm_397, [-1, 8, 64]);  addmm_397 = None
        view_576 = torch.ops.aten.view.default(addmm_398, [-1, 8, 64]);  addmm_398 = None
        view_577 = torch.ops.aten.view.default(addmm_399, [-1, 8, 64]);  addmm_399 = None
        rope_4 = torch.ops.qianchuan_triton.rope.default(view_575, ascend_position_concat_5, add_1169, 656, 10000.0, False);  view_575 = None
        rope_5 = torch.ops.qianchuan_triton.rope.default(view_576, ascend_position_concat_5, add_1169, 656, 10000.0, False);  view_576 = None
        ascend_flash_attention_16 = torch.ops.ascend_triton.ascend_flash_attention.default(rope_4, rope_5, view_577, ascend_seq_tensor_concat_16, ascend_seq_tensor_concat_16, add_1169, add_1169, 656, 656, 0.125, 1);  rope_4 = rope_5 = view_577 = None
        view_578 = torch.ops.aten.view.default(ascend_flash_attention_16, [-1, 512]);  ascend_flash_attention_16 = None
        addmm_400 = torch.ops.aten.addmm.default(arg1391_1, view_578, arg1390_1);  arg1391_1 = view_578 = arg1390_1 = None
        add_1394 = torch.ops.aten.add.Tensor(addmm_400, softcap_31);  addmm_400 = softcap_31 = None
        softcap_32 = torch.ops.qianchuan_triton.softcap.default(add_1394, 50.0);  add_1394 = None
        native_layer_norm_127 = torch.ops.aten.native_layer_norm.default(softcap_32, [512], arg1392_1, arg1393_1, 1e-06);  arg1392_1 = arg1393_1 = None
        getitem_385 = native_layer_norm_127[0];  native_layer_norm_127 = None
        fused_swiglu_16 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_385, arg1394_1, arg1395_1, arg1396_1, arg1397_1, False, False);  getitem_385 = arg1394_1 = arg1395_1 = arg1396_1 = arg1397_1 = None
        addmm_401 = torch.ops.aten.addmm.default(arg1399_1, fused_swiglu_16, arg1398_1);  arg1399_1 = fused_swiglu_16 = arg1398_1 = None
        add_1416 = torch.ops.aten.add.Tensor(addmm_401, softcap_32);  addmm_401 = softcap_32 = None
        softcap_33 = torch.ops.qianchuan_triton.softcap.default(add_1416, 50.0);  add_1416 = None
        native_layer_norm_128 = torch.ops.aten.native_layer_norm.default(softcap_33, [512], arg1400_1, arg1401_1, 1e-06);  arg1400_1 = arg1401_1 = None
        getitem_388 = native_layer_norm_128[0];  native_layer_norm_128 = None
        addmm_402 = torch.ops.aten.addmm.default(arg1403_1, getitem_388, arg1402_1);  arg1403_1 = arg1402_1 = None
        addmm_403 = torch.ops.aten.addmm.default(arg1405_1, getitem_388, arg1404_1);  arg1405_1 = arg1404_1 = None
        addmm_404 = torch.ops.aten.addmm.default(arg1407_1, getitem_388, arg1406_1);  arg1407_1 = getitem_388 = arg1406_1 = None
        view_579 = torch.ops.aten.view.default(addmm_402, [-1, 8, 64]);  addmm_402 = None
        view_580 = torch.ops.aten.view.default(addmm_403, [-1, 8, 64]);  addmm_403 = None
        view_581 = torch.ops.aten.view.default(addmm_404, [-1, 8, 64]);  addmm_404 = None
        rope_6 = torch.ops.qianchuan_triton.rope.default(view_579, ascend_position_concat_5, add_1169, 656, 10000.0, False);  view_579 = None
        rope_7 = torch.ops.qianchuan_triton.rope.default(view_580, ascend_position_concat_5, add_1169, 656, 10000.0, False);  view_580 = None
        ascend_flash_attention_17 = torch.ops.ascend_triton.ascend_flash_attention.default(rope_6, rope_7, view_581, ascend_seq_tensor_concat_16, ascend_seq_tensor_concat_16, add_1169, add_1169, 656, 656, 0.125, 1);  rope_6 = rope_7 = view_581 = None
        view_582 = torch.ops.aten.view.default(ascend_flash_attention_17, [-1, 512]);  ascend_flash_attention_17 = None
        addmm_405 = torch.ops.aten.addmm.default(arg1409_1, view_582, arg1408_1);  arg1409_1 = view_582 = arg1408_1 = None
        add_1471 = torch.ops.aten.add.Tensor(addmm_405, softcap_33);  addmm_405 = softcap_33 = None
        softcap_34 = torch.ops.qianchuan_triton.softcap.default(add_1471, 50.0);  add_1471 = None
        native_layer_norm_129 = torch.ops.aten.native_layer_norm.default(softcap_34, [512], arg1410_1, arg1411_1, 1e-06);  arg1410_1 = arg1411_1 = None
        getitem_391 = native_layer_norm_129[0];  native_layer_norm_129 = None
        fused_swiglu_17 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_391, arg1412_1, arg1413_1, arg1414_1, arg1415_1, False, False);  getitem_391 = arg1412_1 = arg1413_1 = arg1414_1 = arg1415_1 = None
        addmm_406 = torch.ops.aten.addmm.default(arg1417_1, fused_swiglu_17, arg1416_1);  arg1417_1 = fused_swiglu_17 = arg1416_1 = None
        add_1493 = torch.ops.aten.add.Tensor(addmm_406, softcap_34);  addmm_406 = softcap_34 = None
        softcap_35 = torch.ops.qianchuan_triton.softcap.default(add_1493, 50.0);  add_1493 = None
        native_layer_norm_130 = torch.ops.aten.native_layer_norm.default(softcap_35, [512], arg1418_1, arg1419_1, 1e-06);  arg1418_1 = arg1419_1 = None
        getitem_394 = native_layer_norm_130[0];  native_layer_norm_130 = None
        addmm_407 = torch.ops.aten.addmm.default(arg1421_1, getitem_394, arg1420_1);  arg1421_1 = arg1420_1 = None
        addmm_408 = torch.ops.aten.addmm.default(arg1423_1, getitem_394, arg1422_1);  arg1423_1 = arg1422_1 = None
        addmm_409 = torch.ops.aten.addmm.default(arg1425_1, getitem_394, arg1424_1);  arg1425_1 = getitem_394 = arg1424_1 = None
        view_583 = torch.ops.aten.view.default(addmm_407, [-1, 8, 64]);  addmm_407 = None
        view_584 = torch.ops.aten.view.default(addmm_408, [-1, 8, 64]);  addmm_408 = None
        view_585 = torch.ops.aten.view.default(addmm_409, [-1, 8, 64]);  addmm_409 = None
        rope_8 = torch.ops.qianchuan_triton.rope.default(view_583, ascend_position_concat_5, add_1169, 656, 10000.0, False);  view_583 = None
        rope_9 = torch.ops.qianchuan_triton.rope.default(view_584, ascend_position_concat_5, add_1169, 656, 10000.0, False);  view_584 = None
        ascend_flash_attention_18 = torch.ops.ascend_triton.ascend_flash_attention.default(rope_8, rope_9, view_585, ascend_seq_tensor_concat_16, ascend_seq_tensor_concat_16, add_1169, add_1169, 656, 656, 0.125, 1);  rope_8 = rope_9 = view_585 = None
        view_586 = torch.ops.aten.view.default(ascend_flash_attention_18, [-1, 512]);  ascend_flash_attention_18 = None
        addmm_410 = torch.ops.aten.addmm.default(arg1427_1, view_586, arg1426_1);  arg1427_1 = view_586 = arg1426_1 = None
        add_1548 = torch.ops.aten.add.Tensor(addmm_410, softcap_35);  addmm_410 = softcap_35 = None
        softcap_36 = torch.ops.qianchuan_triton.softcap.default(add_1548, 50.0);  add_1548 = None
        native_layer_norm_131 = torch.ops.aten.native_layer_norm.default(softcap_36, [512], arg1428_1, arg1429_1, 1e-06);  arg1428_1 = arg1429_1 = None
        getitem_397 = native_layer_norm_131[0];  native_layer_norm_131 = None
        fused_swiglu_18 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_397, arg1430_1, arg1431_1, arg1432_1, arg1433_1, False, False);  getitem_397 = arg1430_1 = arg1431_1 = arg1432_1 = arg1433_1 = None
        addmm_411 = torch.ops.aten.addmm.default(arg1435_1, fused_swiglu_18, arg1434_1);  arg1435_1 = fused_swiglu_18 = arg1434_1 = None
        add_1570 = torch.ops.aten.add.Tensor(addmm_411, softcap_36);  addmm_411 = softcap_36 = None
        softcap_37 = torch.ops.qianchuan_triton.softcap.default(add_1570, 50.0);  add_1570 = None
        index_67 = torch.ops.aten.index.Tensor(softcap_37, [squeeze_217])
        native_layer_norm_132 = torch.ops.aten.native_layer_norm.default(softcap_37, [512], arg1436_1, arg1437_1, 1e-06);  softcap_37 = arg1436_1 = arg1437_1 = None
        getitem_400 = native_layer_norm_132[0];  native_layer_norm_132 = None
        index_71 = torch.ops.aten.index.Tensor(getitem_400, [squeeze_217])
        full_277 = torch.ops.aten.full.default([sym_size_int_98], False, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False);  sym_size_int_98 = None
        full_default_258 = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        index_put_11 = torch.ops.aten.index_put.default(full_277, [squeeze_217], full_default_258);  full_277 = full_default_258 = None
        convert_element_type_151 = torch.ops.prims.convert_element_type.default(index_put_11, torch.int64);  index_put_11 = None
        cumsum_38 = torch.ops.aten.cumsum.default(convert_element_type_151, 0);  convert_element_type_151 = None
        constant_pad_nd_25 = torch.ops.aten.constant_pad_nd.default(cumsum_38, [1, 0], 0.0);  cumsum_38 = None
        index_72 = torch.ops.aten.index.Tensor(constant_pad_nd_25, [add_1169]);  constant_pad_nd_25 = None
        slice_723 = torch.ops.aten.slice.Tensor(index_72, 0, 1, 9223372036854775807)
        slice_724 = torch.ops.aten.slice.Tensor(index_72, 0, 0, -1)
        sub_534 = torch.ops.aten.sub.Tensor(slice_723, slice_724);  slice_723 = slice_724 = None
        max_12 = torch.ops.aten.max.default(sub_534);  sub_534 = None
        _local_scalar_dense_5 = torch.ops.aten._local_scalar_dense.default(max_12);  max_12 = None
        index_73 = torch.ops.aten.index.Tensor(ascend_position_concat_5, [squeeze_217])
        index_74 = torch.ops.aten.index.Tensor(ascend_seq_tensor_concat_16, [squeeze_217]);  squeeze_217 = None
        addmm_412 = torch.ops.aten.addmm.default(arg1439_1, index_71, arg1438_1);  arg1439_1 = index_71 = arg1438_1 = None
        addmm_413 = torch.ops.aten.addmm.default(arg1441_1, getitem_400, arg1440_1);  arg1441_1 = arg1440_1 = None
        addmm_414 = torch.ops.aten.addmm.default(arg1443_1, getitem_400, arg1442_1);  arg1443_1 = getitem_400 = arg1442_1 = None
        view_587 = torch.ops.aten.view.default(addmm_412, [-1, 16, 32]);  addmm_412 = None
        view_588 = torch.ops.aten.view.default(addmm_413, [-1, 16, 32]);  addmm_413 = None
        view_589 = torch.ops.aten.view.default(addmm_414, [-1, 16, 32]);  addmm_414 = None
        rope_10 = torch.ops.qianchuan_triton.rope.default(view_587, index_73, index_72, _local_scalar_dense_5, 10000.0, False);  view_587 = index_73 = None
        rope_11 = torch.ops.qianchuan_triton.rope.default(view_588, ascend_position_concat_5, add_1169, 656, 10000.0, False);  view_588 = ascend_position_concat_5 = None
        ascend_flash_attention_19 = torch.ops.ascend_triton.ascend_flash_attention.default(rope_10, rope_11, view_589, index_74, ascend_seq_tensor_concat_16, index_72, add_1169, _local_scalar_dense_5, 656, 0.17677669529663687, 1);  rope_10 = rope_11 = view_589 = index_74 = ascend_seq_tensor_concat_16 = index_72 = add_1169 = _local_scalar_dense_5 = None
        view_590 = torch.ops.aten.view.default(ascend_flash_attention_19, [-1, 512]);  ascend_flash_attention_19 = None
        addmm_415 = torch.ops.aten.addmm.default(arg1445_1, view_590, arg1444_1);  arg1445_1 = view_590 = arg1444_1 = None
        add_1628 = torch.ops.aten.add.Tensor(addmm_415, index_67);  addmm_415 = index_67 = None
        softcap_38 = torch.ops.qianchuan_triton.softcap.default(add_1628, 50.0);  add_1628 = None
        native_layer_norm_133 = torch.ops.aten.native_layer_norm.default(softcap_38, [512], arg1446_1, arg1447_1, 1e-06);  arg1446_1 = arg1447_1 = None
        getitem_403 = native_layer_norm_133[0];  native_layer_norm_133 = None
        fused_swiglu_19 = torch.ops.qianchuan_triton.fused_swiglu.default(getitem_403, arg1448_1, arg1449_1, arg1450_1, arg1451_1, False, False);  getitem_403 = arg1448_1 = arg1449_1 = arg1450_1 = arg1451_1 = None
        addmm_416 = torch.ops.aten.addmm.default(arg1453_1, fused_swiglu_19, arg1452_1);  arg1453_1 = fused_swiglu_19 = arg1452_1 = None
        add_1629 = torch.ops.aten.add.Tensor(addmm_416, softcap_38);  addmm_416 = softcap_38 = None
        softcap_39 = torch.ops.qianchuan_triton.softcap.default(add_1629, 50.0);  add_1629 = None
        view_591 = torch.ops.aten.view.default(softcap_39, [batch_size, 2, 512]);  softcap_39 = None
        view_592 = torch.ops.aten.view.default(view_591, [-1, 1024]);  view_591 = None
        full_default_259 = torch.ops.aten.full.default([batch_size, 1024], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_256 = torch.ops.aten.where.self(logical_or_12, full_default_259, view_592);  logical_or_12 = full_default_259 = view_592 = None
        addmm_417 = torch.ops.aten.addmm.default(arg1455_1, where_256, arg1454_1);  arg1455_1 = arg1454_1 = None
        relu_107 = torch.ops.aten.relu.default(addmm_417);  addmm_417 = None
        addmm_418 = torch.ops.aten.addmm.default(arg1457_1, relu_107, arg1456_1);  arg1457_1 = relu_107 = arg1456_1 = None
        squeeze_218 = torch.ops.aten.squeeze.dim(addmm_418, 1);  addmm_418 = None
        addmm_419 = torch.ops.aten.addmm.default(arg1459_1, where_256, arg1458_1);  arg1459_1 = arg1458_1 = None
        relu_108 = torch.ops.aten.relu.default(addmm_419);  addmm_419 = None
        addmm_420 = torch.ops.aten.addmm.default(arg1461_1, relu_108, arg1460_1);  arg1461_1 = relu_108 = arg1460_1 = None
        squeeze_219 = torch.ops.aten.squeeze.dim(addmm_420, 1);  addmm_420 = None
        cat_154 = torch.ops.aten.cat.default([mul_14, mul_16, cat_44, cat_46, addmm_30, addmm_31, addmm_32, cat_52, cat_126, cat_85, cat_88, where_196, where_192, where_200, where_198, where_194, cat_97, cat_116, cat_103, cat_100, where_204, where_206, where_208, where_210, where_212, cat_94, mul_1317, addmm_297, where_239, addmm_294, mul_1286, addmm_301, tanh_8, where_256], 1);  tanh_8 = where_256 = None
        cat_155 = torch.ops.aten.cat.default([cat_154, where_239], 1)
        addmm_421 = torch.ops.aten.addmm.default(arg1463_1, cat_154, arg1462_1);  arg1463_1 = arg1462_1 = None
        relu_109 = torch.ops.aten.relu.default(addmm_421);  addmm_421 = None
        split = torch.ops.aten.split.Tensor(relu_109, 512, 1);  relu_109 = None
        getitem_406 = split[0]
        getitem_407 = split[1]
        getitem_408 = split[2]
        getitem_409 = split[3]
        getitem_410 = split[4];  split = None
        addmm_422 = torch.ops.aten.addmm.default(arg1465_1, getitem_406, arg1464_1);  arg1465_1 = getitem_406 = arg1464_1 = None
        relu_110 = torch.ops.aten.relu.default(addmm_422);  addmm_422 = None
        slice_725 = torch.ops.aten.slice.Tensor(addmm_33, 1, 0, 256)
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(slice_725, -1);  slice_725 = None
        slice_726 = torch.ops.aten.slice.Tensor(addmm_33, 1, 256, 9223372036854775807)
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(arg1466_1, 0);  arg1466_1 = None
        expand_94 = torch.ops.aten.expand.default(unsqueeze_61, [batch_size, -1, -1]);  unsqueeze_61 = None
        add_1630 = torch.ops.aten.add.Tensor(expand_94, unsqueeze_60);  expand_94 = unsqueeze_60 = None
        add_1631 = torch.ops.aten.add.Tensor(slice_726, arg1467_1);  slice_726 = arg1467_1 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(relu_110, 1);  relu_110 = None
        expand_95 = torch.ops.aten.expand.default(unsqueeze_62, [batch_size, 1, 256]);  unsqueeze_62 = None
        expand_96 = torch.ops.aten.expand.default(add_1630, [batch_size, 256, 1]);  add_1630 = None
        bmm_40 = torch.ops.aten.bmm.default(expand_95, expand_96);  expand_95 = expand_96 = None
        squeeze_220 = torch.ops.aten.squeeze.dim(bmm_40, 2);  bmm_40 = None
        add_1632 = torch.ops.aten.add.Tensor(squeeze_220, add_1631);  squeeze_220 = add_1631 = None
        sum_195 = torch.ops.aten.sum.dim_IntList(add_1632, [1]);  add_1632 = None
        addmm_423 = torch.ops.aten.addmm.default(arg1469_1, getitem_407, arg1468_1);  arg1469_1 = getitem_407 = arg1468_1 = None
        relu_111 = torch.ops.aten.relu.default(addmm_423);  addmm_423 = None
        slice_727 = torch.ops.aten.slice.Tensor(addmm_33, 1, 0, 256)
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(slice_727, -1);  slice_727 = None
        slice_728 = torch.ops.aten.slice.Tensor(addmm_33, 1, 256, 9223372036854775807)
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(arg1470_1, 0);  arg1470_1 = None
        expand_97 = torch.ops.aten.expand.default(unsqueeze_64, [batch_size, -1, -1]);  unsqueeze_64 = None
        add_1633 = torch.ops.aten.add.Tensor(expand_97, unsqueeze_63);  expand_97 = unsqueeze_63 = None
        add_1634 = torch.ops.aten.add.Tensor(slice_728, arg1471_1);  slice_728 = arg1471_1 = None
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(relu_111, 1);  relu_111 = None
        expand_98 = torch.ops.aten.expand.default(unsqueeze_65, [batch_size, 1, 256]);  unsqueeze_65 = None
        expand_99 = torch.ops.aten.expand.default(add_1633, [batch_size, 256, 1]);  add_1633 = None
        bmm_41 = torch.ops.aten.bmm.default(expand_98, expand_99);  expand_98 = expand_99 = None
        squeeze_221 = torch.ops.aten.squeeze.dim(bmm_41, 2);  bmm_41 = None
        add_1635 = torch.ops.aten.add.Tensor(squeeze_221, add_1634);  squeeze_221 = add_1634 = None
        sum_196 = torch.ops.aten.sum.dim_IntList(add_1635, [1]);  add_1635 = None
        addmm_424 = torch.ops.aten.addmm.default(arg1473_1, getitem_408, arg1472_1);  arg1473_1 = getitem_408 = arg1472_1 = None
        relu_112 = torch.ops.aten.relu.default(addmm_424);  addmm_424 = None
        slice_729 = torch.ops.aten.slice.Tensor(addmm_33, 1, 0, 256)
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(slice_729, -1);  slice_729 = None
        slice_730 = torch.ops.aten.slice.Tensor(addmm_33, 1, 256, 9223372036854775807)
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(arg1474_1, 0);  arg1474_1 = None
        expand_100 = torch.ops.aten.expand.default(unsqueeze_67, [batch_size, -1, -1]);  unsqueeze_67 = None
        add_1636 = torch.ops.aten.add.Tensor(expand_100, unsqueeze_66);  expand_100 = unsqueeze_66 = None
        add_1637 = torch.ops.aten.add.Tensor(slice_730, arg1475_1);  slice_730 = arg1475_1 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(relu_112, 1);  relu_112 = None
        expand_101 = torch.ops.aten.expand.default(unsqueeze_68, [batch_size, 1, 256]);  unsqueeze_68 = None
        expand_102 = torch.ops.aten.expand.default(add_1636, [batch_size, 256, 1]);  add_1636 = None
        bmm_42 = torch.ops.aten.bmm.default(expand_101, expand_102);  expand_101 = expand_102 = None
        squeeze_222 = torch.ops.aten.squeeze.dim(bmm_42, 2);  bmm_42 = None
        add_1638 = torch.ops.aten.add.Tensor(squeeze_222, add_1637);  squeeze_222 = add_1637 = None
        sum_197 = torch.ops.aten.sum.dim_IntList(add_1638, [1]);  add_1638 = None
        addmm_425 = torch.ops.aten.addmm.default(arg1477_1, getitem_409, arg1476_1);  arg1477_1 = getitem_409 = arg1476_1 = None
        relu_113 = torch.ops.aten.relu.default(addmm_425);  addmm_425 = None
        slice_731 = torch.ops.aten.slice.Tensor(addmm_33, 1, 0, 256)
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(slice_731, -1);  slice_731 = None
        slice_732 = torch.ops.aten.slice.Tensor(addmm_33, 1, 256, 9223372036854775807)
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(arg1478_1, 0);  arg1478_1 = None
        expand_103 = torch.ops.aten.expand.default(unsqueeze_70, [batch_size, -1, -1]);  unsqueeze_70 = None
        add_1639 = torch.ops.aten.add.Tensor(expand_103, unsqueeze_69);  expand_103 = unsqueeze_69 = None
        add_1640 = torch.ops.aten.add.Tensor(slice_732, arg1479_1);  slice_732 = arg1479_1 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(relu_113, 1);  relu_113 = None
        expand_104 = torch.ops.aten.expand.default(unsqueeze_71, [batch_size, 1, 256]);  unsqueeze_71 = None
        expand_105 = torch.ops.aten.expand.default(add_1639, [batch_size, 256, 1]);  add_1639 = None
        bmm_43 = torch.ops.aten.bmm.default(expand_104, expand_105);  expand_104 = expand_105 = None
        squeeze_223 = torch.ops.aten.squeeze.dim(bmm_43, 2);  bmm_43 = None
        add_1641 = torch.ops.aten.add.Tensor(squeeze_223, add_1640);  squeeze_223 = add_1640 = None
        sum_198 = torch.ops.aten.sum.dim_IntList(add_1641, [1]);  add_1641 = None
        addmm_426 = torch.ops.aten.addmm.default(arg1481_1, getitem_410, arg1480_1);  arg1481_1 = getitem_410 = arg1480_1 = None
        relu_114 = torch.ops.aten.relu.default(addmm_426);  addmm_426 = None
        slice_733 = torch.ops.aten.slice.Tensor(addmm_33, 1, 0, 256)
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(slice_733, -1);  slice_733 = None
        slice_734 = torch.ops.aten.slice.Tensor(addmm_33, 1, 256, 9223372036854775807);  addmm_33 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(arg1482_1, 0);  arg1482_1 = None
        expand_106 = torch.ops.aten.expand.default(unsqueeze_73, [batch_size, -1, -1]);  unsqueeze_73 = None
        add_1642 = torch.ops.aten.add.Tensor(expand_106, unsqueeze_72);  expand_106 = unsqueeze_72 = None
        add_1643 = torch.ops.aten.add.Tensor(slice_734, arg1483_1);  slice_734 = arg1483_1 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(relu_114, 1);  relu_114 = None
        expand_107 = torch.ops.aten.expand.default(unsqueeze_74, [batch_size, 1, 256]);  unsqueeze_74 = None
        expand_108 = torch.ops.aten.expand.default(add_1642, [batch_size, 256, 1]);  add_1642 = None
        bmm_44 = torch.ops.aten.bmm.default(expand_107, expand_108);  expand_107 = expand_108 = None
        squeeze_224 = torch.ops.aten.squeeze.dim(bmm_44, 2);  bmm_44 = None
        add_1644 = torch.ops.aten.add.Tensor(squeeze_224, add_1643);  squeeze_224 = add_1643 = None
        sum_199 = torch.ops.aten.sum.dim_IntList(add_1644, [1]);  add_1644 = None
        addmm_427 = torch.ops.aten.addmm.default(arg1485_1, cat_154, arg1484_1);  arg1485_1 = arg1484_1 = None
        split_1 = torch.ops.aten.split.Tensor(addmm_427, 1, 1);  addmm_427 = None
        getitem_411 = split_1[0]
        getitem_412 = split_1[1]
        getitem_413 = split_1[2]
        getitem_414 = split_1[3]
        getitem_415 = split_1[4];  split_1 = None
        sum_206 = torch.ops.aten.sum.dim_IntList(getitem_411, [1]);  getitem_411 = None
        sum_207 = torch.ops.aten.sum.dim_IntList(getitem_412, [1]);  getitem_412 = None
        sum_208 = torch.ops.aten.sum.dim_IntList(getitem_413, [1]);  getitem_413 = None
        sum_209 = torch.ops.aten.sum.dim_IntList(getitem_414, [1]);  getitem_414 = None
        sum_210 = torch.ops.aten.sum.dim_IntList(getitem_415, [1]);  getitem_415 = None
        view_612 = torch.ops.aten.view.default(arg1487_1, [1, batch_size, 1]);  arg1487_1 = None
        sum_211 = torch.ops.aten.sum.dim_IntList(view_612, [0]);  view_612 = None
        view_615 = torch.ops.aten.view.default(arg1488_1, [1, batch_size, 2]);  arg1488_1 = None
        sum_212 = torch.ops.aten.sum.dim_IntList(view_615, [0]);  view_615 = None
        slice_735 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43248, 43249)
        clone_74 = torch.ops.aten.clone.default(slice_735);  slice_735 = None
        slice_736 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42400, 42401)
        clone_75 = torch.ops.aten.clone.default(slice_736);  slice_736 = None
        slice_737 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43912, 43913)
        clone_76 = torch.ops.aten.clone.default(slice_737);  slice_737 = None
        cat_156 = torch.ops.aten.cat.default([sum_212, arg1486_1], 1);  sum_212 = arg1486_1 = None
        addmm_428 = torch.ops.aten.addmm.default(arg1490_1, cat_156, arg1489_1);  arg1490_1 = cat_156 = arg1489_1 = None
        relu_115 = torch.ops.aten.relu.default(addmm_428);  addmm_428 = None
        addmm_429 = torch.ops.aten.addmm.default(arg1492_1, relu_115, arg1491_1);  arg1492_1 = relu_115 = arg1491_1 = None
        relu_116 = torch.ops.aten.relu.default(addmm_429);  addmm_429 = None
        addmm_430 = torch.ops.aten.addmm.default(arg1494_1, relu_116, arg1493_1);  arg1494_1 = relu_116 = arg1493_1 = None
        add_1645 = torch.ops.aten.add.Tensor(clone_74, clone_75);  clone_74 = clone_75 = None
        add_1646 = torch.ops.aten.add.Tensor(add_1645, sum_211);  add_1645 = sum_211 = None
        add_1647 = torch.ops.aten.add.Tensor(add_1646, clone_76);  add_1646 = clone_76 = None
        add_1648 = torch.ops.aten.add.Tensor(add_1647, addmm_430);  add_1647 = addmm_430 = None
        sum_213 = torch.ops.aten.sum.dim_IntList(add_1648, [1]);  add_1648 = None
        squeeze_231 = torch.ops.aten.squeeze.dim(arg4_1, 1)
        eq_1262 = torch.ops.aten.eq.Scalar(squeeze_231, 172);  squeeze_231 = None
        full_default_260 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_257 = torch.ops.aten.where.self(eq_1262, sum_213, full_default_260);  eq_1262 = sum_213 = full_default_260 = None
        view_620 = torch.ops.aten.view.default(arg1496_1, [1, batch_size, 1]);  arg1496_1 = None
        sum_214 = torch.ops.aten.sum.dim_IntList(view_620, [0]);  view_620 = None
        view_623 = torch.ops.aten.view.default(arg1497_1, [1, batch_size, 2]);  arg1497_1 = None
        sum_215 = torch.ops.aten.sum.dim_IntList(view_623, [0]);  view_623 = None
        slice_738 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43248, 43249)
        clone_80 = torch.ops.aten.clone.default(slice_738);  slice_738 = None
        slice_739 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42400, 42401)
        clone_81 = torch.ops.aten.clone.default(slice_739);  slice_739 = None
        slice_740 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43912, 43913)
        clone_82 = torch.ops.aten.clone.default(slice_740);  slice_740 = None
        cat_157 = torch.ops.aten.cat.default([sum_215, arg1495_1], 1);  sum_215 = arg1495_1 = None
        addmm_431 = torch.ops.aten.addmm.default(arg1499_1, cat_157, arg1498_1);  arg1499_1 = cat_157 = arg1498_1 = None
        relu_117 = torch.ops.aten.relu.default(addmm_431);  addmm_431 = None
        addmm_432 = torch.ops.aten.addmm.default(arg1501_1, relu_117, arg1500_1);  arg1501_1 = relu_117 = arg1500_1 = None
        relu_118 = torch.ops.aten.relu.default(addmm_432);  addmm_432 = None
        addmm_433 = torch.ops.aten.addmm.default(arg1503_1, relu_118, arg1502_1);  arg1503_1 = relu_118 = arg1502_1 = None
        add_1649 = torch.ops.aten.add.Tensor(clone_80, clone_81);  clone_80 = clone_81 = None
        add_1650 = torch.ops.aten.add.Tensor(add_1649, sum_214);  add_1649 = sum_214 = None
        add_1651 = torch.ops.aten.add.Tensor(add_1650, clone_82);  add_1650 = clone_82 = None
        add_1652 = torch.ops.aten.add.Tensor(add_1651, addmm_433);  add_1651 = addmm_433 = None
        sum_216 = torch.ops.aten.sum.dim_IntList(add_1652, [1]);  add_1652 = None
        squeeze_238 = torch.ops.aten.squeeze.dim(arg4_1, 1)
        eq_1263 = torch.ops.aten.eq.Scalar(squeeze_238, 169);  squeeze_238 = None
        full_default_261 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_258 = torch.ops.aten.where.self(eq_1263, sum_216, full_default_261);  eq_1263 = sum_216 = full_default_261 = None
        view_628 = torch.ops.aten.view.default(arg1505_1, [1, batch_size, 1]);  arg1505_1 = None
        sum_217 = torch.ops.aten.sum.dim_IntList(view_628, [0]);  view_628 = None
        view_631 = torch.ops.aten.view.default(arg1506_1, [1, batch_size, 2]);  arg1506_1 = None
        sum_218 = torch.ops.aten.sum.dim_IntList(view_631, [0]);  view_631 = None
        slice_741 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43248, 43249)
        clone_86 = torch.ops.aten.clone.default(slice_741);  slice_741 = None
        slice_742 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42400, 42401)
        clone_87 = torch.ops.aten.clone.default(slice_742);  slice_742 = None
        slice_743 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43912, 43913)
        clone_88 = torch.ops.aten.clone.default(slice_743);  slice_743 = None
        cat_158 = torch.ops.aten.cat.default([sum_218, arg1504_1], 1);  sum_218 = arg1504_1 = None
        addmm_434 = torch.ops.aten.addmm.default(arg1508_1, cat_158, arg1507_1);  arg1508_1 = cat_158 = arg1507_1 = None
        relu_119 = torch.ops.aten.relu.default(addmm_434);  addmm_434 = None
        addmm_435 = torch.ops.aten.addmm.default(arg1510_1, relu_119, arg1509_1);  arg1510_1 = relu_119 = arg1509_1 = None
        relu_120 = torch.ops.aten.relu.default(addmm_435);  addmm_435 = None
        addmm_436 = torch.ops.aten.addmm.default(arg1512_1, relu_120, arg1511_1);  arg1512_1 = relu_120 = arg1511_1 = None
        add_1653 = torch.ops.aten.add.Tensor(clone_86, clone_87);  clone_86 = clone_87 = None
        add_1654 = torch.ops.aten.add.Tensor(add_1653, sum_217);  add_1653 = sum_217 = None
        add_1655 = torch.ops.aten.add.Tensor(add_1654, clone_88);  add_1654 = clone_88 = None
        add_1656 = torch.ops.aten.add.Tensor(add_1655, addmm_436);  add_1655 = addmm_436 = None
        sum_219 = torch.ops.aten.sum.dim_IntList(add_1656, [1]);  add_1656 = None
        squeeze_245 = torch.ops.aten.squeeze.dim(arg4_1, 1)
        eq_1264 = torch.ops.aten.eq.Scalar(squeeze_245, 167);  squeeze_245 = None
        full_default_262 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_259 = torch.ops.aten.where.self(eq_1264, sum_219, full_default_262);  eq_1264 = sum_219 = full_default_262 = None
        view_636 = torch.ops.aten.view.default(arg1514_1, [1, batch_size, 1]);  arg1514_1 = None
        sum_220 = torch.ops.aten.sum.dim_IntList(view_636, [0]);  view_636 = None
        view_639 = torch.ops.aten.view.default(arg1515_1, [1, batch_size, 2]);  arg1515_1 = None
        sum_221 = torch.ops.aten.sum.dim_IntList(view_639, [0]);  view_639 = None
        slice_744 = torch.ops.aten.slice.Tensor(arg15_1, 1, 6826, 6827)
        clone_92 = torch.ops.aten.clone.default(slice_744);  slice_744 = None
        slice_745 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42400, 42401)
        clone_93 = torch.ops.aten.clone.default(slice_745);  slice_745 = None
        slice_746 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43912, 43913)
        clone_94 = torch.ops.aten.clone.default(slice_746);  slice_746 = None
        cat_159 = torch.ops.aten.cat.default([sum_221, arg1513_1], 1);  sum_221 = arg1513_1 = None
        addmm_437 = torch.ops.aten.addmm.default(arg1517_1, cat_159, arg1516_1);  arg1517_1 = cat_159 = arg1516_1 = None
        relu_121 = torch.ops.aten.relu.default(addmm_437);  addmm_437 = None
        addmm_438 = torch.ops.aten.addmm.default(arg1519_1, relu_121, arg1518_1);  arg1519_1 = relu_121 = arg1518_1 = None
        relu_122 = torch.ops.aten.relu.default(addmm_438);  addmm_438 = None
        addmm_439 = torch.ops.aten.addmm.default(arg1521_1, relu_122, arg1520_1);  arg1521_1 = relu_122 = arg1520_1 = None
        add_1657 = torch.ops.aten.add.Tensor(clone_92, clone_93);  clone_92 = clone_93 = None
        add_1658 = torch.ops.aten.add.Tensor(add_1657, sum_220);  add_1657 = sum_220 = None
        add_1659 = torch.ops.aten.add.Tensor(add_1658, clone_94);  add_1658 = clone_94 = None
        add_1660 = torch.ops.aten.add.Tensor(add_1659, addmm_439);  add_1659 = addmm_439 = None
        sum_222 = torch.ops.aten.sum.dim_IntList(add_1660, [1]);  add_1660 = None
        squeeze_252 = torch.ops.aten.squeeze.dim(arg4_1, 1)
        eq_1265 = torch.ops.aten.eq.Scalar(squeeze_252, 96);  squeeze_252 = None
        full_default_263 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_260 = torch.ops.aten.where.self(eq_1265, sum_222, full_default_263);  eq_1265 = sum_222 = full_default_263 = None
        view_644 = torch.ops.aten.view.default(arg1523_1, [1, batch_size, 1]);  arg1523_1 = None
        sum_223 = torch.ops.aten.sum.dim_IntList(view_644, [0]);  view_644 = None
        view_647 = torch.ops.aten.view.default(arg1524_1, [1, batch_size, 2]);  arg1524_1 = None
        sum_224 = torch.ops.aten.sum.dim_IntList(view_647, [0]);  view_647 = None
        slice_747 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43248, 43249)
        clone_98 = torch.ops.aten.clone.default(slice_747);  slice_747 = None
        slice_748 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42400, 42401)
        clone_99 = torch.ops.aten.clone.default(slice_748);  slice_748 = None
        slice_749 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43912, 43913)
        clone_100 = torch.ops.aten.clone.default(slice_749);  slice_749 = None
        cat_160 = torch.ops.aten.cat.default([sum_224, arg1522_1], 1);  sum_224 = arg1522_1 = None
        addmm_440 = torch.ops.aten.addmm.default(arg1526_1, cat_160, arg1525_1);  arg1526_1 = cat_160 = arg1525_1 = None
        relu_123 = torch.ops.aten.relu.default(addmm_440);  addmm_440 = None
        addmm_441 = torch.ops.aten.addmm.default(arg1528_1, relu_123, arg1527_1);  arg1528_1 = relu_123 = arg1527_1 = None
        relu_124 = torch.ops.aten.relu.default(addmm_441);  addmm_441 = None
        addmm_442 = torch.ops.aten.addmm.default(arg1530_1, relu_124, arg1529_1);  arg1530_1 = relu_124 = arg1529_1 = None
        add_1661 = torch.ops.aten.add.Tensor(clone_98, clone_99);  clone_98 = clone_99 = None
        add_1662 = torch.ops.aten.add.Tensor(add_1661, sum_223);  add_1661 = sum_223 = None
        add_1663 = torch.ops.aten.add.Tensor(add_1662, clone_100);  add_1662 = clone_100 = None
        add_1664 = torch.ops.aten.add.Tensor(add_1663, addmm_442);  add_1663 = addmm_442 = None
        sum_225 = torch.ops.aten.sum.dim_IntList(add_1664, [1]);  add_1664 = None
        squeeze_259 = torch.ops.aten.squeeze.dim(arg4_1, 1)
        eq_1266 = torch.ops.aten.eq.Scalar(squeeze_259, 412);  squeeze_259 = None
        full_default_264 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_261 = torch.ops.aten.where.self(eq_1266, sum_225, full_default_264);  eq_1266 = sum_225 = full_default_264 = None
        view_652 = torch.ops.aten.view.default(arg1532_1, [1, batch_size, 1]);  arg1532_1 = None
        sum_226 = torch.ops.aten.sum.dim_IntList(view_652, [0]);  view_652 = None
        view_655 = torch.ops.aten.view.default(arg1533_1, [1, batch_size, 2]);  arg1533_1 = None
        sum_227 = torch.ops.aten.sum.dim_IntList(view_655, [0]);  view_655 = None
        slice_750 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43248, 43249)
        clone_104 = torch.ops.aten.clone.default(slice_750);  slice_750 = None
        slice_751 = torch.ops.aten.slice.Tensor(arg15_1, 1, 42400, 42401)
        clone_105 = torch.ops.aten.clone.default(slice_751);  slice_751 = None
        slice_752 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43912, 43913)
        clone_106 = torch.ops.aten.clone.default(slice_752);  slice_752 = None
        cat_161 = torch.ops.aten.cat.default([sum_227, arg1531_1], 1);  sum_227 = arg1531_1 = None
        addmm_443 = torch.ops.aten.addmm.default(arg1535_1, cat_161, arg1534_1);  arg1535_1 = cat_161 = arg1534_1 = None
        relu_125 = torch.ops.aten.relu.default(addmm_443);  addmm_443 = None
        addmm_444 = torch.ops.aten.addmm.default(arg1537_1, relu_125, arg1536_1);  arg1537_1 = relu_125 = arg1536_1 = None
        relu_126 = torch.ops.aten.relu.default(addmm_444);  addmm_444 = None
        addmm_445 = torch.ops.aten.addmm.default(arg1539_1, relu_126, arg1538_1);  arg1539_1 = relu_126 = arg1538_1 = None
        add_1665 = torch.ops.aten.add.Tensor(clone_104, clone_105);  clone_104 = clone_105 = None
        add_1666 = torch.ops.aten.add.Tensor(add_1665, sum_226);  add_1665 = sum_226 = None
        add_1667 = torch.ops.aten.add.Tensor(add_1666, clone_106);  add_1666 = clone_106 = None
        add_1668 = torch.ops.aten.add.Tensor(add_1667, addmm_445);  add_1667 = addmm_445 = None
        sum_228 = torch.ops.aten.sum.dim_IntList(add_1668, [1]);  add_1668 = None
        squeeze_266 = torch.ops.aten.squeeze.dim(arg4_1, 1)
        eq_1267 = torch.ops.aten.eq.Scalar(squeeze_266, 412);  squeeze_266 = None
        squeeze_267 = torch.ops.aten.squeeze.dim(eq_4, 1);  eq_4 = None
        logical_and_5 = torch.ops.aten.logical_and.default(eq_1267, squeeze_267);  eq_1267 = squeeze_267 = None
        where_262 = torch.ops.aten.where.self(logical_and_5, sum_228, where_261);  logical_and_5 = sum_228 = where_261 = None
        squeeze_268 = torch.ops.aten.squeeze.default(logical_not)
        view_656 = torch.ops.aten.view.default(squeeze_268, [-1, 1]);  squeeze_268 = None
        full_default_265 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_263 = torch.ops.aten.where.self(view_656, full_default_265, arg1540_1);  view_656 = full_default_265 = arg1540_1 = None
        cat_162 = torch.ops.aten.cat.default([arg1541_1, where_263, arg1542_1], -1);  arg1541_1 = where_263 = arg1542_1 = None
        squeeze_270 = torch.ops.aten.squeeze.default(logical_not)
        view_658 = torch.ops.aten.view.default(squeeze_270, [-1, 1]);  squeeze_270 = None
        full_default_266 = torch.ops.aten.full.default([batch_size, 40], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_264 = torch.ops.aten.where.self(view_658, full_default_266, arg1543_1);  view_658 = full_default_266 = arg1543_1 = None
        cat_163 = torch.ops.aten.cat.default([arg1544_1, where_264], -1);  arg1544_1 = where_264 = None
        cat_164 = torch.ops.aten.cat.default([cat_162, cat_163], -1);  cat_162 = cat_163 = None
        addmm_446 = torch.ops.aten.addmm.default(arg1546_1, cat_164, arg1545_1);  arg1546_1 = cat_164 = arg1545_1 = None
        relu_127 = torch.ops.aten.relu.default(addmm_446);  addmm_446 = None
        addmm_447 = torch.ops.aten.addmm.default(arg1548_1, relu_127, arg1547_1);  arg1548_1 = relu_127 = arg1547_1 = None
        sub_540 = torch.ops.aten.sub.Tensor(0.0, addmm_447);  addmm_447 = None
        exp = torch.ops.aten.exp.default(sub_540);  sub_540 = None
        add_1669 = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        log = torch.ops.aten.log.default(add_1669);  add_1669 = None
        neg = torch.ops.aten.neg.default(log);  log = None
        eq_1268 = torch.ops.aten.eq.Scalar(where_8, 1)
        full_default_267 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_265 = torch.ops.aten.where.self(eq_1268, neg, full_default_267);  eq_1268 = full_default_267 = None
        squeeze_271 = torch.ops.aten.squeeze.dim(where_265, 1);  where_265 = None
        squeeze_272 = torch.ops.aten.squeeze.dim(neg, 1);  neg = None
        squeeze_273 = torch.ops.aten.squeeze.default(logical_not)
        view_659 = torch.ops.aten.view.default(squeeze_273, [-1, 1]);  squeeze_273 = None
        full_default_268 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_266 = torch.ops.aten.where.self(view_659, full_default_268, arg1549_1);  view_659 = full_default_268 = arg1549_1 = None
        cat_165 = torch.ops.aten.cat.default([arg1550_1, where_266, arg1551_1], -1);  arg1550_1 = where_266 = arg1551_1 = None
        squeeze_275 = torch.ops.aten.squeeze.default(logical_not)
        view_661 = torch.ops.aten.view.default(squeeze_275, [-1, 1]);  squeeze_275 = None
        full_default_269 = torch.ops.aten.full.default([batch_size, 40], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_267 = torch.ops.aten.where.self(view_661, full_default_269, arg1552_1);  view_661 = full_default_269 = arg1552_1 = None
        cat_166 = torch.ops.aten.cat.default([arg1553_1, where_267], -1);  arg1553_1 = where_267 = None
        cat_167 = torch.ops.aten.cat.default([cat_165, cat_166], -1);  cat_165 = cat_166 = None
        addmm_448 = torch.ops.aten.addmm.default(arg1555_1, cat_167, arg1554_1);  arg1555_1 = cat_167 = arg1554_1 = None
        relu_128 = torch.ops.aten.relu.default(addmm_448);  addmm_448 = None
        addmm_449 = torch.ops.aten.addmm.default(arg1557_1, relu_128, arg1556_1);  arg1557_1 = relu_128 = arg1556_1 = None
        sub_541 = torch.ops.aten.sub.Tensor(0.0, addmm_449);  addmm_449 = None
        exp_1 = torch.ops.aten.exp.default(sub_541);  sub_541 = None
        add_1670 = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        log_1 = torch.ops.aten.log.default(add_1670);  add_1670 = None
        neg_1 = torch.ops.aten.neg.default(log_1);  log_1 = None
        eq_1269 = torch.ops.aten.eq.Scalar(where_8, 1)
        full_default_270 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_268 = torch.ops.aten.where.self(eq_1269, neg_1, full_default_270);  eq_1269 = full_default_270 = None
        squeeze_276 = torch.ops.aten.squeeze.dim(where_268, 1);  where_268 = None
        squeeze_277 = torch.ops.aten.squeeze.dim(neg_1, 1);  neg_1 = None
        squeeze_278 = torch.ops.aten.squeeze.default(logical_not)
        view_662 = torch.ops.aten.view.default(squeeze_278, [-1, 1]);  squeeze_278 = None
        full_default_271 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_269 = torch.ops.aten.where.self(view_662, full_default_271, arg1558_1);  view_662 = full_default_271 = arg1558_1 = None
        cat_168 = torch.ops.aten.cat.default([arg1559_1, where_269, arg1560_1], -1);  arg1559_1 = where_269 = arg1560_1 = None
        squeeze_280 = torch.ops.aten.squeeze.default(logical_not)
        view_664 = torch.ops.aten.view.default(squeeze_280, [-1, 1]);  squeeze_280 = None
        full_default_272 = torch.ops.aten.full.default([batch_size, 40], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_270 = torch.ops.aten.where.self(view_664, full_default_272, arg1561_1);  view_664 = full_default_272 = arg1561_1 = None
        cat_169 = torch.ops.aten.cat.default([arg1562_1, where_270], -1);  arg1562_1 = where_270 = None
        cat_170 = torch.ops.aten.cat.default([cat_168, cat_169], -1);  cat_168 = cat_169 = None
        addmm_450 = torch.ops.aten.addmm.default(arg1564_1, cat_170, arg1563_1);  arg1564_1 = cat_170 = arg1563_1 = None
        relu_129 = torch.ops.aten.relu.default(addmm_450);  addmm_450 = None
        addmm_451 = torch.ops.aten.addmm.default(arg1566_1, relu_129, arg1565_1);  arg1566_1 = relu_129 = arg1565_1 = None
        sub_542 = torch.ops.aten.sub.Tensor(0.0, addmm_451);  addmm_451 = None
        exp_2 = torch.ops.aten.exp.default(sub_542);  sub_542 = None
        add_1671 = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        log_2 = torch.ops.aten.log.default(add_1671);  add_1671 = None
        neg_2 = torch.ops.aten.neg.default(log_2);  log_2 = None
        eq_1270 = torch.ops.aten.eq.Scalar(where_8, 1)
        full_default_273 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_271 = torch.ops.aten.where.self(eq_1270, neg_2, full_default_273);  eq_1270 = full_default_273 = None
        squeeze_281 = torch.ops.aten.squeeze.dim(where_271, 1);  where_271 = None
        squeeze_282 = torch.ops.aten.squeeze.dim(neg_2, 1);  neg_2 = None
        squeeze_283 = torch.ops.aten.squeeze.default(logical_not)
        view_665 = torch.ops.aten.view.default(squeeze_283, [-1, 1]);  squeeze_283 = None
        full_default_274 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_272 = torch.ops.aten.where.self(view_665, full_default_274, arg1567_1);  view_665 = full_default_274 = arg1567_1 = None
        cat_171 = torch.ops.aten.cat.default([arg1568_1, where_272, arg1569_1], -1);  arg1568_1 = where_272 = arg1569_1 = None
        squeeze_285 = torch.ops.aten.squeeze.default(logical_not)
        view_667 = torch.ops.aten.view.default(squeeze_285, [-1, 1]);  squeeze_285 = None
        full_default_275 = torch.ops.aten.full.default([batch_size, 40], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_273 = torch.ops.aten.where.self(view_667, full_default_275, arg1570_1);  view_667 = full_default_275 = arg1570_1 = None
        cat_172 = torch.ops.aten.cat.default([arg1571_1, where_273], -1);  arg1571_1 = where_273 = None
        cat_173 = torch.ops.aten.cat.default([cat_171, cat_172], -1);  cat_171 = cat_172 = None
        addmm_452 = torch.ops.aten.addmm.default(arg1573_1, cat_173, arg1572_1);  arg1573_1 = cat_173 = arg1572_1 = None
        relu_130 = torch.ops.aten.relu.default(addmm_452);  addmm_452 = None
        addmm_453 = torch.ops.aten.addmm.default(arg1575_1, relu_130, arg1574_1);  arg1575_1 = relu_130 = arg1574_1 = None
        sub_543 = torch.ops.aten.sub.Tensor(0.0, addmm_453);  addmm_453 = None
        exp_3 = torch.ops.aten.exp.default(sub_543);  sub_543 = None
        add_1672 = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        log_3 = torch.ops.aten.log.default(add_1672);  add_1672 = None
        neg_3 = torch.ops.aten.neg.default(log_3);  log_3 = None
        eq_1271 = torch.ops.aten.eq.Scalar(where_8, 1)
        full_default_276 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_274 = torch.ops.aten.where.self(eq_1271, neg_3, full_default_276);  eq_1271 = full_default_276 = None
        squeeze_286 = torch.ops.aten.squeeze.dim(where_274, 1);  where_274 = None
        squeeze_287 = torch.ops.aten.squeeze.dim(neg_3, 1);  neg_3 = None
        squeeze_288 = torch.ops.aten.squeeze.default(logical_not)
        view_668 = torch.ops.aten.view.default(squeeze_288, [-1, 1]);  squeeze_288 = None
        slice_753 = torch.ops.aten.slice.Tensor(arg15_1, 1, 910, 918)
        full_default_277 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_275 = torch.ops.aten.where.self(view_668, full_default_277, arg1576_1);  view_668 = full_default_277 = arg1576_1 = None
        cat_174 = torch.ops.aten.cat.default([arg1577_1, where_275, arg1578_1], -1);  arg1577_1 = where_275 = arg1578_1 = None
        squeeze_290 = torch.ops.aten.squeeze.default(logical_not)
        view_670 = torch.ops.aten.view.default(squeeze_290, [-1, 1]);  squeeze_290 = None
        full_default_278 = torch.ops.aten.full.default([batch_size, 40], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_276 = torch.ops.aten.where.self(view_670, full_default_278, arg1579_1);  view_670 = full_default_278 = arg1579_1 = None
        cat_175 = torch.ops.aten.cat.default([arg1580_1, where_276], -1);  arg1580_1 = where_276 = None
        cat_176 = torch.ops.aten.cat.default([cat_174, cat_175], -1);  cat_174 = cat_175 = None
        addmm_454 = torch.ops.aten.addmm.default(arg1582_1, cat_176, arg1581_1);  arg1582_1 = cat_176 = arg1581_1 = None
        relu_131 = torch.ops.aten.relu.default(addmm_454);  addmm_454 = None
        addmm_455 = torch.ops.aten.addmm.default(arg1584_1, relu_131, arg1583_1);  arg1584_1 = relu_131 = arg1583_1 = None
        sub_544 = torch.ops.aten.sub.Tensor(0.0, addmm_455);  addmm_455 = None
        exp_4 = torch.ops.aten.exp.default(sub_544);  sub_544 = None
        add_1673 = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        log_4 = torch.ops.aten.log.default(add_1673);  add_1673 = None
        neg_4 = torch.ops.aten.neg.default(log_4);  log_4 = None
        eq_1272 = torch.ops.aten.eq.Scalar(where_8, 1);  where_8 = None
        full_default_279 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_277 = torch.ops.aten.where.self(eq_1272, neg_4, full_default_279);  eq_1272 = full_default_279 = None
        squeeze_291 = torch.ops.aten.squeeze.dim(where_277, 1);  where_277 = None
        squeeze_292 = torch.ops.aten.squeeze.dim(neg_4, 1);  neg_4 = None
        full_default_280 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        eq_1273 = torch.ops.aten.eq.Scalar(where_7, 1)
        full_default_281 = torch.ops.aten.full.default([batch_size, 1], 1, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_278 = torch.ops.aten.where.self(eq_1273, full_default_281, full_default_280);  eq_1273 = full_default_281 = full_default_280 = None
        eq_1274 = torch.ops.aten.eq.Scalar(where_7, 2)
        full_default_282 = torch.ops.aten.full.default([batch_size, 1], 2, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_279 = torch.ops.aten.where.self(eq_1274, full_default_282, where_278);  eq_1274 = full_default_282 = where_278 = None
        eq_1275 = torch.ops.aten.eq.Scalar(where_7, 3)
        full_default_283 = torch.ops.aten.full.default([batch_size, 1], 3, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_280 = torch.ops.aten.where.self(eq_1275, full_default_283, where_279);  eq_1275 = full_default_283 = where_279 = None
        eq_1276 = torch.ops.aten.eq.Scalar(where_7, 9998)
        full_default_284 = torch.ops.aten.full.default([batch_size, 1], 4, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_281 = torch.ops.aten.where.self(eq_1276, full_default_284, where_280);  eq_1276 = full_default_284 = where_280 = None
        eq_1277 = torch.ops.aten.eq.Scalar(where_7, 9999)
        full_default_285 = torch.ops.aten.full.default([batch_size, 1], 5, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_282 = torch.ops.aten.where.self(eq_1277, full_default_285, where_281);  eq_1277 = full_default_285 = where_281 = None
        embedding_3 = torch.ops.aten.embedding.default(arg1585_1, where_282);  arg1585_1 = where_282 = None
        squeeze_293 = torch.ops.aten.squeeze.dim(embedding_3, 1);  embedding_3 = None
        slice_754 = torch.ops.aten.slice.Tensor(arg15_1, 1, 918, 934)
        squeeze_299 = torch.ops.aten.squeeze.default(logical_not)
        view_676 = torch.ops.aten.view.default(squeeze_299, [-1, 1]);  squeeze_299 = None
        full_default_286 = torch.ops.aten.full.default([batch_size, 16], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_283 = torch.ops.aten.where.self(view_676, full_default_286, arg1589_1);  view_676 = full_default_286 = arg1589_1 = None
        cat_177 = torch.ops.aten.cat.default([where_283, arg1590_1], -1);  where_283 = arg1590_1 = None
        squeeze_301 = torch.ops.aten.squeeze.default(logical_not)
        view_678 = torch.ops.aten.view.default(squeeze_301, [-1, 1]);  squeeze_301 = None
        full_default_287 = torch.ops.aten.full.default([batch_size, 80], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_284 = torch.ops.aten.where.self(view_678, full_default_287, arg1591_1);  view_678 = full_default_287 = arg1591_1 = None
        cat_178 = torch.ops.aten.cat.default([arg1586_1, squeeze_293, arg1587_1, arg1588_1], 1);  arg1587_1 = arg1588_1 = None
        cat_179 = torch.ops.aten.cat.default([arg1586_1, squeeze_293, cat_177, where_284], 1);  arg1586_1 = squeeze_293 = cat_177 = where_284 = None
        addmm_456 = torch.ops.aten.addmm.default(arg1593_1, cat_178, arg1592_1);  arg1593_1 = cat_178 = arg1592_1 = None
        addmm_457 = torch.ops.aten.addmm.default(arg1595_1, cat_179, arg1594_1);  arg1595_1 = cat_179 = arg1594_1 = None
        eq_1278 = torch.ops.aten.eq.Scalar(where_7, 9998)
        eq_1279 = torch.ops.aten.eq.Scalar(where_7, 3);  where_7 = None
        logical_or_35 = torch.ops.aten.logical_or.default(eq_1278, eq_1279);  eq_1278 = eq_1279 = None
        repeat_44 = torch.ops.aten.repeat.default(logical_or_35, [1, 256]);  logical_or_35 = None
        where_285 = torch.ops.aten.where.self(repeat_44, addmm_457, addmm_456);  repeat_44 = addmm_457 = addmm_456 = None
        addmm_458 = torch.ops.aten.addmm.default(arg1597_1, where_285, arg1596_1);  arg1597_1 = arg1596_1 = None
        relu_132 = torch.ops.aten.relu.default(addmm_458);  addmm_458 = None
        addmm_459 = torch.ops.aten.addmm.default(arg1599_1, relu_132, arg1598_1);  arg1599_1 = relu_132 = arg1598_1 = None
        relu_133 = torch.ops.aten.relu.default(addmm_459);  addmm_459 = None
        addmm_460 = torch.ops.aten.addmm.default(arg1601_1, relu_133, arg1600_1);  arg1601_1 = relu_133 = arg1600_1 = None
        relu_134 = torch.ops.aten.relu.default(addmm_460);  addmm_460 = None
        addmm_461 = torch.ops.aten.addmm.default(arg1603_1, relu_134, arg1602_1);  arg1603_1 = relu_134 = arg1602_1 = None
        addmm_462 = torch.ops.aten.addmm.default(arg1605_1, where_285, arg1604_1);  arg1605_1 = arg1604_1 = None
        relu_135 = torch.ops.aten.relu.default(addmm_462);  addmm_462 = None
        addmm_463 = torch.ops.aten.addmm.default(arg1607_1, relu_135, arg1606_1);  arg1607_1 = relu_135 = arg1606_1 = None
        relu_136 = torch.ops.aten.relu.default(addmm_463);  addmm_463 = None
        addmm_464 = torch.ops.aten.addmm.default(arg1609_1, relu_136, arg1608_1);  arg1609_1 = relu_136 = arg1608_1 = None
        relu_137 = torch.ops.aten.relu.default(addmm_464);  addmm_464 = None
        addmm_465 = torch.ops.aten.addmm.default(arg1611_1, relu_137, arg1610_1);  arg1611_1 = relu_137 = arg1610_1 = None
        addmm_466 = torch.ops.aten.addmm.default(arg1613_1, where_285, arg1612_1);  arg1613_1 = arg1612_1 = None
        relu_138 = torch.ops.aten.relu.default(addmm_466);  addmm_466 = None
        addmm_467 = torch.ops.aten.addmm.default(arg1615_1, relu_138, arg1614_1);  arg1615_1 = relu_138 = arg1614_1 = None
        relu_139 = torch.ops.aten.relu.default(addmm_467);  addmm_467 = None
        addmm_468 = torch.ops.aten.addmm.default(arg1617_1, relu_139, arg1616_1);  arg1617_1 = relu_139 = arg1616_1 = None
        relu_140 = torch.ops.aten.relu.default(addmm_468);  addmm_468 = None
        addmm_469 = torch.ops.aten.addmm.default(arg1619_1, relu_140, arg1618_1);  arg1619_1 = relu_140 = arg1618_1 = None
        addmm_470 = torch.ops.aten.addmm.default(arg1621_1, where_285, arg1620_1);  arg1621_1 = arg1620_1 = None
        relu_141 = torch.ops.aten.relu.default(addmm_470);  addmm_470 = None
        addmm_471 = torch.ops.aten.addmm.default(arg1623_1, relu_141, arg1622_1);  arg1623_1 = relu_141 = arg1622_1 = None
        relu_142 = torch.ops.aten.relu.default(addmm_471);  addmm_471 = None
        addmm_472 = torch.ops.aten.addmm.default(arg1625_1, relu_142, arg1624_1);  arg1625_1 = relu_142 = arg1624_1 = None
        relu_143 = torch.ops.aten.relu.default(addmm_472);  addmm_472 = None
        addmm_473 = torch.ops.aten.addmm.default(arg1627_1, relu_143, arg1626_1);  arg1627_1 = relu_143 = arg1626_1 = None
        addmm_474 = torch.ops.aten.addmm.default(arg1629_1, where_285, arg1628_1);  arg1629_1 = where_285 = arg1628_1 = None
        relu_144 = torch.ops.aten.relu.default(addmm_474);  addmm_474 = None
        addmm_475 = torch.ops.aten.addmm.default(arg1631_1, relu_144, arg1630_1);  arg1631_1 = relu_144 = arg1630_1 = None
        relu_145 = torch.ops.aten.relu.default(addmm_475);  addmm_475 = None
        addmm_476 = torch.ops.aten.addmm.default(arg1633_1, relu_145, arg1632_1);  arg1633_1 = relu_145 = arg1632_1 = None
        relu_146 = torch.ops.aten.relu.default(addmm_476);  addmm_476 = None
        addmm_477 = torch.ops.aten.addmm.default(arg1635_1, relu_146, arg1634_1);  arg1635_1 = relu_146 = arg1634_1 = None
        cat_180 = torch.ops.aten.cat.default([sum_4, sum_7, sum_10, sum_13, sum_16, sum_19, sum_22, sum_25, sum_28, sum_31, sum_34, sum_166])
        view_679 = torch.ops.aten.view.default(cat_180, [12, batch_size]);  cat_180 = None
        sum_229 = torch.ops.aten.sum.dim_IntList(view_679, [0]);  view_679 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(sum_229, 1);  sum_229 = None
        sigmoid_24 = torch.ops.aten.sigmoid.default(addmm_461)
        mul_1987 = torch.ops.aten.mul.Tensor(sigmoid_24, 2.0);  sigmoid_24 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(add_1097, 1)
        cat_181 = torch.ops.aten.cat.default([unsqueeze_75, unsqueeze_77], 1);  unsqueeze_75 = unsqueeze_77 = None
        mul_1988 = torch.ops.aten.mul.Tensor(cat_181, mul_1987);  cat_181 = mul_1987 = None
        sum_230 = torch.ops.aten.sum.dim_IntList(mul_1988, [1]);  mul_1988 = None
        cat_182 = torch.ops.aten.cat.default([sum_195, sum_230, squeeze_271, where_260]);  sum_230 = None
        view_680 = torch.ops.aten.view.default(cat_182, [4, batch_size]);  cat_182 = None
        sum_231 = torch.ops.aten.sum.dim_IntList(view_680, [0]);  view_680 = None
        convert_element_type_174 = torch.ops.prims.convert_element_type.default(sum_231, torch.float32);  sum_231 = None
        clamp_min_11 = torch.ops.aten.clamp_min.default(convert_element_type_174, -15);  convert_element_type_174 = None
        clamp_max_24 = torch.ops.aten.clamp_max.default(clamp_min_11, 15);  clamp_min_11 = None
        convert_element_type_175 = torch.ops.prims.convert_element_type.default(clamp_max_24, torch.float16);  clamp_max_24 = None
        full_default_288 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        isnan = torch.ops.aten.isnan.default(convert_element_type_175)
        where_286 = torch.ops.aten.where.self(isnan, full_default_288, convert_element_type_175);  isnan = full_default_288 = convert_element_type_175 = None
        full_default_289 = torch.ops.aten.full.default([], -2.197265625, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        add_1674 = torch.ops.aten.add.Tensor(where_286, full_default_289);  where_286 = full_default_289 = None
        sub_546 = torch.ops.aten.sub.Tensor(squeeze_272, squeeze_271)
        add_1675 = torch.ops.aten.add.Tensor(add_1674, sub_546);  sub_546 = None
        eq_1280 = torch.ops.aten.eq.Tensor(squeeze_272, squeeze_271)
        isinf = torch.ops.aten.isinf.default(squeeze_272)
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(isinf, eq_1280);  isinf = None
        full_default_290 = torch.ops.aten.full.default([batch_size], -11.0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        le_7 = torch.ops.aten.le.Tensor(squeeze_272, full_default_290);  full_default_290 = None
        bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(eq_1280, le_7);  eq_1280 = le_7 = None
        bitwise_or_10 = torch.ops.aten.bitwise_or.Tensor(bitwise_and, bitwise_and_1);  bitwise_and = bitwise_and_1 = None
        full_default_291 = torch.ops.aten.full.default([batch_size], nan, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        convert_element_type_176 = torch.ops.prims.convert_element_type.default(add_1675, torch.float32);  add_1675 = None
        clamp_min_12 = torch.ops.aten.clamp_min.default(convert_element_type_176, -15);  convert_element_type_176 = None
        clamp_max_25 = torch.ops.aten.clamp_max.default(clamp_min_12, 15);  clamp_min_12 = None
        convert_element_type_177 = torch.ops.prims.convert_element_type.default(clamp_max_25, torch.float16);  clamp_max_25 = None
        where_287 = torch.ops.aten.where.self(bitwise_or_10, full_default_291, convert_element_type_177);  bitwise_or_10 = full_default_291 = convert_element_type_177 = None
        cat_183 = torch.ops.aten.cat.default([sum_4, sum_7, sum_10, sum_13, sum_16, sum_19, sum_22, sum_25, sum_28, sum_31, sum_34, sum_166])
        view_681 = torch.ops.aten.view.default(cat_183, [12, batch_size]);  cat_183 = None
        sum_232 = torch.ops.aten.sum.dim_IntList(view_681, [0]);  view_681 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(sum_232, 1);  sum_232 = None
        sigmoid_25 = torch.ops.aten.sigmoid.default(addmm_465)
        mul_1990 = torch.ops.aten.mul.Tensor(sigmoid_25, 2.0);  sigmoid_25 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(add_1097, 1)
        cat_184 = torch.ops.aten.cat.default([unsqueeze_78, unsqueeze_80], 1);  unsqueeze_78 = unsqueeze_80 = None
        mul_1991 = torch.ops.aten.mul.Tensor(cat_184, mul_1990);  cat_184 = mul_1990 = None
        sum_233 = torch.ops.aten.sum.dim_IntList(mul_1991, [1]);  mul_1991 = None
        cat_185 = torch.ops.aten.cat.default([sum_196, sum_233, squeeze_276, where_259]);  sum_233 = None
        view_682 = torch.ops.aten.view.default(cat_185, [4, batch_size]);  cat_185 = None
        sum_234 = torch.ops.aten.sum.dim_IntList(view_682, [0]);  view_682 = None
        convert_element_type_178 = torch.ops.prims.convert_element_type.default(sum_234, torch.float32);  sum_234 = None
        clamp_min_13 = torch.ops.aten.clamp_min.default(convert_element_type_178, -15);  convert_element_type_178 = None
        clamp_max_26 = torch.ops.aten.clamp_max.default(clamp_min_13, 15);  clamp_min_13 = None
        convert_element_type_179 = torch.ops.prims.convert_element_type.default(clamp_max_26, torch.float16);  clamp_max_26 = None
        full_default_292 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        isnan_1 = torch.ops.aten.isnan.default(convert_element_type_179)
        where_288 = torch.ops.aten.where.self(isnan_1, full_default_292, convert_element_type_179);  isnan_1 = full_default_292 = convert_element_type_179 = None
        full_default_293 = torch.ops.aten.full.default([], -2.197265625, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        add_1676 = torch.ops.aten.add.Tensor(where_288, full_default_293);  where_288 = full_default_293 = None
        sub_548 = torch.ops.aten.sub.Tensor(squeeze_277, squeeze_276)
        add_1677 = torch.ops.aten.add.Tensor(add_1676, sub_548);  sub_548 = None
        eq_1281 = torch.ops.aten.eq.Tensor(squeeze_277, squeeze_276)
        isinf_1 = torch.ops.aten.isinf.default(squeeze_277)
        bitwise_and_2 = torch.ops.aten.bitwise_and.Tensor(isinf_1, eq_1281);  isinf_1 = None
        full_default_294 = torch.ops.aten.full.default([batch_size], -11.0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        le_8 = torch.ops.aten.le.Tensor(squeeze_277, full_default_294);  full_default_294 = None
        bitwise_and_3 = torch.ops.aten.bitwise_and.Tensor(eq_1281, le_8);  eq_1281 = le_8 = None
        bitwise_or_11 = torch.ops.aten.bitwise_or.Tensor(bitwise_and_2, bitwise_and_3);  bitwise_and_2 = bitwise_and_3 = None
        full_default_295 = torch.ops.aten.full.default([batch_size], nan, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        convert_element_type_180 = torch.ops.prims.convert_element_type.default(add_1677, torch.float32);  add_1677 = None
        clamp_min_14 = torch.ops.aten.clamp_min.default(convert_element_type_180, -15);  convert_element_type_180 = None
        clamp_max_27 = torch.ops.aten.clamp_max.default(clamp_min_14, 15);  clamp_min_14 = None
        convert_element_type_181 = torch.ops.prims.convert_element_type.default(clamp_max_27, torch.float16);  clamp_max_27 = None
        where_289 = torch.ops.aten.where.self(bitwise_or_11, full_default_295, convert_element_type_181);  bitwise_or_11 = full_default_295 = convert_element_type_181 = None
        cat_186 = torch.ops.aten.cat.default([sum_4, sum_7, sum_10, sum_13, sum_16, sum_19, sum_22, sum_25, sum_28, sum_31, sum_34, sum_166])
        view_683 = torch.ops.aten.view.default(cat_186, [12, batch_size]);  cat_186 = None
        sum_235 = torch.ops.aten.sum.dim_IntList(view_683, [0]);  view_683 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(sum_235, 1);  sum_235 = None
        sigmoid_26 = torch.ops.aten.sigmoid.default(addmm_469)
        mul_1993 = torch.ops.aten.mul.Tensor(sigmoid_26, 2.0);  sigmoid_26 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(add_1097, 1)
        cat_187 = torch.ops.aten.cat.default([unsqueeze_81, unsqueeze_83], 1);  unsqueeze_81 = unsqueeze_83 = None
        mul_1994 = torch.ops.aten.mul.Tensor(cat_187, mul_1993);  cat_187 = mul_1993 = None
        sum_236 = torch.ops.aten.sum.dim_IntList(mul_1994, [1]);  mul_1994 = None
        cat_188 = torch.ops.aten.cat.default([sum_197, sum_236, squeeze_281, where_257]);  sum_236 = None
        view_684 = torch.ops.aten.view.default(cat_188, [4, batch_size]);  cat_188 = None
        sum_237 = torch.ops.aten.sum.dim_IntList(view_684, [0]);  view_684 = None
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(sum_237, torch.float32);  sum_237 = None
        clamp_min_15 = torch.ops.aten.clamp_min.default(convert_element_type_182, -15);  convert_element_type_182 = None
        clamp_max_28 = torch.ops.aten.clamp_max.default(clamp_min_15, 15);  clamp_min_15 = None
        convert_element_type_183 = torch.ops.prims.convert_element_type.default(clamp_max_28, torch.float16);  clamp_max_28 = None
        full_default_296 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        isnan_2 = torch.ops.aten.isnan.default(convert_element_type_183)
        where_290 = torch.ops.aten.where.self(isnan_2, full_default_296, convert_element_type_183);  isnan_2 = full_default_296 = convert_element_type_183 = None
        full_default_297 = torch.ops.aten.full.default([], -2.197265625, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        add_1678 = torch.ops.aten.add.Tensor(where_290, full_default_297);  where_290 = full_default_297 = None
        sub_550 = torch.ops.aten.sub.Tensor(squeeze_282, squeeze_281)
        add_1679 = torch.ops.aten.add.Tensor(add_1678, sub_550);  sub_550 = None
        eq_1282 = torch.ops.aten.eq.Tensor(squeeze_282, squeeze_281)
        isinf_2 = torch.ops.aten.isinf.default(squeeze_282)
        bitwise_and_4 = torch.ops.aten.bitwise_and.Tensor(isinf_2, eq_1282);  isinf_2 = None
        full_default_298 = torch.ops.aten.full.default([batch_size], -11.0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        le_9 = torch.ops.aten.le.Tensor(squeeze_282, full_default_298);  full_default_298 = None
        bitwise_and_5 = torch.ops.aten.bitwise_and.Tensor(eq_1282, le_9);  eq_1282 = le_9 = None
        bitwise_or_12 = torch.ops.aten.bitwise_or.Tensor(bitwise_and_4, bitwise_and_5);  bitwise_and_4 = bitwise_and_5 = None
        full_default_299 = torch.ops.aten.full.default([batch_size], nan, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        convert_element_type_184 = torch.ops.prims.convert_element_type.default(add_1679, torch.float32);  add_1679 = None
        clamp_min_16 = torch.ops.aten.clamp_min.default(convert_element_type_184, -15);  convert_element_type_184 = None
        clamp_max_29 = torch.ops.aten.clamp_max.default(clamp_min_16, 15);  clamp_min_16 = None
        convert_element_type_185 = torch.ops.prims.convert_element_type.default(clamp_max_29, torch.float16);  clamp_max_29 = None
        where_291 = torch.ops.aten.where.self(bitwise_or_12, full_default_299, convert_element_type_185);  bitwise_or_12 = full_default_299 = convert_element_type_185 = None
        cat_189 = torch.ops.aten.cat.default([sum_4, sum_7, sum_10, sum_13, sum_16, sum_19, sum_22, sum_25, sum_28, sum_31, sum_34, sum_166])
        view_685 = torch.ops.aten.view.default(cat_189, [12, batch_size]);  cat_189 = None
        sum_238 = torch.ops.aten.sum.dim_IntList(view_685, [0]);  view_685 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(sum_238, 1);  sum_238 = None
        sigmoid_27 = torch.ops.aten.sigmoid.default(addmm_473)
        mul_1996 = torch.ops.aten.mul.Tensor(sigmoid_27, 2.0);  sigmoid_27 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(add_1097, 1)
        cat_190 = torch.ops.aten.cat.default([unsqueeze_84, unsqueeze_86], 1);  unsqueeze_84 = unsqueeze_86 = None
        mul_1997 = torch.ops.aten.mul.Tensor(cat_190, mul_1996);  cat_190 = mul_1996 = None
        sum_239 = torch.ops.aten.sum.dim_IntList(mul_1997, [1]);  mul_1997 = None
        cat_191 = torch.ops.aten.cat.default([sum_198, sum_239, squeeze_286, where_258]);  sum_239 = None
        view_686 = torch.ops.aten.view.default(cat_191, [4, batch_size]);  cat_191 = None
        sum_240 = torch.ops.aten.sum.dim_IntList(view_686, [0]);  view_686 = None
        convert_element_type_186 = torch.ops.prims.convert_element_type.default(sum_240, torch.float32);  sum_240 = None
        clamp_min_17 = torch.ops.aten.clamp_min.default(convert_element_type_186, -15);  convert_element_type_186 = None
        clamp_max_30 = torch.ops.aten.clamp_max.default(clamp_min_17, 15);  clamp_min_17 = None
        convert_element_type_187 = torch.ops.prims.convert_element_type.default(clamp_max_30, torch.float16);  clamp_max_30 = None
        full_default_300 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        isnan_3 = torch.ops.aten.isnan.default(convert_element_type_187)
        where_292 = torch.ops.aten.where.self(isnan_3, full_default_300, convert_element_type_187);  isnan_3 = full_default_300 = convert_element_type_187 = None
        full_default_301 = torch.ops.aten.full.default([], -2.197265625, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        add_1680 = torch.ops.aten.add.Tensor(where_292, full_default_301);  where_292 = full_default_301 = None
        sub_552 = torch.ops.aten.sub.Tensor(squeeze_287, squeeze_286)
        add_1681 = torch.ops.aten.add.Tensor(add_1680, sub_552);  sub_552 = None
        eq_1283 = torch.ops.aten.eq.Tensor(squeeze_287, squeeze_286)
        isinf_3 = torch.ops.aten.isinf.default(squeeze_287)
        bitwise_and_6 = torch.ops.aten.bitwise_and.Tensor(isinf_3, eq_1283);  isinf_3 = None
        full_default_302 = torch.ops.aten.full.default([batch_size], -11.0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        le_10 = torch.ops.aten.le.Tensor(squeeze_287, full_default_302);  full_default_302 = None
        bitwise_and_7 = torch.ops.aten.bitwise_and.Tensor(eq_1283, le_10);  eq_1283 = le_10 = None
        bitwise_or_13 = torch.ops.aten.bitwise_or.Tensor(bitwise_and_6, bitwise_and_7);  bitwise_and_6 = bitwise_and_7 = None
        full_default_303 = torch.ops.aten.full.default([batch_size], nan, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        convert_element_type_188 = torch.ops.prims.convert_element_type.default(add_1681, torch.float32);  add_1681 = None
        clamp_min_18 = torch.ops.aten.clamp_min.default(convert_element_type_188, -15);  convert_element_type_188 = None
        clamp_max_31 = torch.ops.aten.clamp_max.default(clamp_min_18, 15);  clamp_min_18 = None
        convert_element_type_189 = torch.ops.prims.convert_element_type.default(clamp_max_31, torch.float16);  clamp_max_31 = None
        where_293 = torch.ops.aten.where.self(bitwise_or_13, full_default_303, convert_element_type_189);  bitwise_or_13 = full_default_303 = convert_element_type_189 = None
        cat_192 = torch.ops.aten.cat.default([sum_4, sum_7, sum_10, sum_13, sum_16, sum_19, sum_22, sum_25, sum_28, sum_31, sum_34, sum_166])
        view_687 = torch.ops.aten.view.default(cat_192, [12, batch_size]);  cat_192 = None
        sum_241 = torch.ops.aten.sum.dim_IntList(view_687, [0]);  view_687 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(sum_241, 1);  sum_241 = None
        sigmoid_28 = torch.ops.aten.sigmoid.default(addmm_477)
        mul_1999 = torch.ops.aten.mul.Tensor(sigmoid_28, 2.0);  sigmoid_28 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(add_1097, 1)
        cat_193 = torch.ops.aten.cat.default([unsqueeze_87, unsqueeze_89], 1);  unsqueeze_87 = unsqueeze_89 = None
        mul_2000 = torch.ops.aten.mul.Tensor(cat_193, mul_1999);  cat_193 = mul_1999 = None
        sum_242 = torch.ops.aten.sum.dim_IntList(mul_2000, [1]);  mul_2000 = None
        cat_194 = torch.ops.aten.cat.default([sum_199, sum_242, squeeze_291, where_262]);  sum_242 = None
        view_688 = torch.ops.aten.view.default(cat_194, [4, batch_size]);  cat_194 = None
        sum_243 = torch.ops.aten.sum.dim_IntList(view_688, [0]);  view_688 = None
        convert_element_type_190 = torch.ops.prims.convert_element_type.default(sum_243, torch.float32);  sum_243 = None
        clamp_min_19 = torch.ops.aten.clamp_min.default(convert_element_type_190, -15);  convert_element_type_190 = None
        clamp_max_32 = torch.ops.aten.clamp_max.default(clamp_min_19, 15);  clamp_min_19 = None
        convert_element_type_191 = torch.ops.prims.convert_element_type.default(clamp_max_32, torch.float16);  clamp_max_32 = None
        full_default_304 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        isnan_4 = torch.ops.aten.isnan.default(convert_element_type_191)
        where_294 = torch.ops.aten.where.self(isnan_4, full_default_304, convert_element_type_191);  isnan_4 = full_default_304 = convert_element_type_191 = None
        full_default_305 = torch.ops.aten.full.default([], -2.197265625, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        add_1682 = torch.ops.aten.add.Tensor(where_294, full_default_305);  where_294 = full_default_305 = None
        sub_554 = torch.ops.aten.sub.Tensor(squeeze_292, squeeze_291)
        add_1683 = torch.ops.aten.add.Tensor(add_1682, sub_554);  sub_554 = None
        eq_1284 = torch.ops.aten.eq.Tensor(squeeze_292, squeeze_291)
        isinf_4 = torch.ops.aten.isinf.default(squeeze_292)
        bitwise_and_8 = torch.ops.aten.bitwise_and.Tensor(isinf_4, eq_1284);  isinf_4 = None
        full_default_306 = torch.ops.aten.full.default([batch_size], -11.0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        le_11 = torch.ops.aten.le.Tensor(squeeze_292, full_default_306);  full_default_306 = None
        bitwise_and_9 = torch.ops.aten.bitwise_and.Tensor(eq_1284, le_11);  eq_1284 = le_11 = None
        bitwise_or_14 = torch.ops.aten.bitwise_or.Tensor(bitwise_and_8, bitwise_and_9);  bitwise_and_8 = bitwise_and_9 = None
        full_default_307 = torch.ops.aten.full.default([batch_size], nan, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        convert_element_type_192 = torch.ops.prims.convert_element_type.default(add_1683, torch.float32);  add_1683 = None
        clamp_min_20 = torch.ops.aten.clamp_min.default(convert_element_type_192, -15);  convert_element_type_192 = None
        clamp_max_33 = torch.ops.aten.clamp_max.default(clamp_min_20, 15);  clamp_min_20 = None
        convert_element_type_193 = torch.ops.prims.convert_element_type.default(clamp_max_33, torch.float16);  clamp_max_33 = None
        where_295 = torch.ops.aten.where.self(bitwise_or_14, full_default_307, convert_element_type_193);  bitwise_or_14 = full_default_307 = convert_element_type_193 = None
        cat_195 = torch.ops.aten.cat.default([sum_4, sum_7, sum_10, sum_13, sum_16, sum_19, sum_22, sum_25, sum_28, sum_31, sum_34, sum_166])
        view_689 = torch.ops.aten.view.default(cat_195, [12, batch_size]);  cat_195 = None
        sum_244 = torch.ops.aten.sum.dim_IntList(view_689, [0]);  view_689 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(sum_244, 1);  sum_244 = None
        sigmoid_29 = torch.ops.aten.sigmoid.default(addmm_461);  addmm_461 = None
        mul_2002 = torch.ops.aten.mul.Tensor(sigmoid_29, 2.0);  sigmoid_29 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(add_1097, 1)
        cat_196 = torch.ops.aten.cat.default([unsqueeze_90, unsqueeze_92], 1);  unsqueeze_90 = unsqueeze_92 = None
        mul_2003 = torch.ops.aten.mul.Tensor(cat_196, mul_2002);  cat_196 = mul_2002 = None
        sum_245 = torch.ops.aten.sum.dim_IntList(mul_2003, [1]);  mul_2003 = None
        cat_197 = torch.ops.aten.cat.default([sum_206, sum_245, squeeze_271, where_260]);  sum_206 = sum_245 = None
        view_690 = torch.ops.aten.view.default(cat_197, [4, batch_size]);  cat_197 = None
        sum_246 = torch.ops.aten.sum.dim_IntList(view_690, [0]);  view_690 = None
        convert_element_type_194 = torch.ops.prims.convert_element_type.default(sum_246, torch.float32);  sum_246 = None
        clamp_min_21 = torch.ops.aten.clamp_min.default(convert_element_type_194, -15);  convert_element_type_194 = None
        clamp_max_34 = torch.ops.aten.clamp_max.default(clamp_min_21, 15);  clamp_min_21 = None
        convert_element_type_195 = torch.ops.prims.convert_element_type.default(clamp_max_34, torch.float16);  clamp_max_34 = None
        full_default_308 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        isnan_5 = torch.ops.aten.isnan.default(convert_element_type_195)
        where_296 = torch.ops.aten.where.self(isnan_5, full_default_308, convert_element_type_195);  isnan_5 = full_default_308 = convert_element_type_195 = None
        full_default_309 = torch.ops.aten.full.default([], -2.197265625, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        add_1684 = torch.ops.aten.add.Tensor(where_296, full_default_309);  where_296 = full_default_309 = None
        cat_198 = torch.ops.aten.cat.default([sum_4, sum_7, sum_10, sum_13, sum_16, sum_19, sum_22, sum_25, sum_28, sum_31, sum_34, sum_166])
        view_691 = torch.ops.aten.view.default(cat_198, [12, batch_size]);  cat_198 = None
        sum_247 = torch.ops.aten.sum.dim_IntList(view_691, [0]);  view_691 = None
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(sum_247, 1);  sum_247 = None
        sigmoid_30 = torch.ops.aten.sigmoid.default(addmm_465);  addmm_465 = None
        mul_2005 = torch.ops.aten.mul.Tensor(sigmoid_30, 2.0);  sigmoid_30 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(add_1097, 1)
        cat_199 = torch.ops.aten.cat.default([unsqueeze_93, unsqueeze_95], 1);  unsqueeze_93 = unsqueeze_95 = None
        mul_2006 = torch.ops.aten.mul.Tensor(cat_199, mul_2005);  cat_199 = mul_2005 = None
        sum_248 = torch.ops.aten.sum.dim_IntList(mul_2006, [1]);  mul_2006 = None
        cat_200 = torch.ops.aten.cat.default([sum_207, sum_248, squeeze_276, where_259]);  sum_207 = sum_248 = None
        view_692 = torch.ops.aten.view.default(cat_200, [4, batch_size]);  cat_200 = None
        sum_249 = torch.ops.aten.sum.dim_IntList(view_692, [0]);  view_692 = None
        convert_element_type_196 = torch.ops.prims.convert_element_type.default(sum_249, torch.float32);  sum_249 = None
        clamp_min_22 = torch.ops.aten.clamp_min.default(convert_element_type_196, -15);  convert_element_type_196 = None
        clamp_max_35 = torch.ops.aten.clamp_max.default(clamp_min_22, 15);  clamp_min_22 = None
        convert_element_type_197 = torch.ops.prims.convert_element_type.default(clamp_max_35, torch.float16);  clamp_max_35 = None
        full_default_310 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        isnan_6 = torch.ops.aten.isnan.default(convert_element_type_197)
        where_297 = torch.ops.aten.where.self(isnan_6, full_default_310, convert_element_type_197);  isnan_6 = full_default_310 = convert_element_type_197 = None
        full_default_311 = torch.ops.aten.full.default([], -2.197265625, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        add_1685 = torch.ops.aten.add.Tensor(where_297, full_default_311);  where_297 = full_default_311 = None
        cat_201 = torch.ops.aten.cat.default([sum_4, sum_7, sum_10, sum_13, sum_16, sum_19, sum_22, sum_25, sum_28, sum_31, sum_34, sum_166])
        view_693 = torch.ops.aten.view.default(cat_201, [12, batch_size]);  cat_201 = None
        sum_250 = torch.ops.aten.sum.dim_IntList(view_693, [0]);  view_693 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(sum_250, 1);  sum_250 = None
        sigmoid_31 = torch.ops.aten.sigmoid.default(addmm_469);  addmm_469 = None
        mul_2008 = torch.ops.aten.mul.Tensor(sigmoid_31, 2.0);  sigmoid_31 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(add_1097, 1)
        cat_202 = torch.ops.aten.cat.default([unsqueeze_96, unsqueeze_98], 1);  unsqueeze_96 = unsqueeze_98 = None
        mul_2009 = torch.ops.aten.mul.Tensor(cat_202, mul_2008);  cat_202 = mul_2008 = None
        sum_251 = torch.ops.aten.sum.dim_IntList(mul_2009, [1]);  mul_2009 = None
        cat_203 = torch.ops.aten.cat.default([sum_208, sum_251, squeeze_281, where_257]);  sum_208 = sum_251 = None
        view_694 = torch.ops.aten.view.default(cat_203, [4, batch_size]);  cat_203 = None
        sum_252 = torch.ops.aten.sum.dim_IntList(view_694, [0]);  view_694 = None
        convert_element_type_198 = torch.ops.prims.convert_element_type.default(sum_252, torch.float32);  sum_252 = None
        clamp_min_23 = torch.ops.aten.clamp_min.default(convert_element_type_198, -15);  convert_element_type_198 = None
        clamp_max_36 = torch.ops.aten.clamp_max.default(clamp_min_23, 15);  clamp_min_23 = None
        convert_element_type_199 = torch.ops.prims.convert_element_type.default(clamp_max_36, torch.float16);  clamp_max_36 = None
        full_default_312 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        isnan_7 = torch.ops.aten.isnan.default(convert_element_type_199)
        where_298 = torch.ops.aten.where.self(isnan_7, full_default_312, convert_element_type_199);  isnan_7 = full_default_312 = convert_element_type_199 = None
        full_default_313 = torch.ops.aten.full.default([], -2.197265625, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        add_1686 = torch.ops.aten.add.Tensor(where_298, full_default_313);  where_298 = full_default_313 = None
        cat_204 = torch.ops.aten.cat.default([sum_4, sum_7, sum_10, sum_13, sum_16, sum_19, sum_22, sum_25, sum_28, sum_31, sum_34, sum_166])
        view_695 = torch.ops.aten.view.default(cat_204, [12, batch_size]);  cat_204 = None
        sum_253 = torch.ops.aten.sum.dim_IntList(view_695, [0]);  view_695 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(sum_253, 1);  sum_253 = None
        sigmoid_32 = torch.ops.aten.sigmoid.default(addmm_473);  addmm_473 = None
        mul_2011 = torch.ops.aten.mul.Tensor(sigmoid_32, 2.0);  sigmoid_32 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(add_1097, 1)
        cat_205 = torch.ops.aten.cat.default([unsqueeze_99, unsqueeze_101], 1);  unsqueeze_99 = unsqueeze_101 = None
        mul_2012 = torch.ops.aten.mul.Tensor(cat_205, mul_2011);  cat_205 = mul_2011 = None
        sum_254 = torch.ops.aten.sum.dim_IntList(mul_2012, [1]);  mul_2012 = None
        cat_206 = torch.ops.aten.cat.default([sum_209, sum_254, squeeze_286, where_258]);  sum_209 = sum_254 = None
        view_696 = torch.ops.aten.view.default(cat_206, [4, batch_size]);  cat_206 = None
        sum_255 = torch.ops.aten.sum.dim_IntList(view_696, [0]);  view_696 = None
        convert_element_type_200 = torch.ops.prims.convert_element_type.default(sum_255, torch.float32);  sum_255 = None
        clamp_min_24 = torch.ops.aten.clamp_min.default(convert_element_type_200, -15);  convert_element_type_200 = None
        clamp_max_37 = torch.ops.aten.clamp_max.default(clamp_min_24, 15);  clamp_min_24 = None
        convert_element_type_201 = torch.ops.prims.convert_element_type.default(clamp_max_37, torch.float16);  clamp_max_37 = None
        full_default_314 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        isnan_8 = torch.ops.aten.isnan.default(convert_element_type_201)
        where_299 = torch.ops.aten.where.self(isnan_8, full_default_314, convert_element_type_201);  isnan_8 = full_default_314 = convert_element_type_201 = None
        full_default_315 = torch.ops.aten.full.default([], -2.197265625, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        add_1687 = torch.ops.aten.add.Tensor(where_299, full_default_315);  where_299 = full_default_315 = None
        cat_207 = torch.ops.aten.cat.default([sum_4, sum_7, sum_10, sum_13, sum_16, sum_19, sum_22, sum_25, sum_28, sum_31, sum_34, sum_166])
        view_697 = torch.ops.aten.view.default(cat_207, [12, batch_size]);  cat_207 = None
        sum_256 = torch.ops.aten.sum.dim_IntList(view_697, [0]);  view_697 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(sum_256, 1);  sum_256 = None
        sigmoid_33 = torch.ops.aten.sigmoid.default(addmm_477);  addmm_477 = None
        mul_2014 = torch.ops.aten.mul.Tensor(sigmoid_33, 2.0);  sigmoid_33 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(add_1097, 1)
        cat_208 = torch.ops.aten.cat.default([unsqueeze_102, unsqueeze_104], 1);  unsqueeze_102 = unsqueeze_104 = None
        mul_2015 = torch.ops.aten.mul.Tensor(cat_208, mul_2014);  cat_208 = mul_2014 = None
        sum_257 = torch.ops.aten.sum.dim_IntList(mul_2015, [1]);  mul_2015 = None
        cat_209 = torch.ops.aten.cat.default([sum_210, sum_257, squeeze_291, where_262]);  sum_210 = sum_257 = None
        view_698 = torch.ops.aten.view.default(cat_209, [4, batch_size]);  cat_209 = None
        sum_258 = torch.ops.aten.sum.dim_IntList(view_698, [0]);  view_698 = None
        convert_element_type_202 = torch.ops.prims.convert_element_type.default(sum_258, torch.float32);  sum_258 = None
        clamp_min_25 = torch.ops.aten.clamp_min.default(convert_element_type_202, -15);  convert_element_type_202 = None
        clamp_max_38 = torch.ops.aten.clamp_max.default(clamp_min_25, 15);  clamp_min_25 = None
        convert_element_type_203 = torch.ops.prims.convert_element_type.default(clamp_max_38, torch.float16);  clamp_max_38 = None
        full_default_316 = torch.ops.aten.full.default([batch_size], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        isnan_9 = torch.ops.aten.isnan.default(convert_element_type_203)
        where_300 = torch.ops.aten.where.self(isnan_9, full_default_316, convert_element_type_203);  isnan_9 = full_default_316 = convert_element_type_203 = None
        full_default_317 = torch.ops.aten.full.default([], -2.197265625, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        add_1688 = torch.ops.aten.add.Tensor(where_300, full_default_317);  where_300 = full_default_317 = None
        full_default_318 = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        squeeze_302 = torch.ops.aten.squeeze.dim(arg4_1, 1)
        neg_15 = torch.ops.aten.neg.default(add_1674)
        exp_5 = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_1689 = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        reciprocal_10 = torch.ops.aten.reciprocal.default(add_1689);  add_1689 = None
        mul_2017 = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        neg_16 = torch.ops.aten.neg.default(where_287)
        exp_6 = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_1690 = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        reciprocal_11 = torch.ops.aten.reciprocal.default(add_1690);  add_1690 = None
        mul_2018 = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        neg_17 = torch.ops.aten.neg.default(add_1676)
        exp_7 = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_1691 = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        reciprocal_12 = torch.ops.aten.reciprocal.default(add_1691);  add_1691 = None
        mul_2019 = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        neg_18 = torch.ops.aten.neg.default(where_289)
        exp_8 = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        add_1692 = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        reciprocal_13 = torch.ops.aten.reciprocal.default(add_1692);  add_1692 = None
        mul_2020 = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        neg_19 = torch.ops.aten.neg.default(add_1678)
        exp_9 = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_1693 = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        reciprocal_14 = torch.ops.aten.reciprocal.default(add_1693);  add_1693 = None
        mul_2021 = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        neg_20 = torch.ops.aten.neg.default(where_291)
        exp_10 = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_1694 = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        reciprocal_15 = torch.ops.aten.reciprocal.default(add_1694);  add_1694 = None
        mul_2022 = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        neg_21 = torch.ops.aten.neg.default(add_1680)
        exp_11 = torch.ops.aten.exp.default(neg_21);  neg_21 = None
        add_1695 = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        reciprocal_16 = torch.ops.aten.reciprocal.default(add_1695);  add_1695 = None
        mul_2023 = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        neg_22 = torch.ops.aten.neg.default(where_293)
        exp_12 = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_1696 = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        reciprocal_17 = torch.ops.aten.reciprocal.default(add_1696);  add_1696 = None
        mul_2024 = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        neg_23 = torch.ops.aten.neg.default(add_1682)
        exp_13 = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_1697 = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        reciprocal_18 = torch.ops.aten.reciprocal.default(add_1697);  add_1697 = None
        mul_2025 = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        neg_24 = torch.ops.aten.neg.default(where_295)
        exp_14 = torch.ops.aten.exp.default(neg_24);  neg_24 = None
        add_1698 = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        reciprocal_19 = torch.ops.aten.reciprocal.default(add_1698);  add_1698 = None
        mul_2026 = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        eq_1285 = torch.ops.aten.eq.Scalar(squeeze_302, 172)
        where_301 = torch.ops.aten.where.self(eq_1285, mul_2021, mul_2023);  eq_1285 = None
        eq_1286 = torch.ops.aten.eq.Scalar(squeeze_302, 167)
        where_302 = torch.ops.aten.where.self(eq_1286, mul_2019, where_301);  eq_1286 = where_301 = None
        eq_1287 = torch.ops.aten.eq.Scalar(squeeze_302, 96)
        where_303 = torch.ops.aten.where.self(eq_1287, mul_2017, where_302);  eq_1287 = where_302 = None
        eq_1288 = torch.ops.aten.eq.Scalar(squeeze_302, 412)
        squeeze_303 = torch.ops.aten.squeeze.dim(arg5_1, 1);  arg5_1 = None
        eq_1289 = torch.ops.aten.eq.Scalar(squeeze_303, 102);  squeeze_303 = None
        logical_and_6 = torch.ops.aten.logical_and.default(eq_1288, eq_1289);  eq_1288 = eq_1289 = None
        where_304 = torch.ops.aten.where.self(logical_and_6, mul_2017, where_303);  where_303 = None
        squeeze_304 = torch.ops.aten.squeeze.dims(arg4_1, [1]);  arg4_1 = None
        eq_1290 = torch.ops.aten.eq.Scalar(squeeze_304, 412);  squeeze_304 = None
        where_305 = torch.ops.aten.where.self(eq_1290, mul_2025, where_304);  eq_1290 = None
        view_701 = torch.ops.aten.view.default(arg1636_1, [1, batch_size, 8]);  arg1636_1 = None
        sum_259 = torch.ops.aten.sum.dim_IntList(view_701, [0]);  view_701 = None
        squeeze_307 = torch.ops.aten.squeeze.default(logical_not)
        view_702 = torch.ops.aten.view.default(squeeze_307, [-1, 1]);  squeeze_307 = None
        full_default_319 = torch.ops.aten.full.default([batch_size, 8], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_306 = torch.ops.aten.where.self(view_702, full_default_319, arg1637_1);  view_702 = full_default_319 = arg1637_1 = None
        cat_210 = torch.ops.aten.cat.default([arg1638_1, where_306]);  arg1638_1 = where_306 = None
        view_704 = torch.ops.aten.view.default(cat_210, [2, batch_size, 8]);  cat_210 = None
        sum_260 = torch.ops.aten.sum.dim_IntList(view_704, [0]);  view_704 = None
        view_707 = torch.ops.aten.view.default(arg1639_1, [1, batch_size, 8]);  arg1639_1 = None
        sum_261 = torch.ops.aten.sum.dim_IntList(view_707, [0]);  view_707 = None
        view_710 = torch.ops.aten.view.default(arg1640_1, [1, batch_size, 8]);  arg1640_1 = None
        sum_262 = torch.ops.aten.sum.dim_IntList(view_710, [0]);  view_710 = None
        mul_2027 = torch.ops.aten.mul.Tensor(sum_259, sum_260)
        mul_2028 = torch.ops.aten.mul.Tensor(sum_259, sum_262)
        mul_2029 = torch.ops.aten.mul.Tensor(sum_262, sum_260);  sum_260 = None
        cat_211 = torch.ops.aten.cat.default([mul_2027, mul_2028, mul_2029], 1);  mul_2027 = mul_2028 = mul_2029 = None
        mul_2030 = torch.ops.aten.mul.Tensor(sum_259, sum_261)
        mul_2031 = torch.ops.aten.mul.Tensor(sum_259, sum_262);  sum_259 = None
        mul_2032 = torch.ops.aten.mul.Tensor(sum_262, sum_261);  sum_262 = sum_261 = None
        cat_212 = torch.ops.aten.cat.default([mul_2030, mul_2031, mul_2032], 1);  mul_2030 = mul_2031 = mul_2032 = None
        slice_755 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43517, 43525)
        slice_756 = torch.ops.aten.slice.Tensor(arg15_1, 1, 934, 942)
        squeeze_317 = torch.ops.aten.squeeze.default(logical_not)
        view_715 = torch.ops.aten.view.default(squeeze_317, [-1, 1]);  squeeze_317 = None
        full_default_320 = torch.ops.aten.full.default([batch_size, 22], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_307 = torch.ops.aten.where.self(view_715, full_default_320, arg1643_1);  full_default_320 = arg1643_1 = None
        full_default_321 = torch.ops.aten.full.default([batch_size, 52], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_308 = torch.ops.aten.where.self(view_715, full_default_321, arg1644_1);  view_715 = full_default_321 = arg1644_1 = None
        cat_213 = torch.ops.aten.cat.default([where_307, arg1645_1, where_308], -1);  where_307 = arg1645_1 = where_308 = None
        view_719 = torch.ops.aten.view.default(arg1647_1, [1, batch_size, 1]);  arg1647_1 = None
        sum_263 = torch.ops.aten.sum.dim_IntList(view_719, [0]);  view_719 = None
        slice_757 = torch.ops.aten.slice.Tensor(arg15_1, 1, 43525, 43526)
        slice_758 = torch.ops.aten.slice.Tensor(arg15_1, 1, 942, 943);  arg15_1 = None
        view_722 = torch.ops.aten.view.default(arg1649_1, [1, batch_size, 1]);  arg1649_1 = None
        sum_264 = torch.ops.aten.sum.dim_IntList(view_722, [0]);  view_722 = None
        squeeze_323 = torch.ops.aten.squeeze.default(logical_not)
        view_723 = torch.ops.aten.view.default(squeeze_323, [-1, 1]);  squeeze_323 = None
        full_default_322 = torch.ops.aten.full.default([batch_size, 7], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_309 = torch.ops.aten.where.self(view_723, full_default_322, arg1650_1);  full_default_322 = arg1650_1 = None
        full_default_323 = torch.ops.aten.full.default([batch_size, 12], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_310 = torch.ops.aten.where.self(view_723, full_default_323, arg1651_1);  full_default_323 = arg1651_1 = None
        full_default_324 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_311 = torch.ops.aten.where.self(view_723, full_default_324, arg1652_1);  full_default_324 = arg1652_1 = None
        full_default_325 = torch.ops.aten.full.default([batch_size, 1], 0, dtype = torch.float16, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_312 = torch.ops.aten.where.self(view_723, full_default_325, arg1653_1);  view_723 = full_default_325 = arg1653_1 = None
        cat_214 = torch.ops.aten.cat.default([where_309, arg1654_1, where_310], -1);  where_309 = arg1654_1 = where_310 = None
        cat_215 = torch.ops.aten.cat.default([where_311, arg1655_1, where_312]);  where_311 = arg1655_1 = where_312 = None
        view_725 = torch.ops.aten.view.default(cat_215, [3, batch_size, 1]);  cat_215 = None
        sum_265 = torch.ops.aten.sum.dim_IntList(view_725, [0]);  view_725 = None
        add_1699 = torch.ops.aten.add.Tensor(sum_264, sum_263);  sum_264 = None
        add_1700 = torch.ops.aten.add.Tensor(sum_265, sum_263);  sum_265 = sum_263 = None
        cat_216 = torch.ops.aten.cat.default([arg1641_1, arg1642_1], 1);  arg1642_1 = None
        addmm_478 = torch.ops.aten.addmm.default(arg1657_1, cat_216, arg1656_1);  arg1657_1 = cat_216 = arg1656_1 = None
        cat_217 = torch.ops.aten.cat.default([arg1641_1, cat_213], 1);  arg1641_1 = cat_213 = None
        addmm_479 = torch.ops.aten.addmm.default(arg1659_1, cat_217, arg1658_1);  arg1659_1 = cat_217 = arg1658_1 = None
        cat_218 = torch.ops.aten.cat.default([cat_212, addmm_478, arg1646_1, arg1648_1, cat_155], 1);  cat_212 = addmm_478 = arg1648_1 = None
        cat_219 = torch.ops.aten.cat.default([cat_211, addmm_479, arg1646_1, cat_214, cat_155], 1);  cat_211 = addmm_479 = arg1646_1 = cat_214 = None
        addmm_480 = torch.ops.aten.addmm.default(arg1661_1, cat_219, arg1660_1);  arg1661_1 = cat_219 = arg1660_1 = None
        relu_147 = torch.ops.aten.relu.default(addmm_480);  addmm_480 = None
        addmm_481 = torch.ops.aten.addmm.default(arg1663_1, relu_147, arg1662_1);  arg1663_1 = relu_147 = arg1662_1 = None
        relu_148 = torch.ops.aten.relu.default(addmm_481);  addmm_481 = None
        addmm_482 = torch.ops.aten.addmm.default(arg1665_1, cat_218, arg1664_1);  arg1665_1 = cat_218 = arg1664_1 = None
        relu_149 = torch.ops.aten.relu.default(addmm_482);  addmm_482 = None
        addmm_483 = torch.ops.aten.addmm.default(arg1667_1, relu_149, arg1666_1);  arg1667_1 = relu_149 = arg1666_1 = None
        relu_150 = torch.ops.aten.relu.default(addmm_483);  addmm_483 = None
        where_313 = torch.ops.aten.where.self(eq_5, relu_150, relu_148);  relu_150 = relu_148 = None
        where_314 = torch.ops.aten.where.self(eq_5, add_1699, add_1700);  eq_5 = add_1699 = add_1700 = None
        view_728 = torch.ops.aten.view.default(arg1668_1, [-1]);  arg1668_1 = None
        view_729 = torch.ops.aten.view.default(arg1669_1, [-1]);  arg1669_1 = None
        eq_1293 = torch.ops.aten.eq.Scalar(view_728, 26)
        eq_1294 = torch.ops.aten.eq.Scalar(view_728, 30);  view_728 = None
        full_default_326 = torch.ops.aten.full.default([], 1000, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_4 = torch.ops.aten.div.Tensor(view_729, full_default_326);  view_729 = full_default_326 = None
        convert_element_type_204 = torch.ops.prims.convert_element_type.default(div_4, torch.float32);  div_4 = None
        full_default_327 = torch.ops.aten.full.default([], 1024.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_5 = torch.ops.aten.div.Tensor(convert_element_type_204, full_default_327);  convert_element_type_204 = full_default_327 = None
        convert_element_type_205 = torch.ops.prims.convert_element_type.default(div_5, torch.float16);  div_5 = None
        addmm_484 = torch.ops.aten.addmm.default(arg1673_1, where_313, arg1672_1);  arg1673_1 = arg1672_1 = None
        relu_151 = torch.ops.aten.relu.default(addmm_484);  addmm_484 = None
        addmm_485 = torch.ops.aten.addmm.default(arg1675_1, relu_151, arg1674_1);  arg1675_1 = relu_151 = arg1674_1 = None
        relu_152 = torch.ops.aten.relu.default(addmm_485);  addmm_485 = None
        addmm_486 = torch.ops.aten.addmm.default(arg1677_1, relu_152, arg1676_1);  arg1677_1 = relu_152 = arg1676_1 = None
        sum_266 = torch.ops.aten.sum.dim_IntList(addmm_486, [1])
        squeeze_327 = torch.ops.aten.squeeze.dim(where_314, 1)
        mul_2033 = torch.ops.aten.mul.Tensor(squeeze_327, arg1678_1);  squeeze_327 = arg1678_1 = None
        add_1701 = torch.ops.aten.add.Tensor(sum_266, mul_2033);  sum_266 = mul_2033 = None
        addmm_487 = torch.ops.aten.addmm.default(arg1680_1, where_313, arg1679_1);  arg1680_1 = arg1679_1 = None
        relu_153 = torch.ops.aten.relu.default(addmm_487);  addmm_487 = None
        addmm_488 = torch.ops.aten.addmm.default(arg1682_1, relu_153, arg1681_1);  arg1682_1 = relu_153 = arg1681_1 = None
        relu_154 = torch.ops.aten.relu.default(addmm_488);  addmm_488 = None
        addmm_489 = torch.ops.aten.addmm.default(arg1684_1, relu_154, arg1683_1);  arg1684_1 = relu_154 = arg1683_1 = None
        sum_267 = torch.ops.aten.sum.dim_IntList(addmm_489, [1])
        squeeze_328 = torch.ops.aten.squeeze.dim(where_314, 1)
        mul_2034 = torch.ops.aten.mul.Tensor(squeeze_328, arg1685_1);  squeeze_328 = arg1685_1 = None
        add_1702 = torch.ops.aten.add.Tensor(sum_267, mul_2034);  sum_267 = mul_2034 = None
        addmm_490 = torch.ops.aten.addmm.default(arg1687_1, where_313, arg1686_1);  arg1687_1 = arg1686_1 = None
        relu_155 = torch.ops.aten.relu.default(addmm_490);  addmm_490 = None
        addmm_491 = torch.ops.aten.addmm.default(arg1689_1, relu_155, arg1688_1);  arg1689_1 = relu_155 = arg1688_1 = None
        relu_156 = torch.ops.aten.relu.default(addmm_491);  addmm_491 = None
        addmm_492 = torch.ops.aten.addmm.default(arg1691_1, relu_156, arg1690_1);  arg1691_1 = relu_156 = arg1690_1 = None
        sum_268 = torch.ops.aten.sum.dim_IntList(addmm_492, [1])
        squeeze_329 = torch.ops.aten.squeeze.dim(where_314, 1)
        mul_2035 = torch.ops.aten.mul.Tensor(squeeze_329, arg1692_1);  squeeze_329 = arg1692_1 = None
        add_1703 = torch.ops.aten.add.Tensor(sum_268, mul_2035);  sum_268 = mul_2035 = None
        addmm_493 = torch.ops.aten.addmm.default(arg1694_1, where_313, arg1693_1);  arg1694_1 = arg1693_1 = None
        relu_157 = torch.ops.aten.relu.default(addmm_493);  addmm_493 = None
        addmm_494 = torch.ops.aten.addmm.default(arg1696_1, relu_157, arg1695_1);  arg1696_1 = relu_157 = arg1695_1 = None
        relu_158 = torch.ops.aten.relu.default(addmm_494);  addmm_494 = None
        addmm_495 = torch.ops.aten.addmm.default(arg1698_1, relu_158, arg1697_1);  arg1698_1 = relu_158 = arg1697_1 = None
        sum_269 = torch.ops.aten.sum.dim_IntList(addmm_495, [1])
        squeeze_330 = torch.ops.aten.squeeze.dim(where_314, 1)
        mul_2036 = torch.ops.aten.mul.Tensor(squeeze_330, arg1699_1);  squeeze_330 = arg1699_1 = None
        add_1704 = torch.ops.aten.add.Tensor(sum_269, mul_2036);  sum_269 = mul_2036 = None
        addmm_496 = torch.ops.aten.addmm.default(arg1701_1, where_313, arg1700_1);  arg1701_1 = arg1700_1 = None
        relu_159 = torch.ops.aten.relu.default(addmm_496);  addmm_496 = None
        addmm_497 = torch.ops.aten.addmm.default(arg1703_1, relu_159, arg1702_1);  arg1703_1 = relu_159 = arg1702_1 = None
        relu_160 = torch.ops.aten.relu.default(addmm_497);  addmm_497 = None
        addmm_498 = torch.ops.aten.addmm.default(arg1705_1, relu_160, arg1704_1);  arg1705_1 = relu_160 = arg1704_1 = None
        sum_270 = torch.ops.aten.sum.dim_IntList(addmm_498, [1])
        squeeze_331 = torch.ops.aten.squeeze.dim(where_314, 1)
        mul_2037 = torch.ops.aten.mul.Tensor(squeeze_331, arg1706_1);  squeeze_331 = arg1706_1 = None
        add_1705 = torch.ops.aten.add.Tensor(sum_270, mul_2037);  sum_270 = mul_2037 = None
        mul_2038 = torch.ops.aten.mul.Tensor(add_1703, convert_element_type_205)
        add_1706 = torch.ops.aten.add.Tensor(add_1702, mul_2038);  mul_2038 = None
        mul_2039 = torch.ops.aten.mul.Tensor(add_1705, convert_element_type_205);  convert_element_type_205 = None
        add_1707 = torch.ops.aten.add.Tensor(add_1704, mul_2039);  mul_2039 = None
        where_315 = torch.ops.aten.where.self(eq_1293, add_1706, add_1701);  eq_1293 = add_1706 = None
        where_316 = torch.ops.aten.where.self(eq_1294, add_1707, where_315);  eq_1294 = add_1707 = where_315 = None
        add_1708 = torch.ops.aten.add.Tensor(add_1674, where_316)
        convert_element_type_206 = torch.ops.prims.convert_element_type.default(add_1708, torch.float32);  add_1708 = None
        clamp_min_26 = torch.ops.aten.clamp_min.default(convert_element_type_206, -15);  convert_element_type_206 = None
        clamp_max_39 = torch.ops.aten.clamp_max.default(clamp_min_26, 15);  clamp_min_26 = None
        convert_element_type_207 = torch.ops.prims.convert_element_type.default(clamp_max_39, torch.float16);  clamp_max_39 = None
        neg_25 = torch.ops.aten.neg.default(convert_element_type_207)
        exp_15 = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_1709 = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        reciprocal_20 = torch.ops.aten.reciprocal.default(add_1709);  add_1709 = None
        mul_2040 = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        add_1710 = torch.ops.aten.add.Tensor(add_1678, where_316)
        convert_element_type_208 = torch.ops.prims.convert_element_type.default(add_1710, torch.float32);  add_1710 = None
        clamp_min_27 = torch.ops.aten.clamp_min.default(convert_element_type_208, -15);  convert_element_type_208 = None
        clamp_max_40 = torch.ops.aten.clamp_max.default(clamp_min_27, 15);  clamp_min_27 = None
        convert_element_type_209 = torch.ops.prims.convert_element_type.default(clamp_max_40, torch.float16);  clamp_max_40 = None
        neg_26 = torch.ops.aten.neg.default(convert_element_type_209)
        exp_16 = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_1711 = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        reciprocal_21 = torch.ops.aten.reciprocal.default(add_1711);  add_1711 = None
        mul_2041 = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        add_1712 = torch.ops.aten.add.Tensor(add_1680, where_316)
        convert_element_type_210 = torch.ops.prims.convert_element_type.default(add_1712, torch.float32);  add_1712 = None
        clamp_min_28 = torch.ops.aten.clamp_min.default(convert_element_type_210, -15);  convert_element_type_210 = None
        clamp_max_41 = torch.ops.aten.clamp_max.default(clamp_min_28, 15);  clamp_min_28 = None
        convert_element_type_211 = torch.ops.prims.convert_element_type.default(clamp_max_41, torch.float16);  clamp_max_41 = None
        neg_27 = torch.ops.aten.neg.default(convert_element_type_211)
        exp_17 = torch.ops.aten.exp.default(neg_27);  neg_27 = None
        add_1713 = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        reciprocal_22 = torch.ops.aten.reciprocal.default(add_1713);  add_1713 = None
        mul_2042 = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        add_1714 = torch.ops.aten.add.Tensor(add_1682, where_316);  where_316 = None
        convert_element_type_212 = torch.ops.prims.convert_element_type.default(add_1714, torch.float32);  add_1714 = None
        clamp_min_29 = torch.ops.aten.clamp_min.default(convert_element_type_212, -15);  convert_element_type_212 = None
        clamp_max_42 = torch.ops.aten.clamp_max.default(clamp_min_29, 15);  clamp_min_29 = None
        convert_element_type_213 = torch.ops.prims.convert_element_type.default(clamp_max_42, torch.float16);  clamp_max_42 = None
        neg_28 = torch.ops.aten.neg.default(convert_element_type_213)
        exp_18 = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_1715 = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        reciprocal_23 = torch.ops.aten.reciprocal.default(add_1715);  add_1715 = None
        mul_2043 = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        eq_1295 = torch.ops.aten.eq.Scalar(squeeze_302, 172)
        where_317 = torch.ops.aten.where.self(eq_1295, mul_2022, mul_2024);  eq_1295 = None
        eq_1296 = torch.ops.aten.eq.Scalar(squeeze_302, 167)
        where_318 = torch.ops.aten.where.self(eq_1296, mul_2020, where_317);  eq_1296 = where_317 = None
        eq_1297 = torch.ops.aten.eq.Scalar(squeeze_302, 96)
        where_319 = torch.ops.aten.where.self(eq_1297, mul_2018, where_318);  eq_1297 = where_318 = None
        eq_1298 = torch.ops.aten.eq.Scalar(squeeze_302, 412)
        where_320 = torch.ops.aten.where.self(eq_1298, mul_2026, where_319);  eq_1298 = where_319 = None
        eq_1299 = torch.ops.aten.eq.Scalar(squeeze_302, 412)
        bitwise_and_10 = torch.ops.aten.bitwise_and.Tensor(eq_1299, logical_and_6);  eq_1299 = logical_and_6 = None
        where_321 = torch.ops.aten.where.self(bitwise_and_10, mul_2018, where_320);  bitwise_and_10 = where_320 = None
        eq_1300 = torch.ops.aten.eq.Scalar(squeeze_302, 172)
        where_322 = torch.ops.aten.where.self(eq_1300, add_1678, add_1680);  eq_1300 = None
        eq_1301 = torch.ops.aten.eq.Scalar(squeeze_302, 412)
        where_323 = torch.ops.aten.where.self(eq_1301, add_1682, where_322);  eq_1301 = where_322 = None
        eq_1302 = torch.ops.aten.eq.Scalar(squeeze_302, 96);  squeeze_302 = None
        where_324 = torch.ops.aten.where.self(eq_1302, add_1674, where_323);  eq_1302 = where_323 = None
        return (logical_not, logical_not_2, where_12, where_13, where_14, where_15, where_16, cat_1, sum_1, addmm_7, mul_14, mul_16, mul_17, mul_18, sum_4, mul_19, mul_20, sum_7, mul_21, mul_22, sum_10, mul_23, mul_24, sum_13, mul_25, mul_26, sum_16, mul_27, mul_28, sum_19, mul_29, mul_30, sum_22, mul_31, mul_32, sum_25, mul_33, mul_34, sum_28, mul_35, mul_36, sum_31, mul_37, mul_38, sum_34, cat_44, cat_45, cat_46, cat_47, addmm_30, cat_49, clone_4, cat_51, clone_5, cat_52, addmm_31, addmm_32, cat_53, clone_6, cat_54, cat_55, cat_58, where_156, sum_41, where_192, where_193, where_170, sum_51, cat_85, cat_86, where_177, sum_61, cat_88, cat_89, where_135, sum_71, where_194, where_195, where_184, sum_81, cat_94, cat_95, sum_91, cat_97, cat_98, where_191, sum_101, cat_100, cat_101, where_149, sum_111, cat_103, cat_104, where_163, sum_121, where_196, where_197, where_128, sum_131, where_198, where_199, where_121, sum_141, where_200, where_201, where_142, sum_145, sum_146, sum_147, sum_148, sum_149, sum_150, cat_116, cat_117, where_204, squeeze_138, squeeze_139, where_206, squeeze_141, squeeze_142, where_208, squeeze_144, squeeze_145, where_210, squeeze_147, squeeze_148, where_212, squeeze_150, squeeze_151, cat_123, cat_124, mul_1286, sum_166, cat_125, cat_126, add_1097, where_239, div, div_1, addmm_294, sum_189, sum_190, addmm_297, cat_138, mul_1317, cat_139, addmm_301, cat_140, clone_44, getitem_99, getitem_100, getitem_101, getitem_102, addmm_302, addmm_303, cat_143, cat_144, cat_145, cat_146, cat_147, cat_148, clone_45, clone_46, cat_149, clone_47, clone_48, clone_49, convert_element_type_146, squeeze_218, squeeze_219, cat_154, cat_155, sum_195, sum_196, sum_197, sum_198, sum_199, where_257, where_258, where_259, where_260, where_262, squeeze_271, squeeze_272, squeeze_276, squeeze_277, squeeze_281, squeeze_282, squeeze_286, squeeze_287, squeeze_291, squeeze_292, add_1674, where_287, add_1676, where_289, add_1678, where_291, add_1680, where_293, add_1682, where_295, add_1684, add_1685, add_1686, add_1687, add_1688, view_2, view, view_1, view_3, full_default_318, convert_element_type_1, convert_element_type_2, mul_2017, mul_2018, mul_2019, mul_2020, mul_2021, mul_2022, mul_2023, mul_2024, mul_2025, mul_2026, where_313, where_314, addmm_486, add_1701, addmm_489, add_1702, addmm_492, add_1703, addmm_495, add_1704, addmm_498, add_1705, convert_element_type_207, mul_2040, convert_element_type_209, mul_2041, convert_element_type_211, mul_2042, convert_element_type_213, mul_2043, slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13, slice_14, slice_15, slice_16, slice_17, slice_18, slice_19, slice_20, slice_21, slice_22, slice_23, slice_24, slice_25, slice_26, slice_27, slice_28, slice_29, slice_41, slice_42, slice_43, slice_44, slice_45, slice_46, slice_47, slice_48, slice_55, slice_56, slice_57, slice_58, slice_59, slice_60, slice_61, slice_62, slice_63, slice_64, slice_65, slice_66, slice_67, where_37, slice_69, slice_70, slice_71, slice_72, slice_73, slice_74, slice_75, slice_76, slice_77, slice_78, slice_79, slice_80, slice_81, slice_82, slice_83, slice_84, slice_85, slice_86, slice_87, slice_88, slice_89, slice_90, slice_91, slice_92, slice_93, slice_94, slice_95, slice_96, slice_97, slice_98, slice_99, where_38, where_39, where_40, where_41, where_42, where_43, where_44, where_45, where_46, where_47, where_48, where_49, where_50, where_51, where_52, where_53, where_54, where_55, where_56, where_57, where_58, where_59, where_60, where_61, where_62, where_63, where_64, slice_136, slice_137, slice_138, slice_139, slice_140, slice_141, slice_142, slice_143, slice_144, slice_145, slice_146, slice_147, slice_148, slice_149, slice_150, slice_151, slice_152, slice_153, slice_154, slice_155, slice_156, slice_157, slice_158, slice_159, slice_160, slice_161, slice_162, slice_163, slice_164, slice_165, slice_166, slice_167, slice_168, slice_169, slice_170, slice_171, where_65, where_66, where_67, slice_175, slice_176, slice_177, slice_178, slice_179, slice_180, slice_181, slice_182, slice_183, slice_184, where_68, where_69, where_70, where_71, where_72, where_73, where_74, where_75, slice_193, slice_194, slice_195, where_76, where_77, slice_198, where_78, where_79, slice_201, where_80, where_81, where_82, slice_210, slice_211, slice_212, slice_213, slice_214, slice_215, slice_216, slice_217, slice_218, slice_219, slice_220, slice_221, slice_222, slice_223, slice_224, slice_225, slice_226, slice_227, slice_228, slice_229, slice_230, slice_231, slice_232, slice_233, slice_234, slice_235, slice_236, slice_237, slice_238, slice_239, slice_240, slice_241, slice_242, slice_243, slice_244, slice_245, slice_246, slice_247, slice_248, slice_249, slice_250, slice_251, slice_252, slice_253, slice_254, slice_255, slice_256, slice_257, slice_258, slice_259, slice_260, slice_261, slice_262, slice_263, slice_264, slice_265, slice_266, slice_267, slice_268, slice_269, slice_270, slice_271, slice_272, slice_273, where_115, slice_276, slice_282, slice_283, slice_284, slice_285, slice_286, slice_287, slice_288, slice_289, slice_290, slice_291, slice_292, slice_293, slice_294, slice_295, slice_296, where_122, slice_299, slice_305, slice_306, slice_307, slice_308, slice_309, slice_310, slice_311, slice_312, slice_313, slice_314, slice_315, slice_316, slice_317, slice_318, slice_319, where_129, slice_322, slice_328, slice_329, slice_330, slice_331, slice_332, slice_333, slice_334, slice_335, slice_336, slice_337, slice_338, slice_339, slice_340, slice_341, slice_342, where_136, slice_345, slice_351, slice_352, slice_353, slice_354, slice_355, slice_356, slice_357, slice_358, slice_359, slice_360, slice_361, slice_362, slice_363, slice_364, slice_365, where_143, slice_368, slice_374, slice_375, slice_376, slice_377, slice_378, slice_379, slice_380, slice_381, slice_382, slice_383, slice_384, slice_385, slice_386, slice_387, slice_388, where_150, slice_391, slice_397, slice_398, slice_399, slice_400, slice_401, slice_402, slice_403, slice_404, slice_405, slice_406, slice_407, slice_408, slice_409, slice_410, slice_411, where_157, slice_414, slice_420, slice_421, slice_422, slice_423, slice_424, slice_425, slice_426, slice_427, slice_428, slice_429, slice_430, slice_431, slice_432, slice_433, slice_434, where_164, slice_437, slice_443, slice_444, slice_445, slice_446, slice_447, slice_448, slice_449, slice_450, slice_451, slice_452, slice_453, slice_454, slice_455, slice_456, slice_457, where_171, slice_460, slice_466, slice_467, slice_468, slice_469, slice_470, slice_471, slice_472, slice_473, slice_474, slice_475, slice_476, slice_477, slice_478, slice_479, slice_480, where_178, slice_483, slice_489, slice_490, slice_491, slice_492, slice_493, slice_494, slice_495, slice_496, slice_497, slice_498, slice_499, slice_500, slice_501, slice_502, slice_503, where_185, slice_506, slice_715, slice_716, slice_717, slice_753, slice_754, slice_755, slice_756, slice_757, slice_758, where_304, where_305, where_321, where_324)
        
def load_args(reader):
    buf0 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf3, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf4, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf5, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf6, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf7, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf8, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf9, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf10, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf11, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf12, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf13, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf14, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 27950400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf15, (batch_size_hint, 69876), dtype=torch.float16, is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 204800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf16, (batch_size_hint, 512), dtype=torch.float16, is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 12800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf17, (batch_size_hint, 32), dtype=torch.float16, is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 12800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf18, (batch_size_hint, 32), dtype=torch.float16, is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 12800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf19, (batch_size_hint, 32), dtype=torch.float16, is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 12800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf20, (batch_size_hint, 32), dtype=torch.float16, is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 327680, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf21, (640, 256), dtype=torch.float16, is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf22, (256,), dtype=torch.float16, is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf23, (256, 128), dtype=torch.float16, is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf24, (128,), dtype=torch.float16, is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 131200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf25, (batch_size_hint, 328), dtype=torch.float16, is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf26, (128, 128), dtype=torch.float16, is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf27, (128,), dtype=torch.float16, is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf28, (128, 64), dtype=torch.float16, is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf29, (64,), dtype=torch.float16, is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 102400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf30, (batch_size_hint, 256), dtype=torch.float16, is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf31, (256, 128), dtype=torch.float16, is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf32, (128,), dtype=torch.float16, is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf33, (128, 64), dtype=torch.float16, is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf34, (64,), dtype=torch.float16, is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 102400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf35, (batch_size_hint, 256), dtype=torch.float16, is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf36, (1,), dtype=torch.float16, is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf37, (batch_size_hint, 128), dtype=torch.float16, is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf38, (128, 128), dtype=torch.float16, is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf39, (128,), dtype=torch.float16, is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf40, (128, 64), dtype=torch.float16, is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf41, (64,), dtype=torch.float16, is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf42, (1,), dtype=torch.float16, is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 131072, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf43, (256, 256), dtype=torch.float16, is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf44, (256,), dtype=torch.float16, is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 983040, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf45, (1920, 256), dtype=torch.float16, is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf46, (256,), dtype=torch.float16, is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 180224, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf47, (352, 256), dtype=torch.float16, is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf48, (256,), dtype=torch.float16, is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 147456, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf49, (288, 256), dtype=torch.float16, is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf50, (256,), dtype=torch.float16, is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 24576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf51, (48, 256), dtype=torch.float16, is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf52, (256,), dtype=torch.float16, is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 98304, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf53, (192, 256), dtype=torch.float16, is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf54, (256,), dtype=torch.float16, is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf55, (32, 256), dtype=torch.float16, is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf56, (256,), dtype=torch.float16, is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 73728, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf57, (144, 256), dtype=torch.float16, is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf58, (256,), dtype=torch.float16, is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf59, (128, 256), dtype=torch.float16, is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf60, (256,), dtype=torch.float16, is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf61, (160, 256), dtype=torch.float16, is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf62, (256,), dtype=torch.float16, is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf63, (160, 256), dtype=torch.float16, is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf64, (256,), dtype=torch.float16, is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf65, (160, 256), dtype=torch.float16, is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf66, (256,), dtype=torch.float16, is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf67, (64, 256), dtype=torch.float16, is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf68, (256,), dtype=torch.float16, is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 73728, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf69, (144, 256), dtype=torch.float16, is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf70, (256,), dtype=torch.float16, is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf71, (64, 256), dtype=torch.float16, is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf72, (256,), dtype=torch.float16, is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 49152, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf73, (96, 256), dtype=torch.float16, is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf74, (256,), dtype=torch.float16, is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf75, (16, 256), dtype=torch.float16, is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf76, (256,), dtype=torch.float16, is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf77, (64, 256), dtype=torch.float16, is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf78, (256,), dtype=torch.float16, is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 114688, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf79, (224, 256), dtype=torch.float16, is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf80, (256,), dtype=torch.float16, is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 49152, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf81, (96, 256), dtype=torch.float16, is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf82, (256,), dtype=torch.float16, is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 827392, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf83, (1616, 256), dtype=torch.float16, is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf84, (256,), dtype=torch.float16, is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 188416, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf85, (368, 256), dtype=torch.float16, is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf86, (256,), dtype=torch.float16, is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1286144, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf87, (2512, 256), dtype=torch.float16, is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf88, (256,), dtype=torch.float16, is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1009600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf89, (batch_size_hint, 2524), dtype=torch.float16, is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 4800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf90, (batch_size_hint, 12), dtype=torch.float16, is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf91, (batch_size_hint, 2), dtype=torch.float16, is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf92, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 132000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf93, (batch_size_hint, 330), dtype=torch.float16, is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 11200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf94, (batch_size_hint, 28), dtype=torch.float16, is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 122400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf95, (batch_size_hint, 306), dtype=torch.float16, is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 189600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf96, (batch_size_hint, 474), dtype=torch.float16, is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf97, (batch_size_hint, 128), dtype=torch.float16, is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1292288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf98, (2524, 256), dtype=torch.float16, is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf99, (256,), dtype=torch.float16, is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 657408, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf100, (1284, 256), dtype=torch.float16, is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf101, (256,), dtype=torch.float16, is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf102, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 4800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf103, (batch_size_hint, 12), dtype=torch.float16, is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf104, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf105, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 17600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf106, (batch_size_hint, 44), dtype=torch.float16, is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 12800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf107, (batch_size_hint, 32), dtype=torch.float16, is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 4800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf108, (batch_size_hint, 12), dtype=torch.float16, is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 12800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf109, (batch_size_hint, 32), dtype=torch.float16, is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf110, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf111, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf112, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf113, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf114, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf115, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf116, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf117, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 4800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf118, (batch_size_hint, 12), dtype=torch.float16, is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf119, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf120, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 2400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf121, (batch_size_hint, 6), dtype=torch.float16, is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 6400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf122, (batch_size_hint, 16), dtype=torch.float16, is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 22400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf123, (batch_size_hint, 56), dtype=torch.float16, is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 24800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf124, (batch_size_hint, 62), dtype=torch.float16, is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf125, (batch_size_hint, 2), dtype=torch.float16, is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf126, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf127, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 21600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf128, (batch_size_hint, 54), dtype=torch.float16, is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf129, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 7200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf130, (batch_size_hint, 18), dtype=torch.float16, is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 230400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf131, (batch_size_hint, 576), dtype=torch.float16, is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1792, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf132, (7, 128), dtype=torch.float16, is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1510400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf133, (batch_size_hint, 3776), dtype=torch.float16, is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 256000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf134, (batch_size_hint, 640), dtype=torch.float16, is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 2335616, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf135, (4544, 257), dtype=torch.float16, is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 514, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf136, (257,), dtype=torch.float16, is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 1792, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf137, (7, 128), dtype=torch.float16, is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 25600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf138, (batch_size_hint, 64), dtype=torch.float16, is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 153600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf139, (batch_size_hint, 384), dtype=torch.float16, is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1638400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf140, (batch_size_hint, 4096), dtype=torch.float16, is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 128000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf141, (batch_size_hint, 320), dtype=torch.float16, is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 246400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf142, (batch_size_hint, 616), dtype=torch.float16, is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 249856, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf143, (976, 128), dtype=torch.float16, is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf144, (128,), dtype=torch.float16, is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf145, (128, 128), dtype=torch.float16, is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf146, (128,), dtype=torch.float16, is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 56, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf147, (7,), dtype=torch.int64, is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 143360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf148, (560, 128), dtype=torch.float16, is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf149, (128,), dtype=torch.float16, is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf150, (128, 32), dtype=torch.float16, is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf151, (32,), dtype=torch.float16, is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 90112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf152, (352, 128), dtype=torch.float16, is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf153, (128,), dtype=torch.float16, is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf154, (128, 32), dtype=torch.float16, is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf155, (32,), dtype=torch.float16, is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 143360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf156, (560, 128), dtype=torch.float16, is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf157, (128,), dtype=torch.float16, is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf158, (128, 32), dtype=torch.float16, is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf159, (32,), dtype=torch.float16, is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 90112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf160, (352, 128), dtype=torch.float16, is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf161, (128,), dtype=torch.float16, is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf162, (128, 32), dtype=torch.float16, is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf163, (32,), dtype=torch.float16, is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 143360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf164, (560, 128), dtype=torch.float16, is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf165, (128,), dtype=torch.float16, is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf166, (128, 32), dtype=torch.float16, is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf167, (32,), dtype=torch.float16, is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 90112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf168, (352, 128), dtype=torch.float16, is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf169, (128,), dtype=torch.float16, is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf170, (128, 32), dtype=torch.float16, is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf171, (32,), dtype=torch.float16, is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 143360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf172, (560, 128), dtype=torch.float16, is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf173, (128,), dtype=torch.float16, is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf174, (128, 32), dtype=torch.float16, is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf175, (32,), dtype=torch.float16, is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 90112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf176, (352, 128), dtype=torch.float16, is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf177, (128,), dtype=torch.float16, is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf178, (128, 32), dtype=torch.float16, is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf179, (32,), dtype=torch.float16, is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 143360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf180, (560, 128), dtype=torch.float16, is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf181, (128,), dtype=torch.float16, is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf182, (128, 32), dtype=torch.float16, is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf183, (32,), dtype=torch.float16, is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 90112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf184, (352, 128), dtype=torch.float16, is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf185, (128,), dtype=torch.float16, is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf186, (128, 32), dtype=torch.float16, is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf187, (32,), dtype=torch.float16, is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 143360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf188, (560, 128), dtype=torch.float16, is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf189, (128,), dtype=torch.float16, is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf190, (128, 32), dtype=torch.float16, is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf191, (32,), dtype=torch.float16, is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 90112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf192, (352, 128), dtype=torch.float16, is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf193, (128,), dtype=torch.float16, is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf194, (128, 32), dtype=torch.float16, is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf195, (32,), dtype=torch.float16, is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 253952, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf196, (992, 128), dtype=torch.float16, is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf197, (128,), dtype=torch.float16, is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf198, (128, 64), dtype=torch.float16, is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf199, (64,), dtype=torch.float16, is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 147456, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf200, (576, 128), dtype=torch.float16, is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf201, (128,), dtype=torch.float16, is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf202, (128, 64), dtype=torch.float16, is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf203, (64,), dtype=torch.float16, is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 143360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf204, (560, 128), dtype=torch.float16, is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf205, (128,), dtype=torch.float16, is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf206, (128, 32), dtype=torch.float16, is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf207, (32,), dtype=torch.float16, is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 90112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf208, (352, 128), dtype=torch.float16, is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf209, (128,), dtype=torch.float16, is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf210, (128, 32), dtype=torch.float16, is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf211, (32,), dtype=torch.float16, is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 143360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf212, (560, 128), dtype=torch.float16, is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf213, (128,), dtype=torch.float16, is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf214, (128, 32), dtype=torch.float16, is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf215, (32,), dtype=torch.float16, is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 90112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf216, (352, 128), dtype=torch.float16, is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf217, (128,), dtype=torch.float16, is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf218, (128, 32), dtype=torch.float16, is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf219, (32,), dtype=torch.float16, is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 143360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf220, (560, 128), dtype=torch.float16, is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf221, (128,), dtype=torch.float16, is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf222, (128, 32), dtype=torch.float16, is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf223, (32,), dtype=torch.float16, is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 90112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf224, (352, 128), dtype=torch.float16, is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf225, (128,), dtype=torch.float16, is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf226, (128, 32), dtype=torch.float16, is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf227, (32,), dtype=torch.float16, is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 143360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf228, (560, 128), dtype=torch.float16, is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf229, (128,), dtype=torch.float16, is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf230, (128, 32), dtype=torch.float16, is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf231, (32,), dtype=torch.float16, is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 90112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf232, (352, 128), dtype=torch.float16, is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf233, (128,), dtype=torch.float16, is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf234, (128, 32), dtype=torch.float16, is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf235, (32,), dtype=torch.float16, is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf236, (32, 64), dtype=torch.float16, is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf237, (64,), dtype=torch.float16, is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 228480, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf238, (7, 480, 34), dtype=torch.float16, is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 13440, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf239, (7, 480), dtype=torch.int32, is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 31680, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf240, (1, 480, 33), dtype=torch.float16, is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf241, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf242, (512,), dtype=torch.float16, is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf243, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf244, (1024,), dtype=torch.float16, is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf245, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf246, (512,), dtype=torch.float16, is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf247, (512, 32), dtype=torch.float16, is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf248, (32,), dtype=torch.float16, is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf249, (128, 256), dtype=torch.float16, is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf250, (256,), dtype=torch.float16, is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf251, (256, 128), dtype=torch.float16, is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf252, (128,), dtype=torch.float16, is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf253, (128, 32), dtype=torch.float16, is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf254, (32,), dtype=torch.float16, is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 4)
    reader.tensor(buf255, (), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf256, (32, 64), dtype=torch.float16, is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf257, (64,), dtype=torch.float16, is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 1664000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf258, (batch_size_hint, 32, 130), dtype=torch.float16, is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 25600, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf259, (batch_size_hint, 32), dtype=torch.int32, is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 2112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf260, (1, 32, 33), dtype=torch.float16, is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf261, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf262, (512,), dtype=torch.float16, is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 4194304, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf263, (512, 4096), dtype=torch.float16, is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf264, (4096,), dtype=torch.float16, is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf265, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf266, (512,), dtype=torch.float16, is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf267, (512, 32), dtype=torch.float16, is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf268, (32,), dtype=torch.float16, is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf269, (128, 256), dtype=torch.float16, is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf270, (256,), dtype=torch.float16, is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf271, (256, 128), dtype=torch.float16, is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf272, (128,), dtype=torch.float16, is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf273, (128, 32), dtype=torch.float16, is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf274, (32,), dtype=torch.float16, is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 4)
    reader.tensor(buf275, (), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf276, (32, 64), dtype=torch.float16, is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf277, (64,), dtype=torch.float16, is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 3328000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf278, (batch_size_hint, 64, 130), dtype=torch.float16, is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf279, (batch_size_hint, 64), dtype=torch.int32, is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 4224, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf280, (1, 64, 33), dtype=torch.float16, is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf281, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf282, (512,), dtype=torch.float16, is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 4194304, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf283, (512, 4096), dtype=torch.float16, is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf284, (4096,), dtype=torch.float16, is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf285, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf286, (512,), dtype=torch.float16, is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf287, (512, 32), dtype=torch.float16, is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf288, (32,), dtype=torch.float16, is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf289, (128, 256), dtype=torch.float16, is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf290, (256,), dtype=torch.float16, is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf291, (256, 128), dtype=torch.float16, is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf292, (128,), dtype=torch.float16, is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf293, (128, 32), dtype=torch.float16, is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf294, (32,), dtype=torch.float16, is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 4)
    reader.tensor(buf295, (), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf296, (32, 64), dtype=torch.float16, is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf297, (64,), dtype=torch.float16, is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 205632, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf298, (7, 432, 34), dtype=torch.float16, is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 12096, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf299, (7, 432), dtype=torch.int32, is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 28512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf300, (1, 432, 33), dtype=torch.float16, is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf301, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf302, (512,), dtype=torch.float16, is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf303, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf304, (1024,), dtype=torch.float16, is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf305, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf306, (512,), dtype=torch.float16, is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf307, (512, 32), dtype=torch.float16, is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf308, (32,), dtype=torch.float16, is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf309, (128, 256), dtype=torch.float16, is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf310, (256,), dtype=torch.float16, is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf311, (256, 128), dtype=torch.float16, is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf312, (128,), dtype=torch.float16, is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf313, (128, 32), dtype=torch.float16, is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf314, (32,), dtype=torch.float16, is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 4)
    reader.tensor(buf315, (), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf316, (32, 64), dtype=torch.float16, is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf317, (64,), dtype=torch.float16, is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 870400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf318, (batch_size_hint, 64, 34), dtype=torch.float16, is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf319, (batch_size_hint, 64), dtype=torch.int32, is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 4224, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf320, (1, 64, 33), dtype=torch.float16, is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf321, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf322, (512,), dtype=torch.float16, is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf323, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf324, (1024,), dtype=torch.float16, is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf325, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf326, (512,), dtype=torch.float16, is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf327, (512, 32), dtype=torch.float16, is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf328, (32,), dtype=torch.float16, is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf329, (128, 256), dtype=torch.float16, is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf330, (256,), dtype=torch.float16, is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf331, (256, 128), dtype=torch.float16, is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf332, (128,), dtype=torch.float16, is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf333, (128, 32), dtype=torch.float16, is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf334, (32,), dtype=torch.float16, is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 4)
    reader.tensor(buf335, (), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf336, (32, 64), dtype=torch.float16, is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf337, (64,), dtype=torch.float16, is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 870400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf338, (batch_size_hint, 64, 34), dtype=torch.float16, is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf339, (batch_size_hint, 64), dtype=torch.int32, is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 4224, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf340, (1, 64, 33), dtype=torch.float16, is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf341, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf342, (512,), dtype=torch.float16, is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf343, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf344, (1024,), dtype=torch.float16, is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf345, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf346, (512,), dtype=torch.float16, is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf347, (512, 32), dtype=torch.float16, is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf348, (32,), dtype=torch.float16, is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf349, (128, 256), dtype=torch.float16, is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf350, (256,), dtype=torch.float16, is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf351, (256, 128), dtype=torch.float16, is_leaf=True)  # arg351_1
    buf352 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf352, (128,), dtype=torch.float16, is_leaf=True)  # arg352_1
    buf353 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf353, (128, 32), dtype=torch.float16, is_leaf=True)  # arg353_1
    buf354 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf354, (32,), dtype=torch.float16, is_leaf=True)  # arg354_1
    buf355 = reader.storage(None, 4)
    reader.tensor(buf355, (), is_leaf=True)  # arg355_1
    buf356 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf356, (32, 64), dtype=torch.float16, is_leaf=True)  # arg356_1
    buf357 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf357, (64,), dtype=torch.float16, is_leaf=True)  # arg357_1
    buf358 = reader.storage(None, 3328000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf358, (batch_size_hint, 64, 130), dtype=torch.float16, is_leaf=True)  # arg358_1
    buf359 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf359, (batch_size_hint, 64), dtype=torch.int32, is_leaf=True)  # arg359_1
    buf360 = reader.storage(None, 4224, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf360, (1, 64, 33), dtype=torch.float16, is_leaf=True)  # arg360_1
    buf361 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf361, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg361_1
    buf362 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf362, (512,), dtype=torch.float16, is_leaf=True)  # arg362_1
    buf363 = reader.storage(None, 4194304, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf363, (512, 4096), dtype=torch.float16, is_leaf=True)  # arg363_1
    buf364 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf364, (4096,), dtype=torch.float16, is_leaf=True)  # arg364_1
    buf365 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf365, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg365_1
    buf366 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf366, (512,), dtype=torch.float16, is_leaf=True)  # arg366_1
    buf367 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf367, (512, 32), dtype=torch.float16, is_leaf=True)  # arg367_1
    buf368 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf368, (32,), dtype=torch.float16, is_leaf=True)  # arg368_1
    buf369 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf369, (128, 256), dtype=torch.float16, is_leaf=True)  # arg369_1
    buf370 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf370, (256,), dtype=torch.float16, is_leaf=True)  # arg370_1
    buf371 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf371, (256, 128), dtype=torch.float16, is_leaf=True)  # arg371_1
    buf372 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf372, (128,), dtype=torch.float16, is_leaf=True)  # arg372_1
    buf373 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf373, (128, 32), dtype=torch.float16, is_leaf=True)  # arg373_1
    buf374 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf374, (32,), dtype=torch.float16, is_leaf=True)  # arg374_1
    buf375 = reader.storage(None, 4)
    reader.tensor(buf375, (), is_leaf=True)  # arg375_1
    buf376 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf376, (32, 64), dtype=torch.float16, is_leaf=True)  # arg376_1
    buf377 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf377, (64,), dtype=torch.float16, is_leaf=True)  # arg377_1
    buf378 = reader.storage(None, 2112000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf378, (batch_size_hint, 80, 66), dtype=torch.float16, is_leaf=True)  # arg378_1
    buf379 = reader.storage(None, 64000, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf379, (batch_size_hint, 80), dtype=torch.int32, is_leaf=True)  # arg379_1
    buf380 = reader.storage(None, 5280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf380, (1, 80, 33), dtype=torch.float16, is_leaf=True)  # arg380_1
    buf381 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf381, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg381_1
    buf382 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf382, (512,), dtype=torch.float16, is_leaf=True)  # arg382_1
    buf383 = reader.storage(None, 2097152, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf383, (512, 2048), dtype=torch.float16, is_leaf=True)  # arg383_1
    buf384 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf384, (2048,), dtype=torch.float16, is_leaf=True)  # arg384_1
    buf385 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf385, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg385_1
    buf386 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf386, (512,), dtype=torch.float16, is_leaf=True)  # arg386_1
    buf387 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf387, (512, 32), dtype=torch.float16, is_leaf=True)  # arg387_1
    buf388 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf388, (32,), dtype=torch.float16, is_leaf=True)  # arg388_1
    buf389 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf389, (128, 256), dtype=torch.float16, is_leaf=True)  # arg389_1
    buf390 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf390, (256,), dtype=torch.float16, is_leaf=True)  # arg390_1
    buf391 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf391, (256, 128), dtype=torch.float16, is_leaf=True)  # arg391_1
    buf392 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf392, (128,), dtype=torch.float16, is_leaf=True)  # arg392_1
    buf393 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf393, (128, 32), dtype=torch.float16, is_leaf=True)  # arg393_1
    buf394 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf394, (32,), dtype=torch.float16, is_leaf=True)  # arg394_1
    buf395 = reader.storage(None, 4)
    reader.tensor(buf395, (), is_leaf=True)  # arg395_1
    buf396 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf396, (64, 128), dtype=torch.float16, is_leaf=True)  # arg396_1
    buf397 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf397, (128,), dtype=torch.float16, is_leaf=True)  # arg397_1
    buf398 = reader.storage(None, 4435200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf398, (7, 4800, 66), dtype=torch.float16, is_leaf=True)  # arg398_1
    buf399 = reader.storage(None, 134400, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf399, (7, 4800), dtype=torch.int32, is_leaf=True)  # arg399_1
    buf400 = reader.storage(None, 624000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf400, (1, 4800, 65), dtype=torch.float16, is_leaf=True)  # arg400_1
    buf401 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf401, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg401_1
    buf402 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf402, (512,), dtype=torch.float16, is_leaf=True)  # arg402_1
    buf403 = reader.storage(None, 4194304, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf403, (512, 4096), dtype=torch.float16, is_leaf=True)  # arg403_1
    buf404 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf404, (4096,), dtype=torch.float16, is_leaf=True)  # arg404_1
    buf405 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf405, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg405_1
    buf406 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf406, (512,), dtype=torch.float16, is_leaf=True)  # arg406_1
    buf407 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf407, (512, 64), dtype=torch.float16, is_leaf=True)  # arg407_1
    buf408 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf408, (64,), dtype=torch.float16, is_leaf=True)  # arg408_1
    buf409 = reader.storage(None, 131072, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf409, (256, 256), dtype=torch.float16, is_leaf=True)  # arg409_1
    buf410 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf410, (256,), dtype=torch.float16, is_leaf=True)  # arg410_1
    buf411 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf411, (256, 128), dtype=torch.float16, is_leaf=True)  # arg411_1
    buf412 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf412, (128,), dtype=torch.float16, is_leaf=True)  # arg412_1
    buf413 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf413, (128, 64), dtype=torch.float16, is_leaf=True)  # arg413_1
    buf414 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf414, (64,), dtype=torch.float16, is_leaf=True)  # arg414_1
    buf415 = reader.storage(None, 4)
    reader.tensor(buf415, (), is_leaf=True)  # arg415_1
    buf416 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf416, (32, 64), dtype=torch.float16, is_leaf=True)  # arg416_1
    buf417 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf417, (64,), dtype=torch.float16, is_leaf=True)  # arg417_1
    buf418 = reader.storage(None, 331296, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf418, (7, 696, 34), dtype=torch.float16, is_leaf=True)  # arg418_1
    buf419 = reader.storage(None, 19488, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf419, (7, 696), dtype=torch.int32, is_leaf=True)  # arg419_1
    buf420 = reader.storage(None, 45936, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf420, (1, 696, 33), dtype=torch.float16, is_leaf=True)  # arg420_1
    buf421 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf421, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg421_1
    buf422 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf422, (512,), dtype=torch.float16, is_leaf=True)  # arg422_1
    buf423 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf423, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg423_1
    buf424 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf424, (1024,), dtype=torch.float16, is_leaf=True)  # arg424_1
    buf425 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf425, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg425_1
    buf426 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf426, (512,), dtype=torch.float16, is_leaf=True)  # arg426_1
    buf427 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf427, (512, 32), dtype=torch.float16, is_leaf=True)  # arg427_1
    buf428 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf428, (32,), dtype=torch.float16, is_leaf=True)  # arg428_1
    buf429 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf429, (128, 256), dtype=torch.float16, is_leaf=True)  # arg429_1
    buf430 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf430, (256,), dtype=torch.float16, is_leaf=True)  # arg430_1
    buf431 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf431, (256, 128), dtype=torch.float16, is_leaf=True)  # arg431_1
    buf432 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf432, (128,), dtype=torch.float16, is_leaf=True)  # arg432_1
    buf433 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf433, (128, 32), dtype=torch.float16, is_leaf=True)  # arg433_1
    buf434 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf434, (32,), dtype=torch.float16, is_leaf=True)  # arg434_1
    buf435 = reader.storage(None, 4)
    reader.tensor(buf435, (), is_leaf=True)  # arg435_1
    buf436 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf436, (32, 64), dtype=torch.float16, is_leaf=True)  # arg436_1
    buf437 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf437, (64,), dtype=torch.float16, is_leaf=True)  # arg437_1
    buf438 = reader.storage(None, 327488, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf438, (7, 688, 34), dtype=torch.float16, is_leaf=True)  # arg438_1
    buf439 = reader.storage(None, 19264, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf439, (7, 688), dtype=torch.int32, is_leaf=True)  # arg439_1
    buf440 = reader.storage(None, 45408, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf440, (1, 688, 33), dtype=torch.float16, is_leaf=True)  # arg440_1
    buf441 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf441, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg441_1
    buf442 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf442, (512,), dtype=torch.float16, is_leaf=True)  # arg442_1
    buf443 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf443, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg443_1
    buf444 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf444, (1024,), dtype=torch.float16, is_leaf=True)  # arg444_1
    buf445 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf445, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg445_1
    buf446 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf446, (512,), dtype=torch.float16, is_leaf=True)  # arg446_1
    buf447 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf447, (512, 32), dtype=torch.float16, is_leaf=True)  # arg447_1
    buf448 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf448, (32,), dtype=torch.float16, is_leaf=True)  # arg448_1
    buf449 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf449, (128, 256), dtype=torch.float16, is_leaf=True)  # arg449_1
    buf450 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf450, (256,), dtype=torch.float16, is_leaf=True)  # arg450_1
    buf451 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf451, (256, 128), dtype=torch.float16, is_leaf=True)  # arg451_1
    buf452 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf452, (128,), dtype=torch.float16, is_leaf=True)  # arg452_1
    buf453 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf453, (128, 32), dtype=torch.float16, is_leaf=True)  # arg453_1
    buf454 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf454, (32,), dtype=torch.float16, is_leaf=True)  # arg454_1
    buf455 = reader.storage(None, 4)
    reader.tensor(buf455, (), is_leaf=True)  # arg455_1
    buf456 = reader.storage(None, 286720, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf456, (560, 256), dtype=torch.float16, is_leaf=True)  # arg456_1
    buf457 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf457, (256,), dtype=torch.float16, is_leaf=True)  # arg457_1
    buf458 = reader.storage(None, 180224, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf458, (352, 256), dtype=torch.float16, is_leaf=True)  # arg458_1
    buf459 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf459, (256,), dtype=torch.float16, is_leaf=True)  # arg459_1
    buf460 = reader.storage(None, 3328000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf460, (batch_size_hint, 64, 130), dtype=torch.float16, is_leaf=True)  # arg460_1
    buf461 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf461, (batch_size_hint, 64), dtype=torch.int32, is_leaf=True)  # arg461_1
    buf462 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf462, (1, 64, 1), dtype=torch.float16, is_leaf=True)  # arg462_1
    buf463 = reader.storage(None, 3328000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf463, (batch_size_hint, 64, 130), dtype=torch.float16, is_leaf=True)  # arg463_1
    buf464 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf464, (batch_size_hint, 64), dtype=torch.int32, is_leaf=True)  # arg464_1
    buf465 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf465, (1, 64, 1), dtype=torch.float16, is_leaf=True)  # arg465_1
    buf466 = reader.storage(None, 3328000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf466, (batch_size_hint, 64, 130), dtype=torch.float16, is_leaf=True)  # arg466_1
    buf467 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf467, (batch_size_hint, 64), dtype=torch.int32, is_leaf=True)  # arg467_1
    buf468 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf468, (1, 64, 1), dtype=torch.float16, is_leaf=True)  # arg468_1
    buf469 = reader.storage(None, 3328000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf469, (batch_size_hint, 64, 130), dtype=torch.float16, is_leaf=True)  # arg469_1
    buf470 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf470, (batch_size_hint, 64), dtype=torch.int32, is_leaf=True)  # arg470_1
    buf471 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf471, (1, 64, 1), dtype=torch.float16, is_leaf=True)  # arg471_1
    buf472 = reader.storage(None, 3328000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf472, (batch_size_hint, 64, 130), dtype=torch.float16, is_leaf=True)  # arg472_1
    buf473 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf473, (batch_size_hint, 64), dtype=torch.int32, is_leaf=True)  # arg473_1
    buf474 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf474, (1, 64, 1), dtype=torch.float16, is_leaf=True)  # arg474_1
    buf475 = reader.storage(None, 3328000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf475, (batch_size_hint, 64, 130), dtype=torch.float16, is_leaf=True)  # arg475_1
    buf476 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf476, (batch_size_hint, 64), dtype=torch.int32, is_leaf=True)  # arg476_1
    buf477 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf477, (1, 64, 1), dtype=torch.float16, is_leaf=True)  # arg477_1
    buf478 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf478, (1, 64, 32), dtype=torch.float16, is_leaf=True)  # arg478_1
    buf479 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf479, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg479_1
    buf480 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf480, (512,), dtype=torch.float16, is_leaf=True)  # arg480_1
    buf481 = reader.storage(None, 4194304, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf481, (512, 4096), dtype=torch.float16, is_leaf=True)  # arg481_1
    buf482 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf482, (4096,), dtype=torch.float16, is_leaf=True)  # arg482_1
    buf483 = reader.storage(None, 5111808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf483, (4992, 512), dtype=torch.float16, is_leaf=True)  # arg483_1
    buf484 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf484, (512,), dtype=torch.float16, is_leaf=True)  # arg484_1
    buf485 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf485, (512, 32), dtype=torch.float16, is_leaf=True)  # arg485_1
    buf486 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf486, (32,), dtype=torch.float16, is_leaf=True)  # arg486_1
    buf487 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf487, (128, 256), dtype=torch.float16, is_leaf=True)  # arg487_1
    buf488 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf488, (256,), dtype=torch.float16, is_leaf=True)  # arg488_1
    buf489 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf489, (256, 128), dtype=torch.float16, is_leaf=True)  # arg489_1
    buf490 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf490, (128,), dtype=torch.float16, is_leaf=True)  # arg490_1
    buf491 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf491, (128, 32), dtype=torch.float16, is_leaf=True)  # arg491_1
    buf492 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf492, (32,), dtype=torch.float16, is_leaf=True)  # arg492_1
    buf493 = reader.storage(None, 4)
    reader.tensor(buf493, (), is_leaf=True)  # arg493_1
    buf494 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf494, (64, 512), dtype=torch.float16, is_leaf=True)  # arg494_1
    buf495 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf495, (512,), dtype=torch.float16, is_leaf=True)  # arg495_1
    buf496 = reader.storage(None, 1015808, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf496, (992, 512), dtype=torch.float16, is_leaf=True)  # arg496_1
    buf497 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf497, (512,), dtype=torch.float16, is_leaf=True)  # arg497_1
    buf498 = reader.storage(None, 589824, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf498, (576, 512), dtype=torch.float16, is_leaf=True)  # arg498_1
    buf499 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf499, (512,), dtype=torch.float16, is_leaf=True)  # arg499_1
    buf500 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf500, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg500_1
    buf501 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf501, (512,), dtype=torch.float16, is_leaf=True)  # arg501_1
    buf502 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf502, (64, 256), dtype=torch.float16, is_leaf=True)  # arg502_1
    buf503 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf503, (256,), dtype=torch.float16, is_leaf=True)  # arg503_1
    buf504 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf504, (512,), dtype=torch.float16, is_leaf=True)  # arg504_1
    buf505 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf505, (512,), dtype=torch.float16, is_leaf=True)  # arg505_1
    buf506 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf506, (512,), dtype=torch.float16, is_leaf=True)  # arg506_1
    buf507 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf507, (512,), dtype=torch.float16, is_leaf=True)  # arg507_1
    buf508 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf508, (512, 512), dtype=torch.float16, is_leaf=True)  # arg508_1
    buf509 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf509, (512,), dtype=torch.float16, is_leaf=True)  # arg509_1
    buf510 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf510, (512, 512), dtype=torch.float16, is_leaf=True)  # arg510_1
    buf511 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf511, (512,), dtype=torch.float16, is_leaf=True)  # arg511_1
    buf512 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf512, (512, 512), dtype=torch.float16, is_leaf=True)  # arg512_1
    buf513 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf513, (512,), dtype=torch.float16, is_leaf=True)  # arg513_1
    buf514 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf514, (512, 512), dtype=torch.float16, is_leaf=True)  # arg514_1
    buf515 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf515, (512,), dtype=torch.float16, is_leaf=True)  # arg515_1
    buf516 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf516, (512,), dtype=torch.float16, is_leaf=True)  # arg516_1
    buf517 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf517, (512,), dtype=torch.float16, is_leaf=True)  # arg517_1
    buf518 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf518, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg518_1
    buf519 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf519, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg519_1
    buf520 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf520, (1024,), dtype=torch.float16, is_leaf=True)  # arg520_1
    buf521 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf521, (1024,), dtype=torch.float16, is_leaf=True)  # arg521_1
    buf522 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf522, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg522_1
    buf523 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf523, (512,), dtype=torch.float16, is_leaf=True)  # arg523_1
    buf524 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf524, (512,), dtype=torch.float16, is_leaf=True)  # arg524_1
    buf525 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf525, (512,), dtype=torch.float16, is_leaf=True)  # arg525_1
    buf526 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf526, (512, 512), dtype=torch.float16, is_leaf=True)  # arg526_1
    buf527 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf527, (512,), dtype=torch.float16, is_leaf=True)  # arg527_1
    buf528 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf528, (512, 512), dtype=torch.float16, is_leaf=True)  # arg528_1
    buf529 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf529, (512,), dtype=torch.float16, is_leaf=True)  # arg529_1
    buf530 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf530, (512, 512), dtype=torch.float16, is_leaf=True)  # arg530_1
    buf531 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf531, (512,), dtype=torch.float16, is_leaf=True)  # arg531_1
    buf532 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf532, (512, 512), dtype=torch.float16, is_leaf=True)  # arg532_1
    buf533 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf533, (512,), dtype=torch.float16, is_leaf=True)  # arg533_1
    buf534 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf534, (512,), dtype=torch.float16, is_leaf=True)  # arg534_1
    buf535 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf535, (512,), dtype=torch.float16, is_leaf=True)  # arg535_1
    buf536 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf536, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg536_1
    buf537 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf537, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg537_1
    buf538 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf538, (1024,), dtype=torch.float16, is_leaf=True)  # arg538_1
    buf539 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf539, (1024,), dtype=torch.float16, is_leaf=True)  # arg539_1
    buf540 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf540, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg540_1
    buf541 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf541, (512,), dtype=torch.float16, is_leaf=True)  # arg541_1
    buf542 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf542, (512,), dtype=torch.float16, is_leaf=True)  # arg542_1
    buf543 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf543, (512,), dtype=torch.float16, is_leaf=True)  # arg543_1
    buf544 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf544, (512, 512), dtype=torch.float16, is_leaf=True)  # arg544_1
    buf545 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf545, (512,), dtype=torch.float16, is_leaf=True)  # arg545_1
    buf546 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf546, (512, 512), dtype=torch.float16, is_leaf=True)  # arg546_1
    buf547 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf547, (512,), dtype=torch.float16, is_leaf=True)  # arg547_1
    buf548 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf548, (512, 512), dtype=torch.float16, is_leaf=True)  # arg548_1
    buf549 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf549, (512,), dtype=torch.float16, is_leaf=True)  # arg549_1
    buf550 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf550, (512, 512), dtype=torch.float16, is_leaf=True)  # arg550_1
    buf551 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf551, (512,), dtype=torch.float16, is_leaf=True)  # arg551_1
    buf552 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf552, (512,), dtype=torch.float16, is_leaf=True)  # arg552_1
    buf553 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf553, (512,), dtype=torch.float16, is_leaf=True)  # arg553_1
    buf554 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf554, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg554_1
    buf555 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf555, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg555_1
    buf556 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf556, (1024,), dtype=torch.float16, is_leaf=True)  # arg556_1
    buf557 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf557, (1024,), dtype=torch.float16, is_leaf=True)  # arg557_1
    buf558 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf558, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg558_1
    buf559 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf559, (512,), dtype=torch.float16, is_leaf=True)  # arg559_1
    buf560 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf560, (512,), dtype=torch.float16, is_leaf=True)  # arg560_1
    buf561 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf561, (512,), dtype=torch.float16, is_leaf=True)  # arg561_1
    buf562 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf562, (512, 512), dtype=torch.float16, is_leaf=True)  # arg562_1
    buf563 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf563, (512,), dtype=torch.float16, is_leaf=True)  # arg563_1
    buf564 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf564, (512, 512), dtype=torch.float16, is_leaf=True)  # arg564_1
    buf565 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf565, (512,), dtype=torch.float16, is_leaf=True)  # arg565_1
    buf566 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf566, (512, 512), dtype=torch.float16, is_leaf=True)  # arg566_1
    buf567 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf567, (512,), dtype=torch.float16, is_leaf=True)  # arg567_1
    buf568 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf568, (512, 512), dtype=torch.float16, is_leaf=True)  # arg568_1
    buf569 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf569, (512,), dtype=torch.float16, is_leaf=True)  # arg569_1
    buf570 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf570, (512,), dtype=torch.float16, is_leaf=True)  # arg570_1
    buf571 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf571, (512,), dtype=torch.float16, is_leaf=True)  # arg571_1
    buf572 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf572, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg572_1
    buf573 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf573, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg573_1
    buf574 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf574, (1024,), dtype=torch.float16, is_leaf=True)  # arg574_1
    buf575 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf575, (1024,), dtype=torch.float16, is_leaf=True)  # arg575_1
    buf576 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf576, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg576_1
    buf577 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf577, (512,), dtype=torch.float16, is_leaf=True)  # arg577_1
    buf578 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf578, (512,), dtype=torch.float16, is_leaf=True)  # arg578_1
    buf579 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf579, (512,), dtype=torch.float16, is_leaf=True)  # arg579_1
    buf580 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf580, (512, 512), dtype=torch.float16, is_leaf=True)  # arg580_1
    buf581 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf581, (512,), dtype=torch.float16, is_leaf=True)  # arg581_1
    buf582 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf582, (512, 512), dtype=torch.float16, is_leaf=True)  # arg582_1
    buf583 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf583, (512,), dtype=torch.float16, is_leaf=True)  # arg583_1
    buf584 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf584, (512, 512), dtype=torch.float16, is_leaf=True)  # arg584_1
    buf585 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf585, (512,), dtype=torch.float16, is_leaf=True)  # arg585_1
    buf586 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf586, (512, 512), dtype=torch.float16, is_leaf=True)  # arg586_1
    buf587 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf587, (512,), dtype=torch.float16, is_leaf=True)  # arg587_1
    buf588 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf588, (512,), dtype=torch.float16, is_leaf=True)  # arg588_1
    buf589 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf589, (512,), dtype=torch.float16, is_leaf=True)  # arg589_1
    buf590 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf590, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg590_1
    buf591 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf591, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg591_1
    buf592 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf592, (1024,), dtype=torch.float16, is_leaf=True)  # arg592_1
    buf593 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf593, (1024,), dtype=torch.float16, is_leaf=True)  # arg593_1
    buf594 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf594, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg594_1
    buf595 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf595, (512,), dtype=torch.float16, is_leaf=True)  # arg595_1
    buf596 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf596, (512,), dtype=torch.float16, is_leaf=True)  # arg596_1
    buf597 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf597, (512,), dtype=torch.float16, is_leaf=True)  # arg597_1
    buf598 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf598, (512, 512), dtype=torch.float16, is_leaf=True)  # arg598_1
    buf599 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf599, (512,), dtype=torch.float16, is_leaf=True)  # arg599_1
    buf600 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf600, (512, 512), dtype=torch.float16, is_leaf=True)  # arg600_1
    buf601 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf601, (512,), dtype=torch.float16, is_leaf=True)  # arg601_1
    buf602 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf602, (512, 512), dtype=torch.float16, is_leaf=True)  # arg602_1
    buf603 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf603, (512,), dtype=torch.float16, is_leaf=True)  # arg603_1
    buf604 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf604, (512, 512), dtype=torch.float16, is_leaf=True)  # arg604_1
    buf605 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf605, (512,), dtype=torch.float16, is_leaf=True)  # arg605_1
    buf606 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf606, (512,), dtype=torch.float16, is_leaf=True)  # arg606_1
    buf607 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf607, (512,), dtype=torch.float16, is_leaf=True)  # arg607_1
    buf608 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf608, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg608_1
    buf609 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf609, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg609_1
    buf610 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf610, (1024,), dtype=torch.float16, is_leaf=True)  # arg610_1
    buf611 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf611, (1024,), dtype=torch.float16, is_leaf=True)  # arg611_1
    buf612 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf612, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg612_1
    buf613 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf613, (512,), dtype=torch.float16, is_leaf=True)  # arg613_1
    buf614 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf614, (512, 32), dtype=torch.float16, is_leaf=True)  # arg614_1
    buf615 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf615, (32,), dtype=torch.float16, is_leaf=True)  # arg615_1
    buf616 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf616, (32, 1), dtype=torch.float16, is_leaf=True)  # arg616_1
    buf617 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf617, (1,), dtype=torch.float16, is_leaf=True)  # arg617_1
    buf618 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf618, (512, 32), dtype=torch.float16, is_leaf=True)  # arg618_1
    buf619 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf619, (32,), dtype=torch.float16, is_leaf=True)  # arg619_1
    buf620 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf620, (32, 1), dtype=torch.float16, is_leaf=True)  # arg620_1
    buf621 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf621, (1,), dtype=torch.float16, is_leaf=True)  # arg621_1
    buf622 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf622, (64, 512), dtype=torch.float16, is_leaf=True)  # arg622_1
    buf623 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf623, (512,), dtype=torch.float16, is_leaf=True)  # arg623_1
    buf624 = reader.storage(None, 573440, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf624, (560, 512), dtype=torch.float16, is_leaf=True)  # arg624_1
    buf625 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf625, (512,), dtype=torch.float16, is_leaf=True)  # arg625_1
    buf626 = reader.storage(None, 360448, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf626, (352, 512), dtype=torch.float16, is_leaf=True)  # arg626_1
    buf627 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf627, (512,), dtype=torch.float16, is_leaf=True)  # arg627_1
    buf628 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf628, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg628_1
    buf629 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf629, (512,), dtype=torch.float16, is_leaf=True)  # arg629_1
    buf630 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf630, (32, 64), dtype=torch.float16, is_leaf=True)  # arg630_1
    buf631 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf631, (64,), dtype=torch.float16, is_leaf=True)  # arg631_1
    buf632 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf632, (512,), dtype=torch.float16, is_leaf=True)  # arg632_1
    buf633 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf633, (512,), dtype=torch.float16, is_leaf=True)  # arg633_1
    buf634 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf634, (512,), dtype=torch.float16, is_leaf=True)  # arg634_1
    buf635 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf635, (512,), dtype=torch.float16, is_leaf=True)  # arg635_1
    buf636 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf636, (512, 512), dtype=torch.float16, is_leaf=True)  # arg636_1
    buf637 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf637, (512,), dtype=torch.float16, is_leaf=True)  # arg637_1
    buf638 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf638, (512, 512), dtype=torch.float16, is_leaf=True)  # arg638_1
    buf639 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf639, (512,), dtype=torch.float16, is_leaf=True)  # arg639_1
    buf640 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf640, (512, 512), dtype=torch.float16, is_leaf=True)  # arg640_1
    buf641 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf641, (512,), dtype=torch.float16, is_leaf=True)  # arg641_1
    buf642 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf642, (512, 512), dtype=torch.float16, is_leaf=True)  # arg642_1
    buf643 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf643, (512,), dtype=torch.float16, is_leaf=True)  # arg643_1
    buf644 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf644, (512,), dtype=torch.float16, is_leaf=True)  # arg644_1
    buf645 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf645, (512,), dtype=torch.float16, is_leaf=True)  # arg645_1
    buf646 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf646, (512, 512), dtype=torch.float16, is_leaf=True)  # arg646_1
    buf647 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf647, (512, 512), dtype=torch.float16, is_leaf=True)  # arg647_1
    buf648 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf648, (512,), dtype=torch.float16, is_leaf=True)  # arg648_1
    buf649 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf649, (512,), dtype=torch.float16, is_leaf=True)  # arg649_1
    buf650 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf650, (512, 512), dtype=torch.float16, is_leaf=True)  # arg650_1
    buf651 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf651, (512,), dtype=torch.float16, is_leaf=True)  # arg651_1
    buf652 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf652, (512,), dtype=torch.float16, is_leaf=True)  # arg652_1
    buf653 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf653, (512,), dtype=torch.float16, is_leaf=True)  # arg653_1
    buf654 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf654, (512, 512), dtype=torch.float16, is_leaf=True)  # arg654_1
    buf655 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf655, (512,), dtype=torch.float16, is_leaf=True)  # arg655_1
    buf656 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf656, (512, 512), dtype=torch.float16, is_leaf=True)  # arg656_1
    buf657 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf657, (512,), dtype=torch.float16, is_leaf=True)  # arg657_1
    buf658 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf658, (512, 512), dtype=torch.float16, is_leaf=True)  # arg658_1
    buf659 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf659, (512,), dtype=torch.float16, is_leaf=True)  # arg659_1
    buf660 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf660, (512, 512), dtype=torch.float16, is_leaf=True)  # arg660_1
    buf661 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf661, (512,), dtype=torch.float16, is_leaf=True)  # arg661_1
    buf662 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf662, (512,), dtype=torch.float16, is_leaf=True)  # arg662_1
    buf663 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf663, (512,), dtype=torch.float16, is_leaf=True)  # arg663_1
    buf664 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf664, (512, 512), dtype=torch.float16, is_leaf=True)  # arg664_1
    buf665 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf665, (512, 512), dtype=torch.float16, is_leaf=True)  # arg665_1
    buf666 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf666, (512,), dtype=torch.float16, is_leaf=True)  # arg666_1
    buf667 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf667, (512,), dtype=torch.float16, is_leaf=True)  # arg667_1
    buf668 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf668, (512, 512), dtype=torch.float16, is_leaf=True)  # arg668_1
    buf669 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf669, (512,), dtype=torch.float16, is_leaf=True)  # arg669_1
    buf670 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf670, (512, 32), dtype=torch.float16, is_leaf=True)  # arg670_1
    buf671 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf671, (32,), dtype=torch.float16, is_leaf=True)  # arg671_1
    buf672 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf672, (32, 1), dtype=torch.float16, is_leaf=True)  # arg672_1
    buf673 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf673, (1,), dtype=torch.float16, is_leaf=True)  # arg673_1
    buf674 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf674, (512, 32), dtype=torch.float16, is_leaf=True)  # arg674_1
    buf675 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf675, (32,), dtype=torch.float16, is_leaf=True)  # arg675_1
    buf676 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf676, (32, 1), dtype=torch.float16, is_leaf=True)  # arg676_1
    buf677 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf677, (1,), dtype=torch.float16, is_leaf=True)  # arg677_1
    buf678 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf678, (64, 512), dtype=torch.float16, is_leaf=True)  # arg678_1
    buf679 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf679, (512,), dtype=torch.float16, is_leaf=True)  # arg679_1
    buf680 = reader.storage(None, 573440, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf680, (560, 512), dtype=torch.float16, is_leaf=True)  # arg680_1
    buf681 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf681, (512,), dtype=torch.float16, is_leaf=True)  # arg681_1
    buf682 = reader.storage(None, 360448, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf682, (352, 512), dtype=torch.float16, is_leaf=True)  # arg682_1
    buf683 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf683, (512,), dtype=torch.float16, is_leaf=True)  # arg683_1
    buf684 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf684, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg684_1
    buf685 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf685, (512,), dtype=torch.float16, is_leaf=True)  # arg685_1
    buf686 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf686, (32, 64), dtype=torch.float16, is_leaf=True)  # arg686_1
    buf687 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf687, (64,), dtype=torch.float16, is_leaf=True)  # arg687_1
    buf688 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf688, (512,), dtype=torch.float16, is_leaf=True)  # arg688_1
    buf689 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf689, (512,), dtype=torch.float16, is_leaf=True)  # arg689_1
    buf690 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf690, (512,), dtype=torch.float16, is_leaf=True)  # arg690_1
    buf691 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf691, (512,), dtype=torch.float16, is_leaf=True)  # arg691_1
    buf692 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf692, (512, 512), dtype=torch.float16, is_leaf=True)  # arg692_1
    buf693 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf693, (512,), dtype=torch.float16, is_leaf=True)  # arg693_1
    buf694 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf694, (512, 512), dtype=torch.float16, is_leaf=True)  # arg694_1
    buf695 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf695, (512,), dtype=torch.float16, is_leaf=True)  # arg695_1
    buf696 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf696, (512, 512), dtype=torch.float16, is_leaf=True)  # arg696_1
    buf697 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf697, (512,), dtype=torch.float16, is_leaf=True)  # arg697_1
    buf698 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf698, (512, 512), dtype=torch.float16, is_leaf=True)  # arg698_1
    buf699 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf699, (512,), dtype=torch.float16, is_leaf=True)  # arg699_1
    buf700 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf700, (512,), dtype=torch.float16, is_leaf=True)  # arg700_1
    buf701 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf701, (512,), dtype=torch.float16, is_leaf=True)  # arg701_1
    buf702 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf702, (512, 512), dtype=torch.float16, is_leaf=True)  # arg702_1
    buf703 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf703, (512, 512), dtype=torch.float16, is_leaf=True)  # arg703_1
    buf704 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf704, (512,), dtype=torch.float16, is_leaf=True)  # arg704_1
    buf705 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf705, (512,), dtype=torch.float16, is_leaf=True)  # arg705_1
    buf706 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf706, (512, 512), dtype=torch.float16, is_leaf=True)  # arg706_1
    buf707 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf707, (512,), dtype=torch.float16, is_leaf=True)  # arg707_1
    buf708 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf708, (512,), dtype=torch.float16, is_leaf=True)  # arg708_1
    buf709 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf709, (512,), dtype=torch.float16, is_leaf=True)  # arg709_1
    buf710 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf710, (512, 512), dtype=torch.float16, is_leaf=True)  # arg710_1
    buf711 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf711, (512,), dtype=torch.float16, is_leaf=True)  # arg711_1
    buf712 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf712, (512, 512), dtype=torch.float16, is_leaf=True)  # arg712_1
    buf713 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf713, (512,), dtype=torch.float16, is_leaf=True)  # arg713_1
    buf714 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf714, (512, 512), dtype=torch.float16, is_leaf=True)  # arg714_1
    buf715 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf715, (512,), dtype=torch.float16, is_leaf=True)  # arg715_1
    buf716 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf716, (512, 512), dtype=torch.float16, is_leaf=True)  # arg716_1
    buf717 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf717, (512,), dtype=torch.float16, is_leaf=True)  # arg717_1
    buf718 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf718, (512,), dtype=torch.float16, is_leaf=True)  # arg718_1
    buf719 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf719, (512,), dtype=torch.float16, is_leaf=True)  # arg719_1
    buf720 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf720, (512, 512), dtype=torch.float16, is_leaf=True)  # arg720_1
    buf721 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf721, (512, 512), dtype=torch.float16, is_leaf=True)  # arg721_1
    buf722 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf722, (512,), dtype=torch.float16, is_leaf=True)  # arg722_1
    buf723 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf723, (512,), dtype=torch.float16, is_leaf=True)  # arg723_1
    buf724 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf724, (512, 512), dtype=torch.float16, is_leaf=True)  # arg724_1
    buf725 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf725, (512,), dtype=torch.float16, is_leaf=True)  # arg725_1
    buf726 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf726, (512, 32), dtype=torch.float16, is_leaf=True)  # arg726_1
    buf727 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf727, (32,), dtype=torch.float16, is_leaf=True)  # arg727_1
    buf728 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf728, (32, 1), dtype=torch.float16, is_leaf=True)  # arg728_1
    buf729 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf729, (1,), dtype=torch.float16, is_leaf=True)  # arg729_1
    buf730 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf730, (512, 32), dtype=torch.float16, is_leaf=True)  # arg730_1
    buf731 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf731, (32,), dtype=torch.float16, is_leaf=True)  # arg731_1
    buf732 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf732, (32, 1), dtype=torch.float16, is_leaf=True)  # arg732_1
    buf733 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf733, (1,), dtype=torch.float16, is_leaf=True)  # arg733_1
    buf734 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf734, (64, 512), dtype=torch.float16, is_leaf=True)  # arg734_1
    buf735 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf735, (512,), dtype=torch.float16, is_leaf=True)  # arg735_1
    buf736 = reader.storage(None, 573440, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf736, (560, 512), dtype=torch.float16, is_leaf=True)  # arg736_1
    buf737 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf737, (512,), dtype=torch.float16, is_leaf=True)  # arg737_1
    buf738 = reader.storage(None, 360448, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf738, (352, 512), dtype=torch.float16, is_leaf=True)  # arg738_1
    buf739 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf739, (512,), dtype=torch.float16, is_leaf=True)  # arg739_1
    buf740 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf740, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg740_1
    buf741 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf741, (512,), dtype=torch.float16, is_leaf=True)  # arg741_1
    buf742 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf742, (32, 64), dtype=torch.float16, is_leaf=True)  # arg742_1
    buf743 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf743, (64,), dtype=torch.float16, is_leaf=True)  # arg743_1
    buf744 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf744, (512,), dtype=torch.float16, is_leaf=True)  # arg744_1
    buf745 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf745, (512,), dtype=torch.float16, is_leaf=True)  # arg745_1
    buf746 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf746, (512,), dtype=torch.float16, is_leaf=True)  # arg746_1
    buf747 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf747, (512,), dtype=torch.float16, is_leaf=True)  # arg747_1
    buf748 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf748, (512, 512), dtype=torch.float16, is_leaf=True)  # arg748_1
    buf749 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf749, (512,), dtype=torch.float16, is_leaf=True)  # arg749_1
    buf750 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf750, (512, 512), dtype=torch.float16, is_leaf=True)  # arg750_1
    buf751 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf751, (512,), dtype=torch.float16, is_leaf=True)  # arg751_1
    buf752 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf752, (512, 512), dtype=torch.float16, is_leaf=True)  # arg752_1
    buf753 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf753, (512,), dtype=torch.float16, is_leaf=True)  # arg753_1
    buf754 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf754, (512, 512), dtype=torch.float16, is_leaf=True)  # arg754_1
    buf755 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf755, (512,), dtype=torch.float16, is_leaf=True)  # arg755_1
    buf756 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf756, (512,), dtype=torch.float16, is_leaf=True)  # arg756_1
    buf757 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf757, (512,), dtype=torch.float16, is_leaf=True)  # arg757_1
    buf758 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf758, (512, 512), dtype=torch.float16, is_leaf=True)  # arg758_1
    buf759 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf759, (512, 512), dtype=torch.float16, is_leaf=True)  # arg759_1
    buf760 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf760, (512,), dtype=torch.float16, is_leaf=True)  # arg760_1
    buf761 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf761, (512,), dtype=torch.float16, is_leaf=True)  # arg761_1
    buf762 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf762, (512, 512), dtype=torch.float16, is_leaf=True)  # arg762_1
    buf763 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf763, (512,), dtype=torch.float16, is_leaf=True)  # arg763_1
    buf764 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf764, (512,), dtype=torch.float16, is_leaf=True)  # arg764_1
    buf765 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf765, (512,), dtype=torch.float16, is_leaf=True)  # arg765_1
    buf766 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf766, (512, 512), dtype=torch.float16, is_leaf=True)  # arg766_1
    buf767 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf767, (512,), dtype=torch.float16, is_leaf=True)  # arg767_1
    buf768 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf768, (512, 512), dtype=torch.float16, is_leaf=True)  # arg768_1
    buf769 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf769, (512,), dtype=torch.float16, is_leaf=True)  # arg769_1
    buf770 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf770, (512, 512), dtype=torch.float16, is_leaf=True)  # arg770_1
    buf771 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf771, (512,), dtype=torch.float16, is_leaf=True)  # arg771_1
    buf772 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf772, (512, 512), dtype=torch.float16, is_leaf=True)  # arg772_1
    buf773 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf773, (512,), dtype=torch.float16, is_leaf=True)  # arg773_1
    buf774 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf774, (512,), dtype=torch.float16, is_leaf=True)  # arg774_1
    buf775 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf775, (512,), dtype=torch.float16, is_leaf=True)  # arg775_1
    buf776 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf776, (512, 512), dtype=torch.float16, is_leaf=True)  # arg776_1
    buf777 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf777, (512, 512), dtype=torch.float16, is_leaf=True)  # arg777_1
    buf778 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf778, (512,), dtype=torch.float16, is_leaf=True)  # arg778_1
    buf779 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf779, (512,), dtype=torch.float16, is_leaf=True)  # arg779_1
    buf780 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf780, (512, 512), dtype=torch.float16, is_leaf=True)  # arg780_1
    buf781 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf781, (512,), dtype=torch.float16, is_leaf=True)  # arg781_1
    buf782 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf782, (512, 32), dtype=torch.float16, is_leaf=True)  # arg782_1
    buf783 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf783, (32,), dtype=torch.float16, is_leaf=True)  # arg783_1
    buf784 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf784, (32, 1), dtype=torch.float16, is_leaf=True)  # arg784_1
    buf785 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf785, (1,), dtype=torch.float16, is_leaf=True)  # arg785_1
    buf786 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf786, (512, 32), dtype=torch.float16, is_leaf=True)  # arg786_1
    buf787 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf787, (32,), dtype=torch.float16, is_leaf=True)  # arg787_1
    buf788 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf788, (32, 1), dtype=torch.float16, is_leaf=True)  # arg788_1
    buf789 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf789, (1,), dtype=torch.float16, is_leaf=True)  # arg789_1
    buf790 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf790, (64, 512), dtype=torch.float16, is_leaf=True)  # arg790_1
    buf791 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf791, (512,), dtype=torch.float16, is_leaf=True)  # arg791_1
    buf792 = reader.storage(None, 573440, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf792, (560, 512), dtype=torch.float16, is_leaf=True)  # arg792_1
    buf793 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf793, (512,), dtype=torch.float16, is_leaf=True)  # arg793_1
    buf794 = reader.storage(None, 360448, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf794, (352, 512), dtype=torch.float16, is_leaf=True)  # arg794_1
    buf795 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf795, (512,), dtype=torch.float16, is_leaf=True)  # arg795_1
    buf796 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf796, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg796_1
    buf797 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf797, (512,), dtype=torch.float16, is_leaf=True)  # arg797_1
    buf798 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf798, (32, 64), dtype=torch.float16, is_leaf=True)  # arg798_1
    buf799 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf799, (64,), dtype=torch.float16, is_leaf=True)  # arg799_1
    buf800 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf800, (512,), dtype=torch.float16, is_leaf=True)  # arg800_1
    buf801 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf801, (512,), dtype=torch.float16, is_leaf=True)  # arg801_1
    buf802 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf802, (512,), dtype=torch.float16, is_leaf=True)  # arg802_1
    buf803 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf803, (512,), dtype=torch.float16, is_leaf=True)  # arg803_1
    buf804 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf804, (512, 512), dtype=torch.float16, is_leaf=True)  # arg804_1
    buf805 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf805, (512,), dtype=torch.float16, is_leaf=True)  # arg805_1
    buf806 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf806, (512, 512), dtype=torch.float16, is_leaf=True)  # arg806_1
    buf807 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf807, (512,), dtype=torch.float16, is_leaf=True)  # arg807_1
    buf808 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf808, (512, 512), dtype=torch.float16, is_leaf=True)  # arg808_1
    buf809 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf809, (512,), dtype=torch.float16, is_leaf=True)  # arg809_1
    buf810 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf810, (512, 512), dtype=torch.float16, is_leaf=True)  # arg810_1
    buf811 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf811, (512,), dtype=torch.float16, is_leaf=True)  # arg811_1
    buf812 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf812, (512,), dtype=torch.float16, is_leaf=True)  # arg812_1
    buf813 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf813, (512,), dtype=torch.float16, is_leaf=True)  # arg813_1
    buf814 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf814, (512, 512), dtype=torch.float16, is_leaf=True)  # arg814_1
    buf815 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf815, (512, 512), dtype=torch.float16, is_leaf=True)  # arg815_1
    buf816 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf816, (512,), dtype=torch.float16, is_leaf=True)  # arg816_1
    buf817 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf817, (512,), dtype=torch.float16, is_leaf=True)  # arg817_1
    buf818 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf818, (512, 512), dtype=torch.float16, is_leaf=True)  # arg818_1
    buf819 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf819, (512,), dtype=torch.float16, is_leaf=True)  # arg819_1
    buf820 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf820, (512,), dtype=torch.float16, is_leaf=True)  # arg820_1
    buf821 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf821, (512,), dtype=torch.float16, is_leaf=True)  # arg821_1
    buf822 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf822, (512, 512), dtype=torch.float16, is_leaf=True)  # arg822_1
    buf823 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf823, (512,), dtype=torch.float16, is_leaf=True)  # arg823_1
    buf824 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf824, (512, 512), dtype=torch.float16, is_leaf=True)  # arg824_1
    buf825 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf825, (512,), dtype=torch.float16, is_leaf=True)  # arg825_1
    buf826 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf826, (512, 512), dtype=torch.float16, is_leaf=True)  # arg826_1
    buf827 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf827, (512,), dtype=torch.float16, is_leaf=True)  # arg827_1
    buf828 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf828, (512, 512), dtype=torch.float16, is_leaf=True)  # arg828_1
    buf829 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf829, (512,), dtype=torch.float16, is_leaf=True)  # arg829_1
    buf830 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf830, (512,), dtype=torch.float16, is_leaf=True)  # arg830_1
    buf831 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf831, (512,), dtype=torch.float16, is_leaf=True)  # arg831_1
    buf832 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf832, (512, 512), dtype=torch.float16, is_leaf=True)  # arg832_1
    buf833 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf833, (512, 512), dtype=torch.float16, is_leaf=True)  # arg833_1
    buf834 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf834, (512,), dtype=torch.float16, is_leaf=True)  # arg834_1
    buf835 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf835, (512,), dtype=torch.float16, is_leaf=True)  # arg835_1
    buf836 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf836, (512, 512), dtype=torch.float16, is_leaf=True)  # arg836_1
    buf837 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf837, (512,), dtype=torch.float16, is_leaf=True)  # arg837_1
    buf838 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf838, (512, 32), dtype=torch.float16, is_leaf=True)  # arg838_1
    buf839 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf839, (32,), dtype=torch.float16, is_leaf=True)  # arg839_1
    buf840 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf840, (32, 1), dtype=torch.float16, is_leaf=True)  # arg840_1
    buf841 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf841, (1,), dtype=torch.float16, is_leaf=True)  # arg841_1
    buf842 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf842, (512, 32), dtype=torch.float16, is_leaf=True)  # arg842_1
    buf843 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf843, (32,), dtype=torch.float16, is_leaf=True)  # arg843_1
    buf844 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf844, (32, 1), dtype=torch.float16, is_leaf=True)  # arg844_1
    buf845 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf845, (1,), dtype=torch.float16, is_leaf=True)  # arg845_1
    buf846 = reader.storage(None, 32000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf846, (batch_size_hint, 80), dtype=torch.float16, is_leaf=True)  # arg846_1
    buf847 = reader.storage(None, 32000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf847, (batch_size_hint, 80), dtype=torch.float16, is_leaf=True)  # arg847_1
    buf848 = reader.storage(None, 32000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf848, (batch_size_hint, 80), dtype=torch.float16, is_leaf=True)  # arg848_1
    buf849 = reader.storage(None, 32000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf849, (batch_size_hint, 80), dtype=torch.float16, is_leaf=True)  # arg849_1
    buf850 = reader.storage(None, 44800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf850, (batch_size_hint, 112), dtype=torch.float16, is_leaf=True)  # arg850_1
    buf851 = reader.storage(None, 44800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf851, (batch_size_hint, 112), dtype=torch.float16, is_leaf=True)  # arg851_1
    buf852 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf852, (batch_size_hint, 128), dtype=torch.float16, is_leaf=True)  # arg852_1
    buf853 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf853, (batch_size_hint, 128), dtype=torch.float16, is_leaf=True)  # arg853_1
    buf854 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf854, (batch_size_hint, 128), dtype=torch.float16, is_leaf=True)  # arg854_1
    buf855 = reader.storage(None, 51200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf855, (batch_size_hint, 128), dtype=torch.float16, is_leaf=True)  # arg855_1
    buf856 = reader.storage(None, 384400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf856, (batch_size_hint, 961), dtype=torch.float16, is_leaf=True)  # arg856_1
    buf857 = reader.storage(None, 14, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf857, (7, 1), dtype=torch.float16, is_leaf=True)  # arg857_1
    buf858 = reader.storage(None, 12800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf858, (batch_size_hint, 32), dtype=torch.float16, is_leaf=True)  # arg858_1
    buf859 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf859, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg859_1
    buf860 = reader.storage(None, 19200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf860, (batch_size_hint, 48), dtype=torch.float16, is_leaf=True)  # arg860_1
    buf861 = reader.storage(None, 28800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf861, (batch_size_hint, 72), dtype=torch.float16, is_leaf=True)  # arg861_1
    buf862 = reader.storage(None, 36800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf862, (batch_size_hint, 92), dtype=torch.float16, is_leaf=True)  # arg862_1
    buf863 = reader.storage(None, 38400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf863, (batch_size_hint, 96), dtype=torch.float16, is_leaf=True)  # arg863_1
    buf864 = reader.storage(None, 64000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf864, (batch_size_hint, 160), dtype=torch.float16, is_leaf=True)  # arg864_1
    buf865 = reader.storage(None, 32000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf865, (batch_size_hint, 80), dtype=torch.float16, is_leaf=True)  # arg865_1
    buf866 = reader.storage(None, 6400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf866, (batch_size_hint, 16), dtype=torch.float16, is_leaf=True)  # arg866_1
    buf867 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf867, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg867_1
    buf868 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf868, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg868_1
    buf869 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf869, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg869_1
    buf870 = reader.storage(None, 19200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf870, (batch_size_hint, 48), dtype=torch.float16, is_leaf=True)  # arg870_1
    buf871 = reader.storage(None, 9600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf871, (batch_size_hint, 24), dtype=torch.float16, is_leaf=True)  # arg871_1
    buf872 = reader.storage(None, 9600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf872, (batch_size_hint, 24), dtype=torch.float16, is_leaf=True)  # arg872_1
    buf873 = reader.storage(None, 9600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf873, (batch_size_hint, 24), dtype=torch.float16, is_leaf=True)  # arg873_1
    buf874 = reader.storage(None, 1107072, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf874, (744, 744), dtype=torch.float16, is_leaf=True)  # arg874_1
    buf875 = reader.storage(None, 1488, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf875, (1, 744), dtype=torch.float16, is_leaf=True)  # arg875_1
    buf876 = reader.storage(None, 1107072, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf876, (744, 744), dtype=torch.float16, is_leaf=True)  # arg876_1
    buf877 = reader.storage(None, 1488, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf877, (1, 744), dtype=torch.float16, is_leaf=True)  # arg877_1
    buf878 = reader.storage(None, 190464, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf878, (744, 128), dtype=torch.float16, is_leaf=True)  # arg878_1
    buf879 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf879, (128,), dtype=torch.float16, is_leaf=True)  # arg879_1
    buf880 = reader.storage(None, 214400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf880, (batch_size_hint, 536), dtype=torch.float16, is_leaf=True)  # arg880_1
    buf881 = reader.storage(None, 67200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf881, (batch_size_hint, 168), dtype=torch.float16, is_leaf=True)  # arg881_1
    buf882 = reader.storage(None, 991232, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf882, (704, 704), dtype=torch.float16, is_leaf=True)  # arg882_1
    buf883 = reader.storage(None, 1408, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf883, (1, 704), dtype=torch.float16, is_leaf=True)  # arg883_1
    buf884 = reader.storage(None, 991232, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf884, (704, 704), dtype=torch.float16, is_leaf=True)  # arg884_1
    buf885 = reader.storage(None, 1408, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf885, (1, 704), dtype=torch.float16, is_leaf=True)  # arg885_1
    buf886 = reader.storage(None, 180224, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf886, (704, 128), dtype=torch.float16, is_leaf=True)  # arg886_1
    buf887 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf887, (128,), dtype=torch.float16, is_leaf=True)  # arg887_1
    buf888 = reader.storage(None, 113600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf888, (batch_size_hint, 284), dtype=torch.float16, is_leaf=True)  # arg888_1
    buf889 = reader.storage(None, 28800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf889, (batch_size_hint, 72), dtype=torch.float16, is_leaf=True)  # arg889_1
    buf890 = reader.storage(None, 364544, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf890, (356, 512), dtype=torch.float16, is_leaf=True)  # arg890_1
    buf891 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf891, (512,), dtype=torch.float16, is_leaf=True)  # arg891_1
    buf892 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf892, (512, 64), dtype=torch.float16, is_leaf=True)  # arg892_1
    buf893 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf893, (64,), dtype=torch.float16, is_leaf=True)  # arg893_1
    buf894 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf894, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg894_1
    buf895 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf895, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg895_1
    buf896 = reader.storage(None, 64000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf896, (batch_size_hint, 160), dtype=torch.float16, is_leaf=True)  # arg896_1
    buf897 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf897, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg897_1
    buf898 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf898, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg898_1
    buf899 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf899, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg899_1
    buf900 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf900, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg900_1
    buf901 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf901, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg901_1
    buf902 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf902, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg902_1
    buf903 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf903, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg903_1
    buf904 = reader.storage(None, 8000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf904, (batch_size_hint, 20), dtype=torch.float16, is_leaf=True)  # arg904_1
    buf905 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf905, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg905_1
    buf906 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf906, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg906_1
    buf907 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf907, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg907_1
    buf908 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf908, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg908_1
    buf909 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf909, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg909_1
    buf910 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf910, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg910_1
    buf911 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf911, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg911_1
    buf912 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf912, (batch_size_hint, 4), dtype=torch.float16, is_leaf=True)  # arg912_1
    buf913 = reader.storage(None, 129024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf913, (252, 256), dtype=torch.float16, is_leaf=True)  # arg913_1
    buf914 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf914, (256,), dtype=torch.float16, is_leaf=True)  # arg914_1
    buf915 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf915, (256, 64), dtype=torch.float16, is_leaf=True)  # arg915_1
    buf916 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf916, (64,), dtype=torch.float16, is_leaf=True)  # arg916_1
    buf917 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf917, (320, 128), dtype=torch.float16, is_leaf=True)  # arg917_1
    buf918 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf918, (128,), dtype=torch.float16, is_leaf=True)  # arg918_1
    buf919 = reader.storage(None, 6400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf919, (batch_size_hint, 16), dtype=torch.float16, is_leaf=True)  # arg919_1
    buf920 = reader.storage(None, 6400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf920, (batch_size_hint, 16), dtype=torch.float16, is_leaf=True)  # arg920_1
    buf921 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf921, (64, 64), dtype=torch.float16, is_leaf=True)  # arg921_1
    buf922 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf922, (64,), dtype=torch.float16, is_leaf=True)  # arg922_1
    buf923 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf923, (64, 32), dtype=torch.float16, is_leaf=True)  # arg923_1
    buf924 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf924, (32,), dtype=torch.float16, is_leaf=True)  # arg924_1
    buf925 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf925, (32, 16), dtype=torch.float16, is_leaf=True)  # arg925_1
    buf926 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf926, (16,), dtype=torch.float16, is_leaf=True)  # arg926_1
    buf927 = reader.storage(None, 12800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf927, (batch_size_hint, 32), dtype=torch.float16, is_leaf=True)  # arg927_1
    buf928 = reader.storage(None, 185600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf928, (batch_size_hint, 464), dtype=torch.float16, is_leaf=True)  # arg928_1
    buf929 = reader.storage(None, 12800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf929, (batch_size_hint, 32), dtype=torch.float16, is_leaf=True)  # arg929_1
    buf930 = reader.storage(None, 64000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf930, (batch_size_hint, 160), dtype=torch.float16, is_leaf=True)  # arg930_1
    buf931 = reader.storage(None, 352256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf931, (688, 256), dtype=torch.float16, is_leaf=True)  # arg931_1
    buf932 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf932, (256,), dtype=torch.float16, is_leaf=True)  # arg932_1
    buf933 = reader.storage(None, 131072, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf933, (256, 256), dtype=torch.float16, is_leaf=True)  # arg933_1
    buf934 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf934, (256,), dtype=torch.float16, is_leaf=True)  # arg934_1
    buf935 = reader.storage(None, 102400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf935, (batch_size_hint, 256), dtype=torch.float16, is_leaf=True)  # arg935_1
    buf936 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf936, (256, 128), dtype=torch.float16, is_leaf=True)  # arg936_1
    buf937 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf937, (128, 256), dtype=torch.float16, is_leaf=True)  # arg937_1
    buf938 = reader.storage(None, 6400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf938, (batch_size_hint, 16), dtype=torch.float16, is_leaf=True)  # arg938_1
    buf939 = reader.storage(None, 6400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf939, (batch_size_hint, 16), dtype=torch.float16, is_leaf=True)  # arg939_1
    buf940 = reader.storage(None, 6400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf940, (batch_size_hint, 16), dtype=torch.float16, is_leaf=True)  # arg940_1
    buf941 = reader.storage(None, 501760, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf941, (490, 512), dtype=torch.float16, is_leaf=True)  # arg941_1
    buf942 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf942, (512,), dtype=torch.float16, is_leaf=True)  # arg942_1
    buf943 = reader.storage(None, 262144, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf943, (512, 256), dtype=torch.float16, is_leaf=True)  # arg943_1
    buf944 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf944, (256,), dtype=torch.float16, is_leaf=True)  # arg944_1
    buf945 = reader.storage(None, 1687200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf945, (batch_size_hint, 4218), dtype=torch.float16, is_leaf=True)  # arg945_1
    buf946 = reader.storage(None, 1308672, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf946, (2556, 256), dtype=torch.float16, is_leaf=True)  # arg946_1
    buf947 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf947, (256,), dtype=torch.float16, is_leaf=True)  # arg947_1
    buf948 = reader.storage(None, 548864, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf948, (1072, 256), dtype=torch.float16, is_leaf=True)  # arg948_1
    buf949 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf949, (256,), dtype=torch.float16, is_leaf=True)  # arg949_1
    buf950 = reader.storage(None, 640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf950, (320,), dtype=torch.float16, is_leaf=True)  # arg950_1
    buf951 = reader.storage(None, 640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf951, (320,), dtype=torch.float16, is_leaf=True)  # arg951_1
    buf952 = reader.storage(None, 409600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf952, (320, 640), dtype=torch.float16, is_leaf=True)  # arg952_1
    buf953 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf953, (640,), dtype=torch.float16, is_leaf=True)  # arg953_1
    buf954 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf954, (512,), dtype=torch.float16, is_leaf=True)  # arg954_1
    buf955 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf955, (512,), dtype=torch.float16, is_leaf=True)  # arg955_1
    buf956 = reader.storage(None, 655360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf956, (512, 640), dtype=torch.float16, is_leaf=True)  # arg956_1
    buf957 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf957, (640,), dtype=torch.float16, is_leaf=True)  # arg957_1
    buf958 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf958, (256,), dtype=torch.float16, is_leaf=True)  # arg958_1
    buf959 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf959, (256,), dtype=torch.float16, is_leaf=True)  # arg959_1
    buf960 = reader.storage(None, 327680, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf960, (256, 640), dtype=torch.float16, is_leaf=True)  # arg960_1
    buf961 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf961, (640,), dtype=torch.float16, is_leaf=True)  # arg961_1
    buf962 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf962, (256,), dtype=torch.float16, is_leaf=True)  # arg962_1
    buf963 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf963, (256,), dtype=torch.float16, is_leaf=True)  # arg963_1
    buf964 = reader.storage(None, 327680, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf964, (256, 640), dtype=torch.float16, is_leaf=True)  # arg964_1
    buf965 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf965, (64,), dtype=torch.float16, is_leaf=True)  # arg965_1
    buf966 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf966, (64,), dtype=torch.float16, is_leaf=True)  # arg966_1
    buf967 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf967, (64, 640), dtype=torch.float16, is_leaf=True)  # arg967_1
    buf968 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf968, (640,), dtype=torch.float16, is_leaf=True)  # arg968_1
    buf969 = reader.storage(None, 3072, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf969, (1536,), dtype=torch.float16, is_leaf=True)  # arg969_1
    buf970 = reader.storage(None, 3072, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf970, (1536,), dtype=torch.float16, is_leaf=True)  # arg970_1
    buf971 = reader.storage(None, 1966080, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf971, (1536, 640), dtype=torch.float16, is_leaf=True)  # arg971_1
    buf972 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf972, (640,), dtype=torch.float16, is_leaf=True)  # arg972_1
    buf973 = reader.storage(None, 5024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf973, (2512,), dtype=torch.float16, is_leaf=True)  # arg973_1
    buf974 = reader.storage(None, 5024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf974, (2512,), dtype=torch.float16, is_leaf=True)  # arg974_1
    buf975 = reader.storage(None, 3215360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf975, (2512, 640), dtype=torch.float16, is_leaf=True)  # arg975_1
    buf976 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf976, (640,), dtype=torch.float16, is_leaf=True)  # arg976_1
    buf977 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf977, (256,), dtype=torch.float16, is_leaf=True)  # arg977_1
    buf978 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf978, (256,), dtype=torch.float16, is_leaf=True)  # arg978_1
    buf979 = reader.storage(None, 327680, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf979, (256, 640), dtype=torch.float16, is_leaf=True)  # arg979_1
    buf980 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf980, (640,), dtype=torch.float16, is_leaf=True)  # arg980_1
    buf981 = reader.storage(None, 2568, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf981, (1284,), dtype=torch.float16, is_leaf=True)  # arg981_1
    buf982 = reader.storage(None, 2568, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf982, (1284,), dtype=torch.float16, is_leaf=True)  # arg982_1
    buf983 = reader.storage(None, 1643520, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf983, (1284, 640), dtype=torch.float16, is_leaf=True)  # arg983_1
    buf984 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf984, (640,), dtype=torch.float16, is_leaf=True)  # arg984_1
    buf985 = reader.storage(None, 2056, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf985, (1028,), dtype=torch.float16, is_leaf=True)  # arg985_1
    buf986 = reader.storage(None, 2056, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf986, (1028,), dtype=torch.float16, is_leaf=True)  # arg986_1
    buf987 = reader.storage(None, 1315840, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf987, (1028, 640), dtype=torch.float16, is_leaf=True)  # arg987_1
    buf988 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf988, (640,), dtype=torch.float16, is_leaf=True)  # arg988_1
    buf989 = reader.storage(None, 1792, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf989, (896,), dtype=torch.float16, is_leaf=True)  # arg989_1
    buf990 = reader.storage(None, 1792, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf990, (896,), dtype=torch.float16, is_leaf=True)  # arg990_1
    buf991 = reader.storage(None, 1146880, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf991, (896, 640), dtype=torch.float16, is_leaf=True)  # arg991_1
    buf992 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf992, (640,), dtype=torch.float16, is_leaf=True)  # arg992_1
    buf993 = reader.storage(None, 640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf993, (320,), dtype=torch.float16, is_leaf=True)  # arg993_1
    buf994 = reader.storage(None, 640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf994, (320,), dtype=torch.float16, is_leaf=True)  # arg994_1
    buf995 = reader.storage(None, 409600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf995, (320, 640), dtype=torch.float16, is_leaf=True)  # arg995_1
    buf996 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf996, (640,), dtype=torch.float16, is_leaf=True)  # arg996_1
    buf997 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf997, (640,), dtype=torch.float16, is_leaf=True)  # arg997_1
    buf998 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf998, (640,), dtype=torch.float16, is_leaf=True)  # arg998_1
    buf999 = reader.storage(None, 819200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf999, (640, 640), dtype=torch.float16, is_leaf=True)  # arg999_1
    buf1000 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1000, (640,), dtype=torch.float16, is_leaf=True)  # arg1000_1
    buf1001 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1001, (512,), dtype=torch.float16, is_leaf=True)  # arg1001_1
    buf1002 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1002, (512,), dtype=torch.float16, is_leaf=True)  # arg1002_1
    buf1003 = reader.storage(None, 655360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1003, (512, 640), dtype=torch.float16, is_leaf=True)  # arg1003_1
    buf1004 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1004, (640,), dtype=torch.float16, is_leaf=True)  # arg1004_1
    buf1005 = reader.storage(None, 640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1005, (320,), dtype=torch.float16, is_leaf=True)  # arg1005_1
    buf1006 = reader.storage(None, 640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1006, (320,), dtype=torch.float16, is_leaf=True)  # arg1006_1
    buf1007 = reader.storage(None, 409600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1007, (320, 640), dtype=torch.float16, is_leaf=True)  # arg1007_1
    buf1008 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1008, (640,), dtype=torch.float16, is_leaf=True)  # arg1008_1
    buf1009 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1009, (512,), dtype=torch.float16, is_leaf=True)  # arg1009_1
    buf1010 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1010, (512,), dtype=torch.float16, is_leaf=True)  # arg1010_1
    buf1011 = reader.storage(None, 655360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1011, (512, 640), dtype=torch.float16, is_leaf=True)  # arg1011_1
    buf1012 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1012, (640,), dtype=torch.float16, is_leaf=True)  # arg1012_1
    buf1013 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1013, (32,), dtype=torch.float16, is_leaf=True)  # arg1013_1
    buf1014 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1014, (32,), dtype=torch.float16, is_leaf=True)  # arg1014_1
    buf1015 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1015, (32, 640), dtype=torch.float16, is_leaf=True)  # arg1015_1
    buf1016 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1016, (640,), dtype=torch.float16, is_leaf=True)  # arg1016_1
    buf1017 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1017, (32,), dtype=torch.float16, is_leaf=True)  # arg1017_1
    buf1018 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1018, (32,), dtype=torch.float16, is_leaf=True)  # arg1018_1
    buf1019 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1019, (32, 640), dtype=torch.float16, is_leaf=True)  # arg1019_1
    buf1020 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1020, (640,), dtype=torch.float16, is_leaf=True)  # arg1020_1
    buf1021 = reader.storage(None, 640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1021, (320,), dtype=torch.float16, is_leaf=True)  # arg1021_1
    buf1022 = reader.storage(None, 640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1022, (320,), dtype=torch.float16, is_leaf=True)  # arg1022_1
    buf1023 = reader.storage(None, 409600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1023, (320, 640), dtype=torch.float16, is_leaf=True)  # arg1023_1
    buf1024 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1024, (640,), dtype=torch.float16, is_leaf=True)  # arg1024_1
    buf1025 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1025, (512,), dtype=torch.float16, is_leaf=True)  # arg1025_1
    buf1026 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1026, (512,), dtype=torch.float16, is_leaf=True)  # arg1026_1
    buf1027 = reader.storage(None, 655360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1027, (512, 640), dtype=torch.float16, is_leaf=True)  # arg1027_1
    buf1028 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1028, (640,), dtype=torch.float16, is_leaf=True)  # arg1028_1
    buf1029 = reader.storage(None, 640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1029, (320,), dtype=torch.float16, is_leaf=True)  # arg1029_1
    buf1030 = reader.storage(None, 640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1030, (320,), dtype=torch.float16, is_leaf=True)  # arg1030_1
    buf1031 = reader.storage(None, 409600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1031, (320, 640), dtype=torch.float16, is_leaf=True)  # arg1031_1
    buf1032 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1032, (640,), dtype=torch.float16, is_leaf=True)  # arg1032_1
    buf1033 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1033, (512,), dtype=torch.float16, is_leaf=True)  # arg1033_1
    buf1034 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1034, (512,), dtype=torch.float16, is_leaf=True)  # arg1034_1
    buf1035 = reader.storage(None, 655360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1035, (512, 640), dtype=torch.float16, is_leaf=True)  # arg1035_1
    buf1036 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1036, (640,), dtype=torch.float16, is_leaf=True)  # arg1036_1
    buf1037 = reader.storage(None, 1792, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1037, (896,), dtype=torch.float16, is_leaf=True)  # arg1037_1
    buf1038 = reader.storage(None, 1792, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1038, (896,), dtype=torch.float16, is_leaf=True)  # arg1038_1
    buf1039 = reader.storage(None, 1146880, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1039, (896, 640), dtype=torch.float16, is_leaf=True)  # arg1039_1
    buf1040 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1040, (640,), dtype=torch.float16, is_leaf=True)  # arg1040_1
    buf1041 = reader.storage(None, 640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1041, (320,), dtype=torch.float16, is_leaf=True)  # arg1041_1
    buf1042 = reader.storage(None, 640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1042, (320,), dtype=torch.float16, is_leaf=True)  # arg1042_1
    buf1043 = reader.storage(None, 409600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1043, (320, 640), dtype=torch.float16, is_leaf=True)  # arg1043_1
    buf1044 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1044, (640,), dtype=torch.float16, is_leaf=True)  # arg1044_1
    buf1045 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1045, (512,), dtype=torch.float16, is_leaf=True)  # arg1045_1
    buf1046 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1046, (512,), dtype=torch.float16, is_leaf=True)  # arg1046_1
    buf1047 = reader.storage(None, 655360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1047, (512, 640), dtype=torch.float16, is_leaf=True)  # arg1047_1
    buf1048 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1048, (640,), dtype=torch.float16, is_leaf=True)  # arg1048_1
    buf1049 = reader.storage(None, 1792, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1049, (896,), dtype=torch.float16, is_leaf=True)  # arg1049_1
    buf1050 = reader.storage(None, 1792, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1050, (896,), dtype=torch.float16, is_leaf=True)  # arg1050_1
    buf1051 = reader.storage(None, 1146880, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1051, (896, 640), dtype=torch.float16, is_leaf=True)  # arg1051_1
    buf1052 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1052, (640,), dtype=torch.float16, is_leaf=True)  # arg1052_1
    buf1053 = reader.storage(None, 980, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1053, (490,), dtype=torch.float16, is_leaf=True)  # arg1053_1
    buf1054 = reader.storage(None, 980, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1054, (490,), dtype=torch.float16, is_leaf=True)  # arg1054_1
    buf1055 = reader.storage(None, 627200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1055, (490, 640), dtype=torch.float16, is_leaf=True)  # arg1055_1
    buf1056 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1056, (640,), dtype=torch.float16, is_leaf=True)  # arg1056_1
    buf1057 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1057, (256,), dtype=torch.float16, is_leaf=True)  # arg1057_1
    buf1058 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1058, (256,), dtype=torch.float16, is_leaf=True)  # arg1058_1
    buf1059 = reader.storage(None, 327680, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1059, (256, 640), dtype=torch.float16, is_leaf=True)  # arg1059_1
    buf1060 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1060, (640,), dtype=torch.float16, is_leaf=True)  # arg1060_1
    buf1061 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1061, (128,), dtype=torch.float16, is_leaf=True)  # arg1061_1
    buf1062 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1062, (128,), dtype=torch.float16, is_leaf=True)  # arg1062_1
    buf1063 = reader.storage(None, 163840, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1063, (128, 640), dtype=torch.float16, is_leaf=True)  # arg1063_1
    buf1064 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1064, (640,), dtype=torch.float16, is_leaf=True)  # arg1064_1
    buf1065 = reader.storage(None, 10072, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1065, (5036,), dtype=torch.float16, is_leaf=True)  # arg1065_1
    buf1066 = reader.storage(None, 10072, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1066, (5036,), dtype=torch.float16, is_leaf=True)  # arg1066_1
    buf1067 = reader.storage(None, 6446080, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1067, (5036, 640), dtype=torch.float16, is_leaf=True)  # arg1067_1
    buf1068 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1068, (640,), dtype=torch.float16, is_leaf=True)  # arg1068_1
    buf1069 = reader.storage(None, 3548, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1069, (1774,), dtype=torch.float16, is_leaf=True)  # arg1069_1
    buf1070 = reader.storage(None, 3548, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1070, (1774,), dtype=torch.float16, is_leaf=True)  # arg1070_1
    buf1071 = reader.storage(None, 2270720, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1071, (1774, 640), dtype=torch.float16, is_leaf=True)  # arg1071_1
    buf1072 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1072, (640,), dtype=torch.float16, is_leaf=True)  # arg1072_1
    buf1073 = reader.storage(None, 704, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1073, (352,), dtype=torch.float16, is_leaf=True)  # arg1073_1
    buf1074 = reader.storage(None, 704, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1074, (352,), dtype=torch.float16, is_leaf=True)  # arg1074_1
    buf1075 = reader.storage(None, 450560, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1075, (352, 640), dtype=torch.float16, is_leaf=True)  # arg1075_1
    buf1076 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1076, (640,), dtype=torch.float16, is_leaf=True)  # arg1076_1
    buf1077 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1077, (128,), dtype=torch.float16, is_leaf=True)  # arg1077_1
    buf1078 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1078, (128,), dtype=torch.float16, is_leaf=True)  # arg1078_1
    buf1079 = reader.storage(None, 163840, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1079, (128, 640), dtype=torch.float16, is_leaf=True)  # arg1079_1
    buf1080 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1080, (640,), dtype=torch.float16, is_leaf=True)  # arg1080_1
    buf1081 = reader.storage(None, 384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1081, (192,), dtype=torch.float16, is_leaf=True)  # arg1081_1
    buf1082 = reader.storage(None, 384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1082, (192,), dtype=torch.float16, is_leaf=True)  # arg1082_1
    buf1083 = reader.storage(None, 245760, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1083, (192, 640), dtype=torch.float16, is_leaf=True)  # arg1083_1
    buf1084 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1084, (640,), dtype=torch.float16, is_leaf=True)  # arg1084_1
    buf1085 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1085, (128,), dtype=torch.float16, is_leaf=True)  # arg1085_1
    buf1086 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1086, (128,), dtype=torch.float16, is_leaf=True)  # arg1086_1
    buf1087 = reader.storage(None, 163840, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1087, (128, 640), dtype=torch.float16, is_leaf=True)  # arg1087_1
    buf1088 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1088, (640,), dtype=torch.float16, is_leaf=True)  # arg1088_1
    buf1089 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1089, (512,), dtype=torch.float16, is_leaf=True)  # arg1089_1
    buf1090 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1090, (512,), dtype=torch.float16, is_leaf=True)  # arg1090_1
    buf1091 = reader.storage(None, 655360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1091, (512, 640), dtype=torch.float16, is_leaf=True)  # arg1091_1
    buf1092 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1092, (640,), dtype=torch.float16, is_leaf=True)  # arg1092_1
    buf1093 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1093, (64,), dtype=torch.float16, is_leaf=True)  # arg1093_1
    buf1094 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1094, (64,), dtype=torch.float16, is_leaf=True)  # arg1094_1
    buf1095 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1095, (64, 640), dtype=torch.float16, is_leaf=True)  # arg1095_1
    buf1096 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1096, (640,), dtype=torch.float16, is_leaf=True)  # arg1096_1
    buf1097 = reader.storage(None, 2704, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1097, (1352,), dtype=torch.float16, is_leaf=True)  # arg1097_1
    buf1098 = reader.storage(None, 2704, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1098, (1352,), dtype=torch.float16, is_leaf=True)  # arg1098_1
    buf1099 = reader.storage(None, 1730560, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1099, (1352, 640), dtype=torch.float16, is_leaf=True)  # arg1099_1
    buf1100 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1100, (640,), dtype=torch.float16, is_leaf=True)  # arg1100_1
    buf1101 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1101, (64,), dtype=torch.float16, is_leaf=True)  # arg1101_1
    buf1102 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1102, (64,), dtype=torch.float16, is_leaf=True)  # arg1102_1
    buf1103 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1103, (64, 640), dtype=torch.float16, is_leaf=True)  # arg1103_1
    buf1104 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1104, (640,), dtype=torch.float16, is_leaf=True)  # arg1104_1
    buf1105 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1105, (512,), dtype=torch.float16, is_leaf=True)  # arg1105_1
    buf1106 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1106, (512,), dtype=torch.float16, is_leaf=True)  # arg1106_1
    buf1107 = reader.storage(None, 655360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1107, (512, 640), dtype=torch.float16, is_leaf=True)  # arg1107_1
    buf1108 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1108, (640,), dtype=torch.float16, is_leaf=True)  # arg1108_1
    buf1109 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1109, (128,), dtype=torch.float16, is_leaf=True)  # arg1109_1
    buf1110 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1110, (128,), dtype=torch.float16, is_leaf=True)  # arg1110_1
    buf1111 = reader.storage(None, 163840, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1111, (128, 640), dtype=torch.float16, is_leaf=True)  # arg1111_1
    buf1112 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1112, (640,), dtype=torch.float16, is_leaf=True)  # arg1112_1
    buf1113 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1113, (512,), dtype=torch.float16, is_leaf=True)  # arg1113_1
    buf1114 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1114, (512,), dtype=torch.float16, is_leaf=True)  # arg1114_1
    buf1115 = reader.storage(None, 655360, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1115, (512, 640), dtype=torch.float16, is_leaf=True)  # arg1115_1
    buf1116 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1116, (640,), dtype=torch.float16, is_leaf=True)  # arg1116_1
    buf1117 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1117, (32,), dtype=torch.float16, is_leaf=True)  # arg1117_1
    buf1118 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1118, (32,), dtype=torch.float16, is_leaf=True)  # arg1118_1
    buf1119 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1119, (32, 640), dtype=torch.float16, is_leaf=True)  # arg1119_1
    buf1120 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1120, (640,), dtype=torch.float16, is_leaf=True)  # arg1120_1
    buf1121 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1121, (32,), dtype=torch.float16, is_leaf=True)  # arg1121_1
    buf1122 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1122, (32,), dtype=torch.float16, is_leaf=True)  # arg1122_1
    buf1123 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1123, (32, 640), dtype=torch.float16, is_leaf=True)  # arg1123_1
    buf1124 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1124, (640,), dtype=torch.float16, is_leaf=True)  # arg1124_1
    buf1125 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1125, (32,), dtype=torch.float16, is_leaf=True)  # arg1125_1
    buf1126 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1126, (32,), dtype=torch.float16, is_leaf=True)  # arg1126_1
    buf1127 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1127, (32, 640), dtype=torch.float16, is_leaf=True)  # arg1127_1
    buf1128 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1128, (640,), dtype=torch.float16, is_leaf=True)  # arg1128_1
    buf1129 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1129, (256,), dtype=torch.float16, is_leaf=True)  # arg1129_1
    buf1130 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1130, (256,), dtype=torch.float16, is_leaf=True)  # arg1130_1
    buf1131 = reader.storage(None, 327680, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1131, (256, 640), dtype=torch.float16, is_leaf=True)  # arg1131_1
    buf1132 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1132, (640,), dtype=torch.float16, is_leaf=True)  # arg1132_1
    buf1133 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1133, (256,), dtype=torch.float16, is_leaf=True)  # arg1133_1
    buf1134 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1134, (256,), dtype=torch.float16, is_leaf=True)  # arg1134_1
    buf1135 = reader.storage(None, 327680, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1135, (256, 640), dtype=torch.float16, is_leaf=True)  # arg1135_1
    buf1136 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1136, (640,), dtype=torch.float16, is_leaf=True)  # arg1136_1
    buf1137 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1137, (64,), dtype=torch.float16, is_leaf=True)  # arg1137_1
    buf1138 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1138, (64,), dtype=torch.float16, is_leaf=True)  # arg1138_1
    buf1139 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1139, (64, 640), dtype=torch.float16, is_leaf=True)  # arg1139_1
    buf1140 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1140, (640,), dtype=torch.float16, is_leaf=True)  # arg1140_1
    buf1141 = reader.storage(None, 3072, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1141, (1536,), dtype=torch.float16, is_leaf=True)  # arg1141_1
    buf1142 = reader.storage(None, 3072, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1142, (1536,), dtype=torch.float16, is_leaf=True)  # arg1142_1
    buf1143 = reader.storage(None, 1966080, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1143, (1536, 640), dtype=torch.float16, is_leaf=True)  # arg1143_1
    buf1144 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1144, (640,), dtype=torch.float16, is_leaf=True)  # arg1144_1
    buf1145 = reader.storage(None, 2560, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1145, (1280,), dtype=torch.float16, is_leaf=True)  # arg1145_1
    buf1146 = reader.storage(None, 2560, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1146, (1280,), dtype=torch.float16, is_leaf=True)  # arg1146_1
    buf1147 = reader.storage(None, 1638400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1147, (1280, 640), dtype=torch.float16, is_leaf=True)  # arg1147_1
    buf1148 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1148, (640,), dtype=torch.float16, is_leaf=True)  # arg1148_1
    buf1149 = reader.storage(None, 1152, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1149, (576,), dtype=torch.float16, is_leaf=True)  # arg1149_1
    buf1150 = reader.storage(None, 1152, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1150, (576,), dtype=torch.float16, is_leaf=True)  # arg1150_1
    buf1151 = reader.storage(None, 737280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1151, (576, 640), dtype=torch.float16, is_leaf=True)  # arg1151_1
    buf1152 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1152, (640,), dtype=torch.float16, is_leaf=True)  # arg1152_1
    buf1153 = reader.storage(None, 9088, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1153, (4544,), dtype=torch.float16, is_leaf=True)  # arg1153_1
    buf1154 = reader.storage(None, 9088, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1154, (4544,), dtype=torch.float16, is_leaf=True)  # arg1154_1
    buf1155 = reader.storage(None, 5816320, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1155, (4544, 640), dtype=torch.float16, is_leaf=True)  # arg1155_1
    buf1156 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1156, (640,), dtype=torch.float16, is_leaf=True)  # arg1156_1
    buf1157 = reader.storage(None, 1952, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1157, (976,), dtype=torch.float16, is_leaf=True)  # arg1157_1
    buf1158 = reader.storage(None, 1952, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1158, (976,), dtype=torch.float16, is_leaf=True)  # arg1158_1
    buf1159 = reader.storage(None, 1249280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1159, (976, 640), dtype=torch.float16, is_leaf=True)  # arg1159_1
    buf1160 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1160, (640,), dtype=torch.float16, is_leaf=True)  # arg1160_1
    buf1161 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1161, (32,), dtype=torch.float16, is_leaf=True)  # arg1161_1
    buf1162 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1162, (32,), dtype=torch.float16, is_leaf=True)  # arg1162_1
    buf1163 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1163, (32, 640), dtype=torch.float16, is_leaf=True)  # arg1163_1
    buf1164 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1164, (640,), dtype=torch.float16, is_leaf=True)  # arg1164_1
    buf1165 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1165, (32,), dtype=torch.float16, is_leaf=True)  # arg1165_1
    buf1166 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1166, (32,), dtype=torch.float16, is_leaf=True)  # arg1166_1
    buf1167 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1167, (32, 640), dtype=torch.float16, is_leaf=True)  # arg1167_1
    buf1168 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1168, (640,), dtype=torch.float16, is_leaf=True)  # arg1168_1
    buf1169 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1169, (32,), dtype=torch.float16, is_leaf=True)  # arg1169_1
    buf1170 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1170, (32,), dtype=torch.float16, is_leaf=True)  # arg1170_1
    buf1171 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1171, (32, 640), dtype=torch.float16, is_leaf=True)  # arg1171_1
    buf1172 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1172, (640,), dtype=torch.float16, is_leaf=True)  # arg1172_1
    buf1173 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1173, (32,), dtype=torch.float16, is_leaf=True)  # arg1173_1
    buf1174 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1174, (32,), dtype=torch.float16, is_leaf=True)  # arg1174_1
    buf1175 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1175, (32, 640), dtype=torch.float16, is_leaf=True)  # arg1175_1
    buf1176 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1176, (640,), dtype=torch.float16, is_leaf=True)  # arg1176_1
    buf1177 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1177, (128,), dtype=torch.float16, is_leaf=True)  # arg1177_1
    buf1178 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1178, (128,), dtype=torch.float16, is_leaf=True)  # arg1178_1
    buf1179 = reader.storage(None, 163840, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1179, (128, 640), dtype=torch.float16, is_leaf=True)  # arg1179_1
    buf1180 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1180, (640,), dtype=torch.float16, is_leaf=True)  # arg1180_1
    buf1181 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1181, (32,), dtype=torch.float16, is_leaf=True)  # arg1181_1
    buf1182 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1182, (32,), dtype=torch.float16, is_leaf=True)  # arg1182_1
    buf1183 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1183, (32, 640), dtype=torch.float16, is_leaf=True)  # arg1183_1
    buf1184 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1184, (640,), dtype=torch.float16, is_leaf=True)  # arg1184_1
    buf1185 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1185, (64,), dtype=torch.float16, is_leaf=True)  # arg1185_1
    buf1186 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1186, (64,), dtype=torch.float16, is_leaf=True)  # arg1186_1
    buf1187 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1187, (64, 640), dtype=torch.float16, is_leaf=True)  # arg1187_1
    buf1188 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1188, (640,), dtype=torch.float16, is_leaf=True)  # arg1188_1
    buf1189 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1189, (64,), dtype=torch.float16, is_leaf=True)  # arg1189_1
    buf1190 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1190, (64,), dtype=torch.float16, is_leaf=True)  # arg1190_1
    buf1191 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1191, (64, 640), dtype=torch.float16, is_leaf=True)  # arg1191_1
    buf1192 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1192, (640,), dtype=torch.float16, is_leaf=True)  # arg1192_1
    buf1193 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1193, (64,), dtype=torch.float16, is_leaf=True)  # arg1193_1
    buf1194 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1194, (64,), dtype=torch.float16, is_leaf=True)  # arg1194_1
    buf1195 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1195, (64, 640), dtype=torch.float16, is_leaf=True)  # arg1195_1
    buf1196 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1196, (640,), dtype=torch.float16, is_leaf=True)  # arg1196_1
    buf1197 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1197, (128,), dtype=torch.float16, is_leaf=True)  # arg1197_1
    buf1198 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1198, (128,), dtype=torch.float16, is_leaf=True)  # arg1198_1
    buf1199 = reader.storage(None, 163840, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1199, (128, 640), dtype=torch.float16, is_leaf=True)  # arg1199_1
    buf1200 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1200, (640,), dtype=torch.float16, is_leaf=True)  # arg1200_1
    buf1201 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1201, (128,), dtype=torch.float16, is_leaf=True)  # arg1201_1
    buf1202 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1202, (128,), dtype=torch.float16, is_leaf=True)  # arg1202_1
    buf1203 = reader.storage(None, 163840, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1203, (128, 640), dtype=torch.float16, is_leaf=True)  # arg1203_1
    buf1204 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1204, (640,), dtype=torch.float16, is_leaf=True)  # arg1204_1
    buf1205 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1205, (32,), dtype=torch.float16, is_leaf=True)  # arg1205_1
    buf1206 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1206, (32,), dtype=torch.float16, is_leaf=True)  # arg1206_1
    buf1207 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1207, (32, 640), dtype=torch.float16, is_leaf=True)  # arg1207_1
    buf1208 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1208, (640,), dtype=torch.float16, is_leaf=True)  # arg1208_1
    buf1209 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1209, (32,), dtype=torch.float16, is_leaf=True)  # arg1209_1
    buf1210 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1210, (32,), dtype=torch.float16, is_leaf=True)  # arg1210_1
    buf1211 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1211, (32, 640), dtype=torch.float16, is_leaf=True)  # arg1211_1
    buf1212 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1212, (640,), dtype=torch.float16, is_leaf=True)  # arg1212_1
    buf1213 = reader.storage(None, 2112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1213, (1056,), dtype=torch.float16, is_leaf=True)  # arg1213_1
    buf1214 = reader.storage(None, 2112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1214, (1056,), dtype=torch.float16, is_leaf=True)  # arg1214_1
    buf1215 = reader.storage(None, 1351680, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1215, (1056, 640), dtype=torch.float16, is_leaf=True)  # arg1215_1
    buf1216 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1216, (640,), dtype=torch.float16, is_leaf=True)  # arg1216_1
    buf1217 = reader.storage(None, 2112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1217, (1056,), dtype=torch.float16, is_leaf=True)  # arg1217_1
    buf1218 = reader.storage(None, 2112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1218, (1056,), dtype=torch.float16, is_leaf=True)  # arg1218_1
    buf1219 = reader.storage(None, 1351680, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1219, (1056, 640), dtype=torch.float16, is_leaf=True)  # arg1219_1
    buf1220 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1220, (640,), dtype=torch.float16, is_leaf=True)  # arg1220_1
    buf1221 = reader.storage(None, 1922, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1221, (961,), dtype=torch.float16, is_leaf=True)  # arg1221_1
    buf1222 = reader.storage(None, 1922, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1222, (961,), dtype=torch.float16, is_leaf=True)  # arg1222_1
    buf1223 = reader.storage(None, 1230080, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1223, (961, 640), dtype=torch.float16, is_leaf=True)  # arg1223_1
    buf1224 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1224, (640,), dtype=torch.float16, is_leaf=True)  # arg1224_1
    buf1225 = reader.storage(None, 34, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1225, (17,), dtype=torch.float16, is_leaf=True)  # arg1225_1
    buf1226 = reader.storage(None, 34, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1226, (17,), dtype=torch.float16, is_leaf=True)  # arg1226_1
    buf1227 = reader.storage(None, 21760, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1227, (17, 640), dtype=torch.float16, is_leaf=True)  # arg1227_1
    buf1228 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1228, (640,), dtype=torch.float16, is_leaf=True)  # arg1228_1
    buf1229 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1229, (64,), dtype=torch.float16, is_leaf=True)  # arg1229_1
    buf1230 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1230, (64,), dtype=torch.float16, is_leaf=True)  # arg1230_1
    buf1231 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1231, (64, 640), dtype=torch.float16, is_leaf=True)  # arg1231_1
    buf1232 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1232, (640,), dtype=torch.float16, is_leaf=True)  # arg1232_1
    buf1233 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1233, (64,), dtype=torch.float16, is_leaf=True)  # arg1233_1
    buf1234 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1234, (64,), dtype=torch.float16, is_leaf=True)  # arg1234_1
    buf1235 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1235, (64, 640), dtype=torch.float16, is_leaf=True)  # arg1235_1
    buf1236 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1236, (640,), dtype=torch.float16, is_leaf=True)  # arg1236_1
    buf1237 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1237, (16,), dtype=torch.float16, is_leaf=True)  # arg1237_1
    buf1238 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1238, (16,), dtype=torch.float16, is_leaf=True)  # arg1238_1
    buf1239 = reader.storage(None, 20480, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1239, (16, 640), dtype=torch.float16, is_leaf=True)  # arg1239_1
    buf1240 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1240, (640,), dtype=torch.float16, is_leaf=True)  # arg1240_1
    buf1241 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1241, (16,), dtype=torch.float16, is_leaf=True)  # arg1241_1
    buf1242 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1242, (16,), dtype=torch.float16, is_leaf=True)  # arg1242_1
    buf1243 = reader.storage(None, 20480, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1243, (16, 640), dtype=torch.float16, is_leaf=True)  # arg1243_1
    buf1244 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1244, (640,), dtype=torch.float16, is_leaf=True)  # arg1244_1
    buf1245 = reader.storage(None, 1376, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1245, (688,), dtype=torch.float16, is_leaf=True)  # arg1245_1
    buf1246 = reader.storage(None, 1376, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1246, (688,), dtype=torch.float16, is_leaf=True)  # arg1246_1
    buf1247 = reader.storage(None, 880640, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1247, (688, 640), dtype=torch.float16, is_leaf=True)  # arg1247_1
    buf1248 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1248, (640,), dtype=torch.float16, is_leaf=True)  # arg1248_1
    buf1249 = reader.storage(None, 96, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1249, (48,), dtype=torch.float16, is_leaf=True)  # arg1249_1
    buf1250 = reader.storage(None, 96, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1250, (48,), dtype=torch.float16, is_leaf=True)  # arg1250_1
    buf1251 = reader.storage(None, 61440, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1251, (48, 640), dtype=torch.float16, is_leaf=True)  # arg1251_1
    buf1252 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1252, (640,), dtype=torch.float16, is_leaf=True)  # arg1252_1
    buf1253 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1253, (1024,), dtype=torch.float16, is_leaf=True)  # arg1253_1
    buf1254 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1254, (1024,), dtype=torch.float16, is_leaf=True)  # arg1254_1
    buf1255 = reader.storage(None, 1310720, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1255, (1024, 640), dtype=torch.float16, is_leaf=True)  # arg1255_1
    buf1256 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1256, (640,), dtype=torch.float16, is_leaf=True)  # arg1256_1
    buf1257 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1257, (1024,), dtype=torch.float16, is_leaf=True)  # arg1257_1
    buf1258 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1258, (1024,), dtype=torch.float16, is_leaf=True)  # arg1258_1
    buf1259 = reader.storage(None, 1310720, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1259, (1024, 640), dtype=torch.float16, is_leaf=True)  # arg1259_1
    buf1260 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1260, (640,), dtype=torch.float16, is_leaf=True)  # arg1260_1
    buf1261 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1261, (1024,), dtype=torch.float16, is_leaf=True)  # arg1261_1
    buf1262 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1262, (1024,), dtype=torch.float16, is_leaf=True)  # arg1262_1
    buf1263 = reader.storage(None, 1310720, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1263, (1024, 640), dtype=torch.float16, is_leaf=True)  # arg1263_1
    buf1264 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1264, (640,), dtype=torch.float16, is_leaf=True)  # arg1264_1
    buf1265 = reader.storage(None, 2292, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1265, (1146,), dtype=torch.float16, is_leaf=True)  # arg1265_1
    buf1266 = reader.storage(None, 2292, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1266, (1146,), dtype=torch.float16, is_leaf=True)  # arg1266_1
    buf1267 = reader.storage(None, 1466880, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1267, (1146, 640), dtype=torch.float16, is_leaf=True)  # arg1267_1
    buf1268 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1268, (640,), dtype=torch.float16, is_leaf=True)  # arg1268_1
    buf1269 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1269, (640,), dtype=torch.float16, is_leaf=True)  # arg1269_1
    buf1270 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1270, (640,), dtype=torch.float16, is_leaf=True)  # arg1270_1
    buf1271 = reader.storage(None, 8192000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1271, (80, 640, 80), dtype=torch.float16, is_leaf=True)  # arg1271_1
    buf1272 = reader.storage(None, 12800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1272, (80, 1, 80), dtype=torch.float16, is_leaf=True)  # arg1272_1
    buf1273 = reader.storage(None, 8192000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1273, (80, 80, 640), dtype=torch.float16, is_leaf=True)  # arg1273_1
    buf1274 = reader.storage(None, 102400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1274, (80, 1, 640), dtype=torch.float16, is_leaf=True)  # arg1274_1
    buf1275 = reader.storage(None, 98304000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1275, (80, 640, 960), dtype=torch.float16, is_leaf=True)  # arg1275_1
    buf1276 = reader.storage(None, 153600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1276, (80, 1, 960), dtype=torch.float16, is_leaf=True)  # arg1276_1
    buf1277 = reader.storage(None, 196608000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1277, (80, 960, 1280), dtype=torch.float16, is_leaf=True)  # arg1277_1
    buf1278 = reader.storage(None, 204800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1278, (80, 1, 1280), dtype=torch.float16, is_leaf=True)  # arg1278_1
    buf1279 = reader.storage(None, 131072000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1279, (80, 640, 1280), dtype=torch.float16, is_leaf=True)  # arg1279_1
    buf1280 = reader.storage(None, 204800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1280, (80, 1, 1280), dtype=torch.float16, is_leaf=True)  # arg1280_1
    buf1281 = reader.storage(None, 131072000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1281, (80, 1280, 640), dtype=torch.float16, is_leaf=True)  # arg1281_1
    buf1282 = reader.storage(None, 102400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1282, (80, 1, 640), dtype=torch.float16, is_leaf=True)  # arg1282_1
    buf1283 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1283, (640,), dtype=torch.float16, is_leaf=True)  # arg1283_1
    buf1284 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1284, (640,), dtype=torch.float16, is_leaf=True)  # arg1284_1
    buf1285 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1285, (640,), dtype=torch.float16, is_leaf=True)  # arg1285_1
    buf1286 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1286, (640,), dtype=torch.float16, is_leaf=True)  # arg1286_1
    buf1287 = reader.storage(None, 8192000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1287, (80, 640, 80), dtype=torch.float16, is_leaf=True)  # arg1287_1
    buf1288 = reader.storage(None, 12800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1288, (80, 1, 80), dtype=torch.float16, is_leaf=True)  # arg1288_1
    buf1289 = reader.storage(None, 8192000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1289, (80, 80, 640), dtype=torch.float16, is_leaf=True)  # arg1289_1
    buf1290 = reader.storage(None, 102400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1290, (80, 1, 640), dtype=torch.float16, is_leaf=True)  # arg1290_1
    buf1291 = reader.storage(None, 98304000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1291, (80, 640, 960), dtype=torch.float16, is_leaf=True)  # arg1291_1
    buf1292 = reader.storage(None, 153600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1292, (80, 1, 960), dtype=torch.float16, is_leaf=True)  # arg1292_1
    buf1293 = reader.storage(None, 196608000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1293, (80, 960, 1280), dtype=torch.float16, is_leaf=True)  # arg1293_1
    buf1294 = reader.storage(None, 204800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1294, (80, 1, 1280), dtype=torch.float16, is_leaf=True)  # arg1294_1
    buf1295 = reader.storage(None, 131072000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1295, (80, 640, 1280), dtype=torch.float16, is_leaf=True)  # arg1295_1
    buf1296 = reader.storage(None, 204800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1296, (80, 1, 1280), dtype=torch.float16, is_leaf=True)  # arg1296_1
    buf1297 = reader.storage(None, 131072000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1297, (80, 1280, 640), dtype=torch.float16, is_leaf=True)  # arg1297_1
    buf1298 = reader.storage(None, 102400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1298, (80, 1, 640), dtype=torch.float16, is_leaf=True)  # arg1298_1
    buf1299 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1299, (640,), dtype=torch.float16, is_leaf=True)  # arg1299_1
    buf1300 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1300, (640,), dtype=torch.float16, is_leaf=True)  # arg1300_1
    buf1301 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1301, (1600,), dtype=torch.float16, is_leaf=True)  # arg1301_1
    buf1302 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1302, (1600,), dtype=torch.float16, is_leaf=True)  # arg1302_1
    buf1303 = reader.storage(None, 20480000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1303, (32, 1600, 200), dtype=torch.float16, is_leaf=True)  # arg1303_1
    buf1304 = reader.storage(None, 12800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1304, (32, 1, 200), dtype=torch.float16, is_leaf=True)  # arg1304_1
    buf1305 = reader.storage(None, 20480000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1305, (32, 200, 1600), dtype=torch.float16, is_leaf=True)  # arg1305_1
    buf1306 = reader.storage(None, 102400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1306, (32, 1, 1600), dtype=torch.float16, is_leaf=True)  # arg1306_1
    buf1307 = reader.storage(None, 98304000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1307, (32, 1600, 960), dtype=torch.float16, is_leaf=True)  # arg1307_1
    buf1308 = reader.storage(None, 61440, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1308, (32, 1, 960), dtype=torch.float16, is_leaf=True)  # arg1308_1
    buf1309 = reader.storage(None, 78643200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1309, (32, 960, 1280), dtype=torch.float16, is_leaf=True)  # arg1309_1
    buf1310 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1310, (32, 1, 1280), dtype=torch.float16, is_leaf=True)  # arg1310_1
    buf1311 = reader.storage(None, 131072000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1311, (32, 1600, 1280), dtype=torch.float16, is_leaf=True)  # arg1311_1
    buf1312 = reader.storage(None, 81920, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1312, (32, 1, 1280), dtype=torch.float16, is_leaf=True)  # arg1312_1
    buf1313 = reader.storage(None, 52428800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1313, (32, 1280, 640), dtype=torch.float16, is_leaf=True)  # arg1313_1
    buf1314 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1314, (32, 1, 640), dtype=torch.float16, is_leaf=True)  # arg1314_1
    buf1315 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1315, (640,), dtype=torch.float16, is_leaf=True)  # arg1315_1
    buf1316 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1316, (640,), dtype=torch.float16, is_leaf=True)  # arg1316_1
    buf1317 = reader.storage(None, 2560, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1317, (1280,), dtype=torch.float16, is_leaf=True)  # arg1317_1
    buf1318 = reader.storage(None, 2560, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1318, (1280,), dtype=torch.float16, is_leaf=True)  # arg1318_1
    buf1319 = reader.storage(None, 6553600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1319, (16, 1280, 160), dtype=torch.float16, is_leaf=True)  # arg1319_1
    buf1320 = reader.storage(None, 5120, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1320, (16, 1, 160), dtype=torch.float16, is_leaf=True)  # arg1320_1
    buf1321 = reader.storage(None, 6553600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1321, (16, 160, 1280), dtype=torch.float16, is_leaf=True)  # arg1321_1
    buf1322 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1322, (16, 1, 1280), dtype=torch.float16, is_leaf=True)  # arg1322_1
    buf1323 = reader.storage(None, 39321600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1323, (16, 1280, 960), dtype=torch.float16, is_leaf=True)  # arg1323_1
    buf1324 = reader.storage(None, 30720, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1324, (16, 1, 960), dtype=torch.float16, is_leaf=True)  # arg1324_1
    buf1325 = reader.storage(None, 39321600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1325, (16, 960, 1280), dtype=torch.float16, is_leaf=True)  # arg1325_1
    buf1326 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1326, (16, 1, 1280), dtype=torch.float16, is_leaf=True)  # arg1326_1
    buf1327 = reader.storage(None, 52428800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1327, (16, 1280, 1280), dtype=torch.float16, is_leaf=True)  # arg1327_1
    buf1328 = reader.storage(None, 40960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1328, (16, 1, 1280), dtype=torch.float16, is_leaf=True)  # arg1328_1
    buf1329 = reader.storage(None, 26214400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1329, (16, 1280, 640), dtype=torch.float16, is_leaf=True)  # arg1329_1
    buf1330 = reader.storage(None, 20480, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1330, (16, 1, 640), dtype=torch.float16, is_leaf=True)  # arg1330_1
    buf1331 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1331, (640,), dtype=torch.float16, is_leaf=True)  # arg1331_1
    buf1332 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1332, (640,), dtype=torch.float16, is_leaf=True)  # arg1332_1
    buf1333 = reader.storage(None, 229376, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1333, (7, 16384), dtype=torch.float16, is_leaf=True)  # arg1333_1
    buf1334 = reader.storage(None, 232960, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1334, (7, 256, 65), dtype=torch.float16, is_leaf=True)  # arg1334_1
    buf1335 = reader.storage(None, 7168, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf1335, (7, 256), dtype=torch.int32, is_leaf=True)  # arg1335_1
    buf1336 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1336, (64, 64), dtype=torch.float16, is_leaf=True)  # arg1336_1
    buf1337 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1337, (64,), dtype=torch.float16, is_leaf=True)  # arg1337_1
    buf1338 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1338, (64, 512), dtype=torch.float16, is_leaf=True)  # arg1338_1
    buf1339 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1339, (512,), dtype=torch.float16, is_leaf=True)  # arg1339_1
    buf1340 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1340, (64, 512), dtype=torch.float16, is_leaf=True)  # arg1340_1
    buf1341 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1341, (512,), dtype=torch.float16, is_leaf=True)  # arg1341_1
    buf1342 = reader.storage(None, 8515584, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1342, (8316, 512), dtype=torch.float16, is_leaf=True)  # arg1342_1
    buf1343 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1343, (512,), dtype=torch.float16, is_leaf=True)  # arg1343_1
    buf1344 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1344, (512,), dtype=torch.float16, is_leaf=True)  # arg1344_1
    buf1345 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1345, (512,), dtype=torch.float16, is_leaf=True)  # arg1345_1
    buf1346 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1346, (512,), dtype=torch.float16, is_leaf=True)  # arg1346_1
    buf1347 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1347, (512,), dtype=torch.float16, is_leaf=True)  # arg1347_1
    buf1348 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1348, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1348_1
    buf1349 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1349, (512,), dtype=torch.float16, is_leaf=True)  # arg1349_1
    buf1350 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1350, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1350_1
    buf1351 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1351, (512,), dtype=torch.float16, is_leaf=True)  # arg1351_1
    buf1352 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1352, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1352_1
    buf1353 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1353, (512,), dtype=torch.float16, is_leaf=True)  # arg1353_1
    buf1354 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1354, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1354_1
    buf1355 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1355, (512,), dtype=torch.float16, is_leaf=True)  # arg1355_1
    buf1356 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1356, (512,), dtype=torch.float16, is_leaf=True)  # arg1356_1
    buf1357 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1357, (512,), dtype=torch.float16, is_leaf=True)  # arg1357_1
    buf1358 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1358, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg1358_1
    buf1359 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1359, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg1359_1
    buf1360 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1360, (1024,), dtype=torch.float16, is_leaf=True)  # arg1360_1
    buf1361 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1361, (1024,), dtype=torch.float16, is_leaf=True)  # arg1361_1
    buf1362 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1362, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg1362_1
    buf1363 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1363, (512,), dtype=torch.float16, is_leaf=True)  # arg1363_1
    buf1364 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1364, (512,), dtype=torch.float16, is_leaf=True)  # arg1364_1
    buf1365 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1365, (512,), dtype=torch.float16, is_leaf=True)  # arg1365_1
    buf1366 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1366, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1366_1
    buf1367 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1367, (512,), dtype=torch.float16, is_leaf=True)  # arg1367_1
    buf1368 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1368, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1368_1
    buf1369 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1369, (512,), dtype=torch.float16, is_leaf=True)  # arg1369_1
    buf1370 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1370, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1370_1
    buf1371 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1371, (512,), dtype=torch.float16, is_leaf=True)  # arg1371_1
    buf1372 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1372, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1372_1
    buf1373 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1373, (512,), dtype=torch.float16, is_leaf=True)  # arg1373_1
    buf1374 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1374, (512,), dtype=torch.float16, is_leaf=True)  # arg1374_1
    buf1375 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1375, (512,), dtype=torch.float16, is_leaf=True)  # arg1375_1
    buf1376 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1376, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg1376_1
    buf1377 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1377, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg1377_1
    buf1378 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1378, (1024,), dtype=torch.float16, is_leaf=True)  # arg1378_1
    buf1379 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1379, (1024,), dtype=torch.float16, is_leaf=True)  # arg1379_1
    buf1380 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1380, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg1380_1
    buf1381 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1381, (512,), dtype=torch.float16, is_leaf=True)  # arg1381_1
    buf1382 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1382, (512,), dtype=torch.float16, is_leaf=True)  # arg1382_1
    buf1383 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1383, (512,), dtype=torch.float16, is_leaf=True)  # arg1383_1
    buf1384 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1384, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1384_1
    buf1385 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1385, (512,), dtype=torch.float16, is_leaf=True)  # arg1385_1
    buf1386 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1386, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1386_1
    buf1387 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1387, (512,), dtype=torch.float16, is_leaf=True)  # arg1387_1
    buf1388 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1388, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1388_1
    buf1389 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1389, (512,), dtype=torch.float16, is_leaf=True)  # arg1389_1
    buf1390 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1390, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1390_1
    buf1391 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1391, (512,), dtype=torch.float16, is_leaf=True)  # arg1391_1
    buf1392 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1392, (512,), dtype=torch.float16, is_leaf=True)  # arg1392_1
    buf1393 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1393, (512,), dtype=torch.float16, is_leaf=True)  # arg1393_1
    buf1394 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1394, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg1394_1
    buf1395 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1395, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg1395_1
    buf1396 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1396, (1024,), dtype=torch.float16, is_leaf=True)  # arg1396_1
    buf1397 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1397, (1024,), dtype=torch.float16, is_leaf=True)  # arg1397_1
    buf1398 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1398, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg1398_1
    buf1399 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1399, (512,), dtype=torch.float16, is_leaf=True)  # arg1399_1
    buf1400 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1400, (512,), dtype=torch.float16, is_leaf=True)  # arg1400_1
    buf1401 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1401, (512,), dtype=torch.float16, is_leaf=True)  # arg1401_1
    buf1402 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1402, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1402_1
    buf1403 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1403, (512,), dtype=torch.float16, is_leaf=True)  # arg1403_1
    buf1404 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1404, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1404_1
    buf1405 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1405, (512,), dtype=torch.float16, is_leaf=True)  # arg1405_1
    buf1406 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1406, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1406_1
    buf1407 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1407, (512,), dtype=torch.float16, is_leaf=True)  # arg1407_1
    buf1408 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1408, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1408_1
    buf1409 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1409, (512,), dtype=torch.float16, is_leaf=True)  # arg1409_1
    buf1410 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1410, (512,), dtype=torch.float16, is_leaf=True)  # arg1410_1
    buf1411 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1411, (512,), dtype=torch.float16, is_leaf=True)  # arg1411_1
    buf1412 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1412, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg1412_1
    buf1413 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1413, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg1413_1
    buf1414 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1414, (1024,), dtype=torch.float16, is_leaf=True)  # arg1414_1
    buf1415 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1415, (1024,), dtype=torch.float16, is_leaf=True)  # arg1415_1
    buf1416 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1416, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg1416_1
    buf1417 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1417, (512,), dtype=torch.float16, is_leaf=True)  # arg1417_1
    buf1418 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1418, (512,), dtype=torch.float16, is_leaf=True)  # arg1418_1
    buf1419 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1419, (512,), dtype=torch.float16, is_leaf=True)  # arg1419_1
    buf1420 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1420, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1420_1
    buf1421 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1421, (512,), dtype=torch.float16, is_leaf=True)  # arg1421_1
    buf1422 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1422, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1422_1
    buf1423 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1423, (512,), dtype=torch.float16, is_leaf=True)  # arg1423_1
    buf1424 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1424, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1424_1
    buf1425 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1425, (512,), dtype=torch.float16, is_leaf=True)  # arg1425_1
    buf1426 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1426, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1426_1
    buf1427 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1427, (512,), dtype=torch.float16, is_leaf=True)  # arg1427_1
    buf1428 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1428, (512,), dtype=torch.float16, is_leaf=True)  # arg1428_1
    buf1429 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1429, (512,), dtype=torch.float16, is_leaf=True)  # arg1429_1
    buf1430 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1430, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg1430_1
    buf1431 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1431, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg1431_1
    buf1432 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1432, (1024,), dtype=torch.float16, is_leaf=True)  # arg1432_1
    buf1433 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1433, (1024,), dtype=torch.float16, is_leaf=True)  # arg1433_1
    buf1434 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1434, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg1434_1
    buf1435 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1435, (512,), dtype=torch.float16, is_leaf=True)  # arg1435_1
    buf1436 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1436, (512,), dtype=torch.float16, is_leaf=True)  # arg1436_1
    buf1437 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1437, (512,), dtype=torch.float16, is_leaf=True)  # arg1437_1
    buf1438 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1438, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1438_1
    buf1439 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1439, (512,), dtype=torch.float16, is_leaf=True)  # arg1439_1
    buf1440 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1440, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1440_1
    buf1441 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1441, (512,), dtype=torch.float16, is_leaf=True)  # arg1441_1
    buf1442 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1442, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1442_1
    buf1443 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1443, (512,), dtype=torch.float16, is_leaf=True)  # arg1443_1
    buf1444 = reader.storage(None, 524288, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1444, (512, 512), dtype=torch.float16, is_leaf=True)  # arg1444_1
    buf1445 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1445, (512,), dtype=torch.float16, is_leaf=True)  # arg1445_1
    buf1446 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1446, (512,), dtype=torch.float16, is_leaf=True)  # arg1446_1
    buf1447 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1447, (512,), dtype=torch.float16, is_leaf=True)  # arg1447_1
    buf1448 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1448, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg1448_1
    buf1449 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1449, (512, 1024), dtype=torch.float16, is_leaf=True)  # arg1449_1
    buf1450 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1450, (1024,), dtype=torch.float16, is_leaf=True)  # arg1450_1
    buf1451 = reader.storage(None, 2048, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1451, (1024,), dtype=torch.float16, is_leaf=True)  # arg1451_1
    buf1452 = reader.storage(None, 1048576, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1452, (1024, 512), dtype=torch.float16, is_leaf=True)  # arg1452_1
    buf1453 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1453, (512,), dtype=torch.float16, is_leaf=True)  # arg1453_1
    buf1454 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1454, (1024, 32), dtype=torch.float16, is_leaf=True)  # arg1454_1
    buf1455 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1455, (32,), dtype=torch.float16, is_leaf=True)  # arg1455_1
    buf1456 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1456, (32, 1), dtype=torch.float16, is_leaf=True)  # arg1456_1
    buf1457 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1457, (1,), dtype=torch.float16, is_leaf=True)  # arg1457_1
    buf1458 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1458, (1024, 32), dtype=torch.float16, is_leaf=True)  # arg1458_1
    buf1459 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1459, (32,), dtype=torch.float16, is_leaf=True)  # arg1459_1
    buf1460 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1460, (32, 1), dtype=torch.float16, is_leaf=True)  # arg1460_1
    buf1461 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1461, (1,), dtype=torch.float16, is_leaf=True)  # arg1461_1
    buf1462 = reader.storage(None, 47820800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1462, (9340, 2560), dtype=torch.float16, is_leaf=True)  # arg1462_1
    buf1463 = reader.storage(None, 5120, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1463, (2560,), dtype=torch.float16, is_leaf=True)  # arg1463_1
    buf1464 = reader.storage(None, 262144, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1464, (512, 256), dtype=torch.float16, is_leaf=True)  # arg1464_1
    buf1465 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1465, (256,), dtype=torch.float16, is_leaf=True)  # arg1465_1
    buf1466 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1466, (256, 1), dtype=torch.float16, is_leaf=True)  # arg1466_1
    buf1467 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1467, (1,), dtype=torch.float16, is_leaf=True)  # arg1467_1
    buf1468 = reader.storage(None, 262144, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1468, (512, 256), dtype=torch.float16, is_leaf=True)  # arg1468_1
    buf1469 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1469, (256,), dtype=torch.float16, is_leaf=True)  # arg1469_1
    buf1470 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1470, (256, 1), dtype=torch.float16, is_leaf=True)  # arg1470_1
    buf1471 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1471, (1,), dtype=torch.float16, is_leaf=True)  # arg1471_1
    buf1472 = reader.storage(None, 262144, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1472, (512, 256), dtype=torch.float16, is_leaf=True)  # arg1472_1
    buf1473 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1473, (256,), dtype=torch.float16, is_leaf=True)  # arg1473_1
    buf1474 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1474, (256, 1), dtype=torch.float16, is_leaf=True)  # arg1474_1
    buf1475 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1475, (1,), dtype=torch.float16, is_leaf=True)  # arg1475_1
    buf1476 = reader.storage(None, 262144, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1476, (512, 256), dtype=torch.float16, is_leaf=True)  # arg1476_1
    buf1477 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1477, (256,), dtype=torch.float16, is_leaf=True)  # arg1477_1
    buf1478 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1478, (256, 1), dtype=torch.float16, is_leaf=True)  # arg1478_1
    buf1479 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1479, (1,), dtype=torch.float16, is_leaf=True)  # arg1479_1
    buf1480 = reader.storage(None, 262144, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1480, (512, 256), dtype=torch.float16, is_leaf=True)  # arg1480_1
    buf1481 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1481, (256,), dtype=torch.float16, is_leaf=True)  # arg1481_1
    buf1482 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1482, (256, 1), dtype=torch.float16, is_leaf=True)  # arg1482_1
    buf1483 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1483, (1,), dtype=torch.float16, is_leaf=True)  # arg1483_1
    buf1484 = reader.storage(None, 93400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1484, (9340, 5), dtype=torch.float16, is_leaf=True)  # arg1484_1
    buf1485 = reader.storage(None, 10, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1485, (5,), dtype=torch.float16, is_leaf=True)  # arg1485_1
    buf1486 = reader.storage(None, 7200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1486, (batch_size_hint, 18), dtype=torch.float16, is_leaf=True)  # arg1486_1
    buf1487 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1487, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg1487_1
    buf1488 = reader.storage(None, 800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1488, (batch_size_hint, 2), dtype=torch.float16, is_leaf=True)  # arg1488_1
    buf1489 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1489, (20, 32), dtype=torch.float16, is_leaf=True)  # arg1489_1
    buf1490 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1490, (32,), dtype=torch.float16, is_leaf=True)  # arg1490_1
    buf1491 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1491, (32, 16), dtype=torch.float16, is_leaf=True)  # arg1491_1
    buf1492 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1492, (16,), dtype=torch.float16, is_leaf=True)  # arg1492_1
    buf1493 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1493, (16, 1), dtype=torch.float16, is_leaf=True)  # arg1493_1
    buf1494 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1494, (1,), dtype=torch.float16, is_leaf=True)  # arg1494_1
    buf1495 = reader.storage(None, 7200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1495, (batch_size_hint, 18), dtype=torch.float16, is_leaf=True)  # arg1495_1
    buf1496 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1496, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg1496_1
    buf1497 = reader.storage(None, 800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1497, (batch_size_hint, 2), dtype=torch.float16, is_leaf=True)  # arg1497_1
    buf1498 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1498, (20, 32), dtype=torch.float16, is_leaf=True)  # arg1498_1
    buf1499 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1499, (32,), dtype=torch.float16, is_leaf=True)  # arg1499_1
    buf1500 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1500, (32, 16), dtype=torch.float16, is_leaf=True)  # arg1500_1
    buf1501 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1501, (16,), dtype=torch.float16, is_leaf=True)  # arg1501_1
    buf1502 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1502, (16, 1), dtype=torch.float16, is_leaf=True)  # arg1502_1
    buf1503 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1503, (1,), dtype=torch.float16, is_leaf=True)  # arg1503_1
    buf1504 = reader.storage(None, 7200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1504, (batch_size_hint, 18), dtype=torch.float16, is_leaf=True)  # arg1504_1
    buf1505 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1505, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg1505_1
    buf1506 = reader.storage(None, 800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1506, (batch_size_hint, 2), dtype=torch.float16, is_leaf=True)  # arg1506_1
    buf1507 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1507, (20, 32), dtype=torch.float16, is_leaf=True)  # arg1507_1
    buf1508 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1508, (32,), dtype=torch.float16, is_leaf=True)  # arg1508_1
    buf1509 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1509, (32, 16), dtype=torch.float16, is_leaf=True)  # arg1509_1
    buf1510 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1510, (16,), dtype=torch.float16, is_leaf=True)  # arg1510_1
    buf1511 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1511, (16, 1), dtype=torch.float16, is_leaf=True)  # arg1511_1
    buf1512 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1512, (1,), dtype=torch.float16, is_leaf=True)  # arg1512_1
    buf1513 = reader.storage(None, 7200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1513, (batch_size_hint, 18), dtype=torch.float16, is_leaf=True)  # arg1513_1
    buf1514 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1514, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg1514_1
    buf1515 = reader.storage(None, 800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1515, (batch_size_hint, 2), dtype=torch.float16, is_leaf=True)  # arg1515_1
    buf1516 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1516, (20, 32), dtype=torch.float16, is_leaf=True)  # arg1516_1
    buf1517 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1517, (32,), dtype=torch.float16, is_leaf=True)  # arg1517_1
    buf1518 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1518, (32, 16), dtype=torch.float16, is_leaf=True)  # arg1518_1
    buf1519 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1519, (16,), dtype=torch.float16, is_leaf=True)  # arg1519_1
    buf1520 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1520, (16, 1), dtype=torch.float16, is_leaf=True)  # arg1520_1
    buf1521 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1521, (1,), dtype=torch.float16, is_leaf=True)  # arg1521_1
    buf1522 = reader.storage(None, 7200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1522, (batch_size_hint, 18), dtype=torch.float16, is_leaf=True)  # arg1522_1
    buf1523 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1523, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg1523_1
    buf1524 = reader.storage(None, 800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1524, (batch_size_hint, 2), dtype=torch.float16, is_leaf=True)  # arg1524_1
    buf1525 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1525, (20, 32), dtype=torch.float16, is_leaf=True)  # arg1525_1
    buf1526 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1526, (32,), dtype=torch.float16, is_leaf=True)  # arg1526_1
    buf1527 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1527, (32, 16), dtype=torch.float16, is_leaf=True)  # arg1527_1
    buf1528 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1528, (16,), dtype=torch.float16, is_leaf=True)  # arg1528_1
    buf1529 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1529, (16, 1), dtype=torch.float16, is_leaf=True)  # arg1529_1
    buf1530 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1530, (1,), dtype=torch.float16, is_leaf=True)  # arg1530_1
    buf1531 = reader.storage(None, 7200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1531, (batch_size_hint, 18), dtype=torch.float16, is_leaf=True)  # arg1531_1
    buf1532 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1532, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg1532_1
    buf1533 = reader.storage(None, 800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1533, (batch_size_hint, 2), dtype=torch.float16, is_leaf=True)  # arg1533_1
    buf1534 = reader.storage(None, 1280, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1534, (20, 32), dtype=torch.float16, is_leaf=True)  # arg1534_1
    buf1535 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1535, (32,), dtype=torch.float16, is_leaf=True)  # arg1535_1
    buf1536 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1536, (32, 16), dtype=torch.float16, is_leaf=True)  # arg1536_1
    buf1537 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1537, (16,), dtype=torch.float16, is_leaf=True)  # arg1537_1
    buf1538 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1538, (16, 1), dtype=torch.float16, is_leaf=True)  # arg1538_1
    buf1539 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1539, (1,), dtype=torch.float16, is_leaf=True)  # arg1539_1
    buf1540 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1540, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1540_1
    buf1541 = reader.storage(None, 59200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1541, (batch_size_hint, 148), dtype=torch.float16, is_leaf=True)  # arg1541_1
    buf1542 = reader.storage(None, 40000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1542, (batch_size_hint, 100), dtype=torch.float16, is_leaf=True)  # arg1542_1
    buf1543 = reader.storage(None, 16000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1543, (batch_size_hint, 40), dtype=torch.float16, is_leaf=True)  # arg1543_1
    buf1544 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1544, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1544_1
    buf1545 = reader.storage(None, 19456, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1545, (304, 32), dtype=torch.float16, is_leaf=True)  # arg1545_1
    buf1546 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1546, (32,), dtype=torch.float16, is_leaf=True)  # arg1546_1
    buf1547 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1547, (32, 1), dtype=torch.float16, is_leaf=True)  # arg1547_1
    buf1548 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1548, (1,), dtype=torch.float16, is_leaf=True)  # arg1548_1
    buf1549 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1549, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1549_1
    buf1550 = reader.storage(None, 59200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1550, (batch_size_hint, 148), dtype=torch.float16, is_leaf=True)  # arg1550_1
    buf1551 = reader.storage(None, 40000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1551, (batch_size_hint, 100), dtype=torch.float16, is_leaf=True)  # arg1551_1
    buf1552 = reader.storage(None, 16000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1552, (batch_size_hint, 40), dtype=torch.float16, is_leaf=True)  # arg1552_1
    buf1553 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1553, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1553_1
    buf1554 = reader.storage(None, 19456, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1554, (304, 32), dtype=torch.float16, is_leaf=True)  # arg1554_1
    buf1555 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1555, (32,), dtype=torch.float16, is_leaf=True)  # arg1555_1
    buf1556 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1556, (32, 1), dtype=torch.float16, is_leaf=True)  # arg1556_1
    buf1557 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1557, (1,), dtype=torch.float16, is_leaf=True)  # arg1557_1
    buf1558 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1558, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1558_1
    buf1559 = reader.storage(None, 59200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1559, (batch_size_hint, 148), dtype=torch.float16, is_leaf=True)  # arg1559_1
    buf1560 = reader.storage(None, 40000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1560, (batch_size_hint, 100), dtype=torch.float16, is_leaf=True)  # arg1560_1
    buf1561 = reader.storage(None, 16000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1561, (batch_size_hint, 40), dtype=torch.float16, is_leaf=True)  # arg1561_1
    buf1562 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1562, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1562_1
    buf1563 = reader.storage(None, 19456, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1563, (304, 32), dtype=torch.float16, is_leaf=True)  # arg1563_1
    buf1564 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1564, (32,), dtype=torch.float16, is_leaf=True)  # arg1564_1
    buf1565 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1565, (32, 1), dtype=torch.float16, is_leaf=True)  # arg1565_1
    buf1566 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1566, (1,), dtype=torch.float16, is_leaf=True)  # arg1566_1
    buf1567 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1567, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1567_1
    buf1568 = reader.storage(None, 59200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1568, (batch_size_hint, 148), dtype=torch.float16, is_leaf=True)  # arg1568_1
    buf1569 = reader.storage(None, 40000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1569, (batch_size_hint, 100), dtype=torch.float16, is_leaf=True)  # arg1569_1
    buf1570 = reader.storage(None, 16000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1570, (batch_size_hint, 40), dtype=torch.float16, is_leaf=True)  # arg1570_1
    buf1571 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1571, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1571_1
    buf1572 = reader.storage(None, 19456, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1572, (304, 32), dtype=torch.float16, is_leaf=True)  # arg1572_1
    buf1573 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1573, (32,), dtype=torch.float16, is_leaf=True)  # arg1573_1
    buf1574 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1574, (32, 1), dtype=torch.float16, is_leaf=True)  # arg1574_1
    buf1575 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1575, (1,), dtype=torch.float16, is_leaf=True)  # arg1575_1
    buf1576 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1576, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1576_1
    buf1577 = reader.storage(None, 59200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1577, (batch_size_hint, 148), dtype=torch.float16, is_leaf=True)  # arg1577_1
    buf1578 = reader.storage(None, 40000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1578, (batch_size_hint, 100), dtype=torch.float16, is_leaf=True)  # arg1578_1
    buf1579 = reader.storage(None, 16000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1579, (batch_size_hint, 40), dtype=torch.float16, is_leaf=True)  # arg1579_1
    buf1580 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1580, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1580_1
    buf1581 = reader.storage(None, 19456, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1581, (304, 32), dtype=torch.float16, is_leaf=True)  # arg1581_1
    buf1582 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1582, (32,), dtype=torch.float16, is_leaf=True)  # arg1582_1
    buf1583 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1583, (32, 1), dtype=torch.float16, is_leaf=True)  # arg1583_1
    buf1584 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1584, (1,), dtype=torch.float16, is_leaf=True)  # arg1584_1
    buf1585 = reader.storage(None, 1792, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1585, (7, 128), dtype=torch.float16, is_leaf=True)  # arg1585_1
    buf1586 = reader.storage(None, 246400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1586, (batch_size_hint, 616), dtype=torch.float16, is_leaf=True)  # arg1586_1
    buf1587 = reader.storage(None, 80000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1587, (batch_size_hint, 200), dtype=torch.float16, is_leaf=True)  # arg1587_1
    buf1588 = reader.storage(None, 6400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1588, (batch_size_hint, 16), dtype=torch.float16, is_leaf=True)  # arg1588_1
    buf1589 = reader.storage(None, 6400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1589, (batch_size_hint, 16), dtype=torch.float16, is_leaf=True)  # arg1589_1
    buf1590 = reader.storage(None, 6400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1590, (batch_size_hint, 16), dtype=torch.float16, is_leaf=True)  # arg1590_1
    buf1591 = reader.storage(None, 32000, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1591, (batch_size_hint, 80), dtype=torch.float16, is_leaf=True)  # arg1591_1
    buf1592 = reader.storage(None, 491520, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1592, (960, 256), dtype=torch.float16, is_leaf=True)  # arg1592_1
    buf1593 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1593, (256,), dtype=torch.float16, is_leaf=True)  # arg1593_1
    buf1594 = reader.storage(None, 438272, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1594, (856, 256), dtype=torch.float16, is_leaf=True)  # arg1594_1
    buf1595 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1595, (256,), dtype=torch.float16, is_leaf=True)  # arg1595_1
    buf1596 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1596, (256, 128), dtype=torch.float16, is_leaf=True)  # arg1596_1
    buf1597 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1597, (128,), dtype=torch.float16, is_leaf=True)  # arg1597_1
    buf1598 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1598, (128, 64), dtype=torch.float16, is_leaf=True)  # arg1598_1
    buf1599 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1599, (64,), dtype=torch.float16, is_leaf=True)  # arg1599_1
    buf1600 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1600, (64, 32), dtype=torch.float16, is_leaf=True)  # arg1600_1
    buf1601 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1601, (32,), dtype=torch.float16, is_leaf=True)  # arg1601_1
    buf1602 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1602, (32, 2), dtype=torch.float16, is_leaf=True)  # arg1602_1
    buf1603 = reader.storage(None, 4, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1603, (2,), dtype=torch.float16, is_leaf=True)  # arg1603_1
    buf1604 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1604, (256, 128), dtype=torch.float16, is_leaf=True)  # arg1604_1
    buf1605 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1605, (128,), dtype=torch.float16, is_leaf=True)  # arg1605_1
    buf1606 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1606, (128, 64), dtype=torch.float16, is_leaf=True)  # arg1606_1
    buf1607 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1607, (64,), dtype=torch.float16, is_leaf=True)  # arg1607_1
    buf1608 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1608, (64, 32), dtype=torch.float16, is_leaf=True)  # arg1608_1
    buf1609 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1609, (32,), dtype=torch.float16, is_leaf=True)  # arg1609_1
    buf1610 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1610, (32, 2), dtype=torch.float16, is_leaf=True)  # arg1610_1
    buf1611 = reader.storage(None, 4, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1611, (2,), dtype=torch.float16, is_leaf=True)  # arg1611_1
    buf1612 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1612, (256, 128), dtype=torch.float16, is_leaf=True)  # arg1612_1
    buf1613 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1613, (128,), dtype=torch.float16, is_leaf=True)  # arg1613_1
    buf1614 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1614, (128, 64), dtype=torch.float16, is_leaf=True)  # arg1614_1
    buf1615 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1615, (64,), dtype=torch.float16, is_leaf=True)  # arg1615_1
    buf1616 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1616, (64, 32), dtype=torch.float16, is_leaf=True)  # arg1616_1
    buf1617 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1617, (32,), dtype=torch.float16, is_leaf=True)  # arg1617_1
    buf1618 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1618, (32, 2), dtype=torch.float16, is_leaf=True)  # arg1618_1
    buf1619 = reader.storage(None, 4, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1619, (2,), dtype=torch.float16, is_leaf=True)  # arg1619_1
    buf1620 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1620, (256, 128), dtype=torch.float16, is_leaf=True)  # arg1620_1
    buf1621 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1621, (128,), dtype=torch.float16, is_leaf=True)  # arg1621_1
    buf1622 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1622, (128, 64), dtype=torch.float16, is_leaf=True)  # arg1622_1
    buf1623 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1623, (64,), dtype=torch.float16, is_leaf=True)  # arg1623_1
    buf1624 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1624, (64, 32), dtype=torch.float16, is_leaf=True)  # arg1624_1
    buf1625 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1625, (32,), dtype=torch.float16, is_leaf=True)  # arg1625_1
    buf1626 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1626, (32, 2), dtype=torch.float16, is_leaf=True)  # arg1626_1
    buf1627 = reader.storage(None, 4, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1627, (2,), dtype=torch.float16, is_leaf=True)  # arg1627_1
    buf1628 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1628, (256, 128), dtype=torch.float16, is_leaf=True)  # arg1628_1
    buf1629 = reader.storage(None, 256, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1629, (128,), dtype=torch.float16, is_leaf=True)  # arg1629_1
    buf1630 = reader.storage(None, 16384, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1630, (128, 64), dtype=torch.float16, is_leaf=True)  # arg1630_1
    buf1631 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1631, (64,), dtype=torch.float16, is_leaf=True)  # arg1631_1
    buf1632 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1632, (64, 32), dtype=torch.float16, is_leaf=True)  # arg1632_1
    buf1633 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1633, (32,), dtype=torch.float16, is_leaf=True)  # arg1633_1
    buf1634 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1634, (32, 2), dtype=torch.float16, is_leaf=True)  # arg1634_1
    buf1635 = reader.storage(None, 4, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1635, (2,), dtype=torch.float16, is_leaf=True)  # arg1635_1
    buf1636 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1636, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1636_1
    buf1637 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1637, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1637_1
    buf1638 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1638, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1638_1
    buf1639 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1639, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1639_1
    buf1640 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1640, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1640_1
    buf1641 = reader.storage(None, 38400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1641, (batch_size_hint, 96), dtype=torch.float16, is_leaf=True)  # arg1641_1
    buf1642 = reader.storage(None, 43200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1642, (batch_size_hint, 108), dtype=torch.float16, is_leaf=True)  # arg1642_1
    buf1643 = reader.storage(None, 8800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1643, (batch_size_hint, 22), dtype=torch.float16, is_leaf=True)  # arg1643_1
    buf1644 = reader.storage(None, 20800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1644, (batch_size_hint, 52), dtype=torch.float16, is_leaf=True)  # arg1644_1
    buf1645 = reader.storage(None, 3200, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1645, (batch_size_hint, 8), dtype=torch.float16, is_leaf=True)  # arg1645_1
    buf1646 = reader.storage(None, 12400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1646, (batch_size_hint, 31), dtype=torch.float16, is_leaf=True)  # arg1646_1
    buf1647 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1647, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg1647_1
    buf1648 = reader.storage(None, 9600, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1648, (batch_size_hint, 24), dtype=torch.float16, is_leaf=True)  # arg1648_1
    buf1649 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1649, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg1649_1
    buf1650 = reader.storage(None, 2800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1650, (batch_size_hint, 7), dtype=torch.float16, is_leaf=True)  # arg1650_1
    buf1651 = reader.storage(None, 4800, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1651, (batch_size_hint, 12), dtype=torch.float16, is_leaf=True)  # arg1651_1
    buf1652 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1652, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg1652_1
    buf1653 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1653, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg1653_1
    buf1654 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1654, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg1654_1
    buf1655 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1655, (batch_size_hint, 1), dtype=torch.float16, is_leaf=True)  # arg1655_1
    buf1656 = reader.storage(None, 26112, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1656, (204, 64), dtype=torch.float16, is_leaf=True)  # arg1656_1
    buf1657 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1657, (64,), dtype=torch.float16, is_leaf=True)  # arg1657_1
    buf1658 = reader.storage(None, 22784, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1658, (178, 64), dtype=torch.float16, is_leaf=True)  # arg1658_1
    buf1659 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1659, (64,), dtype=torch.float16, is_leaf=True)  # arg1659_1
    buf1660 = reader.storage(None, 4918784, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1660, (9607, 256), dtype=torch.float16, is_leaf=True)  # arg1660_1
    buf1661 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1661, (256,), dtype=torch.float16, is_leaf=True)  # arg1661_1
    buf1662 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1662, (256, 64), dtype=torch.float16, is_leaf=True)  # arg1662_1
    buf1663 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1663, (64,), dtype=torch.float16, is_leaf=True)  # arg1663_1
    buf1664 = reader.storage(None, 4920832, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1664, (9611, 256), dtype=torch.float16, is_leaf=True)  # arg1664_1
    buf1665 = reader.storage(None, 512, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1665, (256,), dtype=torch.float16, is_leaf=True)  # arg1665_1
    buf1666 = reader.storage(None, 32768, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1666, (256, 64), dtype=torch.float16, is_leaf=True)  # arg1666_1
    buf1667 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1667, (64,), dtype=torch.float16, is_leaf=True)  # arg1667_1
    buf1668 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1668, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg1668_1
    buf1669 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1669, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg1669_1
    buf1670 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1670, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg1670_1
    buf1671 = reader.storage(None, 1600, device=device(type='npu', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1671, (batch_size_hint, 1), dtype=torch.int64, is_leaf=True)  # arg1671_1
    buf1672 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1672, (64, 32), dtype=torch.float16, is_leaf=True)  # arg1672_1
    buf1673 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1673, (32,), dtype=torch.float16, is_leaf=True)  # arg1673_1
    buf1674 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1674, (32, 16), dtype=torch.float16, is_leaf=True)  # arg1674_1
    buf1675 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1675, (16,), dtype=torch.float16, is_leaf=True)  # arg1675_1
    buf1676 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1676, (16, 1), dtype=torch.float16, is_leaf=True)  # arg1676_1
    buf1677 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1677, (1,), dtype=torch.float16, is_leaf=True)  # arg1677_1
    buf1678 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1678, (1,), dtype=torch.float16, is_leaf=True)  # arg1678_1
    buf1679 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1679, (64, 32), dtype=torch.float16, is_leaf=True)  # arg1679_1
    buf1680 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1680, (32,), dtype=torch.float16, is_leaf=True)  # arg1680_1
    buf1681 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1681, (32, 16), dtype=torch.float16, is_leaf=True)  # arg1681_1
    buf1682 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1682, (16,), dtype=torch.float16, is_leaf=True)  # arg1682_1
    buf1683 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1683, (16, 1), dtype=torch.float16, is_leaf=True)  # arg1683_1
    buf1684 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1684, (1,), dtype=torch.float16, is_leaf=True)  # arg1684_1
    buf1685 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1685, (1,), dtype=torch.float16, is_leaf=True)  # arg1685_1
    buf1686 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1686, (64, 32), dtype=torch.float16, is_leaf=True)  # arg1686_1
    buf1687 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1687, (32,), dtype=torch.float16, is_leaf=True)  # arg1687_1
    buf1688 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1688, (32, 16), dtype=torch.float16, is_leaf=True)  # arg1688_1
    buf1689 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1689, (16,), dtype=torch.float16, is_leaf=True)  # arg1689_1
    buf1690 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1690, (16, 1), dtype=torch.float16, is_leaf=True)  # arg1690_1
    buf1691 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1691, (1,), dtype=torch.float16, is_leaf=True)  # arg1691_1
    buf1692 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1692, (1,), dtype=torch.float16, is_leaf=True)  # arg1692_1
    buf1693 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1693, (64, 32), dtype=torch.float16, is_leaf=True)  # arg1693_1
    buf1694 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1694, (32,), dtype=torch.float16, is_leaf=True)  # arg1694_1
    buf1695 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1695, (32, 16), dtype=torch.float16, is_leaf=True)  # arg1695_1
    buf1696 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1696, (16,), dtype=torch.float16, is_leaf=True)  # arg1696_1
    buf1697 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1697, (16, 1), dtype=torch.float16, is_leaf=True)  # arg1697_1
    buf1698 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1698, (1,), dtype=torch.float16, is_leaf=True)  # arg1698_1
    buf1699 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1699, (1,), dtype=torch.float16, is_leaf=True)  # arg1699_1
    buf1700 = reader.storage(None, 4096, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1700, (64, 32), dtype=torch.float16, is_leaf=True)  # arg1700_1
    buf1701 = reader.storage(None, 64, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1701, (32,), dtype=torch.float16, is_leaf=True)  # arg1701_1
    buf1702 = reader.storage(None, 1024, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1702, (32, 16), dtype=torch.float16, is_leaf=True)  # arg1702_1
    buf1703 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1703, (16,), dtype=torch.float16, is_leaf=True)  # arg1703_1
    buf1704 = reader.storage(None, 32, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1704, (16, 1), dtype=torch.float16, is_leaf=True)  # arg1704_1
    buf1705 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1705, (1,), dtype=torch.float16, is_leaf=True)  # arg1705_1
    buf1706 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1706, (1,), dtype=torch.float16, is_leaf=True)  # arg1706_1
    buf1707 = reader.storage(None, 400, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1707, (batch_size_hint,), dtype=torch.float16, is_leaf=True)  # arg1707_1
load_args._version = 0


class DeferredStorage:
    def __init__(self, storage_hash, nbytes, device, dtype_hint):
        self.storage_hash = storage_hash
        self.nbytes = nbytes
        self.device = device
        self.dtype_hint = dtype_hint


class SymbolicInputReader:
    def __init__(self, reader, mark_dynamic_dims):
        self.reader = reader
        self.mark_dynamic_dims = mark_dynamic_dims
        self.marked_dims = 0

    @property
    def args(self):
        return self.reader.args

    def storage(self, storage_hash, nbytes, *, device=None, dtype_hint=None):
        return DeferredStorage(storage_hash, nbytes, device, dtype_hint)

    def tensor(
        self,
        storage,
        shape,
        stride=None,
        *,
        storage_offset=None,
        dtype=None,
        **kwargs,
    ):
        concrete_shape = tuple(int(size) for size in shape)
        concrete_stride = stride
        if concrete_stride is None:
            concrete_stride = make_contiguous_strides_for(concrete_shape)
        offset = storage_offset or 0
        if any(size == 0 for size in concrete_shape):
            required_numel = 0
        else:
            required_numel = offset + 1
            required_numel += sum((size - 1) * abs(step) for size, step in zip(concrete_shape, concrete_stride))
        tensor_dtype = dtype or torch.float32
        required_nbytes = required_numel * tensor_dtype.itemsize
        actual_storage = self.reader.storage(
            storage.storage_hash,
            max(storage.nbytes, required_nbytes),
            device=storage.device,
            dtype_hint=storage.dtype_hint,
        )
        value = self.reader.tensor(
            actual_storage,
            concrete_shape,
            stride,
            storage_offset=storage_offset,
            dtype=dtype,
            **kwargs,
        )
        for dim, size in enumerate(shape):
            if isinstance(size, ShapeHint):
                if self.mark_dynamic_dims:
                    torch._dynamo.mark_dynamic(value, dim)
                    self.marked_dims += 1
        return value

    def __getattr__(self, name):
        return getattr(self.reader, name)


def latest_output_code_path():
    output_codes = [path for path in RUN_ROOT.rglob("output_code.py") if path.is_file()]
    if not output_codes:
        return None
    return max(output_codes, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def profile_compiled(compiled, args):
    warmup = int(os.environ.get("WARMUP", "1"))
    active = int(os.environ.get("ACTIVE", "10"))
    repeat = int(os.environ.get("REPEAT", "1"))
    if warmup < 0 or active <= 0 or repeat <= 0:
        raise ValueError("WARMUP must be non-negative; ACTIVE and REPEAT must be positive")

    profile_dir = RUN_ROOT / "profiles"
    profiler = TorchNpuProfiler(
        profile_dir,
        wait=0,
        warmup=warmup,
        active=active,
        repeat=repeat,
        with_stack=False,
    )
    with torch.no_grad():
        profiler.run_steps(lambda: compiled(*args))

    parser = ProfileResultParser(profile_dir)
    kernel_summaries = parser.kernel_time_by_name()
    kernel_count = sum(summary.count for summary in kernel_summaries)
    if kernel_count == 0:
        raise RuntimeError(f"No NPU device kernels found in {profile_dir}")

    call_count = active * repeat
    device_total_us = sum(summary.total_us for summary in kernel_summaries)
    result = {
        "run_id": RUN_ID,
        "execution": SCRIPT_ARGS.execution,
        "device": "npu:0",
        "bs": SCRIPT_ARGS.bs,
        "group": os.environ[GROUP_AUTOTUNE_ENV],
        "dynamic": "False" if SCRIPT_ARGS.execution == "static" else "None",
        "warmup": warmup,
        "active": active,
        "repeat": repeat,
        "profile_calls": call_count,
        "kernel_count": kernel_count,
        "kernels_per_call": f"{kernel_count / call_count:.3f}",
        "device_total_us": f"{device_total_us:.3f}",
        "device_mean_us": f"{device_total_us / call_count:.3f}",
        "step_mean_us": f"{parser.average_step_time_us():.3f}",
        "profile_dir": str(profile_dir),
    }
    result_path = RUN_ROOT / "performance.csv"
    with result_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=result)
        writer.writeheader()
        writer.writerow(result)

    print(f"profile_dir={profile_dir}")
    print(f"performance_csv={result_path}")
    print(f"device_mean_us={result['device_mean_us']} step_mean_us={result['step_mean_us']}")


def main():
    from torch._dynamo.debug_utils import InputReader

    torch.npu.set_device(0)
    debug_root = RUN_ROOT / "torch_compile_debug"
    torch._dynamo.config.debug_dir_root = str(debug_root)
    torch._inductor.config.trace.debug_dir = str(debug_root)

    symbolic = SCRIPT_ARGS.execution != "static"
    reader = SymbolicInputReader(InputReader(save_dir=None), mark_dynamic_dims=symbolic)
    load_args(reader)
    args = reader.args
    compile_dynamic = False if SCRIPT_ARGS.execution == "static" else None
    print(
        f"device=npu execution={SCRIPT_ARGS.execution} bs={SCRIPT_ARGS.bs} "
        f"group={os.environ[GROUP_AUTOTUNE_ENV]} dynamic={compile_dynamic} "
        f"marked_dims={reader.marked_dims} inputs={len(args)}"
    )
    print(f"result_dir={RUN_ROOT}")

    compiled = torch.compile(
        Repro(),
        backend="inductor",
        dynamic=compile_dynamic,
        fullgraph=True,
    )
    with torch.no_grad():
        outputs = compiled(*args)
    torch.npu.synchronize()
    print(f"compiled run completed: outputs={len(outputs)}")
    output_code = latest_output_code_path()
    print(f"output_code={output_code or 'not found'}")
    profile_compiled(compiled, args)


if __name__ == "__main__":
    main()
