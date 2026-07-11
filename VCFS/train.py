import argparse
import datetime
import os
from functools import partial

if os.environ.get("VCFS_CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["VCFS_CUDA_VISIBLE_DEVICES"]
###0414###VLM-FP2

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader_latest import DeeplabDataset, deeplab_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch
from utils.split_utils import load_split_lines
from utils.class_config import load_class_config

'''
鐠侇厾绮岄懛顏勭箒閻ㄥ嫯顕㈡稊澶婂瀻閸撳弶膩閸ㄥ绔寸€规岸娓剁憰浣规暈閹板繋浜掓稉瀣殤閻愮櫢绱?
1閵嗕浇顔勭紒鍐ㄥ娴犳梻绮忓Λ鈧弻銉ㄥ殰瀹歌京娈戦弽鐓庣础閺勵垰鎯佸陇鍐荤憰浣圭湴閿涘矁顕氭惔鎾诡洣濮瑰倹鏆熼幑顕€娉﹂弽鐓庣础娑撶OC閺嶇厧绱￠敍宀勬付鐟曚礁鍣径鍥с偨閻ㄥ嫬鍞寸€硅婀佹潏鎾冲弳閸ュ墽澧栭崪灞剧垼缁?
   鏉堟挸鍙嗛崶鍓у娑?jpg閸ュ墽澧栭敍灞炬￥闂団偓閸ュ搫鐣炬径褍鐨敍灞肩炊閸忋儴顔勭紒鍐ㄥ娴兼俺鍤滈崝銊ㄧ箻鐞涘esize閵?
   閻忔澘瀹抽崶鍙ョ窗閼奉亜濮╂潪顒佸灇RGB閸ュ墽澧栨潻娑滎攽鐠侇厾绮岄敍灞炬￥闂団偓閼奉亜绻佹穱顔芥暭閵?
   鏉堟挸鍙嗛崶鍓у婵″倹鐏夐崥搴ｇ磻闂堢€損g閿涘矂娓剁憰浣藉殰瀹歌鲸澹掗柌蹇氭祮閹存亾pg閸氬骸鍟€瀵偓婵顔勭紒鍐︹偓?

   閺嶅洨顒锋稉绨唍g閸ュ墽澧栭敍灞炬￥闂団偓閸ュ搫鐣炬径褍鐨敍灞肩炊閸忋儴顔勭紒鍐ㄥ娴兼俺鍤滈崝銊ㄧ箻鐞涘esize閵?
   閻㈠彉绨拋绋款樋閸氬苯顒熼惃鍕殶閹诡噣娉﹂弰顖滅秹缂佹粈绗傛稉瀣祰閻ㄥ嫸绱濋弽鍥╊劮閺嶇厧绱￠獮鏈电瑝缁楋箑鎮庨敍宀勬付鐟曚礁鍟€鎼达箑顦╅悶鍡愨偓鍌欑鐎规俺顩﹀▔銊﹀壈閿涗焦鐖ｇ粵鍓ф畱濮ｅ繋閲滈崓蹇曠閻愬湱娈戦崐鐓庢皑閺勵垵绻栨稉顏勫剼缁辩姷鍋ｉ幍鈧仦鐐垫畱缁夊秶琚妴?
   缂冩垳绗傜敮姝岊潌閻ㄥ嫭鏆熼幑顕€娉﹂幀璇插彙鐎电绶崗銉ユ禈閻楀洤鍨庢稉銈囪閿涘矁鍎楅弲顖滄畱閸嶅繒绀岄悙鐟扳偓闂磋礋0閿涘瞼娲伴弽鍥╂畱閸嶅繒绀岄悙鐟扳偓闂磋礋255閵嗗倽绻栭弽椋庢畱閺佺増宓侀梿鍡楀讲娴犮儲顒滅敮姝岀箥鐞涘奔绲鹃弰顖烆暕濞村妲稿▽鈩冩箒閺佸牊鐏夐惃鍕剁磼
   闂団偓鐟曚焦鏁奸幋鎰剁礉閼冲本娅欓惃鍕剼缁辩姷鍋ｉ崐闂磋礋0閿涘瞼娲伴弽鍥╂畱閸嶅繒绀岄悙鐟扳偓闂磋礋1閵?
   婵″倹鐏夐弽鐓庣础閺堝顕ら敍灞藉棘閼板喛绱癶ttps://github.com/bubbliiiing/segmentation-format-fix

2閵嗕焦宕径鍗炩偓鑲╂畱婢堆冪毈閻劋绨崚銈嗘焽閺勵垰鎯侀弨鑸垫殐閿涘本鐦潏鍐櫢鐟曚胶娈戦弰顖涙箒閺€鑸垫殐閻ㄥ嫯绉奸崝鍖＄礉閸楁娊鐛欑拠渚€娉﹂幑鐔枫亼娑撳秵鏌囨稉瀣閿涘苯顩ч弸婊堢崣鐠囦線娉﹂幑鐔枫亼閸╃儤婀版稉濠佺瑝閺€鐟板綁閻ㄥ嫯鐦介敍灞灸侀崹瀣唨閺堫兛绗傜亸杈ㄦ暪閺佹稐绨￠妴?
   閹圭喎銇戦崐鑲╂畱閸忚渹缍嬫径褍鐨獮鑸电梾閺堝绮堟稊鍫熷壈娑斿绱濇径褍鎷扮亸蹇撳涧閸︺劋绨幑鐔枫亼閻ㄥ嫯顓哥粻妤佹煙瀵骏绱濋獮鏈电瑝閺勵垱甯存潻鎴滅艾0閹靛秴銈介妴鍌氼洤閺嬫粍鍏傜憰浣筋唨閹圭喎銇戞總鐣屾箙閻愮櫢绱濋崣顖欎簰閻╁瓨甯撮崚鏉款嚠鎼存梻娈戦幑鐔枫亼閸戣姤鏆熼柌宀勬桨闂勩倓绗?0000閵?
   鐠侇厾绮屾潻鍥┾柤娑擃厾娈戦幑鐔枫亼閸婇棿绱版穱婵嗙摠閸︹暔ogs閺傚洣娆㈡径閫涚瑓閻ㄥ埐oss_%Y_%m_%d_%H_%M_%S閺傚洣娆㈡径閫涜厬
   
3閵嗕浇顔勭紒鍐ㄣ偨閻ㄥ嫭娼堥崐鍏兼瀮娴犳湹绻氱€涙ê婀猯ogs閺傚洣娆㈡径閫涜厬閿涘本鐦℃稉顏囶唲缂佸啩绗樻禒锝忕礄Epoch閿涘瀵橀崥顐ュ楠炶尪顔勭紒鍐╊劄闂€鍖＄礄Step閿涘绱濆В蹇庨嚋鐠侇厾绮屽銉╂毐閿涘湯tep閿涘绻樼悰灞肩濞嗏剝顫惔锔跨瑓闂勫秲鈧?
   婵″倹鐏夐崣顏呮Ц鐠侇厾绮屾禍鍡楀殤娑撶寗tep閺勵垯绗夋导姘箽鐎涙娈戦敍瀛峱och閸滃tep閻ㄥ嫭顩ц箛浣冾洣閹瑰绔诲Δ姘娑撳鈧?
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the VCFS facade parser.")
    parser.add_argument("--class-config", default=None, help="Path to the dataset class JSON.")
    parser.add_argument("--dataset-path", default=None, help="VOC-style dataset directory.")
    parser.add_argument("--model-path", default=None, help="Checkpoint path. Use an empty string to skip loading.")
    parser.add_argument("--input-shape", default=None, help="Input size, for example 512,512 or 512x512.")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Unfreeze-stage batch size.")
    parser.add_argument("--freeze-batch-size", type=int, default=None, help="Freeze-stage batch size.")
    parser.add_argument("--save-dir", default=None, help="Training output directory.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count.")
    parser.add_argument("--num-classes", type=int, default=None, help="Override class count.")
    parser.add_argument("--eval-flag", choices=("true", "false"), default=None, help="Enable or disable periodic mIoU evaluation.")
    cli_args, _ = parser.parse_known_args()
    #---------------------------------#
    #   Cuda    閺勵垰鎯佹担璺ㄦ暏Cuda
    #           濞屸剝婀丟PU閸欘垯浜掔拋鍓х枂閹存€揳lse
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    閻劋绨崶鍝勭暰闂呭繑婧€缁夊秴鐡?
    #           娴ｅ灝绶卞В蹇旑偧閻欘剛鐝涚拋顓犵矊闁棄褰叉禒銉ㄥ箯瀵版ぞ绔撮弽椋庢畱缂佹挻鐏?
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     閻劋绨幐鍥х暰閺勵垰鎯佹担璺ㄦ暏閸楁洘婧€婢舵艾宕遍崚鍡楃瀵繗绻嶇悰?
    #                   缂佸牏顏幐鍥︽姢娴犲懏鏁幐涔乥untu閵嗕景UDA_VISIBLE_DEVICES閻劋绨崷鈺慴untu娑撳瀵氱€规碍妯夐崡掳鈧?
    #                   Windows缁崵绮烘稉瀣帛鐠併倓濞囬悽鈥昉濡€崇础鐠嬪啰鏁ら幍鈧張澶嬫▔閸椻槄绱濇稉宥嗘暜閹镐笍DP閵?
    #   DP濡€崇础閿?
    #       鐠佸墽鐤?           distributed = False
    #       閸︺劎绮撶粩顖欒厬鏉堟挸鍙?   CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP濡€崇础閿?
    #       鐠佸墽鐤?           distributed = True
    #       閸︺劎绮撶粩顖欒厬鏉堟挸鍙?   CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     閺勵垰鎯佹担璺ㄦ暏sync_bn閿涘瓕DP濡€崇础婢舵艾宕遍崣顖滄暏
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        閺勵垰鎯佹担璺ㄦ暏濞ｅ嘲鎮庣划鎯у鐠侇厾绮?
    #               閸欘垰鍣虹亸鎴犲娑撯偓閸楀﹦娈戦弰鎯х摠閵嗕線娓剁憰涔竬torch1.7.1娴犮儰绗?
    #---------------------------------------------------------------------#
    fp16            = False
    #-----------------------------------------------------#
    #   num_classes     鐠侇厾绮岄懛顏勭箒閻ㄥ嫭鏆熼幑顕€娉﹁箛鍛淬€忕憰浣锋叏閺€鍦畱
    #                   閼奉亜绻侀棁鈧憰浣烘畱閸掑棛琚稉顏呮殶+1閿涘苯顩?+1
    #-----------------------------------------------------#
    num_classes     = 7#7
    #---------------------------------#
    #   閹碘偓娴ｈ法鏁ら惃鍕畱娑撹鍏辩純鎴犵捕閿?
    #   mobilenet
    #   xception
    #---------------------------------#
    backbone        = "mobilenet"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      閺勵垰鎯佹担璺ㄦ暏娑撹鍏辩純鎴犵捕閻ㄥ嫰顣╃拋顓犵矊閺夊啴鍣搁敍灞绢劃婢跺嫪濞囬悽銊ф畱閺勵垯瀵岄獮鑼畱閺夊啴鍣搁敍灞芥礈濮濄倖妲搁崷銊δ侀崹瀣€铏规畱閺冭泛鈧瑨绻樼悰灞藉鏉炵晫娈戦妴?
    #                   婵″倹鐏夌拋鍓х枂娴滃攲odel_path閿涘苯鍨稉璇插叡閻ㄥ嫭娼堥崐鍏兼￥闂団偓閸旂姾娴囬敍瀹瞨etrained閻ㄥ嫬鈧吋妫ら幇蹇庣疅閵?
    #                   婵″倹鐏夋稉宥堫啎缂冪斂odel_path閿涘retrained = True閿涘本顒濋弮鏈电矌閸旂姾娴囨稉璇插叡瀵偓婵顔勭紒鍐︹偓?
    #                   婵″倹鐏夋稉宥堫啎缂冪斂odel_path閿涘retrained = False閿涘瓗reeze_Train = Fasle閿涘本顒濋弮鏈电矤0瀵偓婵顔勭紒鍐跨礉娑撴梹鐥呴張澶婂枙缂佹挷瀵岄獮鑼畱鏉╁洨鈻奸妴?
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = True
    #----------------------------------------------------------------------------------------------------------------------------#
    #   閺夊啫鈧吋鏋冩禒鍓佹畱娑撳娴囩拠椋庢箙README閿涘苯褰叉禒銉┾偓姘崇箖缂冩垹娲忔稉瀣祰閵嗗倹膩閸ㄥ娈?妫板嫯顔勭紒鍐╂綀闁?鐎甸€涚瑝閸氬本鏆熼幑顕€娉﹂弰顖炩偓姘辨暏閻ㄥ嫸绱濋崶鐘辫礋閻楃懓绶涢弰顖炩偓姘辨暏閻ㄥ嫨鈧?
    #   濡€崇€烽惃?妫板嫯顔勭紒鍐╂綀闁?濮ｆ棁绶濋柌宥堫洣閻ㄥ嫰鍎撮崚鍡樻Ц 娑撹鍏遍悧鐟扮窙閹绘劕褰囩純鎴犵捕閻ㄥ嫭娼堥崐濂稿劥閸掑棴绱濋悽銊ょ艾鏉╂稖顢戦悧鐟扮窙閹绘劕褰囬妴?
    #   妫板嫯顔勭紒鍐╂綀闁插秴顕禍?9%閻ㄥ嫭鍎忛崘鐢稿厴韫囧懘銆忕憰浣烘暏閿涘奔绗夐悽銊ф畱鐠囨繀瀵岄獮鏌ュ劥閸掑棛娈戦弶鍐ㄢ偓鐓庛亰鏉╁洭娈㈤張鐚寸礉閻楃懓绶涢幓鎰絿閺佸牊鐏夋稉宥嗘閺勬拝绱濈純鎴犵捕鐠侇厾绮岄惃鍕波閺嬫粈绡冩稉宥勭窗婵?
    #   鐠侇厾绮岄懛顏勭箒閻ㄥ嫭鏆熼幑顕€娉﹂弮鑸靛絹缁€铏规樊鎼达缚绗夐崠褰掑帳濮濓絽鐖堕敍宀勵暕濞村娈戞稉婊嗐偪闁垝绗夋稉鈧弽铚傜啊閼奉亞鍔х紒鏉戝娑撳秴灏柊?
    #
    #   婵″倹鐏夌拋顓犵矊鏉╁洨鈻兼稉顓炵摠閸︺劋鑵戦弬顓☆唲缂佸啰娈戦幙宥勭稊閿涘苯褰叉禒銉ョ殺model_path鐠佸墽鐤嗛幋鎭杘gs閺傚洣娆㈡径閫涚瑓閻ㄥ嫭娼堥崐鍏兼瀮娴犺绱濈亸鍡楀嚒缂佸繗顔勭紒鍐х啊娑撯偓闁劌鍨庨惃鍕綀閸婄厧鍟€濞喡ゆ祰閸忋儯鈧?
    #   閸氬本妞傛穱顔芥暭娑撳鏌熼惃?閸愯崵绮ㄩ梼鑸殿唽 閹存牞鈧?鐟欙絽鍠曢梼鑸殿唽 閻ㄥ嫬寮弫甯礉閺夈儰绻氱拠浣鼓侀崹濯弍och閻ㄥ嫯绻涚紒顓熲偓褋鈧?
    #   
    #   瑜版悎odel_path = ''閻ㄥ嫭妞傞崐娆庣瑝閸旂姾娴囬弫缈犻嚋濡€崇€烽惃鍕綀閸婄鈧?
    #
    #   濮濄倕顦╂担璺ㄦ暏閻ㄥ嫭妲搁弫缈犻嚋濡€崇€烽惃鍕綀闁插稄绱濋崶鐘愁劃閺勵垰婀猼rain.py鏉╂稖顢戦崝鐘烘祰閻ㄥ嫸绱漰retrain娑撳秴濂栭崫宥嗩劃婢跺嫮娈戦弶鍐ㄢ偓鐓庡鏉炲鈧?
    #   婵″倹鐏夐幆瀹狀洣鐠佲晜膩閸ㄥ绮犳稉璇插叡閻ㄥ嫰顣╃拋顓犵矊閺夊啫鈧厧绱戞慨瀣唲缂佸喛绱濋崚娆掝啎缂冪斂odel_path = ''閿涘retrain = True閿涘本顒濋弮鏈电矌閸旂姾娴囨稉璇插叡閵?
    #   婵″倹鐏夐幆瀹狀洣鐠佲晜膩閸ㄥ绮?瀵偓婵顔勭紒鍐跨礉閸掓瑨顔曠純鐢縪del_path = ''閿涘retrain = Fasle閿涘瓗reeze_Train = Fasle閿涘本顒濋弮鏈电矤0瀵偓婵顔勭紒鍐跨礉娑撴梹鐥呴張澶婂枙缂佹挷瀵岄獮鑼畱鏉╁洨鈻奸妴?
    #   
    #   娑撯偓閼割剚娼电拋璇х礉缂冩垹绮舵禒?瀵偓婵娈戠拋顓犵矊閺佸牊鐏夋导姘发瀹割噯绱濋崶鐘辫礋閺夊啫鈧厧銇婃潻鍥閺堢尨绱濋悧鐟扮窙閹绘劕褰囬弫鍫熺亯娑撳秵妲戦弰鎾呯礉閸ョ姵顒濋棃鐐茬埗閵嗕線娼敮鎼炩偓渚€娼敮闀愮瑝瀵ら缚顔呮径褍顔嶆禒?瀵偓婵顔勭紒鍐跨磼
    #   婵″倹鐏夋稉鈧€规俺顩︽禒?瀵偓婵绱濋崣顖欎簰娴滃棜袙imagenet閺佺増宓侀梿鍡礉妫ｆ牕鍘涚拋顓犵矊閸掑棛琚Ο鈥崇€烽敍宀冨箯瀵版缍夌紒婊呮畱娑撹鍏遍柈銊ュ瀻閺夊啫鈧》绱濋崚鍡欒濡€崇€烽惃?娑撹鍏遍柈銊ュ瀻 閸滃矁顕氬Ο鈥崇€烽柅姘辨暏閿涘苯鐔€娴滃孩顒濇潻娑滎攽鐠侇厾绮岄妴?
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = "model_data/deeplab_mobilenetv2.pth"
    #model_path      =""
    #model_path      = "logs/ep170-loss0.126-val_loss0.360.pth"
    #---------------------------------------------------------#
    #   downsample_factor   娑撳鍣伴弽椋庢畱閸婂秵鏆?閵?6 
    #                       8娑撳鍣伴弽椋庢畱閸婂秵鏆熸潏鍐ㄧ毈閵嗕胶鎮婄拋杞扮瑐閺佸牊鐏夐弴鏉戙偨閵?
    #                       娴ｅ棔绡冪憰浣圭湴閺囨潙銇囬惃鍕▔鐎?
    #---------------------------------------------------------#
    downsample_factor   = 8
    #------------------------------#
    #   鏉堟挸鍙嗛崶鍓у閻ㄥ嫬銇囩亸?
    #------------------------------#
    input_shape         = [512, 512]#512,512
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   鐠侇厾绮岄崚鍡曡礋娑撱倓閲滈梼鑸殿唽閿涘苯鍨庨崚顐ｆЦ閸愯崵绮ㄩ梼鑸殿唽閸滃矁袙閸愬妯佸▓鐐光偓鍌濐啎缂冾喖鍠曠紒鎾绘▉濞堝灚妲告稉杞扮啊濠娐ゅ喕閺堝搫娅掗幀褑鍏樻稉宥堝喕閻ㄥ嫬鎮撶€涳妇娈戠拋顓犵矊闂団偓濮瑰倶鈧?
    #   閸愯崵绮ㄧ拋顓犵矊闂団偓鐟曚胶娈戦弰鎯х摠鏉堝啫鐨敍灞炬▔閸楋繝娼敮绋挎▕閻ㄥ嫭鍎忛崘鍏哥瑓閿涘苯褰茬拋鍓х枂Freeze_Epoch缁涘绨琔nFreeze_Epoch閿涘本顒濋弮鏈电矌娴犲懓绻樼悰灞藉枙缂佹捁顔勭紒鍐︹偓?
    #      
    #   閸︺劍顒濋幓鎰返閼汇儱鍏遍崣鍌涙殶鐠佸墽鐤嗗楦款唴閿涘苯鎮囨担宥堫唲缂佸啳鈧懏鐗撮幑顔垮殰瀹歌京娈戦棁鈧Ч鍌濈箻鐞涘瞼浼掑ú鏄忕殶閺佽揪绱?
    #   閿涘牅绔撮敍澶夌矤閺佺繝閲滃Ο鈥崇€烽惃鍕暕鐠侇厾绮岄弶鍐櫢瀵偓婵顔勭紒鍐跨窗 
    #       Adam閿?
    #           Init_Epoch = 0閿涘瓗reeze_Epoch = 50閿涘nFreeze_Epoch = 100閿涘瓗reeze_Train = True閿涘ptimizer_type = 'adam'閿涘瓥nit_lr = 5e-4閿涘瘍eight_decay = 0閵嗗偊绱欓崘鑽ょ波閿?
    #           Init_Epoch = 0閿涘nFreeze_Epoch = 100閿涘瓗reeze_Train = False閿涘ptimizer_type = 'adam'閿涘瓥nit_lr = 5e-4閿涘瘍eight_decay = 0閵嗗偊绱欐稉宥呭枙缂佹搫绱?
    #       SGD閿?
    #           Init_Epoch = 0閿涘瓗reeze_Epoch = 50閿涘nFreeze_Epoch = 100閿涘瓗reeze_Train = True閿涘ptimizer_type = 'sgd'閿涘瓥nit_lr = 7e-3閿涘瘍eight_decay = 1e-4閵嗗偊绱欓崘鑽ょ波閿?
    #           Init_Epoch = 0閿涘nFreeze_Epoch = 100閿涘瓗reeze_Train = False閿涘ptimizer_type = 'sgd'閿涘瓥nit_lr = 7e-3閿涘瘍eight_decay = 1e-4閵嗗偊绱欐稉宥呭枙缂佹搫绱?
    #       閸忔湹鑵戦敍姝巒Freeze_Epoch閸欘垯浜掗崷?00-300娑斿妫跨拫鍐╂殻閵?
    #   閿涘牅绨╅敍澶夌矤娑撹鍏辩純鎴犵捕閻ㄥ嫰顣╃拋顓犵矊閺夊啴鍣稿鈧慨瀣唲缂佸喛绱?
    #       Adam閿?
    #           Init_Epoch = 0閿涘瓗reeze_Epoch = 50閿涘nFreeze_Epoch = 100閿涘瓗reeze_Train = True閿涘ptimizer_type = 'adam'閿涘瓥nit_lr = 5e-4閿涘瘍eight_decay = 0閵嗗偊绱欓崘鑽ょ波閿?
    #           Init_Epoch = 0閿涘nFreeze_Epoch = 100閿涘瓗reeze_Train = False閿涘ptimizer_type = 'adam'閿涘瓥nit_lr = 5e-4閿涘瘍eight_decay = 0閵嗗偊绱欐稉宥呭枙缂佹搫绱?
    #       SGD閿?
    #           Init_Epoch = 0閿涘瓗reeze_Epoch = 50閿涘nFreeze_Epoch = 120閿涘瓗reeze_Train = True閿涘ptimizer_type = 'sgd'閿涘瓥nit_lr = 7e-3閿涘瘍eight_decay = 1e-4閵嗗偊绱欓崘鑽ょ波閿?
    #           Init_Epoch = 0閿涘nFreeze_Epoch = 120閿涘瓗reeze_Train = False閿涘ptimizer_type = 'sgd'閿涘瓥nit_lr = 7e-3閿涘瘍eight_decay = 1e-4閵嗗偊绱欐稉宥呭枙缂佹搫绱?
    #       閸忔湹鑵戦敍姘辨暠娴滃簼绮犳稉璇插叡缂冩垹绮堕惃鍕暕鐠侇厾绮岄弶鍐櫢瀵偓婵顔勭紒鍐跨礉娑撹鍏遍惃鍕綀閸婇棿绗夋稉鈧€规岸鈧倸鎮庣拠顓濈疅閸掑棗澹婇敍宀勬付鐟曚焦娲挎径姘辨畱鐠侇厾绮岀捄鍐插毉鐏炩偓闁劍娓舵导妯啃掗妴?
    #             UnFreeze_Epoch閸欘垯浜掗崷?20-300娑斿妫跨拫鍐╂殻閵?
    #             Adam閻╂瓕绶濇禍宥碐D閺€鑸垫殐閻ㄥ嫬鎻╂稉鈧禍娑栤偓鍌氭礈濮濐椈nFreeze_Epoch閻炲棜顔戞稉濠傚讲娴犮儱鐨稉鈧悙鐧哥礉娴ｅ棔绶烽悞鑸靛腹閼芥劖娲挎径姘辨畱Epoch閵?
    #   閿涘牅绗侀敍濉╝tch_size閻ㄥ嫯顔曠純顕嗙窗
    #       閸︺劍妯夐崡陇鍏樻径鐔稿复閸欐娈戦懠鍐ㄦ纯閸愬拑绱濇禒銉ャ亣娑撳搫銈介妴鍌涙▔鐎涙ü绗夌搾鍏呯瑢閺佺増宓侀梿鍡椼亣鐏忓繑妫ら崗绛圭礉閹绘劗銇氶弰鎯х摠娑撳秷鍐婚敍鍦M閹存牞鈧寯UDA out of memory閿涘顕拫鍐ㄧ毈batch_size閵?
    #       閸欐鍩孊atchNorm鐏炲倸濂栭崫宥忕礉batch_size閺堚偓鐏忓繋璐?閿涘奔绗夐懗鎴掕礋1閵?
    #       濮濓絽鐖堕幆鍛枌娑撳┅reeze_batch_size瀵ら缚顔呮稉绡freeze_batch_size閻?-2閸婂秲鈧倷绗夊楦款唴鐠佸墽鐤嗛惃鍕▕鐠烘繆绻冩径褝绱濋崶鐘辫礋閸忓磭閮撮崚鏉款劅娑旂姷宸奸惃鍕殰閸斻劏鐨熼弫娣偓?
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   閸愯崵绮ㄩ梼鑸殿唽鐠侇厾绮岄崣鍌涙殶
    #   濮濄倖妞傚Ο鈥崇€烽惃鍕瘜楠炶尪顫﹂崘鑽ょ波娴滃棴绱濋悧鐟扮窙閹绘劕褰囩純鎴犵捕娑撳秴褰傞悽鐔告暭閸?
    #   閸楃姷鏁ら惃鍕▔鐎涙绶濈亸蹇ョ礉娴犲懎顕純鎴犵捕鏉╂稖顢戝顔跨殶
    #   Init_Epoch          濡€崇€疯ぐ鎾冲瀵偓婵娈戠拋顓犵矊娑撴牔鍞敍灞藉従閸婄厧褰叉禒銉ャ亣娴滃锭reeze_Epoch閿涘苯顩х拋鍓х枂閿?
    #                       Init_Epoch = 60閵嗕笚reeze_Epoch = 50閵嗕箒nFreeze_Epoch = 100
    #                       娴兼俺鐑︽潻鍥у枙缂佹捇妯佸▓纰夌礉閻╁瓨甯存禒?0娴狅絽绱戞慨瀣剁礉楠炴儼鐨熼弫鏉戭嚠鎼存梻娈戠€涳缚绡勯悳鍥モ偓?
    #                       閿涘牊鏌囬悙鍦敾缂佸啯妞傛担璺ㄦ暏閿?
    #   Freeze_Epoch        濡€崇€烽崘鑽ょ波鐠侇厾绮岄惃鍑eeze_Epoch
    #                       (瑜版強reeze_Train=False閺冭泛銇戦弫?
    #   Freeze_batch_size   濡€崇€烽崘鑽ょ波鐠侇厾绮岄惃鍒tch_size
    #                       (瑜版強reeze_Train=False閺冭泛銇戦弫?
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 2
    #------------------------------------------------------------------#
    #   鐟欙絽鍠曢梼鑸殿唽鐠侇厾绮岄崣鍌涙殶
    #   濮濄倖妞傚Ο鈥崇€烽惃鍕瘜楠炶弓绗夌悮顐㈠枙缂佹挷绨￠敍宀€澹掑浣瑰絹閸欐牜缍夌紒婊€绱伴崣鎴犳晸閺€鐟板綁
    #   閸楃姷鏁ら惃鍕▔鐎涙绶濇径褝绱濈純鎴犵捕閹碘偓閺堝娈戦崣鍌涙殶闁垝绱伴崣鎴犳晸閺€鐟板綁
    #   UnFreeze_Epoch          濡€崇€烽幀璇插彙鐠侇厾绮岄惃鍒och
    #   Unfreeze_batch_size     濡€崇€烽崷銊ㄐ掗崘璇叉倵閻ㄥ垺atch_size閿涙稑甯弶?
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 200
    Unfreeze_batch_size = 4
    #------------------------------------------------------------------#
    #   Freeze_Train    閺勵垰鎯佹潻娑滎攽閸愯崵绮ㄧ拋顓犵矊
    #                   姒涙顓婚崗鍫濆枙缂佹挷瀵岄獮鑼额唲缂佸啫鎮楃憴锝呭枙鐠侇厾绮岄妴?
    #------------------------------------------------------------------#
    Freeze_Train        = False

    #------------------------------------------------------------------#
    #   閸忚泛鐣犵拋顓犵矊閸欏倹鏆熼敍姘劅娑旂姷宸奸妴浣风喘閸栨牕娅掗妴浣割劅娑旂姷宸兼稉瀣閺堝鍙?
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         濡€崇€烽惃鍕付婢堆冾劅娑旂姷宸?
    #                   瑜版挷濞囬悽藡dam娴兼ê瀵查崳銊︽瀵ら缚顔呯拋鍓х枂  Init_lr=5e-4
    #                   瑜版挷濞囬悽鈯縂D娴兼ê瀵查崳銊︽瀵ら缚顔呯拋鍓х枂   Init_lr=7e-3
    #   Min_lr          濡€崇€烽惃鍕付鐏忓繐顒熸稊鐘靛芳閿涘矂绮拋銈勮礋閺堚偓婢堆冾劅娑旂姷宸奸惃?.01
    #------------------------------------------------------------------#
    Init_lr             = 2e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  娴ｈ法鏁ら崚鎵畱娴兼ê瀵查崳銊ь潚缁紮绱濋崣顖炩偓澶屾畱閺堝¨dam閵嗕够gd
    #                   瑜版挷濞囬悽藡dam娴兼ê瀵查崳銊︽瀵ら缚顔呯拋鍓х枂  Init_lr=5e-4
    #                   瑜版挷濞囬悽鈯縂D娴兼ê瀵查崳銊︽瀵ら缚顔呯拋鍓х枂   Init_lr=7e-3
    #   momentum        娴兼ê瀵查崳銊ュ敶闁劋濞囬悽銊ュ煂閻ㄥ埓omentum閸欏倹鏆?
    #   weight_decay    閺夊啫鈧壈鈥滈崙蹇ョ礉閸欘垶妲诲銏ｇ箖閹风喎鎮?
    #                   adam娴兼艾顕遍懛纾渆ight_decay闁挎瑨顕ら敍灞煎▏閻⑩暆dam閺冭泛缂撶拋顔款啎缂冾喕璐?閵?
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0.0001
    #------------------------------------------------------------------#
    #   lr_decay_type   娴ｈ法鏁ら崚鎵畱鐎涳缚绡勯悳鍥︾瑓闂勫秵鏌熷蹇ョ礉閸欘垶鈧娈戦張?step'閵?          'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     婢舵艾鐨稉鐚爌och娣囨繂鐡ㄦ稉鈧▎鈩冩綀閸?
    #------------------------------------------------------------------#
    save_period         = 5
    #------------------------------------------------------------------#
    #   save_dir        閺夊啫鈧棿绗岄弮銉ョ箶閺傚洣娆㈡穱婵嗙摠閻ㄥ嫭鏋冩禒璺恒仚
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       閺勵垰鎯侀崷銊唲缂佸啯妞傛潻娑滎攽鐠囧嫪鍙婇敍宀冪槑娴兼澘顕挒鈥茶礋妤犲矁鐦夐梿?
    #   eval_period     娴狅綀銆冩径姘毌娑撶尃poch鐠囧嫪鍙婃稉鈧▎鈽呯礉娑撳秴缂撶拋顕€顣剁换浣烘畱鐠囧嫪鍙?
    #                   鐠囧嫪鍙婇棁鈧憰浣圭Х閼版绶濇径姘辨畱閺冨爼妫块敍宀勵暥缁讳浇鐦庢导棰佺窗鐎佃壈鍤х拋顓犵矊闂堢偛鐖堕幈?
    #   濮濄倕顦╅懢宄扮繁閻ㄥ埓AP娴兼矮绗実et_map.py閼惧嘲绶遍惃鍕窗閺堝澧嶆稉宥呮倱閿涘苯甯崶鐘虫箒娴滃矉绱?
    #   閿涘牅绔撮敍澶嬵劃婢跺嫯骞忓妤冩畱mAP娑撴椽鐛欑拠渚€娉﹂惃鍒碅P閵?
    #   閿涘牅绨╅敍澶嬵劃婢跺嫯顔曠純顔跨槑娴兼澘寮弫鎷岀窛娑撹桨绻氱€瑰牞绱濋惄顔炬畱閺勵垰濮炶箛顐ョ槑娴间即鈧喎瀹抽妴?
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5

    #------------------------------------------------------------------#
    #   VOCdevkit_path  閺佺増宓侀梿鍡氱熅瀵?
    #------------------------------------------------------------------#
    VOCdevkit_path  ='facadewhu_extend'
    #------------------------------------------------------------------#
    #   瀵ら缚顔呴柅澶愩€嶉敍?
    #   缁夊秶琚亸鎴礄閸戠姷琚敍澶嬫閿涘矁顔曠純顔昏礋True+
    #   缁夊秶琚径姘剧礄閸椾礁鍤戠猾浼欑礆閺冭绱濇俊鍌涚亯batch_size濮ｆ棁绶濇径褝绱?0娴犮儰绗傞敍澶涚礉闁絼绠炵拋鍓х枂娑撶rue
    #   缁夊秶琚径姘剧礄閸椾礁鍤戠猾浼欑礆閺冭绱濇俊鍌涚亯batch_size濮ｆ棁绶濈亸蹇ョ礄10娴犮儰绗呴敍澶涚礉闁絼绠炵拋鍓х枂娑撶瘞alse
    #------------------------------------------------------------------#
    dice_loss       = False
    #------------------------------------------------------------------#
    #   閺勵垰鎯佹担璺ㄦ暏focal loss閺夈儵妲诲銏☆劀鐠愮喐鐗遍張顑跨瑝楠炲疇銆€
    #------------------------------------------------------------------#
    focal_loss      = False

    inverfreq_loss  = False
    #------------------------------------------------------------------#
    #   閺勵垰鎯佺紒娆庣瑝閸氬瞼顫掔猾鏄忕ゴ娴滃牅绗夐崥宀€娈戦幑鐔枫亼閺夊啫鈧》绱濇妯款吇閺勵垰閽╃悰锛勬畱閵?
    #   鐠佸墽鐤嗛惃鍕樈閿涘本鏁為幇蹇氼啎缂冾喗鍨歯umpy瑜般垹绱￠惃鍕剁礉闂€鍨閸滃um_classes娑撯偓閺嶆灚鈧?
    #   婵″偊绱?
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    class_config = load_class_config(cli_args.class_config or os.environ.get("VCFS_CLASS_CONFIG"))
    num_classes_value = cli_args.num_classes if cli_args.num_classes is not None else os.environ.get("VCFS_NUM_CLASSES")
    num_classes = int(num_classes_value or class_config["num_classes"])
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    #   num_workers     閻劋绨拋鍓х枂閺勵垰鎯佹担璺ㄦ暏婢舵氨鍤庣粙瀣嚢閸欐牗鏆熼幑顕嗙礉1娴狅綀銆冮崗鎶芥４婢舵氨鍤庣粙?
    #                   瀵偓閸氼垰鎮楁导姘韫囶偅鏆熼幑顔款嚢閸欐牠鈧喎瀹抽敍灞肩稻閺勵垯绱伴崡鐘垫暏閺囨潙顦块崘鍛摠
    #                   keras闁插苯绱戦崥顖氼樋缁捐法鈻奸張澶夌昂閺冭泛鈧瑩鈧喎瀹抽崣宥堚偓灞惧弮娴滃棜顔忔径?
    #                   閸︹問O娑撹櫣鎽辨０鍫㈡畱閺冭泛鈧瑥鍟€瀵偓閸氼垰顦跨痪璺ㄢ柤閿涘苯宓咷PU鏉╂劗鐣婚柅鐔峰鏉╂粌銇囨禍搴ゎ嚢閸欐牕娴橀悧鍥╂畱闁喎瀹抽妴?
    #------------------------------------------------------------------#
    num_workers         = 4


    def _cli_env_int(cli_value, name, current):
        if cli_value is not None:
            return int(cli_value)
        value = os.environ.get(name)
        return current if value in (None, "") else int(value)

    def _cli_env_bool(cli_value, name, current):
        if cli_value is not None:
            return cli_value.lower() in ("1", "true", "yes", "y", "on")
        value = os.environ.get(name)
        if value in (None, ""):
            return current
        return value.lower() in ("1", "true", "yes", "y", "on")

    def _cli_env_shape(cli_value, name, current):
        value = cli_value if cli_value is not None else os.environ.get(name)
        if value in (None, ""):
            return current
        parts = [int(part.strip()) for part in value.replace("x", ",").split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError("{} must be like 512,512 or 512x512".format(name))
        return parts

    model_path = cli_args.model_path if cli_args.model_path is not None else os.environ.get("VCFS_MODEL_PATH", model_path)
    input_shape = _cli_env_shape(cli_args.input_shape, "VCFS_INPUT_SHAPE", input_shape)
    UnFreeze_Epoch = _cli_env_int(cli_args.epochs, "VCFS_EPOCHS", UnFreeze_Epoch)
    Freeze_batch_size = _cli_env_int(cli_args.freeze_batch_size, "VCFS_FREEZE_BATCH_SIZE", Freeze_batch_size)
    Unfreeze_batch_size = _cli_env_int(cli_args.batch_size, "VCFS_BATCH_SIZE", Unfreeze_batch_size)
    save_dir = cli_args.save_dir if cli_args.save_dir is not None else os.environ.get("VCFS_SAVE_DIR", save_dir)
    eval_flag = _cli_env_bool(cli_args.eval_flag, "VCFS_EVAL_FLAG", eval_flag)
    VOCdevkit_path = cli_args.dataset_path if cli_args.dataset_path is not None else os.environ.get("VCFS_DATASET_PATH", VOCdevkit_path)
    num_workers = _cli_env_int(cli_args.num_workers, "VCFS_NUM_WORKERS", num_workers)
    print("Using class config: {}".format(class_config["path"]))

    seed_everything(seed)
    #------------------------------------------------------#
    #   鐠佸墽鐤嗛悽銊ュ煂閻ㄥ嫭妯夐崡?
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #----------------------------------------------------#
    #   娑撳娴囨０鍕唲缂佸啯娼堥柌?
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   閺夊啫鈧吋鏋冩禒鎯邦嚞閻┗EADME閿涘瞼娅ㄦ惔锔剧秹閻╂ü绗呮潪?
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   閺嶈宓佹０鍕唲缂佸啯娼堥柌宥囨畱Key閸滃本膩閸ㄥ娈慘ey鏉╂稖顢戦崝鐘烘祰
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   閺勫墽銇氬▽鈩冩箒閸栧綊鍘ゆ稉濠勬畱Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "閳ワ腹鈧泜nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "閳ワ腹鈧泜nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m濞撯晠螛閹绘劗銇氶敍瀹ad闁劌鍨庡▽鈩冩箒鏉炶棄鍙嗛弰顖涱劀鐢摜骞囩挒鈽呯礉Backbone闁劌鍨庡▽鈩冩箒鏉炶棄鍙嗛弰顖炴晩鐠囶垳娈戦妴淇?33[0m")

    #----------------------#
    #   鐠佹澘缍峀oss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    #------------------------------------------------------------------#
    #   torch 1.2娑撳秵鏁幐涔p閿涘苯缂撶拋顔诲▏閻⑩暟orch 1.7.1閸欏﹣浜掓稉濠冾劀绾喕濞囬悽鈺16
    #   閸ョ姵顒漷orch1.2鏉╂瑩鍣烽弰鍓с仛"could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   婢舵艾宕遍崥灞绢劄Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   婢舵艾宕遍獮瀹狀攽鏉╂劘顢?
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   鐠囪褰囬弫鐗堝祦闂嗗棗顕惔鏃傛畱txt
    #---------------------------#
    train_lines, train_split_path = load_split_lines(
        VOCdevkit_path,
        "train_1601.txt",
        fallback_name="train.txt",
        auto_create_train=True,
    )
    val_lines, val_split_path = load_split_lines(
        VOCdevkit_path,
        "val.txt",
        fallback_name="test.txt",
    )
    if local_rank == 0:
        print("Train split: {}".format(train_split_path))
        print("Val split: {}".format(val_split_path))

    train_limit = _cli_env_int(None, "VCFS_TRAIN_LIMIT", 0)
    val_limit = _cli_env_int(None, "VCFS_VAL_LIMIT", 0)
    if train_limit > 0:
        train_lines = train_lines[:train_limit]
    if val_limit > 0:
        val_lines = val_lines[:val_limit]

    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, class_config = class_config["path"], backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        #---------------------------------------------------------#
        #   閹槒顔勭紒鍐х瑯娴狅絾瀵氶惃鍕Ц闁秴宸婚崗銊╁劥閺佺増宓侀惃鍕偓缁橆偧閺?
        #   閹槒顔勭紒鍐╊劄闂€鎸庡瘹閻ㄥ嫭妲稿顖氬娑撳妾烽惃鍕偓缁橆偧閺?
        #   濮ｅ繋閲滅拋顓犵矊娑撴牔鍞崠鍛儓閼汇儱鍏辩拋顓犵矊濮濄儵鏆遍敍灞剧槨娑擃亣顔勭紒鍐╊劄闂€鑳箻鐞涘奔绔村▎鈩冾潽鎼达缚绗呴梽宥冣偓?
        #   濮濄倕顦╂禒鍛紦鐠侇喗娓舵担搴ゎ唲缂佸啩绗樻禒锝忕礉娑撳﹣绗夌亸渚€銆婇敍宀冾吀缁犳妞傞崣顏団偓鍐娴滃棜袙閸愬鍎撮崚?
        #----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError("Dataset is too small for training. Please add more data.")
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print(
                "\n[Warning] With optimizer {}, recommended total steps are at least {}.".format(
                    optimizer_type, int(wanted_step)
                )
            )
            print(
                "[Warning] Current run: num_train={}, batch_size={}, epochs={}, total_steps={}.".format(
                    num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step
                )
            )
            print("[Warning] Consider increasing epochs to about {} for a full experiment.".format(int(wanted_epoch)))
        
    #------------------------------------------------------#
    #   娑撹鍏遍悧鐟扮窙閹绘劕褰囩純鎴犵捕閻楃懓绶涢柅姘辨暏閿涘苯鍠曠紒鎾诡唲缂佸啫褰叉禒銉ュ韫囶偉顔勭紒鍐偓鐔峰
    #   娑旂喎褰叉禒銉ユ躬鐠侇厾绮岄崚婵囨埂闂冨弶顒涢弶鍐ㄢ偓鑹邦潶閻潙娼栭妴?
    #   Init_Epoch娑撻缚鎹ｆ慨瀣╃瑯娴?
    #   Interval_Epoch娑撳搫鍠曠紒鎾诡唲缂佸啰娈戞稉鏍﹀敩
    #   Epoch閹槒顔勭紒鍐х瑯娴?
    #   閹绘劗銇歄OM閹存牞鈧懏妯夌€涙ü绗夌搾瀹狀嚞鐠嬪啫鐨珺atch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   閸愯崵绮ㄦ稉鈧€规岸鍎撮崚鍡氼唲缂?
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #   婵″倹鐏夋稉宥呭枙缂佹捁顔勭紒鍐畱鐠囨繐绱濋惄瀛樺复鐠佸墽鐤哹atch_size娑撶nfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   閸掋倖鏌囪ぐ鎾冲batch_size閿涘矁鍤滈柅鍌氱安鐠嬪啯鏆ｇ€涳缚绡勯悳?
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   閺嶈宓乷ptimizer_type闁瀚ㄦ导妯哄閸?
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   閼惧嘲绶辩€涳缚绡勯悳鍥︾瑓闂勫秶娈戦崗顒€绱?
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   閸掋倖鏌囧В蹇庣娑擃亙绗樻禒锝囨畱闂€鍨
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small for training. Please add more data.")

        train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))#閸愯崵绮ㄧ拋顓犵矊
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #----------------------#
        #   鐠佹澘缍峞val閻ㄥ埓ap閺囪尙鍤?
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   瀵偓婵膩閸ㄥ顔勭紒?
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   婵″倹鐏夊Ο鈥崇€烽張澶婂枙缂佹挸顒熸稊鐘诲劥閸?
            #   閸掓瑨袙閸愪紮绱濋獮鎯邦啎缂冾喖寮弫?
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:#鐟欙絽鍠?
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   閸掋倖鏌囪ぐ鎾冲batch_size閿涘矁鍤滈柅鍌氱安鐠嬪啯鏆ｇ€涳缚绡勯悳?
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   閼惧嘲绶辩€涳缚绡勯悳鍥︾瑓闂勫秶娈戦崗顒€绱?
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset is too small for training. Please add more data.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss,inverfreq_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
