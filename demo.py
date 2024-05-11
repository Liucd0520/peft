from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from supervised_dataset import LazySupervisedDataset, SupervisedDataset
import json
import os 
from model_save import safe_save_model_for_hf_trainer

# 2卡机器CUDA版本过低或其他问题导致无法分布式训练
# torchrun     --nproc_per_node 2    --nnodes 1    --node_rank 0     --master_addr localhost    --master_port 1123   demo.py

model_name_or_path = '/data/liucd/BigModel/qwen/qwen/Qwen-1_8B-Chat'
device_map = None #  单机多卡（非deepspeed） 用auto，可以将显存分布到多张卡上
                    # 单机单卡

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    ) 
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

@dataclass
class ModelArguments:
    model_name_or_path: str = model_name_or_path

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_lora: bool = True 
    bf16: bool = False
    output_dir: str = 'qwen_output'
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )  # 微调时最大序列长度
    gradient_checkpointing: bool = True 
    report_to: str = 'none'
    # deepspeed: str = '/data/liucd/BigModel/qwen/Qwen/finetune/ds_config_zero2.json'

@dataclass
class DataArguments:
    data_path: str = field(default='/data/liucd/BigModel/qwen/Qwen/finetune/data.json', metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False



args_lora = LoraArguments()
args_model = ModelArguments()
args_train = TrainingArguments()
args_data = DataArguments()

print(args_train)

# tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    args_model.model_name_or_path,
    model_max_length=args_train.model_max_length,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True,
)
tokenizer.pad_token_id = tokenizer.eod_id #  部分tokenizer没有pad_token，例如qwen，将pad_token置为eos_token

# model

model = transformers.AutoModelForCausalLM.from_pretrained(
        args_model.model_name_or_path, 
        device_map=device_map,    # 分布式需要注释掉或者设置为None
        quantization_config=GPTQConfig(bits=4, disable_exllama=True) \
            if args_train.use_lora  and args_lora.q_lora 
            else None,  # 只有使用了qlora 才会载入量化模型, lora是不能用量化模型的，因为lora分支是用的FP16
        trust_remote_code=True
)
lora_config = LoraConfig(
            r=args_lora.lora_r,
            lora_alpha=args_lora.lora_alpha,
            target_modules=args_lora.lora_target_modules,
            lora_dropout=args_lora.lora_dropout,
            bias=args_lora.lora_bias,
            task_type="CAUSAL_LM",

        )

if args_lora.q_lora:
     model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args_train.gradient_checkpointing
            )  # 将某些的LN层等从FP16变成FP32

model = get_peft_model(model, peft_config=lora_config) 
model.print_trainable_parameters()

# 调用 model.enable_input_require_grads() 是为了确保在使用 grad_checkpoint 时，模型的输入能够被要求梯度，以便在检查点处能够正确地重新计算梯度。
if args_train.gradient_checkpointing:
    model.enable_input_require_grads()


#  数据集制作
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    
    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

# Load Data
data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=args_data, max_len=args_train.model_max_length
    )

# Start trainner

trainer = Trainer(
    model=model, tokenizer=tokenizer, args=args_train, **data_module
)

trainer.train()
trainer.save_state()  # 保存状态

safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args_train.output_dir, bias=args_lora.lora_bias)


