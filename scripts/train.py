from transformers import Seq2SeqTrainingArguments
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig, WhisperConfig
import torch
from dagshub import get_repo_bucket_client
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers import EarlyStoppingCallback
import pandas as pd
from transformers.integrations import WandbCallback
import evaluate
from typing import Any, Dict, List, Union
from dataclasses import dataclass
from transformers import WhisperProcessor
from transformers import WhisperTokenizerFast
from transformers import WhisperFeatureExtractor
from datasets import Audio
import regex as re
from datasets import Dataset, Audio
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import wandb
import json

login()
wandb.login()

CONFIG_PATH = "configs/large.json"

LANGUAGE = "Russian"
TASK = "transcribe"

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

run_name = cfg["run_name"]
base_model = cfg["base_model"]


ds = load_dataset("csv", data_files="data/whole_dataset.csv")

ds = DatasetDict({'train': ds.filter(lambda x: not (x["val"] or x["test"]))["train"],
                  'val': ds.filter(lambda x: x["val"])["train"],
                  'test': ds.filter(lambda x: x["test"])["train"]})

prefixes = ["unchecked_packed_dataset/", "checked_packed_dataset/"]


def add_prefix(example):
    example["text"] = "<|notimestamps|>" + example["text"]
    example["audio"] = "data/" + \
        prefixes[example["verified"]] + example["audio"]
    return example


ds = ds.rename_column('binned_audio', "audio").map(add_prefix)


pattern = r'(?: ?/+)|(?: ?\*\p{L})+|(?: ?\(.*?\))|(?: ?[@\|#\?\!])|(?:[^\-_\p{L}\p{N}\s])'
non_text = re.compile(pattern)
translator = str.maketrans({"ё": "е", "_": " "})


def clean_ORD(text):
    text = non_text.sub('', text)
    text = text.lower()
    text = text.translate(translator)
    return text


ds = ds.remove_columns(["Unnamed: 0", "segments", "duration", "n",
                       "speakers", "noise", "informant", 'verified', 'test', 'val'])


ds = ds.cast_column("audio", Audio(sampling_rate=16000))


feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)


tokenizer = WhisperTokenizerFast.from_pretrained(base_model,
                                                 language=LANGUAGE, task=TASK, predict_timestamps=True)
tokenizer.set_prefix_tokens(predict_timestamps=True)


processor = WhisperProcessor.from_pretrained(
    base_model, language=LANGUAGE, task=TASK)


def prepare_dataset(batch):
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels_timed"] = tokenizer(batch["text_timed"]).input_ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def prepare_train_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    return batch


def prepare_val_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["text_timed"]).input_ids
    return batch


def tokenize(batch):
    # randomize timed text
    n = len(batch["text"])
    texts = zip(batch["text"], batch["text_timed"])
    inds = torch.rand(n) < cfg["timed_p"]
    texts = [tup[ind] for ind, tup in zip(inds, texts)]
    batch["labels"] = tokenizer(texts).input_ids
    return batch


num_proc = 2
if cfg["BPEDrop"]:
    ds["val"] = ds["val"].map(
        prepare_val_dataset, remove_columns=ds.column_names["train"], num_proc=num_proc)
    ds["train"] = ds["train"].map(prepare_train_dataset, remove_columns=[
                                  "audio"], num_proc=num_proc)
    tokenizer._tokenizer.model.dropout = cfg["BPEDrop"]
    ds["train"].set_transform(tokenize)
    print(len(ds["train"][0]["labels"]), len(ds["train"][0]["labels"]))
else:
    ds["val"] = ds["val"].map(
        prepare_val_dataset, remove_columns=ds.column_names["train"], num_proc=num_proc)
    ds["train"] = ds["train"].map(
        prepare_dataset, remove_columns=ds.column_names["train"], num_proc=num_proc)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        # get the tokenized label sequences
        if "labels_timed" in features[0]:
            inds = torch.rand(len(features)) < cfg["timed_p"]
            keys = ("labels", "labels_timed")
            label_features = [{"input_ids": feature[keys[ind]]}
                              for ind, feature in zip(inds, features)]
        else:
            label_features = [{"input_ids": feature["labels"]}
                              for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


wer = evaluate.load("wer")
cer = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str_words = [clean_ORD(s) for s in pred_str]
    label_str_words = [clean_ORD(s) for s in label_str]

    # uncomment if the dataset has not been cleaned up of empty strs
#     pred_str_words = [pred_str_words[i] for i in range(len(pred_str_words)) if len(label_str_words[i]) > 0]
#     label_str_words = [label_str_words[i] for i in range(len(label_str_words)) if len(label_str_words[i]) > 0]

    metrics = dict()

    metrics["wer"] = 100 * \
        wer.compute(predictions=pred_str, references=label_str)
    metrics["cer"] = 100 * \
        cer.compute(predictions=pred_str, references=label_str)
    metrics["clean_wer"] = 100 * \
        wer.compute(predictions=pred_str_words, references=label_str_words)
    metrics["clean_cer"] = 100 * \
        cer.compute(predictions=pred_str_words, references=label_str_words)

    return metrics


if cfg["peft"]:
    quantization_config = BitsAndBytesConfig(load_in_8bit=cfg["load_in_8bit"],
                                             load_in_4bit=cfg["load_in_4bit"], bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"])
else:
    quantization_config = None

model = WhisperForConditionalGeneration.from_pretrained(base_model,
                                                        quantization_config=quantization_config,
                                                        ignore_mismatched_sizes=True,
                                                        # attn_implementation = "flash_attention_2"
                                                        )

model.config.encoder_layerdrop, model.config.decoder_layerdrop = (
    cfg["encoder_layerdrop"], cfg["decoder_layerdrop"])
model.config.apply_spec_augment = cfg["apply_spec_augment"]
model.config.mask_time_min_masks = cfg["mask_time_min_masks"]

model.config.mask_time_prob = cfg["mask_time_prob"]
model.config.mask_feature_prob = cfg["mask_feature_prob"]

model.config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(
    language=LANGUAGE, task=TASK, no_timestamps=False
)
model.config.suppress_tokens = []

# force to output the language
model.generation_config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(
    language=LANGUAGE, task=TASK, no_timestamps=False
)
model.generation_config.suppress_tokens = []
model.generation_config.return_timestamps = True

if cfg["peft"]:
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)

    from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

    config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], target_modules=cfg["target_modules"],
                        lora_dropout=cfg["lora_dropout"], bias=cfg["bias"], use_dora=cfg["use_dora"])

    model = get_peft_model(model, config)
    model.print_trainable_parameters()


def decode_predictions(predictions):
    pred_ids = predictions.predictions
    label_ids = predictions.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    pred_str_special = tokenizer.batch_decode(
        pred_ids, skip_special_tokens=False, decode_with_timestamps=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

#     logits = predictions.predictions.argmax(axis=-1)
    return {"labels": label_str, "predictions": pred_str, "predictions_special": pred_str_special}


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset
          for generating predictions.
        num_samples (int, optional): Number of samples to select from
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset,
                 num_samples=8, freq=1):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # generate predictions
        predictions = self.trainer.predict(
            self.sample_dataset, metric_key_prefix="eval_sample/")
        # decode predictions and labels
        predictions = decode_predictions(predictions)
        # add predictions to a wandb.Table
        predictions_df = pd.DataFrame(predictions)
        predictions_df["epoch"] = state.epoch
        records_table = self._wandb.Table(dataframe=predictions_df)
        # log the table to wandb
        self._wandb.log({"sample_predictions": records_table})


if cfg["peft"]:
    lr = 1e-3
    warmup_steps = 50
else:
    if "lr" in cfg:
        lr = cfg["lr"]
    else:
        lr = 1e-5
        print("Set lr automatically")
    warmup_steps = 500

if "eval_bs_divisor" in cfg:
    eval_bs_divisor = cfg["eval_bs_divisor"]
else:
    eval_bs_divisor = 2

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./{run_name}",
    per_device_train_batch_size=bs,  # // torch.cuda.device_count(),
    gradient_accumulation_steps=1,
    learning_rate=lr,
    weight_decay=0.1,
    warmup_steps=warmup_steps,
    num_train_epochs=cfg["num_train_epochs"],
    gradient_checkpointing=True,  # no for peft???
    fp16=True,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=bs // eval_bs_divisor,
    predict_with_generate=True,
    generation_max_length=225,
    save_strategy="epoch",
    save_total_limit=2,
    save_only_model=False,  # TODO: ? change ^ in final train
    # eval_steps=steps_in_train,
    logging_steps=25,
    report_to=[],  # otherwise doubled logs
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    remove_unused_columns=False,
    label_names=["labels"],
    #     deepspeed="ds_config.json",
    #     split_batches = False,
    #     accelerator_config  = AcceleratorConfig(split_batches=False),
)

run = wandb.init(
    project="ORD",
    name=run_name,
    config=cfg | training_args.to_dict()
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=processor.feature_extractor,
    val_dataset=ds["val"],
    num_samples=10,
    freq=1,
)

# Add the callback to the trainer
trainer.add_callback(progress_callback)

processor.save_pretrained(training_args.output_dir)

if cfg["peft"]:
    with torch.autocast("cuda"):
        trainer.train()
else:
    trainer.train()

kwargs = {
    "dataset": "ORD",
    "language": "ru",
    "model_name": "Whisper Large Ru ORD 0.9 Peft PEFT 4-bit Q DoRA - Mizoru ",
    "finetuned_from": base_model,
    "tasks": "automatic-speech-recognition",
}
trainer.push_to_hub(**kwargs)

tokenizer._tokenizer.model.dropout = 0.

ds["test"] = ds["test"].map(prepare_val_dataset, num_proc=num_proc)
test_dataloader = trainer.get_test_dataloader(ds["test"])

eval_loop = trainer.prediction_loop if trainer.args.use_legacy_prediction_loop else trainer.evaluation_loop
output = eval_loop(
    test_dataloader, description="Prediction", metric_key_prefix="test"
)

predictions = decode_predictions(output)
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("predictions.csv")


s3 = get_repo_bucket_client("mizoru/ORD")

s3.upload_file(
    Bucket="ORD",  # name of the repo
    Key="predictions_trained.csv",
    Filename="predictions.csv",
)
