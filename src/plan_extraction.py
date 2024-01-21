import json
from collections import defaultdict

from nltk.stem.porter import PorterStemmer
from tqdm import tqdm

from tokenizer import Tokenizer
import os


def stemming(stemmer: PorterStemmer, tokens):
    stemmed_tokens = []
    for tok in tokens:
        stemmed_tokens.append(stemmer.stem(tok))
    return stemmed_tokens


def load_mentions(folder):
    obs2mentions = {}
    files = os.listdir(folder)
    for fil in files:
        obs = fil.replace(".txt", "")
        path = os.path.join(folder, fil)
        obs2mentions[obs] = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obs2mentions[obs].add(line.strip())
    return obs2mentions


def data_augmentation(report, pos):
    pass


def func(dataset, obs2mentions, pmi_obs2mentions=None):
    # stemmer = PorterStemmer()
    tag_path = "../CheXbert/src/data/%s/id2tag.csv" % dataset
    annotation = json.load(
        open("../%s/annotation.json" % dataset, "r", encoding="utf-8")
    )
    id2reports = {}
    clean_fn = (
        Tokenizer.clean_report_mimic_cxr
        if "mimic" in dataset
        else Tokenizer.clean_report_iu_xray
    )
    print(clean_fn)
    # clean_fn = Tokenizer.clean_report_iu_xray
    split2id = defaultdict(set)
    for split in ["train", "val", "test"]:
        id2reports.update(
            {
                report["id"]: clean_fn(report["report"])
                for report in tqdm(annotation[split], desc="Processing %s set" % split)
            }
        )
        for report in annotation[split]:
            split2id[split].add(report["id"])

    id2tagpos = {}
    id2tags, headers = Tokenizer.load_tag2ids(tag_path, need_header=True)
    progress = tqdm(id2reports, desc="Processing")
    tag_stat = defaultdict(int)
    for idx in progress:
        tokens = id2reports[idx].split()
        report = id2reports[idx]
        tags = id2tags[idx]
        tag2pos = {}
        report_len = len(report)
        for tag in tags:
            if idx in split2id["train"]:
                tag_stat[tag] += 1
            if tag == "No Finding":
                continue
            flag = True
            start = report_len
            for mention in obs2mentions[tag.lower().replace(" ", "_")]:
                if mention in report:
                    new_pos = report.index(mention)
                    start = min(new_pos, start)
                    flag = False
            if flag and pmi_obs2mentions is not None:
                for mentions in pmi_obs2mentions[tag]:
                    mentions = map(
                        lambda x: x[0],
                        sorted(
                            pmi_obs2mentions[tag].items(),
                            key=lambda x: -len(x[0].split()),
                        ),
                    )
                    for mention in mentions:
                        if mention in report:
                            new_pos = report.index(mention)
                            start = min(new_pos, start)
                            flag = False
                            break
            if flag:
                try:
                    inp = tag.lower()
                    start = report.index(inp)
                except Exception:
                    tokens = tag.lower().split()
                    for tok in tokens:
                        try:
                            new_pos = report.index(tok)
                            start = min(new_pos, start)
                        except Exception:
                            pass
            tag2pos[tag] = start
        if "No Finding" in tags:
            tag2pos["No Finding"] = -100000
        sorted_tag_pos = sorted(
            tag2pos.items(),
            key=lambda x: (x[1], x[0]),
        )
        tag2pos = {key: loc for loc, (key, pos) in enumerate(sorted_tag_pos)}
        if dataset == "iu_xray":
            flag1 = False
            flag2 = False
            if "Enlarged Cardiomediastinum" in tag2pos and "Cardiomegaly" in tag2pos:
                if (
                    abs(tag2pos["Enlarged Cardiomediastinum"] - tag2pos["Cardiomegaly"])
                    == 1
                ):
                    max_pos = max(
                        tag2pos["Enlarged Cardiomediastinum"], tag2pos["Cardiomegaly"]
                    )
                    min_pos = min(
                        tag2pos["Enlarged Cardiomediastinum"], tag2pos["Cardiomegaly"]
                    )
                    tag2pos["Cardiomegaly"] = min_pos
                    tag2pos["Enlarged Cardiomediastinum"] = max_pos
                    flag1 = True
            if "Pneumothorax" in tag2pos and "Pleural Effusion" in tag2pos:
                if abs(tag2pos["Pleural Effusion"] - tag2pos["Pneumothorax"]) == 1:
                    max_pos = max(tag2pos["Pleural Effusion"], tag2pos["Pneumothorax"])
                    min_pos = min(tag2pos["Pleural Effusion"], tag2pos["Pneumothorax"])
                    tag2pos["Pleural Effusion"] = min_pos
                    tag2pos["Pneumothorax"] = max_pos
                    flag2 = True
            if flag1 and flag2:
                four_pos = [
                    tag2pos["Cardiomegaly"],
                    tag2pos["Enlarged Cardiomediastinum"],
                    tag2pos["Pleural Effusion"],
                    tag2pos["Pneumothorax"],
                ]
                four_pos = sorted(four_pos)
                (
                    tag2pos["Cardiomegaly"],
                    tag2pos["Enlarged Cardiomediastinum"],
                    tag2pos["Pleural Effusion"],
                    tag2pos["Pneumothorax"],
                ) = four_pos
        id2tagpos[idx] = tag2pos

    with open(
        "./data/%s_id2tagpos.json" % dataset,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(id2tagpos, f, ensure_ascii=False, indent=4)
    for split in split2id:
        if split == "train":
            continue
        ids = split2id[split]
        seqs = set()
        for id_ in ids:
            seqs.add("-".join(id2tagpos[id_].keys()))
        print(split, len(seqs))


def load_pmi_mentions(path, topk=5):
    pmi_obs2mentions = defaultdict(dict)
    pmi_mentions = json.load(open(path, "r", encoding="utf-8"))
    for obs in pmi_mentions:
        if "No Finding" in obs:
            continue
        mentions = pmi_mentions[obs][:topk]
        obs = obs.split("_")[0]
        for priority, mention in enumerate(mentions):
            if mention not in pmi_obs2mentions[obs]:
                pmi_obs2mentions[obs][mention] = 10000
            pmi_obs2mentions[obs][mention] = min(
                priority, pmi_obs2mentions[obs][mention]
            )
    return pmi_obs2mentions


if __name__ == "__main__":
    # mention_folder = "./data/mention/"
    mention_folder = "./chexpert/phrases/mention"
    obs2mentions = load_mentions(mention_folder)
    pmi_obs2mentions = load_pmi_mentions(
        "./data/iu_xray_filter_pmi.json",
        topk=5,
    )
    func("iu_xray", obs2mentions, pmi_obs2mentions=pmi_obs2mentions)
