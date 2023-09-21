from dataclasses import dataclass
import json
import re


from tqdm import tqdm

# # GermEval2014
# DIRECTORY = "/media/data/thielen/nlp-ha/datasets/GermEval2014/"
# FILENAMES = [
#     f"{DIRECTORY}NER-de-train.tsv",
#     f"{DIRECTORY}NER-de-dev.tsv",
#     f"{DIRECTORY}NER-de-test.tsv",
# ]
# OFFSET = 0
# IS_NESTED = True

# # MultiNERD
# DIRECTORY = "/media/data/thielen/nlp-ha/datasets/MultiNERD/de/"
# FILENAMES = [
#     f"{DIRECTORY}train_de.tsv",
#     f"{DIRECTORY}dev_de.tsv",
#     f"{DIRECTORY}test_de.tsv",
# ]
# OFFSET = -1
# IS_NESTED = False

# WikiANN
DIRECTORY = "/media/data/thielen/nlp-ha/datasets/WikiANN/"
FILENAMES = [
    f"{DIRECTORY}train.tsv",
    f"{DIRECTORY}test.tsv",
    f"{DIRECTORY}dev.tsv",
]

OFFSET = -1
IS_NESTED = False


@dataclass
class Span:
    start: int = None
    end: int = None
    tag: str = None


def main():
    for filename in FILENAMES:
        tokens = []
        output = []
        tags = {}
        outside_span = Span()
        inside_span = Span()
        off = OFFSET

        with open(filename, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile, desc=f"Processing {filename}"):
                line = line.strip()

                # empty line is sentence separator
                if not line:
                    if tokens:
                        context = " ".join(tokens)
                        context = re.sub(r"\s{2,}", " ", context)

                        output.append({"context": context, "label": tags})

                        tokens = []
                        tags = {}
                        off = OFFSET
                        outside_span = Span()
                        inside_span = Span()

                    continue

                # ignore comments
                if line.startswith("#"):
                    continue

                # split by tab
                idx, token, label_outside, *rest = line.split("\t")
                if IS_NESTED:
                    label_inside, *rest = rest

                # ignore non-printable characters and increment offset
                if not token.isprintable():
                    off += 1
                    continue

                # adjust index: 0-indexed and offset by non-printable characters
                idx = int(idx) - 1 - off

                # remember token for context
                tokens.append(token)

                if label_outside == "O":
                    if outside_span.start != None:
                        outside_span.end = idx - 1
                        tags[outside_span.tag].append(
                            f"{outside_span.start};{outside_span.end}"
                        )
                        outside_span = Span()

                    if inside_span.start != None:
                        inside_span.end = idx - 1
                        tags[inside_span.tag].append(
                            f"{inside_span.start};{inside_span.end}"
                        )
                        inside_span = Span()

                    continue

                # parse outside tag
                tag_out = re.search(r"(?<=-)[A-Z]{3,}", label_outside).group(0)
                if tag_out not in tags:
                    tags[tag_out] = []

                if outside_span.start == None:
                    outside_span.start = idx
                    outside_span.tag = tag_out

                if not IS_NESTED:
                    continue

                # parse inside tag
                if label_inside == "O":
                    if inside_span.start != None:
                        inside_span.end = idx
                        tags[inside_span.tag].append(
                            f"{inside_span.start};{inside_span.end}"
                        )
                        inside_span = Span()
                else:
                    tag_in = re.search(r"(?<=-)[A-Z]{3,}", label_inside).group(0)

                    if (
                        tag_in not in tags
                    ):  # should not be the case, since we have seen the outside tag
                        tags[tag_in] = []
                    if inside_span.start == None:
                        inside_span.start = idx
                        inside_span.tag = tag_in

        outfile = (
            filename.replace(".tsv", ".json").replace("NER-de-", "").replace("_de", "")
        )
        with open(outfile, "w", encoding="utf-8") as ofile:
            json.dump(output, ofile, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
