"""Keep every proposal in exactly the Report_Gen format (key order, text layout, json_gen.py copy)."""
from __future__ import annotations

import json
import textwrap

# Same order as Report_Gen/json_gen.py -> data_to_save
KEY_ORDER = ["Version", "Year", "Semester", "project_name", "Objective", "Dataset", "Rationale", "Approach",
             "Timeline", "Expected Number Students", "Research Contributions", "Possible Issues",
             "Additional Resources", "Proposed by", "Proposed by email", "instructor", "instructor_email",
             "collaborator", "funding_opportunity", "github_repo"]

# Long text fields are stored like the triple-quoted blocks in json_gen.py:
# "\n" + text indented 12 spaces + "\n            "
LONG = ["Objective", "Dataset", "Rationale", "Approach", "Timeline", "Expected Number Students",
        "Research Contributions", "Possible Issues", "Additional Resources"]
# Short fields that json_gen.py writes as """...""" (single line)
TRIPLE_SHORT = ["Version", "Year", "Semester", "project_name"]
IND = " " * 12
SEP = "        # " + "-" * 119


def _long(text) -> str:
    t = textwrap.dedent(str(text or "").replace("\r\n", "\n")).strip()
    if not t:
        return ""
    return "\n" + textwrap.indent(t, IND) + "\n" + IND


def canonical(data: dict) -> dict:
    """Return data with the Report_Gen key order and text layout (idempotent)."""
    out = {}
    for k in KEY_ORDER:
        v = data.get(k, "")
        out[k] = _long(v) if k in LONG else " ".join(str(v or "").split())
    for k, v in data.items():          # keep any extra keys at the end
        if k not in out:
            out[k] = v
    return out


def input_json(data: dict) -> bytes:
    # identical to json_gen.save_to_json: json.dump(data, f, indent=2)
    return json.dumps(canonical(data), indent=2).encode("utf-8")


def _py_literal(v: str, triple: bool) -> str:
    if triple and '"""' not in v and "\\" not in v and not v.endswith('"'):
        return f'"""{v}"""'
    return json.dumps(v)               # safe, valid Python string literal


def json_gen_py(data: dict, base: str) -> bytes:
    """A json_gen.py in the same layout as Report_Gen/json_gen.py; base is 'Proposals' or 'External'."""
    d = canonical(data)
    body = []
    for k in KEY_ORDER:
        v = d[k]
        if k in LONG or k in TRIPLE_SHORT:
            body.append(SEP)
            body.append(f'        "{k}":\n            {_py_literal(v, True)},')
        else:
            if k == "Proposed by":
                body.append(SEP)
            body.append(f'        "{k}": {_py_literal(v, False)},')
    body.append(SEP)
    path = f"Arxiv{{os.sep}}{base}{{os.sep}}" + \
        '{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}'
    src = f'''import json
import os
import shutil


def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)


data_to_save = \\
    {{
{chr(10).join(body)}
    }}


os.makedirs(
    os.getcwd() + os.sep + f'{path}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'{path}{{os.sep}}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy(__file__, output_file_path)
print(f"Data saved to {{output_file_path}}")
'''
    return src.encode("utf-8")
