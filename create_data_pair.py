import os
import json
import random
from generate_field import generate_field_types
from replace_field_name import update_field_names




def generate_data_pair():
    all_sources_hold = []
    all_target_hold = []

    max_source_seq_length = 0
    max_target_seq_length = 0

    for subdir, dirs, files in os.walk('examples'):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith("vl.json"):
                with open(filepath, 'r') as f:
                    data = json.load(f)

                if "url" in data["data"]:
                    data_file_url = "examplesdata/" + data["data"]["url"].rsplit("/", 1)[-1]

                del data["_info"]
                if "_any" in data["encoding"]:
                    del data["encoding"]["_any"]
                del data["data"]
                del data["config"]

                # if "x" in data["encoding"] and "scale" in data["encoding"]["x"]:
                #     del data["encoding"]["x"]["scale"]

                if "x" in data["encoding"]:
                    if "scale" in data["encoding"]["x"]:
                        del data["encoding"]["x"]["scale"]

                # if "y" in data["encoding"] and "scale" in data["encoding"]["y"]:
                #     del data["encoding"]["y"]["scale"]

                if "y" in data["encoding"]:
                    if "scale" in data["encoding"]["y"]:
                        del data["encoding"]["y"]["scale"]


                target_vega_spec = json.dumps(data)
                target_vega_spec = target_vega_spec.replace(', "_any": false', "")

                if len(target_vega_spec) > max_target_seq_length:
                    max_target_seq_length = len(target_vega_spec)

                with open(data_file_url, 'r') as f:
                    data_content = json.load(f)

                data_holder = []
                # for i in range(0, 1):
                #     selected_index = random.randint(0, len(data_content) - 1)
                #     data_holder.append(data_content[selected_index])

                selected_index = random.randint(0, len(data_content) - 1)
                data_holder.append(data_content[selected_index])
                source_data_spec = json.dumps(data_holder)

                field_name_types = generate_field_types(data_content)

                for i in range(0, 50):
                    data_holder = []
                    for i in range(0, 1):
                        selected_index = random.randint(0, len(data_content) - 1)
                        data_holder.append(data_content[selected_index])
                    source_data_spec = json.dumps(data_holder)

                    target_vega_spec = update_field_names(target_vega_spec, field_name_types)
                    source_data_spec = update_field_names(source_data_spec, field_name_types)

                    if len(source_data_spec) > max_source_seq_length:
                        max_source_seq_length = len(source_data_spec)

                    all_sources_hold.append(source_data_spec)
                    all_target_hold.append(target_vega_spec)

    return all_sources_hold, all_target_hold, max_source_seq_length, max_target_seq_length



