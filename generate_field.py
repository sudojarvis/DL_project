import datetime

def generate_field_types(data):
    field_name_types = {}
    data_labels = {"str": 0, "num": 0, "dt": 0}
    field_name_types_array = []

    for field_name in data[0]:
        current_label = next( (row[field_name] for row in data if row[field_name] is not None), None)
        if current_label is None:
            continue
        if isinstance(current_label, str):
            field_type="str"
        elif isinstance(current_label, (int, float)):
            field_type="num"
        elif isinstance(current_label, datetime.datetime):
            field_type="dt"
        else:
            raise ValueError("Unknown field type")
        
        replace_var=f"{field_type}{data_labels[field_type]}"
        data_labels[field_type] = data_labels[field_type] + 1
        field_name_types[field_name] = replace_var
        field_name_types_array.append({field_name: replace_var})
        
   
    return list(reversed(field_name_types_array))

