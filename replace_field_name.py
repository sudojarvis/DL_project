# def replace_field_names(source_data_spec, field_name_types):
#     for field_name_type in field_name_types:
#         field=list(field_name_type.keys())[0]

#         value=field_name_type[field]
#         # field,value = next(iter(field_name_type.items()))
#         replace_direction=True

#         if(replace_direction):
#             source_data_spec=str(source_data_spec).replace(f"{field}",f"{value}")
#         else:
#             source_data_spec=str(source_data_spec).replace(f"{value}",f"{field}")


#     # print(source_data_spec)
#     return source_data_spec

def update_field_names(source_data_spec, field_name_types):
    for field_name_type in field_name_types:
        field, value = next(iter(field_name_type.items()))

        replace_direction = True
        search_str = field if replace_direction else value
        replace_str = value if replace_direction else field

        source_data_spec = source_data_spec.replace(search_str, replace_str)

    return source_data_spec

