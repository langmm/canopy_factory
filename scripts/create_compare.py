import yggdrasil_rapidjson as rapidjson

id_base = 'B73'
data_year = 2025
# planting_density = 91126.5168339014  # 2024
planting_density = 118134.34279710053  # 2025

output = {
    # 30 inch rows
    # 76.2 cm rows
    # 71712.34743198083 ha**-1
    f'{id_base} WT 71.7k plants/ha': {
        'row_spacing': 76.2,
        'id': f'{id_base}_WT',
        'data_year': data_year,
    },
    f'{id_base} rdla {planting_density / 1000:3.1f} plants/ha': {
        'planting_density': planting_density,
        'id': f'{id_base}_rdla',
        'data_year': data_year,
    }
}
# output = {
#     k.replace(' ', '_'): v for k, v in output.items()
# }

fname = f'compare_rows_{id_base}.json'

with open(fname, 'w') as fd:
    rapidjson.dump(
        output, fd,
        datetime_mode=rapidjson.DM_ISO8601,
        write_mode=rapidjson.WM_PRETTY
    )

print(f"WROTE {fname}")
