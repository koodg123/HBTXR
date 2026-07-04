import numpy as np

# show the statistics of the ViT model
d = np.load("type.npy", allow_pickle=True).item()
for name, (sign_type, bit_width) in d.items():
    # format with 20 spaces
    print(f"    {name:30} {sign_type:10} {bit_width:10}")

print("*" * 50)

d = np.load("range.npy", allow_pickle=True).item()
for name, value in d.items():
    # format with 20 spaces
    # the value is a dict, with each node name mapped to (min, max)
    print(f"    {name:30}")
    for node, (min, max) in value.items():
        print(f"        {node:20} {min:20} {max:20}")
    print()