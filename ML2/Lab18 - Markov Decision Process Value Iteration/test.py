x = [1, 2, 3, 4]
y = x.copy()

y[0] = 5
print(x)
print(y)

value_funcs_arr = [0, 0, 0, 0]
new_value_funcs_arr = [1, 2, 3, 4]
value_funcs_arr = new_value_funcs_arr.copy()
print(value_funcs_arr)
print(new_value_funcs_arr)

new_value_funcs_arr[0] = 100

print(value_funcs_arr)
print(new_value_funcs_arr)