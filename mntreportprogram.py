numerator = 617
denominator = 5000
max_bits = 503
x = numerator
bits = []
for i in range(max_bits):
  x *= 2
  bit = x // denominator
  bits.append(bit)
  x = x % denominator
  if x == 0:
    break
integer_part = numerator // denominator
print(f"{integer_part}.", end="")
for b in bits:
  print(b, end="")
print()
