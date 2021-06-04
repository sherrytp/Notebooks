# THE KARATSUBA ALGORITHM (Karatsuba multiplication)
class Karatsuba:

  def __init__(self):
    # convert input data to strings
    self.x_string = x.to_s
    self.y_string = y.to_s

    # define the longest string and set it up as 'n'
    n = [self.x_string.length, self.y_string.length].max
    # puts
    # puts("n = #{n}")
  def product(self, x, y, n):
    # return simple product if there are one-digit numbers
    if n == 1:
      return x * y
      # puts("return #{x} * #{y} = #{x * y}")
    # end

    # middle = (n.to_f/2).floor
    middle = (n.to_f/2).ceil

    # define a, b, c, d
    if n > self.x_string.length:
      a = 0
      b = x
    else:
      a = self.x_string[0 - middle].to_i
      b = self.x_string[(middle) - self.x_string.length].to_i
    # end

    if n > self.y_string.length:
      c = 0
      d = y
    else:
      c = self.y_string[0 - middle].to_i
      d = self.y_string[(middle) - self.y_string.length].to_i
    # end

    #print out defined attributes
    # puts '=========='
    # puts "x = #{x}, y = #{y}"
    # puts("n/2: #{n.to_f/2} => middle = #{middle}")
    # puts("x length: #{x_string.length}, y length: #{y_string.length}")
    # puts("a,b,c,d: #{a},#{b},#{c},#{d}")

    # recursive calculations
    ac = self.product(a, c)
    bd = self.product(b, d)
    # puts "a + b = #{a} + #{b} = #{a + b}"
    # puts "c + d = #{c} + #{d} = #{c + d}"
    gauss_trick = self.product(a + b, c + d)

    # puts '=========='
    # puts("ac - #{ac}")
    # puts("bd - #{bd}")
    # puts("(a+b) * (c+d) = #{gauss_trick}")
    # puts("(ad + bc) = #{gauss_trick} - #{ac} - #{bd} = #{gauss_trick - ac - bd}")
    gauss_trick = gauss_trick - ac - bd
    # puts("a+b = #{a+b}, c+d = #{c+d}")

    # the power must be always even, so make it this way
    pow = n / 2
    pow = pow * 2

    # puts '========== ac * 10 ^ n ===== (ad + bc) * 10 ^ (n/2) ===== bd'
    # puts ac * (10 ** pow)
    # puts gauss_trick * (10 ** (pow / 2))
    # puts bd
    # puts

    # the main calculation by Karatsuba method
    return ac * (10 ** pow) + gauss_trick * (10 ** (pow / 2)) + bd

  # end
# end