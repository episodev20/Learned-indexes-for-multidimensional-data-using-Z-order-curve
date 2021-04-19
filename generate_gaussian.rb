MIN = 0
MAX = 2**16-1

# thanks to: https://stackoverflow.com/a/6178290
def rnd_gaussian(mean, stddev)
  theta = 2 * Math::PI * rand
  rho = Math.sqrt(-2 * Math.log(1 - rand))
  scale = stddev * rho
  x = mean + scale * Math.cos(theta)
  y = mean + scale * Math.sin(theta)
  [x,y]
end

def adjust(val)
  val = MIN if val < MIN
  val = MAX if val > MAX
  val
end

def create_gaussian_dataset(size)
  size.times do
    x, y = rnd_gaussian(MIN+(MIN+MAX)/2.0, (MIN+MAX)*0.1)
    x = adjust(x.to_i)
    y = adjust(y.to_i)
    puts "#{x},#{y}"
  end
end

create_gaussian_dataset(ARGV.shift.to_i)
