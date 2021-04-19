MIN = 0
MAX = 2**16-1

def rnd_uniform(min, max)
  rand(min..max)
end

def create_uniform_dataset(size)
  size.times do
    x = rnd_uniform(MIN, MAX)
    y = rnd_uniform(MIN, MAX)
    puts "#{x},#{y}"
  end
end

create_uniform_dataset(ARGV.shift.to_i)
