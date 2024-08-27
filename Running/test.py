class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def distance(self, other):  # Missing `self` argument
    return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

p1 = Point(1, 2)
p2 = Point(3, 4)

print(p1.distance(p2))
